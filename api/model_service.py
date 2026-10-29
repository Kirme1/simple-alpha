"""
ModelService: loads PTFRunner from a local checkpoint and serves inference requests.

The checkpoint is self-contained (see ML_layer/prepare_checkpoint.py), so the service
never reads the training data. Loading sequence (runs once at server startup):
  1. torch.load reads the checkpoint: weights, the ``model_config`` (features, window, hidden
     size, ... written at training time) and the ``serving`` bundle.
  2. A template TimeSeriesDataSet is rebuilt from the bundle's dataset parameters (encoders and
     scalers) and one dummy row group. It defines the model architecture and is the template
     predict() builds each request's dataset from.
  3. PTFRunner.setup("predict") builds self.ptf_model from the template; load_state_dict fills it.
  4. The bundle's per-horizon residual std gives the Q10/Q90 uncertainty bands (Gaussian
     approximation), and its ridge coefficients give the companion h1/h5/h20 forecast.
"""

import tempfile
import threading
import logging
from pathlib import Path

from .config import CHECKPOINT_PATH, FRED_API_KEY
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet

from data_pipeline.interfaces.tft_data_interface import add_target_excess_return
from data_pipeline.utils.cache import load_or_refresh_earnings
from data_pipeline.utils.config import load_tickers
from data_pipeline.utils.features import (
    EARNINGS_FEATURES, INTERACTION_FEATURES, KNOWN_CATEGORICAL_FEATURES, KNOWN_REAL_FEATURES, add_calendar_features,
    engineer_features,
)
from data_pipeline.utils.fred import FRED_FEATURES, fetch_fred_context, vix_zscore_from_market
from data_pipeline.yahoo import fetch_market_context, fetch_ohlcv
from ML_layer.prepare_checkpoint import SERVING_VERSION, build_runner, ranks_to_returns
from ML_layer.framework.ridge_forecaster import RidgeForecaster
from ML_layer.framework.runners import PTFRunner

logger = logging.getLogger(__name__)

# Covers predict()'s 2-year OHLCV fetch plus the FRED z-score's 1-year warmup.
_FRED_CACHE_YEARS = 4


class ModelService:
    def __init__(self) -> None:
        self.config: dict | None = None  # read from .env in load()
        self.template: TimeSeriesDataSet | None = None
        self.runner: PTFRunner | None = None
        self._residual_std: np.ndarray | None = None  # shape [forecast_horizon]
        self.ridge: RidgeForecaster | None = None
        self._rank_quantiles: np.ndarray | None = None  # set for xs_rank_target models
        self.model_cfg: dict | None = None  # read from the checkpoint in load()
        self._uses_fred = False
        self._uses_earnings = False
        # (day built, frame) — refreshed once per calendar day, see _fred_context()
        self._fred_cache: tuple[pd.Timestamp, pd.DataFrame] | None = None
        self._fred_lock = threading.Lock()

    def load(self) -> None:

        # Step 1: read the checkpoint; it carries the config and serving bundle it was exported with
        ckpt = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )
        cfg = ckpt.get("model_config")
        if cfg is None:
            raise RuntimeError(
                f"{CHECKPOINT_PATH} has no 'model_config'. It was saved before "
                "configs were embedded; retrain, or run `python -m ML_layer.prepare_checkpoint --checkpoint <path>` on it."
            )
        bundle = ckpt.get("serving")
        if bundle is None or bundle.get("version") != SERVING_VERSION:
            raise RuntimeError(
                f"{CHECKPOINT_PATH} has no current serving bundle. Run "
                "`python -m ML_layer.prepare_checkpoint` (needs the training data)."
            )
        if cfg.get("xs_rank_features"):
            # Ranking needs every ticker's features on the forecast date, but
            # predict() only builds features for the requested ticker.
            raise NotImplementedError(
                "Models trained with xs_rank_features cannot be served yet: "
                "the API builds features for one ticker at a time."
            )
        self.model_cfg = cfg
        if cfg.get("xs_rank_target"):
            if "rank_to_return" not in bundle:
                raise RuntimeError(
                    f"{CHECKPOINT_PATH} was trained with xs_rank_target but its serving bundle has no "
                    "'rank_to_return'. Re-run `python -m ML_layer.prepare_checkpoint` (needs the training data)."
                )
            self._rank_quantiles = np.asarray(bundle["rank_to_return"])
        # The interaction features are built from the VIX, so they need the FRED context too.
        self._uses_fred = any(f in cfg["features"] for f in FRED_FEATURES + INTERACTION_FEATURES)
        self._uses_earnings = any(f in cfg["features"] for f in EARNINGS_FEATURES)
        
        if self._uses_fred and not FRED_API_KEY:
            # Without a key only the VIX can be rebuilt (from yfinance); fail loudly rather
            # than let engineer_features silently skip other FRED features.
            unsupported = [f for f in FRED_FEATURES if f in cfg["features"] and f != "fred_vix"]
            if unsupported:
                raise RuntimeError(f"Model needs FRED_API_KEY for features {unsupported}.")
            logger.info("FRED_API_KEY not set: building fred_vix from yfinance ^VIX.")
        logger.info("Model config from checkpoint: %s", cfg)

        # Step 2: rebuild the dataset template from the stored parameters
        self.template = self._template_dataset(bundle["dataset_parameters"])

        # Step 3: build PTFRunner from the template and load the weights
        self.runner = build_runner(cfg, ckpt["state_dict"], self.template)

        # Step 4: uncertainty bands and the ridge companion model, both precomputed
        self._residual_std = np.asarray(bundle["residual_std"])
        self.ridge = RidgeForecaster.from_state(bundle["ridge"])

        # Step 5: warm the FRED cache so the first request doesn't pay for it
        if self._uses_fred and FRED_API_KEY:
            try:
                self._fred_context(pd.DatetimeIndex([pd.Timestamp.today().normalize()]), FRED_API_KEY)
            except Exception:
                # Only a warm-up: predict() fetches (and caches) the context itself on the first request.
                logger.warning("Could not pre-fetch FRED data at startup; the first request will fetch it.", exc_info=True)


    def _template_dataset(self, params: dict) -> TimeSeriesDataSet:

        """A tiny TimeSeriesDataSet carrying the training dataset's encoders and scalers.

        from_parameters needs some data to build from; it is one group of placeholder rows,
        so only the parameters matter. predict() builds each request's dataset from it.
        """
        encoders = params["categorical_encoders"]
        group_id = params["group_ids"][0]
        rows = params["max_encoder_length"] + params["max_prediction_length"]
        frame = pd.DataFrame({"time_idx": range(rows), params["target"]: 0.0})


        # Placeholder values for every input column; the first class an encoder knows is always valid.
        frame[group_id] = next(iter(encoders[f"__group_id__{group_id}"].classes_))
        for column in params["time_varying_unknown_reals"] + params["time_varying_known_reals"]:
            frame[column] = 0.0
        for column in params["time_varying_known_categoricals"]:
            frame[column] = next(iter(encoders[column].classes_))

        return TimeSeriesDataSet.from_parameters(params, frame, predict=True)


    def _fred_context(self, index: pd.DatetimeIndex, api_key: str) -> pd.DataFrame:
        """Return FRED features for ``index`` from a cache rebuilt once per day.

        fetch_fred_context only uses releases published strictly before each
        date, so a date's value never changes once computed; the only new
        information each day is today's row. Caching one frame over a fixed
        business-day window and reindexing is therefore equivalent to fetching
        for ``index`` directly.
        """
        today = pd.Timestamp.today().normalize()
        with self._fred_lock:
            cached = self._fred_cache
            if (
                cached is None
                or cached[0] != today
                or index.min() < cached[1].index.min()
            ):
                start = min(index.min(), today - pd.DateOffset(years=_FRED_CACHE_YEARS))
                context = fetch_fred_context(
                    pd.bdate_range(start, today), api_key
                )
                if context.empty:
                    raise RuntimeError("FRED returned no data; check FRED_API_KEY.")
                self._fred_cache = cached = (today, context)
        return cached[1].reindex(index)


    def predict(self, ticker: str, as_of_date: str) -> dict:

        # Fetch 2 years of OHLCV — enough for the 60-day encoder window plus MA50 warmup
        raw_df, sector = fetch_ohlcv(ticker, period="2y")
        _, indexes = load_tickers(Path(self.model_cfg['ticker_file']))

        # Start early enough for the VIX z-score's 1-year warmup plus the 2-year OHLCV window.
        market = fetch_market_context(indexes, start=pd.Timestamp.today() - pd.DateOffset(years=_FRED_CACHE_YEARS))
        raw_df = raw_df.join(market, how="left", validate="one_to_one")


        # The model was trained with FRED macro features, so they must be joined
        # here too; engineer_features silently skips any that are absent.
        if self._uses_fred:
            if FRED_API_KEY:
                fred = self._fred_context(raw_df.index, FRED_API_KEY)
            else:
                fred = vix_zscore_from_market(market).reindex(raw_df.index).to_frame()
            raw_df = raw_df.join(fred, how="left", validate="one_to_one")
        earnings = None


        if self._uses_earnings:
            # Weekly-refreshed Yahoo earnings history, cached under the temp directory
            earnings = load_or_refresh_earnings(ticker, Path(tempfile.gettempdir()), force=False)
        features = engineer_features(raw_df, sector, earnings=earnings)
        features = features.reset_index()
        features = features.rename(columns={features.columns[0]: "date"})
        features["date"] = pd.to_datetime(features["date"])
        features.insert(0, "ticker", ticker)
        features = add_target_excess_return(features)


        # Select observations rather than calendar days. This is robust to
        # weekends, market holidays, and other missing trading sessions.
        encoder_df = (
            features[features["date"] <= pd.Timestamp(as_of_date)]
            .tail(self.model_cfg['window_size'])
            .copy()
            .reset_index(drop=True)
        )

        if len(encoder_df) < self.model_cfg['window_size']:
            raise ValueError(
                f"as_of_date {as_of_date} yields only {len(encoder_df)} trading rows "
                f"after feature warmup; need at least {self.model_cfg['window_size']}."
            )

        encoder_df["time_idx"] = range(len(encoder_df))

        # TimeSeriesDataSet needs decoder rows even for inference. The TFT
        # has no decoder variables for this configuration, so carrying the
        # final observed values forward supplies the required structure
        # without exposing future market data.
        future_df = pd.concat(
            [encoder_df.iloc[[-1]].copy()] * self.model_cfg['forecast_horizon'],
            ignore_index=True,
        )
        future_df["time_idx"] = range(
            len(encoder_df),
            len(encoder_df) + self.model_cfg['forecast_horizon'],
        )
        future_df["date"] = pd.bdate_range(
            encoder_df["date"].iloc[-1] + pd.offsets.BDay(),
            periods=self.model_cfg['forecast_horizon'],
        )
        if self.model_cfg.get("calendar_inputs", True):
            calendar = add_calendar_features(pd.DatetimeIndex(future_df["date"]))
            for column in KNOWN_REAL_FEATURES + KNOWN_CATEGORICAL_FEATURES:
                future_df[column] = calendar[column].to_numpy()
        inference_df = pd.concat([encoder_df, future_df], ignore_index=True)

        inf_dataset = TimeSeriesDataSet.from_dataset(
            self.template,
            inference_df,
            predict=True,
            stop_randomization=True,
        )
        inf_loader = inf_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        with torch.no_grad():
            x, _ = next(iter(inf_loader))
            out = self.runner.ptf_model(x)

        # Median daily excess log returns -> cumulative log returns over SPY
        pred_daily = self.runner.point_forecast(out.prediction.squeeze(-1))  # [1, 20]
        if self._rank_quantiles is not None:
            # The model predicts each day's cross-sectional rank; map it back to a return.
            pred_daily = ranks_to_returns(pred_daily, self._rank_quantiles)
        q50 = self.runner.cumulative_returns(pred_daily)[0].tolist()     # [20]

        horizon = self.model_cfg['forecast_horizon']
        q10 = [q50[h] - 1.28 * float(self._residual_std[h]) for h in range(horizon)]
        q90 = [q50[h] + 1.28 * float(self._residual_std[h]) for h in range(horizon)]

        # Interpretability outputs from PTF forward pass
        interp = self.runner.ptf_model.interpret_output(
            out, reduction="none", attention_prediction_horizon=0
        )
        # encoder_variables shape: [B, n_enc_vars, 1] -> flatten to [n_enc_vars]
        encoder_vars = interp["encoder_variables"][0].squeeze(-1).tolist()
        # attention shape: [B, encoder_len]
        attention_weights = interp["attention"][0].tolist()

        encoder_dates = encoder_df["date"].dt.strftime("%Y-%m-%d").tolist()[-self.model_cfg['window_size']:]

        return {
            "encoder_var_names": list(self.runner.ptf_model.encoder_variables),
            "q50": q50,
            "q10": q10,
            "q90": q90,
            "ridge": self.ridge.predict(encoder_df),
            "encoder_vars": encoder_vars,
            "attention": attention_weights,
            "encoder_dates": encoder_dates,
            "encoder_end_date": encoder_dates[-1],
        }
