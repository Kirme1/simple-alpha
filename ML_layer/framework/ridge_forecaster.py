"""
RidgeForecaster: a linear companion forecast served next to the TFT.

It is fitted offline (see ML_layer/prepare_checkpoint.py) on the data the TFT was trained
on: the same raw parquets, tickers, feature list, window size and train split, all read from
the checkpoint's ``model_config``. Inputs are the per-window standardized features flattened
to one vector, the same representation as RidgeBaseline (see the note on the three ridge models in
baselines.py). The targets are cumulative excess
log returns over SPY at a few chosen horizons. Only the fitted coefficients are stored with
the checkpoint (``to_state``), so the API never loads the training data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge

from data_pipeline.utils.config import load_tickers
from data_pipeline.interfaces.basic_data_interface import _assign_global_time_idx, _make_windows, standardize_window
from data_pipeline.interfaces.tft_data_interface import TARGET_SOURCE_COLUMNS, add_target_excess_return

logger = logging.getLogger(__name__)

# 1-based forecast days the ridge model predicts.
RIDGE_HORIZONS = (1, 5, 20)


class RidgeForecaster:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._features: list[str] = []
        self._window_size = 0
        # Fitted linear map, set by fit() or from_state(); predict() applies it directly.
        self._coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None

    def fit(self, data_directory: str, model_cfg: dict) -> None:
        self._features = list(model_cfg["features"])
        self._window_size = model_cfg["window_size"]

        tickers, _ = load_tickers(Path(model_cfg["ticker_file"]))
        files = [Path(data_directory) / f"{t}_raw.parquet" for t in tickers]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(f"No *_raw.parquet files for the model's tickers in {data_directory}")

        # Only the columns ridge uses, in float32 (the close columns stay float64: they are differenced)
        wanted = {"ticker", "date", *TARGET_SOURCE_COLUMNS, *self._features}
        frames = []
        for file in files:
            frame = pd.read_parquet(file, columns=[c for c in pq.read_schema(file).names if c in wanted])
            float64 = [c for c in frame.select_dtypes("float64") if c not in TARGET_SOURCE_COLUMNS]
            frames.append(frame.astype({c: "float32" for c in float64}))

        # Same preprocessing and train cutoff as PTFLightningDataModule.setup
        df = _assign_global_time_idx(add_target_excess_return(pd.concat(frames, ignore_index=True)))
        del frames
        train_cutoff = int(df["time_idx"].max() * model_cfg["split_ratios"][0])
        train_df = df[df["time_idx"] <= train_cutoff]

        xs, ys = [], []
        for _, ticker_df in train_df.groupby("ticker"):
            ticker_df = ticker_df.sort_values("time_idx").reset_index(drop=True)
            x, y, _, _ = _make_windows(
                ticker_df, self._window_size, model_cfg.get("stride", 1), self._features
            )
            if len(x):
                xs.append(x.reshape(len(x), -1))
                ys.append(y[:, [h - 1 for h in RIDGE_HORIZONS]])
        x_train, y_train = np.concatenate(xs), np.concatenate(ys)

        finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
        self._model.fit(x_train[finite], y_train[finite])
        self._coef, self._intercept = self._model.coef_, self._model.intercept_
        logger.info(
            "Ridge fitted on %d windows x %d inputs (alpha=%g)",
            int(finite.sum()), x_train.shape[1], self.alpha,
        )

    def to_state(self) -> dict:
        """Everything predict() needs, as plain arrays, for storing in a checkpoint."""
        return {
            "alpha": self.alpha,
            "features": self._features,
            "window_size": self._window_size,
            "horizons": list(RIDGE_HORIZONS),
            "coef": self._coef,
            "intercept": self._intercept,
        }

    @classmethod
    def from_state(cls, state: dict) -> "RidgeForecaster":
        if list(state["horizons"]) != list(RIDGE_HORIZONS):
            raise ValueError(
                f"Checkpoint ridge horizons {state['horizons']} differ from {list(RIDGE_HORIZONS)}; re-export it."
            )
        forecaster = cls(alpha=state["alpha"])
        forecaster._features = list(state["features"])
        forecaster._window_size = state["window_size"]
        forecaster._coef = np.asarray(state["coef"])
        forecaster._intercept = np.asarray(state["intercept"])
        return forecaster

    def predict(self, encoder_df: pd.DataFrame) -> list[float]:
        """Cumulative excess log return at RIDGE_HORIZONS for the latest window."""
        window = encoder_df[self._features].to_numpy(dtype=np.float32)[-self._window_size:]
        if not np.isfinite(window).all():
            raise ValueError("Encoder window has missing feature values; cannot compute ridge forecast.")
        if self._coef is None:
            raise RuntimeError("RidgeForecaster is not fitted")
        x = standardize_window(window).reshape(-1)
        return (self._coef @ x + self._intercept).tolist()
