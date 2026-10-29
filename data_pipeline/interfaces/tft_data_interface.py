"""
LightningDataModule for pytorch-forecasting's TemporalFusionTransformer.

Loads un-windowed raw parquets ({ticker}_raw.parquet) produced by
data_pipeline/fetch_data.py and builds TimeSeriesDataSet instances.

The training TimeSeriesDataSet is exposed as self.train_dataset so that
PTFRunner can call TemporalFusionTransformer.from_dataset() in its setup().
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet
import numpy as np
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.data.encoders import TorchNormalizer

from ..edgar.features import FUNDAMENTAL_FEATURES
from .basic_data_interface import FORECAST_HORIZON, _assign_global_time_idx, add_fundamentals_and_start, fill_missing_fundamentals
from ..utils.config import load_tickers
from ..utils.features import KNOWN_CATEGORICAL_FEATURES, KNOWN_REAL_FEATURES, cross_sectional_rank

# The TFT forecasts each future day's log return in excess of SPY; summing the
# decoder steps gives the cumulative forward excess log return for every horizon.
TARGET = "target_excess_log_return"
TARGET_SOURCE_COLUMNS = ["target_log_close", "target_spy_log_close"]
# Daily excess return as realised, kept next to the model target when that is a cross-sectional rank.
RAW_TARGET = "raw_excess_log_return"


def add_target_excess_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 1-day log return over SPY realised on each row, per ticker.

    The first row of each ticker has no previous close and is dropped.
    """
    missing = set(TARGET_SOURCE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(
            f"Raw data is missing {sorted(missing)}; re-run `python -m data_pipeline.fetch_data` to rebuild features."
        )
    df = df.sort_values(["ticker", "date"])
    excess_log_level = df["target_log_close"] - df["target_spy_log_close"]
    df[TARGET] = excess_log_level.groupby(df["ticker"]).diff()
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


class PTFLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_directory: str,
        features: list[str],
        ticker_file: str,
        window_size: int,
        split_ratios: list[float],
        stride: int = 1,
        xs_rank_features: bool = False,
        xs_rank_target: bool = False,
        calendar_inputs: bool = True,
        fundamentals_directory: str | None = None,
        start_date: str | None = None,
        fold_test_year: int | None = None,
        split_seed: int = 42,
        apply_normalization: str | None = None,
        compute_stats: bool = False,
        data_transformations=None,
        batch_size: int = 64,
        num_workers: int = 4,
        load_test_split: bool = True,
    ):
        super().__init__()
        self.data_directory = data_directory
        self.features = features
        self.ticker_file = ticker_file
        self.window_size = window_size
        self.split_ratios = split_ratios
        self.stride = stride
        # Rank ticker-specific features against all tickers on the same date.
        self.xs_rank_features = xs_rank_features
        # "raw": the model target is each day's excess return. "xs_rank": it is that return's percentile rank
        # among tickers on the same date, centred on 0, so the model cannot learn volatility from the target scale.
        if not isinstance(xs_rank_target, bool):
            raise ValueError(f"xs_rank_target must be true or false, got {xs_rank_target!r}")
        self.xs_rank_target = xs_rank_target
        # Known-in-advance calendar inputs (day of week, month, days to month end, FOMC and CPI days). Without
        # them the decoder only sees a relative position in the forecast window.
        self.calendar_inputs = calendar_inputs
        # Daily EDGAR fundamentals (merged when `features` include them; missing values become the neutral rank)
        # and the first date used, so the split ratios cut only the range where the features exist.
        self.fundamentals_directory = fundamentals_directory
        self.start_date = start_date
        # Walk-forward fold: train before year Y-1, validate on Y-1, test on Y.
        # None keeps the split_ratios split.
        self.fold_test_year = fold_test_year
        self.batch_size = batch_size
        self.num_workers = num_workers
        # False skips the test dataset (and the copy of the data it keeps), saving memory when only
        # the train and validation splits are used.
        self.load_test_split = load_test_split

        self.train_dataset: TimeSeriesDataSet | None = None
        self.val_dataset: TimeSeriesDataSet | None = None
        self.test_dataset: TimeSeriesDataSet | None = None
        # Global time_idx -> trading date, to key predictions by date for evaluation.
        self.time_idx_dates: pd.Series | None = None

    def _read_raw(self, files: list[str]) -> pd.DataFrame:
        """Concatenate the raw parquets, keeping only the columns the model uses, in float32.

        The files hold every engineered feature, but a model uses a subset; reading the rest
        multiplies memory use several times over for hundreds of tickers. The two close columns
        stay float64 because the target is a difference of log levels.
        """
        wanted = {"ticker", "date", *TARGET_SOURCE_COLUMNS, *self.features}
        if self.calendar_inputs:
            wanted |= {*KNOWN_REAL_FEATURES, *KNOWN_CATEGORICAL_FEATURES}
        frames = []
        for file in files:
            columns = [c for c in pq.read_schema(file).names if c in wanted]
            frame = pd.read_parquet(file, columns=columns)
            float64 = [c for c in frame.select_dtypes("float64") if c not in TARGET_SOURCE_COLUMNS]
            frames.append(frame.astype({c: "float32" for c in float64}))
        return pd.concat(frames, ignore_index=True)

    def setup(self, stage: str | None = None) -> None:
        data_dir = Path(self.data_directory)
        tickers, _ = load_tickers(Path(self.ticker_file))
        raw_files = [data_dir / f"{ticker}_raw.parquet" for ticker in tickers]
        existing_files = [str(path) for path in raw_files if path.exists()]
        if len(existing_files) < len(raw_files):
            print(f'{len(existing_files)} out of {len(raw_files)} files were found in {self.data_directory}')

        if not raw_files:
            raise FileNotFoundError(
                f"No *_raw.parquet files in {self.data_directory}. "
                "Re-run `python -m data_pipeline.fetch_data` to generate them."
            )

        df = self._read_raw(existing_files)
        df = add_fundamentals_and_start(df, self.features, self.fundamentals_directory, self.start_date, self.xs_rank_features)
        df = _assign_global_time_idx(add_target_excess_return(df))
        df[RAW_TARGET] = df[TARGET]
        if self.xs_rank_target:
            df[TARGET] = df.groupby("date")[TARGET].rank(pct=True) - 0.5
        # Per-ticker realised daily excess returns by time_idx, for scoring against the real return.
        self._raw_returns = {
            ticker: block.set_index("time_idx")[RAW_TARGET] for ticker, block in df.groupby("ticker")
        }
        self.time_idx_dates = df.drop_duplicates("time_idx").set_index("time_idx")["date"].sort_index()
        known_reals = KNOWN_REAL_FEATURES if self.calendar_inputs else []
        known_categoricals = KNOWN_CATEGORICAL_FEATURES if self.calendar_inputs else []
        required = {"target_log_close", *self.features, *known_reals, *known_categoricals}
        missing_columns = required.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Raw data is missing required columns: {sorted(missing_columns)}")
        if self.xs_rank_features:
            # The model then sees each stock relative to its peers rather than
            # absolute levels that drift between regimes.
            df = cross_sectional_rank(df, self.features)
        df = fill_missing_fundamentals(df, self.features)

        if self.fold_test_year is None:
            max_time_idx = df["time_idx"].max()
            train_cutoff = int(max_time_idx * self.split_ratios[0])
            val_cutoff = int(max_time_idx * (self.split_ratios[0] + self.split_ratios[1]))
            # Last time_idx a test sample may start forecasting from (None: no limit).
            self.test_last_forecast_idx = None
            test_data_end = max_time_idx
        else:
            train_cutoff, val_cutoff, self.test_last_forecast_idx = self._fold_cutoffs(self.fold_test_year)
            # December forecasts of the test year still need their full horizon.
            test_data_end = self.test_last_forecast_idx + FORECAST_HORIZON

        # Datasets only build samples whose whole decoder lies inside the data
        # they are given, so training targets end at train_cutoff and
        # validation targets at val_cutoff: no target overlaps a later split.
        train_data = df[df["time_idx"] <= train_cutoff]
        val_data = df[df["time_idx"] <= val_cutoff]
        # Test samples forecast from val_cutoff + 1, so only their encoder
        # history is needed before it. Kept so evaluation can rebuild the test
        # set from modified inputs (permutation importance).
        self.val_cutoff = val_cutoff
        self.train_cutoff = train_cutoff
        if self.load_test_split:
            self.test_data = df[
                (df["time_idx"] > val_cutoff - self.window_size) & (df["time_idx"] <= test_data_end)
            ].reset_index(drop=True)

        self.train_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target=TARGET,
            group_ids=["ticker"],
            max_encoder_length=self.window_size,
            max_prediction_length=FORECAST_HORIZON,
            time_varying_unknown_reals=self.features,
            time_varying_known_reals=known_reals,
            time_varying_known_categoricals=known_categoricals,
            # The decoder needs at least one known input; the relative position is the same for every sample.
            add_relative_time_idx=not self.calendar_inputs,
            # EncoderNormalizer scales the target using each encoder window's
            # statistics — analogous to the baseline's per-window standardization.
            # No transformation: returns are signed, so relu/log would clip them.
            # A rank target already has a fixed scale, so it is left unnormalised.
            target_normalizer=(
                TorchNormalizer(method="identity", center=False) if self.xs_rank_target else EncoderNormalizer()
            ),
            allow_missing_timesteps=True,
        )

        if self.stride > 1:
            # Consecutive samples share all but one day, so a stride thins the
            # near-duplicate training windows. Aligning on the global time_idx
            # keeps the same forecast dates for every ticker. Val/test stay dense.
            self.train_dataset.filter(
                lambda index: index["time_idx_first_prediction"] % self.stride == 0, copy=False,
            )

        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset,
            val_data,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=train_cutoff + 1,
        )

        if self.load_test_split:
            self.test_dataset = self.build_test_dataset(self.test_data)

    def raw_cumulative_targets(self, tickers, first_time_idx) -> np.ndarray:
        """Realised cumulative excess log returns [n, FORECAST_HORIZON] from each sample's first forecast day.

        Row i sums the daily excess returns of ``tickers[i]`` over days ``first_time_idx[i]`` onward;
        horizons that run into a missing day are NaN. Used to score a model trained on ranked targets
        against the same raw targets as the baselines.
        """
        out = np.full((len(tickers), FORECAST_HORIZON), np.nan, dtype=np.float32)
        tickers, first_time_idx = np.asarray(tickers), np.asarray(first_time_idx)
        for ticker in np.unique(tickers):
            rows = np.flatnonzero(tickers == ticker)
            series = self._raw_returns[ticker]
            full = series.reindex(np.arange(series.index.min(), series.index.max() + FORECAST_HORIZON + 1)).to_numpy()
            offset = series.index.min()
            for step in range(FORECAST_HORIZON):
                position = first_time_idx[rows] - offset + step
                inside = (position >= 0) & (position < len(full))
                values = np.full(len(rows), np.nan)
                values[inside] = full[position[inside]]
                out[rows, step] = values
        return np.cumsum(out, axis=1)

    def _fold_cutoffs(self, test_year: int) -> tuple[int, int, int]:
        """(train_cutoff, val_cutoff, last test forecast time_idx) for a walk-forward fold.

        Training data ends with year Y-2, validation forecasts start in Y-1, and
        test forecasts start in Y, matching ML_layer/walk_forward.py with
        holdout_years=1.
        """
        years = pd.Series(self.time_idx_dates.dt.year.to_numpy(), index=self.time_idx_dates.index)
        if not (years == test_year).any():
            raise ValueError(f"No trading dates in walk-forward test year {test_year}")
        if not (years < test_year - 1).any():
            raise ValueError(f"No training data before {test_year - 1} for walk-forward test year {test_year}")
        train_cutoff = int(years.index[years < test_year - 1].max())
        val_cutoff = int(years.index[years < test_year].max())
        test_last = int(years.index[years == test_year].max())
        return train_cutoff, val_cutoff, test_last

    def build_test_dataset(self, test_data: pd.DataFrame) -> TimeSeriesDataSet:
        """Test-split dataset over ``test_data``, using the training scalers and encoders."""
        dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset,
            test_data,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=self.val_cutoff + 1,
        )
        if self.test_last_forecast_idx is not None:
            last = self.test_last_forecast_idx
            dataset.filter(lambda index: index["time_idx_first_prediction"] <= last, copy=False)
        return dataset

    def train_dataloader(self) -> DataLoader:
        return self.train_dataset.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return self.val_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.test_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
