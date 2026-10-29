"""
LightningDataModule for the custom TFT baseline.

Reads un-windowed raw parquets ({ticker}_raw.parquet), aligns all tickers to a
shared global time_idx based on the union of trading dates, creates overlapping
windows on-the-fly, normalizes features per-window (mean=0, std=1), and
normalizes targets to unit variance using the training split's std only.

Target: cumulative log returns in excess of SPY,
log(close_{t+n} / close_t) - log(spy_{t+n} / spy_t), calculated from the stored
target_log_close and target_spy_log_close columns (which are not model inputs).
        divided by target_std computed from training windows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

from ..edgar.features import FUNDAMENTAL_FEATURES, fundamentals_start, load_daily_fundamentals
from ..utils.config import load_tickers
from ..utils.features import OBSERVED_FEATURES, cross_sectional_rank


FORECAST_HORIZON = 20
FEATURE_COLS = OBSERVED_FEATURES
NUM_FEATURES = len(FEATURE_COLS)  # derived so adding to FEATURE_COLS propagates automatically
MAX_DATE_GAP_DAYS = 5  # windows spanning more than this many calendar days per step are dropped


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def add_fundamentals_and_start(
    df: pd.DataFrame,
    feature_cols: list[str],
    fundamentals_directory: str | None = None,
    start_date: str | None = None,
    xs_rank_features: bool = False,
) -> pd.DataFrame:
    """Merge the daily EDGAR fundamentals the model uses into ``df`` and drop rows before the start date.

    The fundamentals columns in ``feature_cols`` are merged by ticker and date from ``fundamentals_directory``
    (``data_pipeline.edgar.build``). ``start_date`` is an ISO date, "fundamentals" (the first date half the
    tickers have fundamentals, as in the feature screen) or None (all history). It is applied before the
    global time index is built, so the split ratios then cut the remaining range.
    """
    df = df.assign(date=pd.to_datetime(df["date"]))
    fundamental_cols = [c for c in feature_cols if c in FUNDAMENTAL_FEATURES]
    tickers = list(df["ticker"].unique())
    if fundamental_cols or start_date == "fundamentals":
        if fundamentals_directory is None:
            raise ValueError("fundamentals_directory is required for fundamentals features or start_date: fundamentals")
    if fundamental_cols:
        if not xs_rank_features:
            raise ValueError("fundamentals features need xs_rank_features: true (missing values become the neutral rank)")
        df = df.merge(
            load_daily_fundamentals(fundamentals_directory, tickers)[["ticker", "date", *fundamental_cols]],
            on=["ticker", "date"], how="left",
        )
    if start_date is not None:
        start = fundamentals_start(fundamentals_directory, tickers) if start_date == "fundamentals" else pd.Timestamp(start_date)
        df = df[df["date"] >= start].reset_index(drop=True)
    return df


def fill_missing_fundamentals(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Set missing (already ranked, 0-centred) fundamentals to 0, the neutral rank, as the ridge in the screen does.

    Tickers without EDGAR history, dates before a ticker's first filing, and stale values all end up neutral.
    """
    fundamental_cols = [c for c in feature_cols if c in FUNDAMENTAL_FEATURES]
    if not fundamental_cols:
        return df
    return df.assign(**{c: df[c].fillna(0.0) for c in fundamental_cols})


def _assign_global_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace per-ticker time_idx (if present) with a shared global one.

    Builds the union of all trading dates across tickers and maps each date to
    a single integer. This ensures that the same calendar date has the same
    time_idx in every ticker, so chronological train/val/test splits are
    date-aligned rather than row-count-aligned.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    df["time_idx"] = df["date"].map(date_to_idx)
    return df


def standardize_window(window: np.ndarray) -> np.ndarray:
    """Per-window, per-feature standardization (zero mean, unit std over the window's days)."""
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-8
    return (window - mean) / std


def _make_windows(
    ticker_df: pd.DataFrame,
    window_size: int,
    stride: int,
    feature_cols: list[str] = FEATURE_COLS,
    standardize: bool = True,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over a single ticker's rows (already sorted by time_idx).
    feature_cols defaults to FEATURE_COLS.

    Skips windows that span a date gap (e.g. a ticker was halted), ensuring
    only contiguous trading-day sequences are used.

    Returns:
        X: [num_windows, window_size, len(feature_cols)] — per-window standardized unless ``standardize`` is False
        y: [num_windows, FORECAST_HORIZON] — raw cumulative excess log returns
        forecast_dates: [num_windows] — date of each window's first forecast day
        forecast_time_idx: [num_windows] — global time_idx of that day
    """

    feature_cols = feature_cols or FEATURE_COLS
    num_features = len(feature_cols)
    features = ticker_df[feature_cols].values.astype(np.float32)

    # Differences of this series are log returns in excess of SPY.
    log_close = (ticker_df["target_log_close"] - ticker_df["target_spy_log_close"]).values.astype(np.float32)
    dates = ticker_df["date"].values  # numpy datetime64
    time_idx = ticker_df["time_idx"].values

    n = len(features)
    max_start = n - window_size - FORECAST_HORIZON
    if max_start < 0:
        return (
            np.empty((0, window_size, num_features), dtype=np.float32),
            np.empty((0, FORECAST_HORIZON), dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype=np.int64),
        )

    Xs, ys, forecast_dates, forecast_time_idx = [], [], [], []
    one_day = np.timedelta64(1, "D")

    for start in range(0, max_start + 1, stride):
        end = start + window_size

        # Reject windows with internal date gaps (halted trading, missing data)
        window_dates = dates[start:end]
        max_step_days = int((np.diff(window_dates).max()) / one_day)
        if max_step_days > MAX_DATE_GAP_DAYS:
            continue

        window = features[start:end].copy()
        Xs.append(standardize_window(window) if standardize else window)

        # Excess return over SPY = log_close[end-1+n] - log_close[end-1]
        ref = log_close[end - 1]
        ys.append(log_close[end : end + FORECAST_HORIZON] - ref)
        forecast_dates.append(dates[end])
        forecast_time_idx.append(time_idx[end])

    if not Xs:
        return (
            np.empty((0, window_size, num_features), dtype=np.float32),
            np.empty((0, FORECAST_HORIZON), dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.array(Xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        np.array(forecast_dates, dtype="datetime64[ns]"),
        np.array(forecast_time_idx, dtype=np.int64),
    )


def _prepare_frame(
    data_directory: str,
    split_ratios: list[float],
    ticker_file: str | None = None,
    feature_cols: list[str] = FEATURE_COLS,
    xs_rank_features: bool = False,
    fundamentals_directory: str | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Slim frame of every ticker with a shared global time_idx, and the (train, val) time_idx cutoffs.

    Only the model inputs, target columns and keys are kept so the frame stays small.
    ``ticker_file`` restricts the tickers (as the TFT's data module does); otherwise every raw file is read.
    ``xs_rank_features`` ranks the features among tickers on each date, as the TFT's data module does.
    ``fundamentals_directory`` / ``start_date``: see ``add_fundamentals_and_start``.
    """
    data_dir = Path(data_directory)
    if ticker_file is not None:
        tickers, _ = load_tickers(Path(ticker_file))
        raw_files = [f for f in (data_dir / f"{t}_raw.parquet" for t in tickers) if f.exists()]
    else:
        raw_files = sorted(data_dir.glob("*_raw.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            f"No *_raw.parquet files found in {data_directory}. "
            "Run `python -m data_pipeline.fetch_data` first."
        )
    columns = ["ticker", "date", "target_log_close", "target_spy_log_close",
               *[c for c in feature_cols if c not in FUNDAMENTAL_FEATURES]]
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in raw_files], ignore_index=True)
    df = add_fundamentals_and_start(df, feature_cols, fundamentals_directory, start_date, xs_rank_features)
    df = _assign_global_time_idx(df)
    if xs_rank_features:
        df = cross_sectional_rank(df, feature_cols)
    df = fill_missing_fundamentals(df, feature_cols)
    max_idx = df["time_idx"].max()
    return df, int(max_idx * split_ratios[0]), int(max_idx * (split_ratios[0] + split_ratios[1]))


def _iter_test_windows(df: pd.DataFrame, window_size: int, val_cutoff: int,
                       feature_cols: list[str] = FEATURE_COLS, standardize: bool = True,
                       until: int | None = None):
    """Yield (ticker, X, y, forecast dates) of the test windows one ticker at a time.

    Test windows are those whose first forecast day lies after ``val_cutoff``. The same windows are
    built whether they are collected or streamed, so only one ticker's windows are in memory at once.
    With ``until`` the data stops there, so every window's targets end by then: ``val_cutoff`` and
    ``train_cutoff`` as ``until`` and ``after`` give the validation windows.
    """
    for ticker, ticker_df in df.groupby("ticker"):
        ticker_df = ticker_df.sort_values("time_idx").reset_index(drop=True)
        if until is not None:
            ticker_df = ticker_df[ticker_df["time_idx"] <= until].reset_index(drop=True)
        # Full series; windowing only uses rows with enough future targets.
        X_te, y_te, dates_te, idx_te = _make_windows(ticker_df, window_size, 1, feature_cols, standardize)
        new_te = idx_te > val_cutoff
        if new_te.any():
            yield ticker, X_te[new_te], y_te[new_te], dates_te[new_te]


def _load_and_split(
    data_directory: str,
    window_size: int,
    stride: int,
    split_ratios: list[float],
    include_val: bool = True,
    include_test: bool = True,
    ticker_file: str | None = None,
    feature_cols: list[str] = FEATURE_COLS,
    xs_rank_features: bool = False,
    standardize_windows: bool = True,
    fundamentals_directory: str | None = None,
    start_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]:
    """
    Load the raw parquets, assign a shared global time_idx, split by date
    cutoff, and window each ticker's split independently.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, target_std, test_keys).
    target_std is computed from training targets only. test_keys has one
    (ticker, date) row per test window, date being its first forecast day.
    With ``include_val=False`` / ``include_test=False`` that split's windows are not built (its
    arrays come back empty), which saves memory when only the other splits are needed; the test
    windows can then be streamed with ``_iter_test_windows``.
    """
    df, train_cutoff, val_cutoff = _prepare_frame(
        data_directory, split_ratios, ticker_file, feature_cols, xs_rank_features, fundamentals_directory, start_date,
    )

    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []
    test_keys = []

    for ticker, ticker_df in df.groupby("ticker"):
        ticker_df = ticker_df.sort_values("time_idx").reset_index(drop=True)

        train_rows = ticker_df[ticker_df["time_idx"] <= train_cutoff]
        # Val windows need a lookback buffer of window_size rows before their start
        val_rows = ticker_df[ticker_df["time_idx"] <= val_cutoff]

        # Stride only thins the heavily overlapping training windows; val/test
        # keep every forecast date so they match the TFT's evaluation samples.
        X_tr, y_tr, _, _ = _make_windows(train_rows, window_size, stride, feature_cols, standardize_windows)
        if len(X_tr): X_trains.append(X_tr); y_trains.append(y_tr)

        # Keep only windows whose first forecast day lies after the previous
        # split, as PTF's min_prediction_idx does. Filtering on time_idx rather
        # than window position stays correct when windows are skipped (date gaps).
        if include_val:
            X_va, y_va, _, idx_va = _make_windows(val_rows, window_size, 1, feature_cols, standardize_windows)
            new_va = idx_va > train_cutoff
            X_va, y_va = X_va[new_va], y_va[new_va]
            if len(X_va): X_vals.append(X_va); y_vals.append(y_va)

    if include_test:
        for ticker, X_te, y_te, dates_te in _iter_test_windows(df, window_size, val_cutoff, feature_cols, standardize_windows):
            X_tests.append(X_te); y_tests.append(y_te)
            test_keys.append(pd.DataFrame({"ticker": ticker, "date": dates_te}))

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_val   = np.concatenate(X_vals,  axis=0) if X_vals  else np.empty((0, window_size, len(feature_cols)), dtype=np.float32)
    y_val   = np.concatenate(y_vals,  axis=0) if y_vals  else np.empty((0, FORECAST_HORIZON), dtype=np.float32)
    X_test  = np.concatenate(X_tests, axis=0) if X_tests else np.empty((0, window_size, len(feature_cols)), dtype=np.float32)
    y_test  = np.concatenate(y_tests, axis=0) if y_tests else np.empty((0, FORECAST_HORIZON), dtype=np.float32)

    target_std = float(y_train.std()) + 1e-8
    test_keys = (
        pd.concat(test_keys, ignore_index=True) if test_keys
        else pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, target_std, test_keys


class BasicLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_directory: str,
        window_size: int,
        stride: int,
        split_ratios: list[float],
        split_seed: int = 42,
        apply_normalization: str | None = None,
        compute_stats: bool = False,
        data_transformations=None,
        batch_size: int = 64,
        num_workers: int = 4,
        include_val: bool = True,
        stream_test: bool = False,
        ticker_file: str | None = None,
        pin_memory: bool = True,
        features: list[str] | None = None,
        xs_rank_features: bool = False,
        standardize_windows: bool = True,
        stream_split: str = "test",
        fundamentals_directory: str | None = None,
        start_date: str | None = None,
    ):
        super().__init__()
        self.data_directory = data_directory
        self.include_val = include_val
        # Do not hold the dense test windows in memory; stream them per ticker with iter_test_batches().
        self.stream_test = stream_test
        # Split that iter_test_batches() streams: "test", or "val" (the windows the TFT validates on).
        if stream_split not in ("test", "val"):
            raise ValueError(f"stream_split must be 'test' or 'val', got {stream_split!r}")
        self.stream_split = stream_split
        self.ticker_file = ticker_file
        self.pin_memory = pin_memory
        # Inputs default to every observed feature; the other options let a baseline see the TFT's inputs.
        self.features = list(features) if features is not None else list(FEATURE_COLS)
        self.xs_rank_features = xs_rank_features
        self.standardize_windows = standardize_windows
        # Daily EDGAR fundamentals (merged when `features` include them) and the first date used; see add_fundamentals_and_start.
        self.fundamentals_directory = fundamentals_directory
        self.start_date = start_date
        self.window_size = window_size
        self.stride = stride
        self.split_ratios = split_ratios
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.target_std: float | None = None
        # (ticker, first forecast date) per test sample, aligned with test_dataset
        self.test_keys: pd.DataFrame | None = None
        self.train_dataset: WindowDataset | None = None
        self.val_dataset: WindowDataset | None = None
        self.test_dataset: WindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test, target_std, test_keys = _load_and_split(
            self.data_directory, self.window_size, self.stride, self.split_ratios, self.include_val,
            include_test=not self.stream_test, ticker_file=self.ticker_file,
            feature_cols=self.features, xs_rank_features=self.xs_rank_features,
            standardize_windows=self.standardize_windows,
            fundamentals_directory=self.fundamentals_directory, start_date=self.start_date,
        )
        self.target_std = target_std
        self.test_keys = test_keys

        # Normalize targets to unit variance using training std only
        self.train_dataset = WindowDataset(X_train, y_train / target_std)
        self.val_dataset   = WindowDataset(X_val,   y_val   / target_std)
        self.test_dataset  = None if self.stream_test else WindowDataset(X_test, y_test / target_std)

    def iter_test_batches(self, batch_size: int | None = None):
        """Stream the test split as (X, y/target_std) batches, one ticker's windows in memory at a time.

        Yields exactly the windows ``test_dataset`` would hold, in the same order; ``test_keys`` is
        complete once the generator is exhausted. With ``stream_split="val"`` it yields the validation
        windows instead (first forecast day after the train split, targets inside the val split).
        """
        batch_size = batch_size or self.batch_size
        df, train_cutoff, val_cutoff = _prepare_frame(
            self.data_directory, self.split_ratios, self.ticker_file, self.features, self.xs_rank_features,
            self.fundamentals_directory, self.start_date,
        )
        after, until = (train_cutoff, val_cutoff) if self.stream_split == "val" else (val_cutoff, None)
        keys = []
        for ticker, X, y, dates in _iter_test_windows(df, self.window_size, after, self.features, self.standardize_windows, until):
            keys.append(pd.DataFrame({"ticker": ticker, "date": dates}))
            for start in range(0, len(X), batch_size):
                yield (torch.from_numpy(X[start:start + batch_size]).float(),
                       torch.from_numpy(y[start:start + batch_size] / self.target_std).float())
        self.test_keys = pd.concat(keys, ignore_index=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
