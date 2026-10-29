"""Shared pieces of the feature screener: data, folds, metrics, timing.

Everything reuses ML_layer.walk_forward (data loading, target construction, models)
and ML_layer.framework.metrics (IC, long-short, Newey-West) so numbers are
comparable with the existing walk-forward script.
"""

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path = [os.getcwd()] + sys.path

from .. import walk_forward as wf
from ..framework.metrics import daily_information_coefficient, newey_west_tstat, signal_metrics
from data_pipeline.edgar.features import FUNDAMENTAL_FEATURES
from data_pipeline.utils.features import (EARNINGS_FEATURES, INTERACTION_FEATURES, MACRO_BETA_FEATURES, MARKET_WIDE_FEATURES,
                                   OBSERVED_FEATURES, PRICE_ACTION_FEATURES, TS_NORMALISED_FEATURES,
                                   cross_sectional_rank)

SEED = 0
HORIZONS = (1, 5, 20)
REPO_ROOT = Path(__file__).resolve().parents[2]

# spy_ret_20d is identical for every ticker on a date, so a cross-sectional
# ranking metric cannot use it; it is left out of the screen.
EXCLUDED_MARKET_WIDE = ["spy_ret_20d"]
BASE_FEATURES = [f for f in OBSERVED_FEATURES if f not in EARNINGS_FEATURES and f not in MARKET_WIDE_FEATURES]
CANDIDATE_FEATURES = ["r_12_7", "log_mktcap", "log_turnover"]

FEATURE_GROUPS = {
    "reversal": ["log_return_1d", "log_return_5d", "log_return_20d"],
    "momentum": ["log_return_60d", "momentum_12_1", "dist_52w_high", "r_12_7"],
    "volatility_risk": ["volatility_20d", "idio_vol_60d", "beta_mkt", "beta_sector"],
    "volume_technical": ["volume_rel_20d", "high_ratio", "low_ratio", "rsi", "price_to_ma50"],
    "market_sector": ["sector_vs_market", "stock_vs_sector"],
    "price_action": list(PRICE_ACTION_FEATURES),
    "macro_beta": list(MACRO_BETA_FEATURES),
    "macro_interaction": list(INTERACTION_FEATURES),
    "ts_normalised": list(TS_NORMALISED_FEATURES),
    "fundamentals": list(FUNDAMENTAL_FEATURES),
    "size_liquidity": ["log_mktcap", "log_turnover", "log_dollar_volume"],
}


def groups_for(features: list[str]) -> dict[str, list[str]]:
    groups = {g: [f for f in members if f in features] for g, members in FEATURE_GROUPS.items()}
    return {g: m for g, m in groups.items() if m}


def y_col(h: int) -> str:
    return f"y_h{h:02d}"


def yr_col(h: int) -> str:
    return f"yr_h{h:02d}"


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------

STAGE_SECONDS: dict[str, float] = {}


@contextmanager
def stage(name: str, log: bool = False):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        STAGE_SECONDS[name] = STAGE_SECONDS.get(name, 0.0) + elapsed
        if log:
            print(f"[{elapsed:7.1f}s] {name}", flush=True)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def rank_features(samples: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Percentile-rank each feature among tickers on the same date, centred on 0 (same-date data only)."""
    return cross_sectional_rank(samples, features)


def load_fundamentals(directory: Path, tickers) -> pd.DataFrame:
    """Daily EDGAR fundamentals (data_pipeline.edgar.build) for the tickers that have a file; others stay missing."""
    frames = [pd.read_parquet(directory / f"{t}.parquet").assign(ticker=t) for t in tickers if (directory / f"{t}.parquet").exists()]
    return pd.concat(frames, ignore_index=True)


def load_screen_samples(
    data_directory: Path,
    ticker_file: Path,
    candidates: bool,
    cache_directory: Path,
    min_date: pd.Timestamp | None = None,
    extra_universes: tuple[tuple[Path, Path], ...] = (),
    drop_features: tuple[str, ...] = (),
    fundamentals_directory: Path | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Ranked feature table plus raw and rank-transformed targets. Returns (samples, features, notes).

    ``extra_universes`` are further (data directory, ticker file) pairs appended to the main universe
    (not compatible with ``candidates``, whose Yahoo inputs are loaded from the main directory only).
    """
    samples = wf.load_samples(data_directory, ticker_file)
    if extra_universes:
        assert not candidates, "candidate features are not supported with extra universes"
        samples = pd.concat([samples, *[wf.load_samples(d, t) for d, t in extra_universes]], ignore_index=True)
    notes: list[str] = []
    features = [f for f in BASE_FEATURES if f not in drop_features]
    if fundamentals_directory is not None:
        samples = samples.merge(load_fundamentals(fundamentals_directory, samples["ticker"].unique()), on=["ticker", "date"], how="left")
        features += FUNDAMENTAL_FEATURES
    if candidates:
        from .candidates import add_candidate_features

        samples, candidate_features, notes = add_candidate_features(samples, data_directory, cache_directory)
        features += candidate_features
    if min_date is not None:
        samples = samples[samples["date"] >= min_date]
    keep = ["ticker", "date", "forecast_date", *[y_col(h) for h in HORIZONS], *features]
    samples = samples[keep].sort_values(["date", "ticker"]).reset_index(drop=True)
    samples = rank_features(samples, features)
    for h in HORIZONS:
        samples[yr_col(h)] = samples.groupby("date")[y_col(h)].rank(pct=True) - 0.5
    samples["date_pos"] = samples["date"].rank(method="dense").astype(int) - 1
    samples["year"] = samples["forecast_date"].dt.year
    return samples, features, notes


# ----------------------------------------------------------------------
# Folds
# ----------------------------------------------------------------------

@dataclass
class Fold:
    year: int
    train: np.ndarray  # row indices
    test: np.ndarray


def make_folds(
    samples: pd.DataFrame,
    horizon: int,
    first_test_year: int,
    purge: bool = True,
    embargo: int | None = None,
    last_test_year: int | None = None,
) -> list[Fold]:
    """Expanding-window yearly folds (as in walk_forward.py) with purge and embargo.

    A training sample at trading-day position p has a target window ending at
    p + h. With ``purge`` it is kept only if p + h + embargo < first test
    position, so no training target window overlaps the test period or the
    embargo. ``embargo`` defaults to h days. Without purge, every earlier row
    is used (overlapping targets leak into the test year).
    """
    embargo = horizon if embargo is None else embargo
    pos = samples["date_pos"].to_numpy()
    year = samples["year"].to_numpy()
    last = year.max() if last_test_year is None else last_test_year
    folds = []
    for y in range(first_test_year, last + 1):
        test = year == y
        if not test.any():
            continue
        first_test_pos = pos[test].min()
        cutoff = first_test_pos - horizon - embargo if purge else first_test_pos
        folds.append(Fold(y, np.flatnonzero(pos < cutoff), np.flatnonzero(test)))
    return folds


def stride_rows(samples_pos: np.ndarray, rows: np.ndarray, stride: int) -> np.ndarray:
    """Keep rows on every ``stride``-th trading day (thins overlapping targets and cuts compute)."""
    return rows[samples_pos[rows] % stride == 0]


# ----------------------------------------------------------------------
# Metrics (thin wrappers around ML_layer.framework.metrics)
# ----------------------------------------------------------------------

def nw_lags(h: int) -> int:
    """Newey-West lags: h-1 for overlapping h-day targets, never fewer than 1."""
    return max(h - 1, 1)


def nw_t(values, h: int) -> float:
    return newey_west_tstat(np.asarray(values, dtype=float), lags=nw_lags(h))


def daily_ic(pred, y, dates) -> pd.Series:
    ok = np.isfinite(pred) & np.isfinite(y)
    return daily_information_coefficient(np.asarray(pred)[ok], np.asarray(y)[ok], np.asarray(dates)[ok])


def evaluate(pred, y, dates, h: int) -> dict[str, float]:
    """IC, NW t, IC-IR, share of positive-IC dates, long-short spread and annualised Sharpe at horizon h."""
    pred, y, dates = np.asarray(pred, float), np.asarray(y, float), np.asarray(dates)
    ok = np.isfinite(pred) & np.isfinite(y)
    pred, y, dates = pred[ok], y[ok], dates[ok]
    P = np.full((len(pred), max(HORIZONS)), np.nan)
    Y = P.copy()
    P[:, h - 1], Y[:, h - 1] = pred, y
    m = signal_metrics(P, Y, dates, (h,))
    ic = daily_information_coefficient(pred, y, dates)
    return {
        "ic": m[f"ic_h{h:02d}"],
        "ic_t": nw_t(ic.to_numpy(), h),
        "ic_ir": float(ic.mean() / ic.std(ddof=1)),
        "pct_pos": float((ic > 0).mean()),
        "ls_spread": m[f"ls_spread_h{h:02d}"],
        "ls_sharpe": m[f"ls_sharpe_h{h:02d}"],
    }


def ic_difference(pred_a, pred_b, y, dates, h: int) -> tuple[float, float]:
    """Mean daily IC(A) - IC(B) and its Newey-West t-statistic."""
    ic_a, ic_b = daily_ic(pred_a, y, dates), daily_ic(pred_b, y, dates)
    diff = (ic_a - ic_b).dropna()
    return float(diff.mean()), nw_t(diff.to_numpy(), h)
