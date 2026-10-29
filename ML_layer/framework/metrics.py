"""Metrics shared by the PTFRunner, the baselines and the evaluation scripts. Everything is numpy/pandas.

Point-forecast metrics take cumulative returns [N, H]. The signal metrics further down are cross-sectional:
see the note above them.
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252.0


def directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of predictions with correct sign across all horizons."""
    return float(np.mean(np.sign(pred) == np.sign(target)))


def sharpe_ratio(pred: np.ndarray, target: np.ndarray, horizon_idx: int = 0, annualization: float = ANNUALIZATION) -> float:
    """
    Annualised Sharpe of a simple long/short strategy on a single forecast horizon.
    Position: +1 if pred > 0 else -1. P&L = position * actual_return.
    """
    pnl = np.sign(pred[:, horizon_idx]) * target[:, horizon_idx]
    return float(pnl.mean() / (pnl.std(ddof=1) + 1e-8) * np.sqrt(annualization))


def point_forecast_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """The ``test_*`` metrics logged for every model (except test_loss, which is each model's own loss).

    Includes per-horizon MSE and direction accuracy, zero-padded (h01..h20) so MLflow sorts them correctly.
    """
    metrics = {
        "test_mae": float(np.mean(np.abs(pred - target))),
        "test_directional_accuracy": directional_accuracy(pred, target),
        "test_sharpe": sharpe_ratio(pred, target, horizon_idx=0),
    }
    for h in range(pred.shape[1]):
        metrics[f"test_mse_h{h + 1:02d}"] = float(np.mean((pred[:, h] - target[:, h]) ** 2))
        metrics[f"test_dir_acc_h{h + 1:02d}"] = directional_accuracy(pred[:, h], target[:, h])
    return metrics


# ----------------------------------------------------------------------
# Cross-sectional signal metrics (used by evaluate.py and the validation epoch)
#
# Predictions for different tickers on the same date are strongly correlated
# and consecutive horizons overlap, so these metrics aggregate per date and
# use Newey-West standard errors rather than treating samples as independent.
# ----------------------------------------------------------------------

SIGNAL_HORIZONS = (1, 5, 20)  # 1-based forecast days the signal metrics are reported for
MIN_TICKERS_PER_DATE = 10
LONG_SHORT_QUANTILE = 0.2


def newey_west_tstat(values: np.ndarray, lags: int) -> float:
    """t-statistic of the mean of a time series with Newey-West (Bartlett) errors."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("nan")
    centred = values - values.mean()
    variance = centred @ centred / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        variance += 2.0 * weight * (centred[lag:] @ centred[:-lag]) / n
    if variance <= 0:
        return float("nan")
    return float(values.mean() / np.sqrt(variance / n))


def daily_information_coefficient(pred: np.ndarray, target: np.ndarray, dates: np.ndarray) -> pd.Series:
    """Spearman rank correlation between prediction and outcome across tickers, per date."""
    frame = pd.DataFrame({"date": dates, "pred": pred, "target": target})
    counts = frame.groupby("date")["pred"].transform("size")
    frame = frame[counts >= MIN_TICKERS_PER_DATE]
    by_date = frame["date"]
    ranks = frame.groupby("date")[["pred", "target"]].rank()
    centred = ranks - ranks.groupby(by_date).transform("mean")
    covariance = (centred["pred"] * centred["target"]).groupby(by_date).sum()
    variance = (centred ** 2).groupby(by_date).sum()
    ic = covariance / np.sqrt(variance["pred"] * variance["target"])
    return ic.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def mean_daily_ic(pred: np.ndarray, target: np.ndarray, dates: np.ndarray) -> float:
    """Mean over dates of the daily IC (dates with fewer than MIN_TICKERS_PER_DATE samples are skipped)."""
    return float(daily_information_coefficient(pred, target, dates).mean())


def daily_long_short_spread(pred: np.ndarray, target: np.ndarray, dates: np.ndarray) -> pd.Series:
    """Mean outcome of the top prediction quantile minus the bottom quantile, per date."""
    frame = pd.DataFrame({"date": dates, "pred": pred, "target": target})
    counts = frame.groupby("date")["pred"].transform("size")
    frame = frame[counts >= MIN_TICKERS_PER_DATE]
    pct = frame.groupby("date")["pred"].rank(pct=True)
    top = frame[pct > 1 - LONG_SHORT_QUANTILE].groupby("date")["target"].mean()
    bottom = frame[pct <= LONG_SHORT_QUANTILE].groupby("date")["target"].mean()
    return (top - bottom).dropna().sort_index()


def signal_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    horizons: tuple[int, ...] = SIGNAL_HORIZONS,
    annualization: float = 252.0,
) -> dict[str, float]:
    """IC, out-of-sample R² against a zero forecast, and long-short spread per horizon.

    ``pred``/``target`` are cumulative returns [N, H] in raw units; ``dates`` [N]
    is the first forecast date of each sample. For horizon h, Newey-West uses
    h-1 lags (overlapping outcomes) and the spread Sharpe uses every h-th date
    so that its periods do not overlap.
    """
    metrics = {}
    for h in horizons:
        p, y = pred[:, h - 1], target[:, h - 1]
        ic = daily_information_coefficient(p, y, dates)
        spread = daily_long_short_spread(p, y, dates)
        non_overlapping = spread.iloc[::h]
        metrics[f"ic_h{h:02d}"] = float(ic.mean())
        metrics[f"ic_tstat_h{h:02d}"] = newey_west_tstat(ic.to_numpy(), lags=h - 1)
        metrics[f"r2_oos_h{h:02d}"] = float(1.0 - np.sum((y - p) ** 2) / np.sum(y ** 2))
        metrics[f"mse_h{h:02d}"] = float(np.mean((y - p) ** 2))
        metrics[f"ls_spread_h{h:02d}"] = float(spread.mean())
        metrics[f"ls_sharpe_h{h:02d}"] = float(
            non_overlapping.mean() / (non_overlapping.std(ddof=1) + 1e-12) * np.sqrt(annualization / h)
        )
    return metrics


def paired_ic_difference(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    horizons: tuple[int, ...] = SIGNAL_HORIZONS,
) -> dict[str, float]:
    """Mean daily IC of model A minus model B, with a Newey-West t-statistic."""
    metrics = {}
    for h in horizons:
        ic_a = daily_information_coefficient(pred_a[:, h - 1], target[:, h - 1], dates)
        ic_b = daily_information_coefficient(pred_b[:, h - 1], target[:, h - 1], dates)
        diff = (ic_a - ic_b).dropna()
        metrics[f"ic_diff_h{h:02d}"] = float(diff.mean())
        metrics[f"ic_diff_tstat_h{h:02d}"] = newey_west_tstat(diff.to_numpy(), lags=h - 1)
    return metrics
