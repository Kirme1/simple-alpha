"""Univariate screen: daily rank IC of each (rank-transformed) feature, redundancy, per-year IC."""

import numpy as np
import pandas as pd

from .common import daily_ic, nw_t, y_col


def univariate_screen(test: pd.DataFrame, features: list[str], h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample (test-fold dates only) IC statistics. Returns (summary, per-year IC)."""
    rows, yearly = {}, {}
    dates = test["forecast_date"].to_numpy()
    for f in features:
        ic = daily_ic(test[f].to_numpy(), test[y_col(h)].to_numpy(), dates)
        per_year = ic.groupby(ic.index.year).mean()
        sign = np.sign(ic.mean())
        rows[f] = {
            "ic": ic.mean(),
            "ic_t": nw_t(ic.to_numpy(), h),
            "ic_ir": ic.mean() / ic.std(ddof=1),
            "pct_pos": (ic > 0).mean(),
            "n_dates": len(ic),
            # Share of calendar years whose mean IC has the sign of the overall mean, and the count of sign changes.
            "year_sign_share": float((np.sign(per_year) == sign).mean()),
            "year_sign_flips": int((np.sign(per_year).diff().abs() > 0).sum()),
        }
        yearly[f] = per_year
    return pd.DataFrame(rows).T, pd.DataFrame(yearly).T


def average_rank_correlation(test: pd.DataFrame, features: list[str], date_stride: int = 5) -> pd.DataFrame:
    """Cross-sectional rank correlation between features, averaged over dates (every ``date_stride``-th date)."""
    dates = np.sort(test["date"].unique())[::date_stride]
    subset = test[test["date"].isin(dates)]
    total = np.zeros((len(features), len(features)))
    count = np.zeros_like(total)
    for _, block in subset.groupby("date"):
        corr = block[features].corr(method="spearman", min_periods=10).to_numpy()
        ok = np.isfinite(corr)
        total[ok] += corr[ok]
        count[ok] += 1
    return pd.DataFrame(total / np.maximum(count, 1), index=features, columns=features)


def max_abs_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """Largest |rho| with any other feature, and which one."""
    off = corr.abs().where(~np.eye(len(corr), dtype=bool))
    partner = off.idxmax(axis=1)
    return pd.DataFrame({"max_abs_corr": off.max(axis=1), "max_corr_with": partner})
