"""Model-based importance: within-date permutation importance for HGB, standardized ridge coefficients."""

import numpy as np
import pandas as pd

from .common import daily_ic


def permute_within_dates(X: np.ndarray, column: int, date_codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy of X with one column shuffled among tickers inside each date (rows must be sorted by date)."""
    order = np.argsort(date_codes + rng.random(len(date_codes)) * 0.5, kind="stable")
    out = X.copy()
    out[:, column] = X[order, column]
    return out


def permutation_importance(model, X, y_raw, dates, features, rng, n_repeats=2) -> pd.Series:
    """Drop in mean daily rank IC when a feature is permuted within each date's cross-section."""
    date_codes = pd.factorize(dates, sort=True)[0]
    assert (np.diff(date_codes) >= 0).all(), "test rows must be sorted by date"
    base = daily_ic(model.predict(X), y_raw, dates).mean()
    drops = {}
    for j, name in enumerate(features):
        drops[name] = np.mean([
            base - daily_ic(model.predict(permute_within_dates(X, j, date_codes, rng)), y_raw, dates).mean()
            for _ in range(n_repeats)
        ])
    return pd.Series(drops)


def ridge_coefficient_summary(coefs: pd.DataFrame) -> pd.DataFrame:
    """coefs: folds x features of standardized coefficients -> mean, std, sign consistency across folds."""
    mean = coefs.mean()
    share_pos = (coefs > 0).mean()
    return pd.DataFrame({
        "coef_mean": mean,
        "coef_std": coefs.std(ddof=1),
        "sign_consistency": np.maximum(share_pos, 1 - share_pos),
        "coef_t": mean / (coefs.std(ddof=1) / np.sqrt(len(coefs))),
    })
