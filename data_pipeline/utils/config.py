"""Shared pipeline settings: feature history window and ticker universe loading."""

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKER_FILE = REPO_ROOT / "data_pipeline" / "tickers_sp500.yaml"

# First day of feature history. Prices and market context start a year earlier
# so 12-month momentum, the 52-week high, betas, and the FRED z-scores are warmed
# up by the first feature row. The start is fixed rather than relative to today:
# Yahoo's earnings dates reach back only to about January 2002 (100 quarters), the
# FOMC/CPI calendars below begin in 2002, and every FRED series in FRED_SERIES
# starts before 2001. Sector ETFs that launched later (XLRE 2015, XLC 2018) simply
# have no features before their first year of data.
FEATURE_START = pd.Timestamp("2002-01-01")
FEATURE_WARMUP_YEARS = 1
CONTEXT_WARMUP_YEARS = 1
# Caches starting later than the requested history (plus holiday slack) are refetched.
HISTORY_START_TOLERANCE = pd.Timedelta(days=7)


def repo_path(path: str | Path) -> Path:
    """``path`` as an absolute path; a relative one is taken relative to the repository root, not the cwd."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_tickers(ticker_file: str | Path) -> tuple[list[str], list[str]]:
    with open(repo_path(ticker_file)) as f:
        loaded = yaml.safe_load(f)
    return loaded["tickers"], loaded["indexes"]


def history_start(extra_years: int = 0) -> pd.Timestamp:
    return FEATURE_START - pd.DateOffset(years=extra_years)
