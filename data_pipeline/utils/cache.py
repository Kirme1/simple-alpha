"""On-disk caches under the output directory: market context, earnings dates and per-ticker OHLCV."""

from pathlib import Path

import pandas as pd

from .config import CONTEXT_WARMUP_YEARS, HISTORY_START_TOLERANCE, history_start
from ..yahoo import fetch_earnings, fetch_market_context, normalise_index

SOURCE_CACHE_DIRNAME = "source_ohlcv"
MARKET_CONTEXT_CACHE_NAME = "_market_context.parquet"
EARNINGS_CACHE_DIRNAME = "earnings"
# Earnings caches older than this are refreshed; dates are scheduled weeks ahead.
EARNINGS_CACHE_MAX_AGE = pd.Timedelta(days=7)


def source_cache_path(ticker: str, output_dir: Path) -> Path:
    return output_dir / SOURCE_CACHE_DIRNAME / f"{ticker}.parquet"


def load_or_refresh_earnings(ticker: str, output_dir: Path, force: bool) -> pd.DataFrame:
    """Weekly-refreshed earnings cache; a stale cache is used if Yahoo fails."""
    cache_path = output_dir / EARNINGS_CACHE_DIRNAME / f"{ticker}.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else None
    fresh_enough = (
        cached is not None and not force
        and pd.Timestamp.now() - pd.Timestamp(cached["fetched_at"].iloc[0]) < EARNINGS_CACHE_MAX_AGE
    )
    if fresh_enough:
        return cached.drop(columns="fetched_at")
    try:
        earnings = fetch_earnings(ticker)
    except ValueError:
        if cached is None:
            raise
        return cached.drop(columns="fetched_at")
    if cached is not None:
        # Yahoo's 100-quarter window slides forward every quarter; keep the older
        # announcements that have dropped out of it.
        older = cached.drop(columns="fetched_at")
        older = older[older["announced_at"] < earnings["announced_at"].min()]
        earnings = pd.concat([older, earnings], ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    earnings.assign(fetched_at=pd.Timestamp.now()).to_parquet(cache_path, index=False)
    return earnings


def load_or_refresh_market_context(
    indexes: list[str], output_dir: Path, force: bool,
) -> pd.DataFrame:
    """Only download context sessions newer than the local context cache."""
    cache_path = output_dir / MARKET_CONTEXT_CACHE_NAME
    cached = pd.read_parquet(cache_path) if cache_path.exists() and not force else None
    required_start = history_start(CONTEXT_WARMUP_YEARS) + HISTORY_START_TOLERANCE
    if cached is not None and pd.Timestamp(cached["date"].min()) > required_start:
        cached = None  # cache predates FEATURE_START; refetch everything
    if cached is None:
        context = fetch_market_context(indexes)
    else:
        cached = normalise_index(cached.set_index("date"))
        latest_date = cached.index.max()
        try:
            fresh = fetch_market_context(indexes, start=latest_date)
        except ValueError as exc:
            # yfinance returns an empty frame when the cache is already current.
            if "no Close prices" not in str(exc):
                raise
            fresh = pd.DataFrame(index=cached.index[:0], columns=cached.columns)
        context = pd.concat([cached, fresh]).loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()
    # Older caches carry a trading_day_index column that is no longer a feature.
    context = context.drop(columns=["trading_day_index"], errors="ignore")
    context.index.name = "date"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    context.reset_index().to_parquet(cache_path, index=False)
    return context
