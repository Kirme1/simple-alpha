"""Yahoo Finance access: OHLCV, sector, earnings dates and market-context closes, with retry/backoff."""

import random
import time

import pandas as pd
import yfinance as yf

from .utils.config import CONTEXT_WARMUP_YEARS, history_start

SECTOR_NAME_MAPPINGS = {
    "Technology": "XLK", "Financial Services": "XLF", "Basic Materials": "XLB",
    "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Energy": "XLE", "Healthcare": "XLV", "Industrials": "XLI", "Real Estate": "XLRE",
    "Real-estate": "XLRE", "Utilities": "XLU",
}

# Yahoo intermittently rejects requests with 401 "Invalid Crumb" / "unable to
# access this feature" when its session cookie is refreshed or requests are
# throttled; a short randomized backoff lets yfinance obtain a fresh crumb.
YAHOO_ATTEMPTS = 4


def yahoo_backoff(attempt: int) -> None:
    time.sleep(2 ** attempt + random.uniform(0, 1))


def normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _close_prices(downloaded: pd.DataFrame) -> pd.DataFrame:
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise ValueError("yfinance response has no Close prices")
        close = downloaded["Close"]
    else:
        close = downloaded[["Close"]]
    return normalise_index(close)


def fetch_sector(ticker: str) -> str:
    """Look up the Yahoo sector; raise rather than return a placeholder on failure."""
    for attempt in range(YAHOO_ATTEMPTS):
        try:
            sector = yf.Ticker(ticker).info.get("sector")
        except Exception as exc:
            error = exc
        else:
            if sector in SECTOR_NAME_MAPPINGS:
                return sector
            error = ValueError(f"Yahoo returned sector {sector!r}")
        if attempt < YAHOO_ATTEMPTS - 1:
            yahoo_backoff(attempt)
    raise ValueError(f"Sector lookup failed for {ticker}: {error}")


def fetch_earnings(ticker: str) -> pd.DataFrame:
    """Past and scheduled earnings announcements: columns announced_at (US/Eastern) and surprise."""
    for attempt in range(YAHOO_ATTEMPTS):
        try:
            # Yahoo caps the scrape at 100 quarters (about 25 years).
            dates = yf.Ticker(ticker).get_earnings_dates(limit=100)
        except Exception as exc:
            error = exc
        else:
            if dates is not None and not dates.empty:
                return pd.DataFrame({
                    "announced_at": dates.index.tz_convert("America/New_York").tz_localize(None),
                    "surprise": pd.to_numeric(dates["Surprise(%)"], errors="coerce").to_numpy() / 100.0,
                }).sort_values("announced_at").reset_index(drop=True)
            error = ValueError("Yahoo returned no earnings dates")
        if attempt < YAHOO_ATTEMPTS - 1:
            yahoo_backoff(attempt)
    raise ValueError(f"Earnings lookup failed for {ticker}: {error}")


def fetch_ohlcv(
    ticker: str, period: str | None = None, start: pd.Timestamp | None = None, sector: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """yfinance periods stop at "10y", so the default history is requested by start date."""

    if start is None and period is None:
        start = history_start()

    download_args = {"start": pd.Timestamp(start).date()} if start is not None else {"period": period}

    for attempt in range(YAHOO_ATTEMPTS):
        # yfinance logs failures (including 401s) and returns an empty frame.
        df = yf.download(ticker, auto_adjust=True, progress=False, **download_args)
        if not df.empty:
            break
        if attempt < YAHOO_ATTEMPTS - 1:
            yahoo_backoff(attempt)

    if df.empty:
        raise ValueError(f"No OHLCV rows returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = normalise_index(df).dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if (df["Close"] <= 0).any() or (df["Volume"] < 0).any():
        raise ValueError(f"Invalid OHLCV values returned for {ticker}")

    return df, sector or fetch_sector(ticker)


def fetch_market_context(tickers: list[str], start: pd.Timestamp | None = None) -> pd.DataFrame:
    start = history_start(CONTEXT_WARMUP_YEARS) if start is None else pd.Timestamp(start)
    start = start.date()
    close = _close_prices(yf.download(tickers, start=start, auto_adjust=True, progress=False))
    required = {"SPY", "QQQ", "^VIX", "^TNX", "^IRX", "DX-Y.NYB", *SECTOR_NAME_MAPPINGS.values()}
    missing = required.difference(close.columns)

    if missing:
        raise ValueError(f"Market-context download missing symbols: {sorted(missing)}")
        
    return close
