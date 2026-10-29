"""Optional candidate features: r_12_7, log_mktcap, log_turnover (or log dollar volume).

Look-ahead and data-quality notes (also written into summary.md):
* All three use only information dated <= t. r_12_7 uses closes 147-252 trading
  days back; the volume average is a trailing 20-day mean including day t (the
  same close-of-day-t convention as the existing features).
* Yahoo's share-count history (get_shares_full) is NOT point-in-time: dates are
  as stamped by Yahoo, values may be restated, and coverage starts late and has
  gaps. Each observation is made available only ``shares_lag_days`` (default 60)
  calendar days after its stamped date, roughly the 10-Q/10-K filing delay, and
  is carried forward without back-filling. Rows before a ticker's first usable
  observation are NaN.
* Share counts are as reported, while Yahoo prices and volumes are split-adjusted
  retroactively. Share counts are therefore multiplied by every split that occurs
  after their date (a unit conversion that carries no return information).
* Market cap uses split-adjusted but dividend-UNadjusted closes (auto_adjust=False).
* The universe is today's constituents, so size is subject to survivorship bias.
* Missing values stay NaN here; models later treat a missing rank as 0 (neutral).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

R_12_7_NEAR, R_12_7_FAR = 147, 252  # ~7 and ~12 months of trading days
VOLUME_WINDOW = 20
SHARES_LAG_DAYS = 60
MIN_SHARES_COVERAGE = 0.8
SHARES_START = "2009-01-01"
ATTEMPTS = 3


def momentum_12_7(log_close: pd.Series) -> pd.Series:
    """Cumulative log return from t-12 months to t-7 months."""
    return log_close.shift(R_12_7_NEAR) - log_close.shift(R_12_7_FAR)


def split_adjust_shares(shares: pd.Series, splits: pd.Series) -> pd.Series:
    """Express as-reported share counts in today's split-adjusted units."""
    if shares.empty or splits is None or splits.empty:
        return shares
    factors = np.array([splits[splits.index > d].prod() for d in shares.index])
    return shares * factors


def shares_as_of(index: pd.DatetimeIndex, shares: pd.Series, lag_days: int = SHARES_LAG_DAYS) -> pd.Series:
    """Latest share count known on each date: observation date + lag <= date. No back-fill."""
    if shares.empty:
        return pd.Series(np.nan, index=index)
    known = pd.DataFrame({"known_from": shares.index + pd.Timedelta(days=lag_days), "shares": shares.to_numpy()})
    known = known.sort_values("known_from")
    left = pd.DataFrame({"date": index})
    merged = pd.merge_asof(left, known, left_on="date", right_on="known_from", direction="backward")
    return pd.Series(merged["shares"].to_numpy(), index=index)


def log_turnover(volume: pd.Series, shares_known: pd.Series) -> pd.Series:
    return np.log(volume.rolling(VOLUME_WINDOW).mean() / shares_known)


def log_dollar_volume(raw_close: pd.Series, volume: pd.Series) -> pd.Series:
    return np.log((raw_close * volume).rolling(VOLUME_WINDOW).mean())


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()


def _retry(call):
    for attempt in range(ATTEMPTS):
        try:
            result = call()
            if result is not None and len(result):
                return result
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None


def fetch_yahoo_inputs(ticker: str, cache_directory: Path) -> dict[str, pd.Series]:
    """Raw close, split-adjusted share counts; cached under outputs"""
    cache = cache_directory / "candidates"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{ticker}.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        return {
            "raw_close": frame["raw_close"].dropna(),
            "shares": frame["shares"].dropna(),
        }
    handle = yf.Ticker(ticker)
    history = _retry(lambda: handle.history(start="2010-01-01", auto_adjust=False))
    raw_close = pd.Series(dtype=float) if history is None else history["Close"].set_axis(_naive(history.index))
    shares = _retry(lambda: handle.get_shares_full(start=SHARES_START))
    shares = pd.Series(dtype=float) if shares is None else shares.set_axis(_naive(shares.index))
    shares = shares[~shares.index.duplicated(keep="last")].sort_index().astype(float)
    splits = _retry(lambda: handle.splits)
    splits = None if splits is None else splits.set_axis(_naive(splits.index))
    shares = split_adjust_shares(shares, splits)
    frame = pd.concat({"raw_close": raw_close, "shares": shares}, axis=1)
    frame.index.name = "date"
    if frame.empty:  # do not cache a failed download
        return {"raw_close": raw_close, "shares": shares}
    frame.to_parquet(path)
    return {"raw_close": raw_close, "shares": shares}


def add_candidate_features(
    samples: pd.DataFrame, data_directory: Path, cache_directory: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge candidate columns into ``samples`` (keyed by ticker, date). Returns (samples, names, notes)."""
    frames = []
    shares_coverage = {}
    for ticker in samples["ticker"].unique():
        raw = pd.read_parquet(data_directory / f"{ticker}_raw.parquet", columns=["date", "target_log_close"])
        raw = raw.sort_values("date").set_index("date")
        source = pd.read_parquet(data_directory / "source_ohlcv" / f"{ticker}.parquet", columns=["date", "Volume"])
        volume = source.set_index("date")["Volume"].astype(float).reindex(raw.index)
        inputs = fetch_yahoo_inputs(ticker, cache_directory)
        shares_known = shares_as_of(raw.index, inputs["shares"])
        raw_close = inputs["raw_close"].reindex(raw.index)
        out = pd.DataFrame(index=raw.index)
        out["r_12_7"] = momentum_12_7(raw["target_log_close"])
        out["log_mktcap"] = np.log(raw_close * shares_known)
        out["log_turnover"] = log_turnover(volume, shares_known)
        out["log_dollar_volume"] = log_dollar_volume(raw_close, volume)
        shares_coverage[ticker] = float(shares_known.notna().mean())
        frames.append(out.replace([np.inf, -np.inf], np.nan).assign(ticker=ticker).reset_index())
    candidates = pd.concat(frames, ignore_index=True)
    samples = samples.merge(candidates, on=["ticker", "date"], how="left", validate="one_to_one")

    coverage = float(samples["log_turnover"].notna().mean())
    names = ["r_12_7", "log_mktcap"]
    notes = [
        "Candidate features: `r_12_7` (log return t-12m to t-7m), `log_mktcap` (split-adjusted unadjusted-for-dividends "
        "close x split-adjusted shares), `log_turnover` (20d mean volume / shares).",
        f"Share-count history is spotty: {coverage:.0%} of sample rows have a known share count "
        f"({sum(v == 0 for v in shares_coverage.values())} tickers have none). Observations are lagged "
        f"{SHARES_LAG_DAYS} calendar days, never back-filled; missing ranks are treated as 0 (neutral) by the models.",
        "Residual look-ahead risks: Yahoo share counts are not point-in-time (restatements, stamped dates); "
        "universe = today's constituents (survivorship bias, strongest for size).",
    ]
    if coverage >= MIN_SHARES_COVERAGE:
        names.append("log_turnover")
    else:
        names.append("log_dollar_volume")
        notes.append(
            f"`log_turnover` coverage {coverage:.0%} < {MIN_SHARES_COVERAGE:.0%}; fell back to `log_dollar_volume` "
            "(log 20d mean of raw close x volume) for the turnover/liquidity slot."
        )
    samples = samples.drop(columns=[c for c in ("log_turnover", "log_dollar_volume") if c not in names])
    low = [t for t, v in shares_coverage.items() if v < 0.5]
    if low:
        notes.append(f"log_mktcap is mostly missing for {len(low)} tickers (<50% rows): {', '.join(sorted(low)[:20])}.")
    return samples, names, notes
