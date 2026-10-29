"""FRED macro series fetched as point-in-time release vintages and turned into trailing z-scores."""

import os

import numpy as np
import pandas as pd
import requests

# Set FRED_API_KEY (or pass --fred-api-key) to add these safely-lagged fields.
# Every series must start before FEATURE_START minus the warm-up years; DFII10 and
# T5YIFR (2003) and DTWEXBGS (2006) do not, so they are not used.
# Moody's spreads replace the ICE BofA OAS series, which FRED truncated to a
# rolling three-year window (with no older vintages) in April 2026.
FRED_SERIES = {
    "fred_credit_spread_baa": "BAA10Y", "fred_credit_spread_aaa": "AAA10Y",
    "fred_yield_curve_slope": "T10Y2Y", "fred_financial_conditions": "NFCI",
    "fred_vix": "VIXCLS", "fred_term_spread_3m": "T10Y3M", "fred_oil": "DCOILWTICO",
}

# Both Moody's spreads embed the rate level; BAA - AAA isolates default risk.
FRED_DEFAULT_LEGS = ("fred_credit_spread_baa", "fred_credit_spread_aaa")
FRED_FEATURES = [c for c in FRED_SERIES if c not in FRED_DEFAULT_LEGS] + ["fred_default_spread"]

# Suffix of the un-z-scored levels, which the macro betas are computed from.
FRED_LEVEL_SUFFIX = "_level"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_PAGE_LIMIT = 100_000

# Covers the gap between a feature date and its latest observation for every
# series in FRED_SERIES (NFCI is weekly; the rest are daily).
FRED_OBSERVATION_LOOKBACK_DAYS = 365

# FRED levels drift across regimes, so features are z-scored against a trailing
# window. A calendar-day window gives identical values on trading-day and
# business-day indexes (training vs. the API's FRED cache).
FRED_ZSCORE_WINDOW = "365D"
FRED_ZSCORE_MIN_PERIODS = 200

# First calendar day a VIXCLS close is usable: released the next day, read strictly after.
FRED_VIX_RELEASE_LAG_DAYS = 2


def _fetch_fred_releases(
    series_id: str,
    api_key: str,
    realtime_start: pd.Timestamp,
    realtime_end: pd.Timestamp,
    observation_start: pd.Timestamp,
) -> pd.DataFrame:
    """Return every release of ``series_id`` in a real-time window, all pages.

    fredapi's ``get_series_all_releases`` cannot bound observation dates, so
    the observations endpoint is called directly.
    """

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": realtime_start.strftime("%Y-%m-%d"),
        "realtime_end": realtime_end.strftime("%Y-%m-%d"),
        "observation_start": observation_start.strftime("%Y-%m-%d"),
        "limit": FRED_PAGE_LIMIT,
    }

    rows, offset = [], 0

    while True:
        response = requests.get(FRED_OBSERVATIONS_URL, params={**params, "offset": offset}, timeout=60)
        payload = response.json()
        if response.status_code != 200:
            raise RuntimeError(f"FRED request for {series_id} failed: {payload.get('error_message', response.text)}")
        rows.extend(payload["observations"])
        offset += FRED_PAGE_LIMIT
        if offset >= payload["count"]:
            break
    return pd.DataFrame(rows, columns=["realtime_start", "realtime_end", "date", "value"])


def fetch_fred_context(
    index: pd.DatetimeIndex, api_key: str | None = None, include_levels: bool = False
) -> pd.DataFrame:
    """Retrieve macro values using release vintages available before each day.

    Returns trailing z-scores of every FRED_FEATURES column. With ``include_levels``
    the un-z-scored levels are added as ``<column>_level`` for the macro betas.

    ``get_series()`` returns today's revised history and would leak revisions
    into historical backtests.  Selecting each observation from the latest
    release available before the feature date avoids that contamination.
    """

    api_key = (api_key or os.environ.get("FRED_API_KEY") or "").strip()

    if not api_key:
        return pd.DataFrame(index=pd.DatetimeIndex(index))

    out = pd.DataFrame(index=pd.DatetimeIndex(index))

    # Keep each request safely below FRED's 2,000-vintage-date limit.
    range_start = out.index.min().normalize()
    range_end = out.index.max().normalize()

    # Only observations shortly before the range can be the latest known value.
    # Without this bound, series revised back to their start (NFCI is re-released
    # weekly from 1971) exceed FRED's 100,000-row page and only old rows return.
    observation_start = range_start - pd.Timedelta(days=FRED_OBSERVATION_LOOKBACK_DAYS)

    for column, series_id in FRED_SERIES.items():
        release_batches = []
        chunk_start = range_start
        try:
            while chunk_start <= range_end:
                chunk_end = min(chunk_start + pd.Timedelta(days=1799), range_end)
                batch = _fetch_fred_releases(series_id, api_key, chunk_start, chunk_end, observation_start)
                if not batch.empty:
                    release_batches.append(batch)
                chunk_start = chunk_end + pd.Timedelta(days=1)
        except RuntimeError as error:
            if "does not exist in ALFRED" not in str(error):
                raise

            # Series without vintages (e.g. ICE BofA spreads) are unrevised market
            # data; treat each observation as released the day after its date.
            today = pd.Timestamp.today().normalize()
            batch = _fetch_fred_releases(series_id, api_key, today, today, observation_start)
            batch["realtime_start"] = (
                pd.to_datetime(batch["date"]) + pd.Timedelta(days=1)
            ).dt.strftime("%Y-%m-%d")
            release_batches = [batch] if not batch.empty else []

        if not release_batches:
            out[column] = np.nan
            continue

        releases = pd.concat(release_batches, ignore_index=True)
        releases["date"] = pd.to_datetime(releases["date"]).dt.normalize()
        releases["realtime_start"] = pd.to_datetime(releases["realtime_start"]).dt.normalize()
        releases["value"] = pd.to_numeric(releases["value"], errors="coerce")


        """ALFRED only tracks vintages from 2014 for the daily Moody's/Treasury
        series (2011 for NFCI). Observations already in the first vintage are
        treated as released the day after their date, as for unvintaged
        series; the daily market series are not materially revised."""

        first_vintage = releases["realtime_start"].min()
        backfill = (releases["realtime_start"] == first_vintage) & (releases["date"] < first_vintage)
        releases.loc[backfill, "realtime_start"] = releases.loc[backfill, "date"] + pd.Timedelta(days=1)
        releases = releases.dropna(subset=["value"]).sort_values(["realtime_start", "date"])
        values = []
        latest_by_observation: dict[pd.Timestamp, float] = {}
        release_rows = list(releases[["realtime_start", "date", "value"]].itertuples(index=False))
        release_position = 0


        for feature_date in out.index:
            # Strictly earlier dates make the feature safe even when a series
            # is released after the market has opened or closed.
            while (
                release_position < len(release_rows)
                and release_rows[release_position].realtime_start < feature_date
            ):
                row = release_rows[release_position]
                latest_by_observation[row.date] = row.value
                release_position += 1
            known_dates = [date for date in latest_by_observation if date < feature_date]
            if not known_dates:
                values.append(np.nan)
                continue
            values.append(latest_by_observation[max(known_dates)])
        out[column] = values
    out["fred_default_spread"] = out["fred_credit_spread_baa"] - out["fred_credit_spread_aaa"]
    levels = out[FRED_FEATURES]


    # Only rows at or before each date enter its window, so this stays point-in-time.
    rolling = levels.rolling(FRED_ZSCORE_WINDOW, min_periods=FRED_ZSCORE_MIN_PERIODS)
    zscores = (levels - rolling.mean()) / rolling.std()
    if include_levels:
        return zscores.join(levels.add_suffix(FRED_LEVEL_SUFFIX))

    return zscores


def vix_zscore_from_market(market: pd.DataFrame) -> pd.Series:
    """Keyless ``fred_vix``: the FRED VIXCLS z-score rebuilt from yfinance ``^VIX`` closes.

    VIXCLS is the unrevised CBOE close. ``fetch_fred_context`` treats an observation as
    released the day after its date and uses releases strictly before the feature date,
    so the close of day D is first visible on row D + 2 calendar days.
    """
    
    close = market["^VIX"].dropna()
    released = pd.Series(close.to_numpy(), index=close.index + pd.Timedelta(days=FRED_VIX_RELEASE_LAG_DAYS))
    level = released.reindex(released.index.union(market.index)).ffill().reindex(market.index)
    rolling = level.rolling(FRED_ZSCORE_WINDOW, min_periods=FRED_ZSCORE_MIN_PERIODS)

    return ((level - rolling.mean()) / rolling.std()).rename("fred_vix")
