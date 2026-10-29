"""The seven point-in-time EDGAR fundamentals features.

Filing-based features are computed at each periodic filing from only the facts filed on
or before that date (``compute_filing_events``), then become usable on the first trading
day strictly after the filing and are forward-filled for at most ``STALE_DAYS``
(``expand_to_calendar``). EAR is built from earnings announcement dates
(``earnings_announcements``, ``earnings_announcement_returns``).
"""

import numpy as np
import pandas as pd

from .facts import CHAINS, ORIGINAL_PERIODIC_FORMS, drop_scale_errors, instants, quarterly_values, resolve, trailing_twelve_months, year_ago

FUNDAMENTAL_FEATURES = [
    "gross_profitability", "asset_growth", "net_share_issuance", "accruals", "book_to_market", "sue", "ear",
]
STALE_DAYS = 200
FINANCIAL_SIC = range(6000, 7000)
MAX_ABS_LOG_SHARE_CHANGE = 1.5
MIN_PLAUSIBLE_SHARES = 1e5  # smaller counts are a unit or per-class reporting artifact (e.g. a cover page reporting 1)
SHARES_YEAR_AGO_TOLERANCE_DAYS = 60
SUE_HISTORY_QUARTERS = 8
SUE_MIN_HISTORY = 6
SUE_STD_FLOOR = 0.01
EARNINGS_8K_ITEM = "2.02"
EARNINGS_8K_LOOKBACK_DAYS = 45
EAR_AVAILABLE_AFTER_DAYS = 2  # trading days after the announcement day


def split_factor_after(splits: pd.Series, date: pd.Timestamp) -> float:
    """Product of split ratios with ex-date after ``date``: converts share counts to today's units.

    ``date`` is the filing date of the count: Alphabet's July 2022 10-Q, filed after the 20:1 split,
    reports a cover count as of a pre-split date that is already post-split.
    """
    if splits is None or splits.empty:
        return 1.0
    return float(splits[splits.index > date].prod())


def _latest_tag(periods: pd.DataFrame) -> str | None:
    return None if periods.empty else periods.sort_values("end")["tag"].iloc[-1]


def sue_at(eps_quarters: pd.Series, current: pd.Timestamp) -> float:
    """SUE of the quarter ending ``current``: its year-over-year EPS change over the std of the previous 8
    quarters' changes. Only quarters ending on or before ``current`` are used.
    """
    eps_quarters = eps_quarters[eps_quarters.index <= current]
    if current not in eps_quarters.index:
        return np.nan
    yoy = pd.Series({e: v - year_ago(eps_quarters[eps_quarters.index < e], e) for e, v in eps_quarters.items()}).dropna()
    if current not in yoy.index:
        return np.nan
    window_start = current - pd.Timedelta(days=SUE_HISTORY_QUARTERS * 95)
    history = yoy[(yoy.index < current) & (yoy.index > window_start)].iloc[-SUE_HISTORY_QUARTERS:]
    if len(history) < SUE_MIN_HISTORY:
        return np.nan
    return float(yoy[current] / max(history.std(ddof=1), SUE_STD_FLOOR))


def compute_filing_events(
    facts: pd.DataFrame, splits: pd.Series | None, financial: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    """Features as known right after each periodic filing. Returns (events, flagged share-count jumps).

    ``events`` has one row per distinct ``filed`` date: the five filing-based features,
    ``book_equity`` and split-adjusted ``shares`` (for book-to-market) and the tag used per item.
    """
    facts, _ = drop_scale_errors(facts)
    by_item = {item: g for item, g in facts.groupby("item")}
    flagged: list[dict] = []
    rows = []
    for filed in sorted(facts["filed"].unique()):
        filed = pd.Timestamp(filed)
        state = {item: resolve(by_item[item], filed) if item in by_item else pd.DataFrame() for item in CHAINS}
        row = {"filed": filed}
        quarters = {item: quarterly_values(state[item]) for item in
                    ("revenue", "cost_of_revenue", "gross_profit", "net_income", "operating_cash_flow", "eps")
                    if not state[item].empty}
        assets = instants(state["assets"]) if not state["assets"].empty else pd.Series(dtype=float)
        if len(assets) and (filed - assets.index[-1]).days > STALE_DAYS:  # the filing no longer reports this item
            assets = assets.iloc[:0]
        assets_now = assets.iloc[-1] if len(assets) else np.nan
        assets_prior = year_ago(assets, assets.index[-1]) if len(assets) else np.nan

        # Gross profit per quarter: reported GrossProfit where present, else revenue - cost of revenue.
        gp = quarters.get("gross_profit", pd.Series(dtype=float))
        if "revenue" in quarters and "cost_of_revenue" in quarters:
            derived = (quarters["revenue"] - quarters["cost_of_revenue"]).dropna()
            derived = derived[~derived.index.isin(gp.index)]
            gp = pd.concat([part for part in (gp, derived) if len(part)]).sort_index() if len(gp) or len(derived) else gp
        ttm_gp = trailing_twelve_months(gp)
        ttm_ni = trailing_twelve_months(quarters.get("net_income", pd.Series(dtype=float)))
        ttm_ocf = trailing_twelve_months(quarters.get("operating_cash_flow", pd.Series(dtype=float)))

        row["gross_profitability"] = np.nan if financial or not assets_now > 0 else ttm_gp / assets_now
        row["asset_growth"] = assets_now / assets_prior - 1.0 if assets_prior > 0 and assets_now > 0 else np.nan
        average_assets = (assets_now + assets_prior) / 2.0
        row["accruals"] = np.nan if financial or not average_assets > 0 else (ttm_ni - ttm_ocf) / average_assets
        row["sue"] = sue_at(quarters["eps"], quarters["eps"].index[-1]) if "eps" in quarters and len(quarters["eps"]) else np.nan

        equity = instants(state["equity"]) if not state["equity"].empty else pd.Series(dtype=float)
        fresh_equity = len(equity) and (filed - equity.index[-1]).days <= STALE_DAYS
        row["book_equity"] = equity.iloc[-1] if fresh_equity and equity.iloc[-1] > 0 else np.nan

        row["net_share_issuance"], row["shares"] = np.nan, np.nan
        shares = state["shares"]
        if not shares.empty:
            observed = shares[shares["tag"] == _latest_tag(shares)].sort_values("end")
            # A count is in the units of its filing date: splits effective before the filing are already in it.
            adjusted = observed["val"].to_numpy() * np.array([split_factor_after(splits, f) for f in observed["filed"]])
            ends = pd.DatetimeIndex(observed["end"])
            now = adjusted[-1]
            # A count far from the previous report is a bad value (wrong scale), not issuance: keep it out of market cap too.
            reference = np.median(adjusted[-4:-1]) if len(adjusted) > 1 else np.nan
            bad_now = len(adjusted) > 1 and reference > 0 and now > 0 and abs(np.log(now / reference)) > MAX_ABS_LOG_SHARE_CHANGE
            reported_recently = (filed - observed["filed"].iloc[-1]).days <= STALE_DAYS
            if now >= MIN_PLAUSIBLE_SHARES and not bad_now and reported_recently:
                row["shares"] = now
            before = ends < ends[-1] - pd.Timedelta(days=200)
            if before.any():
                gaps = np.abs((ends[before] - (ends[-1] - pd.Timedelta(days=365))).days)
                best = gaps.argmin()
                prior = adjusted[before][best]
                if gaps[best] <= SHARES_YEAR_AGO_TOLERANCE_DAYS and prior > 0 and now > 0:
                    change = float(np.log(now / prior))
                    if abs(change) > MAX_ABS_LOG_SHARE_CHANGE or bad_now:
                        flagged.append({"filed": filed, "shares_now": now, "shares_year_ago": prior, "log_change": change,
                                        "current_count_rejected": bool(bad_now)})
                    else:
                        row["net_share_issuance"] = change
        for item in CHAINS:
            row[f"tag_{item}"] = _latest_tag(state[item])
        rows.append(row)
    return pd.DataFrame(rows), flagged


def expand_to_calendar(
    calendar: pd.DatetimeIndex, filed: pd.Series, values: pd.Series, stale_days: int = STALE_DAYS,
) -> pd.Series:
    """Daily series: each value is usable from the first trading day after its filing date.

    A day shows the latest earlier non-missing value; values older than ``stale_days``
    since filing are NaN. Missing values never override an earlier one.
    """
    events = pd.DataFrame({"filed": pd.to_datetime(filed).to_numpy(), "value": values.to_numpy()}).dropna()
    out = pd.Series(np.nan, index=calendar)
    if events.empty:
        return out
    position = calendar.searchsorted(events["filed"].to_numpy(), side="right")  # strictly after the filing date
    events = events[position < len(calendar)]
    events = events.assign(available=calendar[position[position < len(calendar)]]).sort_values("available")
    merged = pd.merge_asof(pd.DataFrame({"date": calendar}), events, left_on="date", right_on="available", direction="backward")
    age = (merged["date"] - merged["filed"]).dt.days
    return pd.Series(np.where(age <= stale_days, merged["value"], np.nan), index=calendar)


def earnings_announcements(periodic_filings: pd.DataFrame, eight_ks: pd.DataFrame) -> pd.DataFrame:
    """Earnings announcement date per 10-Q/10-K: filing date of an 8-K with item 2.02 in the 45 days before.

    ``periodic_filings``: columns form, filed. ``eight_ks``: columns filed, items.
    Falls back to the periodic filing date itself (``fallback`` = True) when no such 8-K exists.
    """
    has_item = np.array([EARNINGS_8K_ITEM in str(items).split(",") for items in eight_ks["items"].fillna("")], dtype=bool)
    earnings = eight_ks[has_item]
    dates = np.sort(pd.to_datetime(earnings["filed"]).unique())
    rows = []
    for filed in sorted(pd.to_datetime(periodic_filings.loc[periodic_filings["form"].isin(ORIGINAL_PERIODIC_FORMS), "filed"]).unique()):
        filed = pd.Timestamp(filed)
        recent = dates[(dates <= filed.to_datetime64()) & (dates >= (filed - pd.Timedelta(days=EARNINGS_8K_LOOKBACK_DAYS)).to_datetime64())]
        rows.append({"filed": filed, "announced": pd.Timestamp(recent[-1]) if len(recent) else filed, "fallback": len(recent) == 0})
    out = pd.DataFrame(rows, columns=["filed", "announced", "fallback"])
    return out.drop_duplicates("announced", keep="first").reset_index(drop=True)


def earnings_announcement_returns(
    announcements: pd.DataFrame, log_close: pd.Series, log_benchmark: pd.Series,
) -> pd.DataFrame:
    """EAR: stock minus benchmark cumulative log return over trading days -1..+1 around the announcement.

    Day 0 is the first trading day on or after the announcement date. The window ends at the day +1
    close, and the value is usable from ``available`` = day +2, so it never overlaps what it predicts.
    """
    calendar = log_close.index
    day0 = calendar.searchsorted(pd.to_datetime(announcements["announced"]).to_numpy(), side="left")
    rows = []
    for (_, ann), p in zip(announcements.iterrows(), day0):
        if p - 2 < 0 or p + EAR_AVAILABLE_AFTER_DAYS >= len(calendar):
            continue
        ear = (log_close.iloc[p + 1] - log_close.iloc[p - 2]) - (log_benchmark.iloc[p + 1] - log_benchmark.iloc[p - 2])
        rows.append({"announced": ann["announced"], "day0": calendar[p], "available": calendar[p + EAR_AVAILABLE_AFTER_DAYS],
                     "ear": float(ear), "fallback": bool(ann["fallback"])})
    return pd.DataFrame(rows, columns=["announced", "day0", "available", "ear", "fallback"])


def expand_ear(calendar: pd.DatetimeIndex, ear: pd.DataFrame, stale_days: int = STALE_DAYS) -> pd.Series:
    """Daily EAR: latest announcement whose ``available`` date has arrived, NaN after ``stale_days``."""
    out = pd.Series(np.nan, index=calendar)
    if ear.empty:
        return out
    events = ear.sort_values("available")
    merged = pd.merge_asof(pd.DataFrame({"date": calendar}), events, left_on="date", right_on="available", direction="backward")
    age = (merged["date"] - merged["announced"]).dt.days
    return pd.Series(np.where(age <= stale_days, merged["ear"], np.nan), index=calendar)


def build_daily_features(
    calendar: pd.DatetimeIndex, events: pd.DataFrame, ear: pd.DataFrame, unadjusted_close: pd.Series,
) -> pd.DataFrame:
    """All seven features on ``calendar`` (trading dates)."""
    out = pd.DataFrame(index=calendar, columns=FUNDAMENTAL_FEATURES, dtype=float)
    if len(events):
        for name in ("gross_profitability", "asset_growth", "net_share_issuance", "accruals", "sue"):
            out[name] = expand_to_calendar(calendar, events["filed"], events[name])
        both = events.dropna(subset=["book_equity", "shares"])
        book = expand_to_calendar(calendar, both["filed"], both["book_equity"])
        shares = expand_to_calendar(calendar, both["filed"], both["shares"])
        market_cap = unadjusted_close.reindex(calendar) * shares
        out["book_to_market"] = book / market_cap
    out["ear"] = expand_ear(calendar, ear)
    return out.replace([np.inf, -np.inf], np.nan)


def fundamentals_start(directory, tickers: list[str], share: float = 0.5, feature: str = "gross_profitability") -> pd.Timestamp:
    """First date on which at least ``share`` of the tickers trading that day have ``feature``."""
    from pathlib import Path

    frames = []
    for ticker in tickers:
        path = Path(directory) / f"{ticker}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["date", feature]).assign(ticker=ticker))
    data = pd.concat(frames)
    coverage = data.groupby("date")[feature].apply(lambda s: s.notna().mean())  # of tickers trading that day
    return coverage[coverage >= share].index.min()


def load_daily_fundamentals(directory, tickers) -> pd.DataFrame:
    """Daily fundamentals (``data_pipeline.edgar.build``) of the tickers that have a file; others stay missing."""
    from pathlib import Path

    frames = [
        pd.read_parquet(Path(directory) / f"{t}.parquet").assign(ticker=t)
        for t in tickers if (Path(directory) / f"{t}.parquet").exists()
    ]
    out = pd.concat(frames, ignore_index=True)
    return out.assign(date=pd.to_datetime(out["date"]))
