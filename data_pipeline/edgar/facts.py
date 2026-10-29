"""Point-in-time handling of SEC company facts: tag chains, vintages, quarterly flows, TTM.

A fact is usable on a trading date only if its ``filed`` date is strictly earlier, so
every function here takes an ``as_of`` filing date and sees nothing filed after it.
"""

import numpy as np
import pandas as pd

# Fallback chain per item, highest priority first. The first tag with a value for a
# reporting period is used for that period (companies switch tags over time).
CHAINS: dict[str, list[str]] = {
    "revenue": ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet"],
    "cost_of_revenue": ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfGoodsSold"],
    "gross_profit": ["us-gaap:GrossProfit"],
    "assets": ["us-gaap:Assets"],
    "equity": ["us-gaap:StockholdersEquity",
               "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "net_income": ["us-gaap:NetIncomeLoss"],
    "operating_cash_flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "eps": ["us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareBasic"],
    "shares": ["dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding"],
}
FLOW_ITEMS = ("revenue", "cost_of_revenue", "gross_profit", "net_income", "operating_cash_flow", "eps")
INSTANT_ITEMS = ("assets", "equity", "shares")
UNITS = {"eps": "USD/shares", "shares": "shares"}  # every other item is in USD
PERIODIC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT")
ORIGINAL_PERIODIC_FORMS = ("10-K", "10-Q", "10-KT", "10-QT")

QUARTER_DAYS = (80, 100)
YTD_DAYS = {3: (80, 100), 6: (170, 200), 9: (260, 290), 12: (350, 380)}
QUARTER_GAP_DAYS = (70, 115)  # spacing of consecutive fiscal quarter ends
YEAR_AGO_TOLERANCE_DAYS = 30
FACT_COLUMNS = ["item", "tag", "priority", "start", "end", "val", "filed", "form", "accn"]


def parse_company_facts(payload: dict) -> pd.DataFrame:
    """Long table of every fact in the tag chains, from one companyfacts JSON."""
    rows = []
    for item, tags in CHAINS.items():
        unit = UNITS.get(item, "USD")
        for priority, tag in enumerate(tags):
            taxonomy, concept = tag.split(":")
            entries = payload.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
            for e in entries:
                if e.get("form") not in PERIODIC_FORMS:
                    continue
                rows.append((item, tag, priority, e.get("start"), e["end"], e["val"], e["filed"], e["form"], e.get("accn")))
    frame = pd.DataFrame(rows, columns=FACT_COLUMNS)
    for column in ("start", "end", "filed"):
        frame[column] = pd.to_datetime(frame[column])
    frame["val"] = frame["val"].astype(float)
    return frame


def resolve(rows: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per (start, end): the highest-priority tag with a value filed on or before ``as_of``, latest vintage.

    A restatement is a later filing of the same period, so the latest vintage
    filed by ``as_of`` is what an investor could see then.
    """
    known = rows[rows["filed"] <= as_of]
    if known.empty:
        return known
    known = known.sort_values(["priority", "filed"], ascending=[True, False], kind="stable")
    return known.drop_duplicates(["start", "end"])


def quarterly_values(periods: pd.DataFrame) -> pd.Series:
    """Discrete fiscal-quarter values (index: quarter end) from reported periods.

    10-Q flows are usually year-to-date. A reported 3-month value is used as is; otherwise
    a quarter is the year-to-date value minus the previous year-to-date value of the same
    fiscal year (same ``start``), so Q4 is the 10-K annual value minus the first nine months.
    """
    p = periods.dropna(subset=["start"])
    if p.empty:
        return pd.Series(dtype=float)
    days = (p["end"] - p["start"]).dt.days
    quarters: dict[pd.Timestamp, float] = {}
    for end, val in zip(p.loc[days.between(*QUARTER_DAYS), "end"], p.loc[days.between(*QUARTER_DAYS), "val"]):
        quarters[end] = val
    for _, group in p.assign(days=days).groupby("start"):
        ytd = {}
        for months, (low, high) in YTD_DAYS.items():
            row = group[group["days"].between(low, high)]
            if len(row):
                ytd[months] = (row["end"].iloc[-1], row["val"].iloc[-1])
        for months, (end, val) in ytd.items():
            if end in quarters:
                continue
            if months == 3:
                quarters[end] = val
            elif months - 3 in ytd:
                quarters[end] = val - ytd[months - 3][1]
    return pd.Series(quarters, dtype=float).sort_index()


def trailing_twelve_months(quarters: pd.Series) -> float:
    """Sum of the last four consecutive quarters; NaN unless all four exist."""
    last = quarters.iloc[-4:]
    if len(last) < 4:
        return np.nan
    gaps = np.diff(last.index.values).astype("timedelta64[D]").astype(int)
    if ((gaps < QUARTER_GAP_DAYS[0]) | (gaps > QUARTER_GAP_DAYS[1])).any():
        return np.nan
    return float(last.sum())


def year_ago(series: pd.Series, end: pd.Timestamp, tolerance: int = YEAR_AGO_TOLERANCE_DAYS) -> float:
    """Value at the period ending about one year before ``end`` (closest within the tolerance), else NaN."""
    if series.empty:
        return np.nan
    gaps = np.abs((series.index - (end - pd.Timedelta(days=365))).days)
    best = gaps.argmin()
    return float(series.iloc[best]) if gaps[best] <= tolerance else np.nan


def instants(periods: pd.DataFrame) -> pd.Series:
    """Point-in-time values (balance-sheet items), indexed by period end."""
    return periods.set_index("end")["val"].sort_index()


SCALE_ERROR_ORDERS = 2.0
SCALE_REFERENCE_FILINGS = 4


def drop_scale_errors(facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop whole filings whose scale is wrong, using only earlier filings as the reference.

    Some early XBRL filings report every value in millions or thousands as if they were units. A filing's
    scale sentinel is its latest balance-sheet Assets value; it is rejected if that is more than
    ``SCALE_ERROR_ORDERS`` orders of magnitude from the median of the last few accepted filings.
    Returns (clean facts, rejected filings with the value and reference).
    """
    assets = facts[facts["item"] == "assets"].sort_values("end")
    sentinel = assets.groupby("accn").agg(filed=("filed", "max"), assets=("val", "last")).sort_values(["filed"], kind="stable")
    accepted: list[float] = []
    rejected = []
    for accn, row in sentinel.iterrows():
        value = float(row["assets"])
        log_value = np.log10(value) if value > 0 else np.nan
        reference = float(np.median(accepted[-SCALE_REFERENCE_FILINGS:])) if accepted else np.nan
        if not np.isfinite(log_value) or (np.isfinite(reference) and abs(log_value - reference) > SCALE_ERROR_ORDERS):
            rejected.append({"accn": accn, "filed": row["filed"], "assets": value, "reference_log10": reference})
        else:
            accepted.append(log_value)
    dropped = pd.DataFrame(rejected, columns=["accn", "filed", "assets", "reference_log10"])
    return facts[~facts["accn"].isin(dropped["accn"])], dropped
