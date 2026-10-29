"""Unit tests for the point-in-time EDGAR features, on synthetic facts."""

import numpy as np
import pandas as pd

from data_pipeline.edgar import features as ef
from data_pipeline.edgar.facts import FACT_COLUMNS, quarterly_values, resolve


def _facts(rows):
    """rows: (item, tag, start, end, val, filed) -> facts table (first tag in each chain)."""
    frame = pd.DataFrame(
        [(i, t, 0, s, e, v, f, "10-Q", f"a{k}") for k, (i, t, s, e, v, f) in enumerate(rows)], columns=FACT_COLUMNS)
    for column in ("start", "end", "filed"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _calendar():
    return pd.bdate_range("2020-01-01", "2023-12-29")


def test_year_to_date_to_quarterly_and_q4_derivation():
    # Fiscal year starting 2020-01-01: Q1 = 10 (3m), YTD6 = 25, YTD9 = 45, annual = 70.
    periods = pd.DataFrame({
        "start": pd.to_datetime(["2020-01-01"] * 4),
        "end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]),
        "val": [10.0, 25.0, 45.0, 70.0],
    })
    q = quarterly_values(periods)
    assert q.to_list() == [10.0, 15.0, 20.0, 25.0]  # Q4 = 70 - 45
    assert list(q.index) == list(periods["end"])


def test_reported_three_month_value_beats_derived():
    periods = pd.DataFrame({
        "start": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]),
        "end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-06-30"]),
        "val": [10.0, 16.0, 25.0],
    })
    assert quarterly_values(periods).to_list() == [10.0, 16.0]


def test_restatement_uses_latest_filing_before_the_date():
    rows = _facts([
        ("assets", "us-gaap:Assets", None, "2020-03-31", 100.0, "2020-05-01"),
        ("assets", "us-gaap:Assets", None, "2020-03-31", 90.0, "2020-08-01"),  # restated later
    ])
    assert resolve(rows, pd.Timestamp("2020-06-01"))["val"].iloc[0] == 100.0
    assert resolve(rows, pd.Timestamp("2020-08-01"))["val"].iloc[0] == 90.0
    assert resolve(rows, pd.Timestamp("2020-04-30")).empty


def test_expanded_feature_never_uses_a_filing_on_or_after_the_date():
    calendar = _calendar()
    filed = pd.Series(pd.to_datetime(["2020-05-04", "2020-08-04"]))
    values = pd.Series([1.0, 2.0])
    daily = ef.expand_to_calendar(calendar, filed, values)
    assert np.isnan(daily[pd.Timestamp("2020-05-04")])  # the filing day itself is not usable
    assert daily[pd.Timestamp("2020-05-05")] == 1.0
    assert daily[pd.Timestamp("2020-08-04")] == 1.0
    assert daily[pd.Timestamp("2020-08-05")] == 2.0
    # A filing on a Friday becomes usable the next Monday.
    friday = ef.expand_to_calendar(calendar, pd.Series([pd.Timestamp("2020-05-08")]), pd.Series([3.0]))
    assert np.isnan(friday[pd.Timestamp("2020-05-08")]) and friday[pd.Timestamp("2020-05-11")] == 3.0


def test_feature_values_are_unchanged_by_truncating_future_filings():
    quarters = pd.date_range("2018-03-31", periods=12, freq="QE")
    rows = []
    for k, end in enumerate(quarters):
        filed = end + pd.Timedelta(days=35)
        rows += [("assets", "us-gaap:Assets", None, end, 100.0 + 5 * k, filed),
                 ("eps", "us-gaap:EarningsPerShareDiluted", end - pd.Timedelta(days=89), end, 1.0 + 0.1 * k + 0.05 * (k % 2), filed),
                 ("equity", "us-gaap:StockholdersEquity", None, end, 50.0 + k, filed)]
    facts = _facts(rows)
    full, _ = ef.compute_filing_events(facts, None, financial=False)
    cut = facts[facts["filed"] <= pd.Timestamp("2019-12-31")]
    partial, _ = ef.compute_filing_events(cut, None, financial=False)
    n = len(partial)
    pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True), partial.reset_index(drop=True))


def test_ear_is_unavailable_before_day_two_and_does_not_overlap_its_window():
    calendar = _calendar()
    log_close = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(calendar))), index=calendar)
    spy = pd.Series(0.0, index=calendar)
    announced = pd.Timestamp("2021-05-05")
    ann = pd.DataFrame({"filed": [announced], "announced": [announced], "fallback": [False]})
    ear = ef.earnings_announcement_returns(ann, log_close, spy)
    day0 = calendar.get_loc(announced)
    assert ear["day0"].iloc[0] == announced
    assert ear["available"].iloc[0] == calendar[day0 + 2]
    assert np.isclose(ear["ear"].iloc[0], log_close.iloc[day0 + 1] - log_close.iloc[day0 - 2])
    daily = ef.expand_ear(calendar, ear)
    assert daily[: calendar[day0 + 2]].iloc[:-1].isna().all()
    assert daily[calendar[day0 + 2]] == ear["ear"].iloc[0]


def test_ear_expires_after_stale_days_and_weekend_announcement_uses_next_trading_day():
    calendar = _calendar()
    log_close = pd.Series(np.arange(len(calendar), dtype=float) * 0.001, index=calendar)
    saturday = pd.Timestamp("2021-05-08")
    ann = pd.DataFrame({"filed": [saturday], "announced": [saturday], "fallback": [True]})
    ear = ef.earnings_announcement_returns(ann, log_close, log_close * 0)
    assert ear["day0"].iloc[0] == pd.Timestamp("2021-05-10") and bool(ear["fallback"].iloc[0])
    daily = ef.expand_ear(calendar, ear)
    assert np.isnan(daily[pd.Timestamp("2021-05-08") + pd.Timedelta(days=ef.STALE_DAYS + 1)])


def test_earnings_announcement_uses_8k_item_202_else_falls_back_with_flag():
    periodic = pd.DataFrame({"form": ["10-Q", "10-Q"], "filed": pd.to_datetime(["2021-05-10", "2021-08-10"])})
    eight_ks = pd.DataFrame({"filed": pd.to_datetime(["2021-05-04", "2021-08-02"]), "items": ["2.02,9.01", "5.02"]})
    out = ef.earnings_announcements(periodic, eight_ks)
    assert out["announced"].to_list() == [pd.Timestamp("2021-05-04"), pd.Timestamp("2021-08-10")]
    assert out["fallback"].to_list() == [False, True]


def test_sue_uses_no_quarters_after_current_and_floors_the_std():
    ends = pd.date_range("2017-03-31", periods=16, freq="QE")
    eps = pd.Series(1.0 + 0.2 * np.arange(16) + 0.1 * np.sin(np.arange(16)), index=ends)
    current = ends[11]
    base = ef.sue_at(eps, current)
    changed = eps.copy()
    changed.iloc[12:] = 99.0  # later quarters must not matter
    assert ef.sue_at(changed, current) == base
    flat = pd.Series(1.0 + 0.0 * np.arange(16), index=ends)
    flat.iloc[11] = 2.0
    assert np.isclose(ef.sue_at(flat, ends[11]), 1.0 / ef.SUE_STD_FLOOR)  # zero history std -> floor
    assert np.isnan(ef.sue_at(eps.iloc[:8], ends[7]))  # not enough history


def test_split_adjustment_of_share_counts_on_a_synthetic_split():
    splits = pd.Series([2.0], index=pd.to_datetime(["2021-06-01"]))
    assert ef.split_factor_after(splits, pd.Timestamp("2021-03-31")) == 2.0
    assert ef.split_factor_after(splits, pd.Timestamp("2021-06-30")) == 1.0
    ends = pd.to_datetime(["2020-12-31", "2021-12-31"])
    rows = []
    for end, shares in zip(ends, [1e8, 2e8]):  # 2:1 split in June 2021: no real issuance
        rows.append(("shares", "dei:EntityCommonStockSharesOutstanding", None, end, shares, end + pd.Timedelta(days=30)))
    events, flagged = ef.compute_filing_events(_facts(rows), splits, financial=False)
    last = events.iloc[-1]
    assert np.isclose(last["net_share_issuance"], 0.0) and not flagged
    assert np.isclose(last["shares"], 2e8)
    # Without the split adjustment the same data would look like a 2x issuance.
    unadjusted, _ = ef.compute_filing_events(_facts(rows), None, financial=False)
    assert np.isclose(unadjusted.iloc[-1]["net_share_issuance"], np.log(2.0))


def test_implausible_share_jump_is_flagged_and_left_missing():
    ends = pd.to_datetime(["2020-12-31", "2021-12-31"])
    rows = [("shares", "dei:EntityCommonStockSharesOutstanding", None, e, s, e + pd.Timedelta(days=30))
            for e, s in zip(ends, [1e8, 1e9])]
    events, flagged = ef.compute_filing_events(_facts(rows), None, financial=False)
    assert len(flagged) == 1 and np.isnan(events.iloc[-1]["net_share_issuance"])


def test_financials_get_no_gross_profitability_or_accruals_and_nonpositive_equity_is_missing():
    q = pd.date_range("2020-03-31", periods=4, freq="QE")
    rows = []
    for k, end in enumerate(q):
        start = end - pd.Timedelta(days=89)
        filed = end + pd.Timedelta(days=30)
        rows += [("revenue", "us-gaap:Revenues", start, end, 100.0, filed),
                 ("cost_of_revenue", "us-gaap:CostOfRevenue", start, end, 60.0, filed),
                 ("net_income", "us-gaap:NetIncomeLoss", start, end, 10.0, filed),
                 ("operating_cash_flow", "us-gaap:NetCashProvidedByUsedInOperatingActivities", start, end, 20.0, filed),
                 ("assets", "us-gaap:Assets", None, end, 1000.0, filed),
                 ("equity", "us-gaap:StockholdersEquity", None, end, -5.0, filed)]
    facts = _facts(rows)
    industrial, _ = ef.compute_filing_events(facts, None, financial=False)
    bank, _ = ef.compute_filing_events(facts, None, financial=True)
    assert np.isclose(industrial.iloc[-1]["gross_profitability"], (4 * 40.0) / 1000.0)
    assert bank[["gross_profitability", "accruals"]].isna().all().all()
    assert industrial["book_equity"].isna().all()


def test_cover_count_uses_the_split_units_of_its_filing_date():
    # Alphabet-style: a 10-Q filed after the split reports a post-split count for a pre-split cover date.
    splits = pd.Series([20.0], index=pd.to_datetime(["2022-07-18"]))
    rows = [("shares", "dei:EntityCommonStockSharesOutstanding", None, pd.Timestamp("2021-07-16"), 6.6e8, pd.Timestamp("2021-07-28")),
            ("shares", "dei:EntityCommonStockSharesOutstanding", None, pd.Timestamp("2022-07-15"), 1.3e10, pd.Timestamp("2022-07-27"))]
    events, flagged = ef.compute_filing_events(_facts(rows), splits, financial=False)
    assert not flagged and np.isclose(events.iloc[-1]["shares"], 1.3e10)
    assert np.isclose(events.iloc[-1]["net_share_issuance"], np.log(1.3e10 / (6.6e8 * 20)))


def test_wrong_scale_count_is_kept_out_of_shares_and_flagged():
    ends = pd.date_range("2020-03-31", periods=6, freq="QE")
    values = [1e8] * 5 + [1e12]
    rows = [("shares", "dei:EntityCommonStockSharesOutstanding", None, e, v, e + pd.Timedelta(days=30)) for e, v in zip(ends, values)]
    events, flagged = ef.compute_filing_events(_facts(rows), None, financial=False)
    assert np.isnan(events.iloc[-1]["shares"]) and np.isnan(events.iloc[-1]["net_share_issuance"])
    assert flagged and flagged[-1]["current_count_rejected"]
    assert events.iloc[-2]["shares"] == 1e8


def test_filing_reported_in_wrong_scale_is_dropped_using_only_earlier_filings():
    from data_pipeline.edgar.facts import drop_scale_errors

    ends = pd.date_range("2018-03-31", periods=8, freq="QE")
    values = [1e10, 1.1e10, 1.2e10, 1.3e10, 1.3e4, 1.4e10, 1.5e10, 1.6e10]  # fifth filing in millions
    rows = [("assets", "us-gaap:Assets", None, e, v, e + pd.Timedelta(days=30)) for e, v in zip(ends, values)]
    facts = _facts(rows)
    clean, dropped = drop_scale_errors(facts)
    assert dropped["assets"].to_list() == [1.3e4] and len(clean) == 7
    # The first filing is never rejected for lack of a reference, and later filings do not change earlier decisions.
    clean_head, dropped_head = drop_scale_errors(facts.iloc[:5])
    assert dropped_head["assets"].to_list() == [1.3e4]


def test_stale_or_implausible_share_counts_and_stale_balance_items_are_not_carried_forward():
    rows = [("shares", "dei:EntityCommonStockSharesOutstanding", None, pd.Timestamp("2019-03-18"), 1.0, pd.Timestamp("2019-03-18")),
            ("equity", "us-gaap:StockholdersEquity", None, pd.Timestamp("2019-03-18"), 50.0, pd.Timestamp("2019-03-18")),
            ("assets", "us-gaap:Assets", None, pd.Timestamp("2019-03-18"), 100.0, pd.Timestamp("2019-03-18")),
            # a later filing that reports none of them
            ("net_income", "us-gaap:NetIncomeLoss", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31"), 5.0, pd.Timestamp("2020-05-01"))]
    events, _ = ef.compute_filing_events(_facts(rows), None, financial=False)
    assert np.isnan(events.iloc[0]["shares"])  # a count of 1 is not a share count
    late = events.iloc[-1]
    assert np.isnan(late["shares"]) and np.isnan(late["book_equity"]) and np.isnan(late["asset_growth"])
