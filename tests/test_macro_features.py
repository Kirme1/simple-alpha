"""Unit tests for the macro betas, interactions and time-series-normalised features."""

import numpy as np
import pandas as pd

from data_pipeline.utils import calendars, config, features, fred


def _inputs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    macro = pd.Series(np.cumsum(rng.normal(0, 0.05, n)) + 3.0, index=index)
    ret = pd.Series(1.5 * macro.diff().shift(-1).fillna(0) + rng.normal(0, 0.002, n), index=index)
    return ret, macro


def test_rolling_macro_beta_recovers_planted_beta():
    ret, macro = _inputs()
    # ret[t] responds to the macro move that the lagged FRED row at t+1 reports.
    beta = features.rolling_macro_beta(ret, features.macro_change(macro, "diff"))
    assert abs(beta.iloc[-1] - 1.5) < 0.1


def test_macro_beta_at_t_ignores_data_after_t():
    ret, macro = _inputs()
    delta = features.macro_change(macro, "diff")
    full = features.rolling_macro_beta(ret, delta)
    truncated = features.rolling_macro_beta(ret.iloc[:400], delta.iloc[:400])
    pd.testing.assert_series_equal(full.iloc[:400], truncated)


def test_macro_beta_does_not_use_return_on_day_t():
    ret, macro = _inputs()
    delta = features.macro_change(macro, "diff")
    base = features.rolling_macro_beta(ret, delta)
    shocked = ret.copy()
    shocked.iloc[500] += 0.5
    changed = features.rolling_macro_beta(shocked, delta)
    assert changed.iloc[500] == base.iloc[500]
    assert changed.iloc[501] != base.iloc[501]


def test_log_change_masks_non_positive_prices():
    level = pd.Series([50.0, -37.0, 20.0, 25.0])
    change = features.macro_change(level, "log")
    assert not np.isinf(change).any()
    assert change.isna().iloc[:3].all() and np.isfinite(change.iloc[3])


def test_rolling_zscore_uses_only_past_window():
    ret, _ = _inputs()
    full = features.rolling_zscore(ret)
    truncated = features.rolling_zscore(ret.iloc[:300])
    pd.testing.assert_series_equal(full.iloc[:300], truncated)


def test_cross_sectional_rank_keeps_unranked_columns():
    dates = pd.bdate_range("2021-01-01", periods=3).repeat(4)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"date": dates, "ticker": list("ABCD") * 3})
    cols = ["beta_oil_252d", "volatility_20d_z252", "fred_vix"]
    for c in cols:
        df[c] = rng.normal(size=len(df))
    ranked = features.cross_sectional_rank(df, cols)
    assert not np.allclose(ranked["beta_oil_252d"], df["beta_oil_252d"])
    for c in ("volatility_20d_z252", "fred_vix"):
        pd.testing.assert_series_equal(ranked[c], df[c])


def test_default_spread_replaces_legs():
    assert "fred_default_spread" in fred.FRED_FEATURES
    assert not set(fred.FRED_DEFAULT_LEGS) & set(fred.FRED_FEATURES)


def test_event_calendars_cover_feature_start():
    start = config.FEATURE_START.strftime("%Y-%m-%d")
    for calendar in (calendars.FOMC_DATES, calendars.CPI_RELEASE_DATES):
        assert min(calendar) >= start and min(calendar) < "2002-03-01"
        years = pd.Series([d[:4] for d in calendar]).value_counts()
        assert years[[str(y) for y in range(2002, 2026)]].between(7, 12).all()
    # 2008-03-10 was an unscheduled conference call; 2005-02-18 the seasonal-factor revision.
    assert "2008-09-16" in calendars.FOMC_DATES and "2008-03-10" not in calendars.FOMC_DATES
    assert "2005-02-23" in calendars.CPI_RELEASE_DATES and "2005-02-18" not in calendars.CPI_RELEASE_DATES


def test_fred_series_start_before_warmup():
    assert {"DFII10", "T5YIFR", "DTWEXBGS"}.isdisjoint(fred.FRED_SERIES.values())
    assert config.history_start(config.FEATURE_WARMUP_YEARS) == pd.Timestamp("2001-01-01")


def _vix_market(n=600, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"^VIX": 20 + np.cumsum(rng.normal(0, 0.5, n))}, index=index)


def test_vix_zscore_at_t_ignores_closes_from_the_last_two_days():
    market = _vix_market()
    base = fred.vix_zscore_from_market(market)
    shocked = market.copy()
    shocked.iloc[500] += 10
    changed = fred.vix_zscore_from_market(shocked)
    pd.testing.assert_series_equal(changed.iloc[:502], base.iloc[:502])
    assert changed.iloc[503] != base.iloc[503]


def test_vix_zscore_uses_close_from_two_calendar_days_back_and_warms_up():
    market = _vix_market()
    z = fred.vix_zscore_from_market(market)
    assert z.iloc[:fred.FRED_ZSCORE_MIN_PERIODS].isna().all()
    # Wednesday's feature row sees Monday's close.
    wed = market.index[market.index.dayofweek == 2][60]
    level = market["^VIX"].copy()
    level.index = level.index + pd.Timedelta(days=2)
    level = level.reindex(level.index.union(market.index)).ffill().reindex(market.index)
    window = level[(level.index > wed - pd.Timedelta(days=365)) & (level.index <= wed)]
    assert np.isclose(z[wed], (window.iloc[-1] - window.mean()) / window.std())
    assert level[wed] == market["^VIX"][wed - pd.Timedelta(days=2)]
