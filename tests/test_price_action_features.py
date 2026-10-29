"""Unit tests for the open/close/range features in features.add_price_action_features."""

import numpy as np
import pandas as pd

from data_pipeline.utils import features as fd


def _ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.bdate_range("2020-01-01", periods=n))


def test_overnight_plus_intraday_equals_log_return():
    df = _ohlc()
    out = fd.add_price_action_features(df)
    log_ret = np.log(df["Close"]).diff()
    for days in (20, 60):
        total = out[f"overnight_ret_{days}d"] + out[f"intraday_ret_{days}d"]
        np.testing.assert_allclose(total.dropna(), log_ret.rolling(days).sum().dropna(), atol=1e-10)


def test_clv_is_half_when_high_equals_low():
    df = _ohlc()
    df.iloc[-5:, df.columns.get_indexer(["Open", "High", "Low", "Close"])] = 100.0
    out = fd.add_price_action_features(df)
    assert out["clv_5d"].iloc[-1] == 0.5


def test_quantile_skew_and_extremes_match_manual_window():
    df = _ohlc()
    out = fd.add_price_action_features(df)
    daily = np.log(df["Close"]).diff()
    w = daily.iloc[-60:]
    q10, q50, q90 = w.quantile([0.1, 0.5, 0.9])
    assert np.isclose(out["skew_q_60d"].iloc[-1], (q90 + q10 - 2 * q50) / (q90 - q10))
    last20 = daily.iloc[-20:]
    assert np.isclose(out["max1_20d"].iloc[-1], last20.max())
    assert np.isclose(out["min1_20d"].iloc[-1], last20.min())
    assert np.isclose(out["max5_20d"].iloc[-1], last20.nlargest(5).mean())


def test_features_use_no_future_rows():
    df = _ohlc()
    full = fd.add_price_action_features(df)
    truncated = fd.add_price_action_features(df.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], truncated)
