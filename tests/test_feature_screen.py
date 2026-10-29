"""Unit tests for the feature screener: no look-ahead, purged folds, Newey-West correctness."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ML_layer.feature_screen import candidates as cand
from ML_layer.feature_screen import group_lasso as gl
from ML_layer.feature_screen.common import make_folds, nw_t, rank_features
from ML_layer.feature_screen.importance import permute_within_dates
from data_pipeline import features as fd


def _panel(n_dates=60, n_tickers=15, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    df = pd.DataFrame({
        "date": np.repeat(dates, n_tickers),
        "ticker": np.tile([f"T{i}" for i in range(n_tickers)], n_dates),
        "a": rng.normal(size=n_dates * n_tickers),
        "b": rng.normal(size=n_dates * n_tickers),
    })
    return df


def test_rank_transform_uses_only_same_date_data():
    df = _panel()
    ranked = rank_features(df, ["a", "b"])
    late = df["date"] >= df["date"].unique()[30]
    # Corrupt every later date wildly: earlier-date ranks must not change.
    corrupted = df.copy()
    corrupted.loc[late, ["a", "b"]] = np.random.default_rng(1).normal(size=(late.sum(), 2)) * 1e6
    reranked = rank_features(corrupted, ["a", "b"])
    pd.testing.assert_frame_equal(ranked.loc[~late, ["a", "b"]], reranked.loc[~late, ["a", "b"]])
    # Each date holds uniform ranks in (-0.5, 0.5], regardless of other dates and of row order.
    shuffled = rank_features(df.sample(frac=1, random_state=3), ["a", "b"]).sort_index()
    pd.testing.assert_frame_equal(ranked[["a", "b"]], shuffled[["a", "b"]])
    per_date = ranked.groupby("date")["a"].agg(["min", "max", "mean"])
    # rank(pct=True) - 0.5 has per-date mean 1/(2n), not exactly 0.
    assert np.allclose(per_date["mean"], 1 / (2 * 15))
    assert per_date["max"].max() <= 0.5 and per_date["min"].min() > -0.5


def _price_frame(n=420, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    walk = lambda scale: np.exp(np.cumsum(rng.normal(0, scale, n)) + 4)
    return pd.DataFrame({
        "Open": walk(0.01), "High": walk(0.01) * 1.01, "Low": walk(0.01) * 0.99, "Close": walk(0.01),
        "Volume": rng.integers(1_000, 5_000, n).astype(float), "SPY": walk(0.008), "XLK": walk(0.009),
    }, index=idx)


def test_existing_features_at_t_use_no_data_after_t():
    df = _price_frame()
    full = fd.engineer_features(df, "Technology", include_calendar=False)
    cutoff = full.index[-40]
    truncated = fd.engineer_features(df[df.index <= cutoff], "Technology", include_calendar=False)
    features = [c for c in full.columns if not c.startswith("target_")]
    pd.testing.assert_frame_equal(full.loc[truncated.index, features], truncated[features], check_exact=False, rtol=1e-10)


def test_candidate_features_at_t_use_no_data_after_t():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2018-01-01", periods=400)
    log_close = pd.Series(np.cumsum(rng.normal(0, 0.01, 400)), index=idx)
    volume = pd.Series(rng.integers(100, 900, 400).astype(float), index=idx)
    shares = pd.Series([1e6, 1.1e6, 1.2e6], index=pd.to_datetime(["2018-03-31", "2018-09-30", "2019-03-31"]))
    raw_close = np.exp(log_close)

    def build(upto):
        lc, v, rc = log_close[:upto], volume[:upto], raw_close[:upto]
        known = cand.shares_as_of(lc.index, shares)
        return pd.DataFrame({
            "r": cand.momentum_12_7(lc), "turn": cand.log_turnover(v, known), "dv": cand.log_dollar_volume(rc, v),
            "cap": np.log(rc * known),
        })

    full, partial = build(400), build(300)
    pd.testing.assert_frame_equal(full.iloc[:300], partial, check_exact=False, rtol=1e-12)
    # Share counts become usable only after the reporting lag, never before.
    known = cand.shares_as_of(idx, shares, lag_days=60)
    assert known[idx < pd.Timestamp("2018-03-31") + pd.Timedelta(days=60)].isna().all()
    assert known[idx >= pd.Timestamp("2018-09-30") + pd.Timedelta(days=60)].iloc[0] == 1.1e6


def test_momentum_12_7_window():
    lc = pd.Series(np.arange(400, dtype=float))
    out = cand.momentum_12_7(lc)
    assert out.iloc[:252].isna().all() and out.iloc[252] == lc.iloc[252 - 147] - lc.iloc[0]


@pytest.mark.parametrize("h", [1, 5, 20])
def test_purged_folds_do_not_overlap_target_windows(h):
    dates = pd.bdate_range("2012-01-01", "2019-12-31")
    n_t = 3
    df = pd.DataFrame({"date": np.repeat(dates, n_t)})
    df["date_pos"] = df["date"].rank(method="dense").astype(int) - 1
    df["forecast_date"] = df["date"] + pd.offsets.BDay(1)
    df["year"] = df["forecast_date"].dt.year
    for embargo in (None, 7):
        eff = h if embargo is None else embargo
        for fold in make_folds(df, h, 2016, purge=True, embargo=embargo):
            train_pos, test_pos = df["date_pos"].to_numpy()[fold.train], df["date_pos"].to_numpy()[fold.test]
            # training target window is (p, p + h]; it must end before the test period starts, with the embargo gap.
            assert train_pos.max() + h < test_pos.min()
            assert test_pos.min() - (train_pos.max() + h) > eff
            assert not set(fold.train) & set(fold.test)
    # Without purging, the last training windows do overlap the test period.
    leaky = make_folds(df, h, 2016, purge=False)[0]
    if h > 1:
        assert df["date_pos"].to_numpy()[leaky.train].max() + h >= df["date_pos"].to_numpy()[leaky.test].min()


def test_newey_west_matches_statsmodels():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=500)
    series = 0.1 + np.convolve(noise, np.ones(5) / 5, mode="same")  # autocorrelated
    for lags in (1, 4, 19):
        fit = sm.OLS(series, np.ones(len(series))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
        assert nw_t(series, lags + 1) == pytest.approx(float(fit.tvalues[0]), rel=1e-9)


def test_permutation_stays_within_dates():
    df = _panel(n_dates=20)
    codes = pd.factorize(df["date"], sort=True)[0]
    X = df[["a", "b"]].to_numpy()
    out = permute_within_dates(X, 0, codes, np.random.default_rng(0))
    assert np.array_equal(out[:, 1], X[:, 1])
    for c in np.unique(codes):
        assert np.allclose(np.sort(out[codes == c, 0]), np.sort(X[codes == c, 0]))
    assert not np.array_equal(out[:, 0], X[:, 0])


def test_spline_basis_and_group_lasso_recovers_nonlinear_feature():
    rng = np.random.default_rng(0)
    n, p = 4000, 5
    X = rng.uniform(-0.5, 0.5, size=(n, p))
    y = 4 * (X[:, 1] ** 2 - (X[:, 1] ** 2).mean()) + rng.normal(0, 0.5, n)  # purely quadratic: invisible to linear fits
    basis = gl.SplineBasis(4)
    B = basis.expand(X)
    assert np.allclose(B.sum(axis=2), 1.0, atol=1e-5)  # partition of unity
    ortho = gl.Orthonormaliser().fit(B)
    Z = ortho.transform(B)
    fit = gl.adaptive_group_lasso(Z, y, np.arange(n) // 10, basis.k, h=1, embargo=1)
    assert fit["selected"][1] and fit["selected"].sum() <= 2
    curve = gl.partial_effects(basis, ortho, fit["coef"])[1]
    assert curve[0] > curve[50] and curve[-1] > curve[50]  # U-shape recovered


def test_orthonormaliser_survives_degenerate_and_inert_groups():
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.5, 0.5, size=(500, 3))
    X[:, 2] = 0.0  # a feature that is missing everywhere (filled with 0)
    B = gl.SplineBasis(4).expand(X)
    Z = gl.Orthonormaliser().fit(B, usable=np.array([True, True, False])).transform(B)
    assert np.isfinite(Z).all() and np.allclose(Z[:, -6:], 0.0)
    assert np.isfinite(gl.Orthonormaliser().fit(B).transform(B)).all()  # constant column without the coverage guard
