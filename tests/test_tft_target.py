import numpy as np
import pandas as pd
import pytest

from data_pipeline.interfaces import PTFLightningDataModule
from data_pipeline.interfaces.tft_data_interface import FORECAST_HORIZON


def module_with_returns(returns: dict[str, pd.Series]) -> PTFLightningDataModule:
    dm = PTFLightningDataModule.__new__(PTFLightningDataModule)
    dm._raw_returns = returns
    return dm


def test_raw_cumulative_targets_sum_daily_returns_from_first_forecast_day():
    values = np.arange(1, 41, dtype=float) / 100
    dm = module_with_returns({"AAA": pd.Series(values, index=np.arange(10, 50))})
    out = dm.raw_cumulative_targets(["AAA"], [15])
    # first forecast day 15 -> returns at 15..34, cumulated
    np.testing.assert_allclose(out[0], np.cumsum(values[5:25]), rtol=1e-5)
    assert out.shape == (1, FORECAST_HORIZON)


def test_raw_cumulative_targets_are_nan_when_a_day_is_missing_or_beyond_the_data():
    series = pd.Series(0.01, index=[0, 1, 2, 4, 5, 6])  # time_idx 3 missing
    dm = module_with_returns({"AAA": series})
    out = dm.raw_cumulative_targets(["AAA"], [1])
    assert np.isfinite(out[0, :2]).all()
    assert np.isnan(out[0, 2:]).all()


@pytest.mark.parametrize("value", ["bogus", "xs_rank", 1])
def test_non_boolean_xs_rank_target_is_rejected(value):
    with pytest.raises(ValueError, match="xs_rank_target"):
        PTFLightningDataModule("d", [], "t", 60, [0.6, 0.2, 0.2], xs_rank_target=value)


def test_boolean_xs_rank_target_is_accepted():
    assert PTFLightningDataModule("d", [], "t", 60, [0.6, 0.2, 0.2], xs_rank_target=True).xs_rank_target is True


def test_make_loss_selects_output_size_and_rejects_unknown_names():
    from ML_layer.framework.runners.runner_ptf import make_loss

    assert len(make_loss("quantile").quantiles) == 7
    assert make_loss("mse").reduction == "mean"
    with pytest.raises(ValueError):
        make_loss("huber")


def test_mean_daily_ic_is_per_date_spearman():
    from ML_layer.framework.metrics import MIN_TICKERS_PER_DATE, mean_daily_ic

    n = MIN_TICKERS_PER_DATE
    dates = np.repeat([0, 1], n)
    ranks = np.arange(1.0, n + 1)
    target = np.tile(ranks, 2)
    pred = np.concatenate([ranks * 10, ranks[::-1]])  # IC +1 on day 0, -1 on day 1
    assert mean_daily_ic(pred, target, dates) == pytest.approx(0.0)
    assert mean_daily_ic(target, target, dates) == pytest.approx(1.0)
