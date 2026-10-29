import pandas as pd
import pytest

from data_pipeline.interfaces.basic_data_interface import add_fundamentals_and_start, fill_missing_fundamentals


def frame(tickers=("AAA", "BBB"), days=6) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame([{"ticker": t, "date": d, "log_return_1d": 0.01} for t in tickers for d in dates])


@pytest.fixture
def fundamentals_dir(tmp_path):
    # AAA has `sue` from the 3rd day on; BBB has no file.
    dates = pd.bdate_range("2020-01-01", periods=6)
    pd.DataFrame({"date": dates, "sue": [None, None, 1.0, 2.0, 3.0, 4.0], "accruals": 0.5}).to_parquet(tmp_path / "AAA.parquet")
    return str(tmp_path)


def test_fundamentals_are_merged_by_ticker_and_date(fundamentals_dir):
    out = add_fundamentals_and_start(frame(), ["log_return_1d", "sue"], fundamentals_dir, None, xs_rank_features=True)
    assert len(out) == 12
    aaa = out[out["ticker"] == "AAA"].sort_values("date")
    assert aaa["sue"].tolist()[2:] == [1.0, 2.0, 3.0, 4.0] and aaa["sue"].iloc[:2].isna().all()
    assert out[out["ticker"] == "BBB"]["sue"].isna().all()  # no EDGAR file -> missing
    assert "accruals" not in out.columns  # only the columns the model uses


def test_start_date_drops_earlier_rows(fundamentals_dir):
    out = add_fundamentals_and_start(frame(), ["log_return_1d"], None, "2020-01-03")
    assert out["date"].min() == pd.Timestamp("2020-01-03") and len(out) == 8


def test_start_date_fundamentals_uses_first_date_with_enough_coverage(fundamentals_dir):
    # half of the tickers trading that day must have gross_profitability; only AAA has a file, so 1 of 2 from day 3
    pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=6), "gross_profitability": [None, None, 1, 1, 1, 1]}).to_parquet(
        fundamentals_dir + "/AAA.parquet")
    out = add_fundamentals_and_start(frame(), ["log_return_1d"], fundamentals_dir, "fundamentals")
    assert out["date"].min() == pd.Timestamp("2020-01-03")


def test_fundamentals_need_a_directory_and_ranked_features(fundamentals_dir):
    with pytest.raises(ValueError, match="fundamentals_directory"):
        add_fundamentals_and_start(frame(), ["sue"], None, None, xs_rank_features=True)
    with pytest.raises(ValueError, match="xs_rank_features"):
        add_fundamentals_and_start(frame(), ["sue"], fundamentals_dir, None, xs_rank_features=False)


def test_missing_fundamentals_become_the_neutral_rank():
    df = pd.DataFrame({"sue": [0.3, None], "log_return_1d": [None, 0.1]})
    out = fill_missing_fundamentals(df, ["sue", "log_return_1d"])
    assert out["sue"].tolist() == [0.3, 0.0]
    assert out["log_return_1d"].isna().sum() == 1  # other features are left as they are
