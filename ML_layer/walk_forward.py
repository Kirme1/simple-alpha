"""
walk_forward.py — Expanding-window, year-by-year out-of-sample test of tabular models.

A single 1.4-year test split is too short to detect a realistic cross-sectional
signal (IC ~0.01–0.03). This script refits fast models once per test year on
all data whose targets end before that year, pools every out-of-sample
prediction, and scores the pool with the same signal metrics as evaluate.py.

Usage:
    python -m ML_layer.walk_forward [--first-test-year 2016] [--target rank] [--xs-rank]
                                    [--features ticker] [--models ridge hgb reversal]

Samples use the features of day t to predict the cumulative excess log return
over SPY from t to t+h, keyed by (ticker, first forecast date t+1) like the
TFT and Ridge test samples.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .framework.metrics import paired_ic_difference, signal_metrics
from data_pipeline.utils.config import load_tickers, repo_path
from data_pipeline.utils.features import (
    KNOWN_CATEGORICAL_FEATURES,
    MARKET_WIDE_FEATURES,
    OBSERVED_FEATURES,
    cross_sectional_rank,
)

HORIZONS = (1, 5, 20)
MAX_HORIZON = max(HORIZONS)
MAX_DATE_GAP_DAYS = 5
# --features ticker keeps only inputs that differ between tickers on a date.
TICKER_FEATURES = [f for f in OBSERVED_FEATURES if f not in MARKET_WIDE_FEATURES]
FACTOR_ALIASES = {"reversal": "factor:-log_return_5d"}


def load_samples(data_directory: Path, ticker_file: Path) -> pd.DataFrame:
    """One row per (ticker, day t) with features at t and forward excess returns."""
    tickers, _ = load_tickers(ticker_file)
    frames = []
    for ticker in tickers:
        path = data_directory / f"{ticker}_raw.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        excess_log_level = df["target_log_close"] - df["target_spy_log_close"]
        for h in HORIZONS:
            df[f"y_h{h:02d}"] = excess_log_level.shift(-h) - excess_log_level
        df["forecast_date"] = df["date"].shift(-1)
        df["target_end_date"] = df["date"].shift(-MAX_HORIZON)
        # Drop samples whose forecast window spans a data gap (halts, missing rows).
        gap = df["date"].diff().dt.days
        df["max_forward_gap"] = gap[::-1].rolling(MAX_HORIZON, min_periods=1).max()[::-1].shift(-1)
        frames.append(df)
    samples = pd.concat(frames, ignore_index=True)
    samples = samples.dropna(subset=[f"y_h{h:02d}" for h in HORIZONS])
    samples = samples[samples["max_forward_gap"] <= MAX_DATE_GAP_DAYS]
    for column in KNOWN_CATEGORICAL_FEATURES:
        samples[column] = pd.to_numeric(samples[column])
    return samples.reset_index(drop=True)


def make_model(name: str):
    """A tabular model on one feature row per ticker and date, unlike the windowed ridge of baselines.py."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "hgb":
        # Shallow, heavily regularised trees: daily returns are mostly noise.
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500,
            l2_regularization=1.0, random_state=0,
        )
    raise ValueError(f"Unknown model {name!r}")


def walk_forward(
    samples: pd.DataFrame,
    features: list[str],
    models: list[str],
    first_test_year: int,
    xs_rank_target: str,
    holdout_years: int = 0,
) -> dict[str, pd.DataFrame]:
    """Out-of-sample predictions per model: columns ticker, date, fold, pred_hXX, y_hXX.

    ``holdout_years`` leaves the years just before each test year out of
    training, e.g. 1 to match the TFT, which spends year Y-1 on validation.
    """
    y_columns = [f"y_h{h:02d}" for h in HORIZONS]
    fit_targets = samples[y_columns]
    if xs_rank_target == "rank":
        fit_targets = samples.groupby("date")[y_columns].rank(pct=True) - 0.5

    last_year = samples["forecast_date"].dt.year.max()
    outputs: dict[str, list[pd.DataFrame]] = {FACTOR_ALIASES.get(name, name): [] for name in models}
    for year in range(first_test_year, last_year + 1):
        train_end = pd.Timestamp(year=year - holdout_years, month=1, day=1)
        # Purge: a training target must be fully realised before the test year
        # (or before the held-out years).
        train = samples["target_end_date"] < train_end
        test = samples["forecast_date"].dt.year == year
        if not test.any():
            continue
        print(f"  fold {year}: {train.sum():>7} train rows, {test.sum():>6} test rows")
        keys = samples.loc[test, ["ticker", "forecast_date", *y_columns]].rename(columns={"forecast_date": "date"})
        for name in models:
            name = FACTOR_ALIASES.get(name, name)
            fold = keys.assign(fold=year)
            for h, y_column in zip(HORIZONS, y_columns):
                if name.startswith("factor:"):
                    # Untrained single-feature signal, e.g. factor:-log_return_5d
                    # (short-term reversal: recent losers outperform).
                    column = name.removeprefix("factor:")
                    sign = -1.0 if column.startswith("-") else 1.0
                    fold[f"pred_h{h:02d}"] = sign * samples.loc[test, column.lstrip("-")].to_numpy()
                    continue
                model = make_model(name)
                model.fit(samples.loc[train, features], fit_targets.loc[train, y_column])
                fold[f"pred_h{h:02d}"] = model.predict(samples.loc[test, features])
            outputs[name].append(fold)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in outputs.items()}


def _as_horizon_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """signal_metrics indexes column h-1 of a [N, 20] array; unused horizons stay NaN."""
    matrix = np.full((len(frame), MAX_HORIZON), np.nan)
    for h in HORIZONS:
        matrix[:, h - 1] = frame[f"{prefix}_h{h:02d}"].to_numpy()
    return matrix


def score(predictions: dict[str, pd.DataFrame], xs_rank_target: str) -> pd.DataFrame:
    rows = {}
    reference = next(iter(predictions.values()))
    target = _as_horizon_matrix(reference, "y")
    dates = reference["date"].to_numpy()
    for name, frame in predictions.items():
        if not frame[["ticker", "date"]].equals(reference[["ticker", "date"]]):
            raise RuntimeError(f"{name} predictions are not aligned with the other models")
        rows[name] = signal_metrics(_as_horizon_matrix(frame, "pred"), target, dates, HORIZONS)
    names = list(predictions)
    if len(names) >= 2:
        best, base = names[-1], names[-2]
        rows[best].update(paired_ic_difference(
            _as_horizon_matrix(predictions[best], "pred"), _as_horizon_matrix(predictions[base], "pred"),
            target, dates, HORIZONS,
        ))
    metrics = pd.DataFrame(rows)
    if xs_rank_target == "rank":
        # Rank-scale predictions have no return units, so error metrics are meaningless.
        metrics = metrics[~metrics.index.str.startswith(("r2_oos", "mse"))]
    return metrics


def per_fold_ic(predictions: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = {}
    for name, frame in predictions.items():
        rows[name] = {
            year: signal_metrics(
                _as_horizon_matrix(fold, "pred"), _as_horizon_matrix(fold, "y"), fold["date"].to_numpy(), (horizon,),
            )[f"ic_h{horizon:02d}"]
            for year, fold in frame.groupby("fold")
        }
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward out-of-sample test of tabular models")
    parser.add_argument("--data-directory", default="data_pipeline/local_data/raw_data",
                        help="relative to the repo root")
    parser.add_argument("--ticker-file", default="data_pipeline/tickers_sp500.yaml",
                        help="relative to the repo root")
    parser.add_argument("--first-test-year", type=int, default=2016)
    parser.add_argument("--models", nargs="+", default=["reversal", "ridge", "hgb"],
                        help="Any of: ridge, hgb, factor:<feature> or factor:-<feature> "
                             "(that feature alone as the prediction), reversal (= factor:-log_return_5d)")
    parser.add_argument("--drop-features", nargs="+", default=[],
                        help="Inputs to leave out of the fitted models, e.g. to measure a new feature's value")
    parser.add_argument("--features", choices=["all", "ticker"], default="all",
                        help="'ticker' drops inputs that are identical across tickers on a date")
    parser.add_argument("--xs-rank", action="store_true",
                        help="Rank ticker-specific features across tickers on each date")
    parser.add_argument("--target", choices=["raw", "rank"], default="raw",
                        help="Fit raw excess returns or their per-date cross-sectional rank")
    parser.add_argument("--holdout-years", type=int, default=0,
                        help="Years before each test year left out of training; 1 matches walk_forward_tft")
    parser.add_argument("--output", default=None, help="Optional parquet path for pooled predictions")
    args = parser.parse_args()

    print("Loading samples...")
    samples = load_samples(repo_path(args.data_directory), args.ticker_file)
    features = TICKER_FEATURES if args.features == "ticker" else TICKER_FEATURES + MARKET_WIDE_FEATURES
    unknown = set(args.drop_features).difference(features)
    if unknown:
        raise ValueError(f"--drop-features not in the feature set: {sorted(unknown)}")
    features = [f for f in features if f not in args.drop_features]
    if args.xs_rank:
        samples = cross_sectional_rank(samples, TICKER_FEATURES)
    print(f"{len(samples)} samples, {samples['ticker'].nunique()} tickers, "
          f"{samples['date'].min().date()} to {samples['date'].max().date()}, {len(features)} features")

    print("Running walk-forward folds...")
    predictions = walk_forward(
        samples, features, args.models, args.first_test_year, args.target, args.holdout_years,
    )

    metrics = score(predictions, args.target)
    pooled = next(iter(predictions.values()))
    title = (f"WALK-FORWARD OUT-OF-SAMPLE — {len(pooled)} samples, {pooled['date'].nunique()} dates, "
             f"target={args.target}, features={args.features}, xs_rank={args.xs_rank}, "
             f"holdout_years={args.holdout_years}")
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(metrics.to_string())
        print(f"\nIC per test year, h05:\n{per_fold_ic(predictions, 5).to_string()}")
    if len(predictions) >= 2:
        print(f"\nic_diff rows compare {list(predictions)[-1]} against {list(predictions)[-2]}.")

    if args.output:
        pd.concat(
            [frame.assign(model=name) for name, frame in predictions.items()], ignore_index=True
        ).to_parquet(args.output, index=False)
        print(f"Saved pooled predictions to {args.output}")


if __name__ == "__main__":
    main()
