"""Paired test of HGB (and ridge) trained on nested feature sets: does adding features raise out-of-sample IC?

Same samples, purged folds, model specifications and scoring as the screener, but only HGB and ridge are
fit (the screener's LASSO stability selection dominates its runtime). HGB with these settings has no
subsampling or early stopping, so it is deterministic and seeds would not change the result. Missing values
are 0 (neutral rank) for ridge and left missing for HGB. Predictions are saved so any pair can be re-tested.
IC differences use the screener's daily-IC difference with Newey-West errors.

Usage: python -m ML_layer.feature_screen.hgb_paired [--experiment price_action|macro_interaction|fundamentals|shortlist]
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.edgar.features import FUNDAMENTAL_FEATURES, fundamentals_start
from data_pipeline.utils.config import load_tickers
from data_pipeline.utils.features import INTERACTION_FEATURES, MACRO_BETA_FEATURES, PRICE_ACTION_FEATURES, TS_NORMALISED_FEATURES
from .common import HORIZONS, REPO_ROOT, evaluate, ic_difference, load_screen_samples, make_folds, stride_rows, y_col, yr_col
from .pipeline import make_hgb, make_ridge
from .screen import md_table

SELECTED = ["overnight_ret_20d", "overnight_ret_60d", "min1_20d"]
SHORTLIST = SELECTED + ["sue", "accruals"]
MOMENTUM = ["log_return_60d", "momentum_12_1", "dist_52w_high"]  # the screener's momentum group among the original 17
CANDIDATE_GROUPS = [PRICE_ACTION_FEATURES, MACRO_BETA_FEATURES, INTERACTION_FEATURES, TS_NORMALISED_FEATURES, FUNDAMENTAL_FEATURES]
# experiment -> (feature sets as "base + added features", pairs tested as (a, b): IC(a) - IC(b)).
# The base is always the original 17 features.
EXPERIMENTS = {
    "price_action": ({"plus3": SELECTED, "all": PRICE_ACTION_FEATURES},
                     [("plus3", "orig"), ("all", "orig"), ("all", "plus3")]),
    "macro_interaction": ({"macro_beta": MACRO_BETA_FEATURES, "interaction": INTERACTION_FEATURES,
                           "both": MACRO_BETA_FEATURES + INTERACTION_FEATURES},
                          [("macro_beta", "orig"), ("interaction", "orig"), ("both", "orig"),
                           ("both", "macro_beta"), ("both", "interaction")]),
    "fundamentals": ({"fundamentals": FUNDAMENTAL_FEATURES}, [("fundamentals", "orig")]),
    "shortlist": ({"shortlist": SHORTLIST}, [("shortlist", "orig"), ("no_momentum", "orig")]),
}
# experiment -> feature sets built by removing features from the original 17 (name -> removed features)
REMOVED = {"shortlist": {"no_momentum": MOMENTUM}}
# experiments that need the EDGAR fundamentals (and so the common window where they exist)
USES_FUNDAMENTALS = ("fundamentals", "shortlist")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-directory", default=str(REPO_ROOT / "data_pipeline/local_data/raw_data"))
    ap.add_argument("--ticker-file", default=str(REPO_ROOT / "data_pipeline/tickers_sp500.yaml"))
    ap.add_argument("--fundamentals-directory", default=str(REPO_ROOT / "data_pipeline/local_data/fundamentals"))
    ap.add_argument("--output-root", default=str(REPO_ROOT / "outputs/feature_screen/hgb_paired"))
    ap.add_argument("--experiment", choices=list(EXPERIMENTS), default="price_action")
    ap.add_argument("--min-date", default=None, help="first sample date; the fundamentals experiment defaults to when half the universe has them")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--hgb-stride", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.output_root) / f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)

    use_fundamentals = args.experiment in USES_FUNDAMENTALS
    min_date = pd.Timestamp(args.min_date) if args.min_date else None
    if use_fundamentals and min_date is None:
        min_date = fundamentals_start(args.fundamentals_directory, load_tickers(Path(args.ticker_file))[0])
    print(f"Window starts {min_date.date() if min_date is not None else 'at the first sample'}", flush=True)
    samples, features, _ = load_screen_samples(
        Path(args.data_directory), Path(args.ticker_file), False, REPO_ROOT / "outputs/feature_screen/_cache", min_date,
        fundamentals_directory=Path(args.fundamentals_directory) if use_fundamentals else None,
    )
    added, pairs = EXPERIMENTS[args.experiment]
    candidates = {f for group in CANDIDATE_GROUPS for f in group}
    orig = [f for f in features if f not in candidates]
    sets = {"orig": orig, **{name: orig + [f for f in features if f in extra] for name, extra in added.items()}}
    sets.update({name: [f for f in orig if f not in removed] for name, removed in REMOVED.get(args.experiment, {}).items()})
    print({k: len(v) for k, v in sets.items()}, f"{len(samples)} samples, {samples['ticker'].nunique()} tickers", flush=True)
    pos = samples["date_pos"].to_numpy()
    rows, tests = [], []
    for h in args.horizons:
        folds = make_folds(samples, h, args.first_test_year)
        idx = np.concatenate([f.test for f in folds])
        y_rank, y_raw = samples[yr_col(h)].to_numpy(), samples[y_col(h)].to_numpy()
        preds = {}
        for name, cols in sets.items():
            X_nan = samples[cols].to_numpy()
            X_zero = np.nan_to_num(X_nan, nan=0.0)
            parts = {"hgb": [], "ridge": []}
            for fold in folds:
                tr = stride_rows(pos, fold.train, args.hgb_stride)
                parts["hgb"].append(make_hgb(False).fit(X_nan[tr], y_rank[tr]).predict(X_nan[fold.test]))
                parts["ridge"].append(make_ridge().fit(X_zero[fold.train], y_rank[fold.train]).predict(X_zero[fold.test]))
            for model, chunks in parts.items():
                preds[(model, name)] = np.concatenate(chunks)
            print(f"h={h} {name} done", flush=True)
        dates, y = samples["forecast_date"].to_numpy()[idx], y_raw[idx]
        pd.DataFrame({"ticker": samples["ticker"].to_numpy()[idx], "date": samples["date"].to_numpy()[idx], "forecast_date": dates,
                      "y": y, **{f"{m}_{k}": v for (m, k), v in preds.items()}}).to_parquet(out / f"predictions_h{h:02d}.parquet", index=False)
        for (model, name), p in preds.items():
            rows.append({"h": h, "model": model, "features": name, "n": len(sets[name]), **evaluate(p, y, dates, h)})
        for model in ("hgb", "ridge"):
            for a, b in pairs:
                d, t = ic_difference(preds[(model, a)], preds[(model, b)], y, dates, h)
                tests.append({"h": h, "model": model, "pair": f"{a} - {b}", "ic_diff": d, "nw_t": t})
    scores, paired = pd.DataFrame(rows), pd.DataFrame(tests)
    scores.to_csv(out / "scores.csv", index=False)
    paired.to_csv(out / "paired_tests.csv", index=False)
    print(md_table(scores.set_index("h"), "{:+.4f}"), "\n", md_table(paired.set_index("h"), "{:+.4f}"), sep="\n")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
