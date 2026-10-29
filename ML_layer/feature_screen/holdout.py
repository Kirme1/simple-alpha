"""Held-out check of feature selection: choose features on 2016-2022 only, score the choice on 2023-2026.

Selection (2016-2022 only, no 2023+ targets): a candidate feature outside the original 17 is kept when its
univariate rank IC has |Newey-West t| >= --select-t at any horizon; the original momentum group is dropped
when removing it raises ridge IC at h20 with paired t >= --select-t over the 2016-2022 folds.
Scoring: HGB and ridge on the 2023-2026 folds (expanding window, same purge/embargo), selected sets vs the original 17.

Usage: python -m ML_layer.feature_screen.holdout
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.edgar.features import fundamentals_start
from data_pipeline.utils.config import load_tickers
from .common import HORIZONS, REPO_ROOT, evaluate, ic_difference, load_screen_samples, make_folds, stride_rows, y_col, yr_col
from .hgb_paired import CANDIDATE_GROUPS, MOMENTUM
from .pipeline import make_hgb, make_ridge
from .screen import md_table
from .univariate import univariate_screen


def fit_predict(samples, cols, folds, h, pos, model, stride):
    X = samples[cols].to_numpy()
    y = samples[yr_col(h)].to_numpy()
    if model == "ridge":
        X = np.nan_to_num(X, nan=0.0)
    out = []
    for fold in folds:
        tr = stride_rows(pos, fold.train, stride) if model == "hgb" else fold.train
        est = make_hgb(False) if model == "hgb" else make_ridge()
        out.append(est.fit(X[tr], y[tr]).predict(X[fold.test]))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ticker-file", default=str(REPO_ROOT / "data_pipeline/tickers_sp500.yaml"))
    ap.add_argument("--data-directory", default=str(REPO_ROOT / "data_pipeline/local_data/raw_data"))
    ap.add_argument("--fundamentals-directory", default=str(REPO_ROOT / "data_pipeline/local_data/fundamentals"))
    ap.add_argument("--output-root", default=str(REPO_ROOT / "outputs/feature_screen/holdout"))
    ap.add_argument("--select-until", type=int, default=2022, help="last year used for selection; scoring starts the year after")
    ap.add_argument("--select-t", type=float, default=2.0)
    ap.add_argument("--hgb-stride", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)

    min_date = fundamentals_start(args.fundamentals_directory, load_tickers(Path(args.ticker_file))[0])
    samples, features, _ = load_screen_samples(
        Path(args.data_directory), Path(args.ticker_file), False, REPO_ROOT / "outputs/feature_screen/_cache", min_date,
        fundamentals_directory=Path(args.fundamentals_directory),
    )
    candidates = [f for f in features if f in {c for group in CANDIDATE_GROUPS for c in group}]
    orig = [f for f in features if f not in candidates]
    pos = samples["date_pos"].to_numpy()
    selection = samples[samples["year"] <= args.select_until]
    print(f"Window starts {min_date.date()}; selecting on years <= {args.select_until} ({len(selection)} rows), "
          f"{len(candidates)} candidates, {len(orig)} original features", flush=True)

    # 1. univariate selection of candidates
    stats = {h: univariate_screen(selection, candidates, h)[0] for h in HORIZONS}
    table = pd.concat({f"h{h}": s[["ic", "ic_t"]] for h, s in stats.items()}, axis=1)
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    best_t = pd.concat([s["ic_t"].abs() for s in stats.values()], axis=1).max(axis=1)
    selected = [f for f in candidates if best_t[f] >= args.select_t]
    table["selected"] = table.index.isin(selected)
    table.to_csv(out / "selection_univariate.csv")
    print(md_table(table, "{:+.4f}"), f"\nSelected: {selected}", sep="\n", flush=True)

    # 2. momentum drop on the selection folds (ridge)
    drop_rows = []
    for h in HORIZONS:
        folds = make_folds(samples, h, 2016, last_test_year=args.select_until)
        idx = np.concatenate([f.test for f in folds])
        dates, y = samples["forecast_date"].to_numpy()[idx], samples[y_col(h)].to_numpy()[idx]
        base = fit_predict(samples, orig, folds, h, pos, "ridge", 1)
        less = fit_predict(samples, [f for f in orig if f not in MOMENTUM], folds, h, pos, "ridge", 1)
        d, t = ic_difference(less, base, y, dates, h)
        drop_rows.append({"h": h, "ic_diff": d, "nw_t": t})
    drops = pd.DataFrame(drop_rows)
    drop_momentum = bool(drops.loc[drops["h"] == 20, "nw_t"].iloc[0] >= args.select_t)
    print("\nDrop-momentum ridge on selection folds (no_momentum - orig):", md_table(drops.set_index("h"), "{:+.4f}"),
          f"Drop momentum: {drop_momentum}", sep="\n", flush=True)

    # 3. score the choices on the held-out years
    sets = {"orig": orig, "selected": orig + selected}
    if drop_momentum:
        sets["no_momentum"] = [f for f in orig if f not in MOMENTUM]
    pairs = [(k, "orig") for k in sets if k != "orig"]
    rows, tests = [], []
    for h in HORIZONS:
        folds = make_folds(samples, h, args.select_until + 1)
        idx = np.concatenate([f.test for f in folds])
        dates, y = samples["forecast_date"].to_numpy()[idx], samples[y_col(h)].to_numpy()[idx]
        preds = {(m, k): fit_predict(samples, cols, folds, h, pos, m, args.hgb_stride) for k, cols in sets.items() for m in ("hgb", "ridge")}
        for (m, k), p in preds.items():
            rows.append({"h": h, "model": m, "features": k, "n": len(sets[k]), **evaluate(p, y, dates, h)})
        for m in ("hgb", "ridge"):
            for a, b in pairs:
                d, t = ic_difference(preds[(m, a)], preds[(m, b)], y, dates, h)
                tests.append({"h": h, "model": m, "pair": f"{a} - {b}", "ic_diff": d, "nw_t": t})
        print(f"h={h} scored", flush=True)
    scores, paired = pd.DataFrame(rows), pd.DataFrame(tests)
    scores.to_csv(out / "scores.csv", index=False)
    paired.to_csv(out / "paired_tests.csv", index=False)
    drops.to_csv(out / "selection_drop_momentum.csv", index=False)
    print(md_table(scores.set_index("h"), "{:+.4f}"), "\n", md_table(paired.set_index("h"), "{:+.4f}"), sep="\n")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
