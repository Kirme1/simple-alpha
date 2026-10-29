"""Learning curve over the number of training tickers: does HGB overtake ridge as the cross-section grows?

Design (isolates the effect of training-set size from test-set noise):
* A fixed held-out set of ``--test-tickers`` tickers is the only thing ever scored.
* Training universes of size n are random draws from the remaining tickers, disjoint from the test set.
* Features and the rank target are re-ranked per date *within* each universe (ranking a rank is a no-op,
  so this equals ranking the raw values among that universe's tickers).
* Same walk-forward folds, purge and embargo as the screener; ridge vs HGB on the rank target, scored on the
  held-out tickers with the screener's IC and Newey-West paired difference.
The log-linear extrapolation to larger n at the end is indicative only.

With ``--extra-ticker-file/--extra-data-directory`` (e.g. mid-caps) there are two held-out sets (large and extra) and two
training pools: "large" (large-cap pool only) and "mixed" (large + extra pool, drawn uniformly), so one can see whether
adding a different kind of ticker helps, and whether the HGB-vs-ridge gap depends on the universe.

Usage: python -m ML_layer.feature_screen.subset_experiment [--sizes 10 20 40 80 120] [--draws 4] [--fast]
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline.utils.config import load_tickers
from .common import REPO_ROOT, SEED, evaluate, ic_difference, load_screen_samples, make_folds, rank_features, stride_rows, y_col, yr_col
from .pipeline import make_hgb, make_ridge
from .screen import md_table


def rerank(frame: pd.DataFrame, features: list[str], horizons) -> pd.DataFrame:
    frame = rank_features(frame, features)
    for h in horizons:
        frame[yr_col(h)] = frame.groupby("date")[yr_col(h)].rank(pct=True) - 0.5
    return frame.reset_index(drop=True)


def score_universe(train_df, test_df, features, h, first_test_year, hgb_stride, fast):
    """Pooled held-out predictions of ridge and HGB trained on train_df, scored on test_df."""
    Xtr, Xte = train_df[features].fillna(0.0).to_numpy(), test_df[features].fillna(0.0).to_numpy()
    ytr = train_df[yr_col(h)].to_numpy()
    ptr, pte = train_df["date_pos"].to_numpy(), test_df["date_pos"].to_numpy()
    preds = {"ridge": [], "hgb": []}
    index = []
    for fold in make_folds(test_df, h, first_test_year):
        last_train_pos = pte[fold.train].max()  # same purged cutoff as the screener's folds
        tr = np.flatnonzero(ptr <= last_train_pos)
        te = fold.test
        preds["ridge"].append(make_ridge().fit(Xtr[tr], ytr[tr]).predict(Xte[te]))
        tr_h = stride_rows(ptr, tr, hgb_stride)
        preds["hgb"].append(make_hgb(fast).fit(Xtr[tr_h], ytr[tr_h]).predict(Xte[te]))
        index.append(te)
    index = np.concatenate(index)
    dates, y = test_df["forecast_date"].to_numpy()[index], test_df[y_col(h)].to_numpy()[index]
    p = {k: np.concatenate(v) for k, v in preds.items()}
    diff, t = ic_difference(p["hgb"], p["ridge"], y, dates, h)
    return {"ic_ridge": evaluate(p["ridge"], y, dates, h)["ic"], "ic_hgb": evaluate(p["hgb"], y, dates, h)["ic"],
            "ic_diff": diff, "ic_diff_t": t}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-directory", default=str(REPO_ROOT / "data_pipeline/local_data/raw_data"))
    ap.add_argument("--ticker-file", default=str(REPO_ROOT / "data_pipeline/tickers_sp500.yaml"))
    ap.add_argument("--extra-data-directory", default=None, help="second universe, e.g. mid-caps")
    ap.add_argument("--extra-ticker-file", default=None)
    ap.add_argument("--output-root", default=str(REPO_ROOT / "outputs/feature_screen/subset_experiment"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80, 120])
    ap.add_argument("--draws", type=int, default=4, help="random training universes per size")
    ap.add_argument("--test-tickers", type=int, default=40, help="held-out tickers per universe")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--hgb-stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fast", action="store_true", help="last 6 years, 2 draws, 3 sizes, light HGB")
    args = ap.parse_args()

    out = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    min_date = pd.Timestamp("2020-10-15") if args.fast else None
    if args.fast:
        args.draws, args.sizes = 2, args.sizes[:3]
    extra, extra_names = (), set()
    if args.extra_ticker_file:
        extra = ((Path(args.extra_data_directory), Path(args.extra_ticker_file)),)
        extra_names = set(load_tickers(Path(args.extra_ticker_file))[0])
    samples, features, _ = load_screen_samples(Path(args.data_directory), Path(args.ticker_file), False,
                                               REPO_ROOT / "outputs/feature_screen/_cache", min_date, extra)
    if args.fast:
        args.first_test_year = int(samples["year"].max()) - 2
    all_tickers = set(samples["ticker"].unique())
    large = np.array(sorted(all_tickers - extra_names)) if extra else np.array(sorted(all_tickers))
    mid = np.array(sorted(all_tickers & extra_names)) if extra else np.array([], dtype=str)
    rng = np.random.default_rng(args.seed)
    test_sets = {"large": rng.choice(large, size=args.test_tickers, replace=False)}
    if len(mid):
        test_sets["extra"] = rng.choice(mid, size=args.test_tickers, replace=False)
    pools = {"large": np.setdiff1d(large, test_sets["large"])}
    if len(mid):
        pools["mixed"] = np.concatenate([pools["large"], np.setdiff1d(mid, test_sets["extra"])])
    print(f"{len(large)} large + {len(mid)} extra tickers; test sets {({k: len(v) for k, v in test_sets.items()})}; "
          f"pools {({k: len(v) for k, v in pools.items()})}", flush=True)
    test_dfs = {name: rerank(samples[samples["ticker"].isin(t)].copy(), features, args.horizons) for name, t in test_sets.items()}

    rows = []
    for pool_name, pool in pools.items():
        for n in [n for n in args.sizes if n <= len(pool)]:
            for draw in range(args.draws if n < len(pool) else 1):
                chosen = np.random.default_rng([args.seed, n, draw, len(pool)]).choice(pool, size=n, replace=False)
                train_df = rerank(samples[samples["ticker"].isin(chosen)].copy(), features, args.horizons)
                for test_name, test_df in test_dfs.items():
                    for h in args.horizons:
                        r = score_universe(train_df, test_df, features, h, args.first_test_year, args.hgb_stride, args.fast)
                        rows.append({"train_pool": pool_name, "test_set": test_name, "n_train_tickers": n, "draw": draw, "h": h, **r})
                        print(f"  train={pool_name} n={n:>4} draw={draw} test={test_name} h={h:>2}: ridge {r['ic_ridge']:+.4f} "
                              f"hgb {r['ic_hgb']:+.4f} diff {r['ic_diff']:+.4f} (t {r['ic_diff_t']:+.2f})", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    keys = ["h", "test_set", "train_pool", "n_train_tickers"]
    agg = res.groupby(keys).agg(
        draws=("draw", "size"), ic_ridge=("ic_ridge", "mean"), ic_hgb=("ic_hgb", "mean"), ic_diff=("ic_diff", "mean"),
        ic_diff_sd=("ic_diff", "std"), t_mean=("ic_diff_t", "mean"), share_t_ge_2=("ic_diff_t", lambda s: (s >= 2).mean()))
    agg.to_csv(out / "summary.csv")

    colours = {"large": "#4c78a8", "mixed": "#f58518"}
    fig, axes = plt.subplots(len(args.horizons), 2 * len(test_dfs), figsize=(5 * 2 * len(test_dfs), 3.6 * len(args.horizons)), squeeze=False)
    notes = []
    for hi, h in enumerate(args.horizons):
        for ti, test_name in enumerate(test_dfs):
            ax_ic, ax_t = axes[hi, 2 * ti], axes[hi, 2 * ti + 1]
            for pool_name in pools:
                d = res[(res.h == h) & (res.test_set == test_name) & (res.train_pool == pool_name)]
                a = agg.loc[(h, test_name, pool_name)]
                c = colours.get(pool_name, "#54a24b")
                ax_ic.plot(a.index, a.ic_ridge, color=c, marker="o", ls="-", label=f"ridge, {pool_name} pool")
                ax_ic.plot(a.index, a.ic_hgb, color=c, marker="s", ls="--", label=f"HGB, {pool_name} pool")
                ax_t.scatter(d.n_train_tickers, d.ic_diff_t, color=c, alpha=0.4, s=14)
                ax_t.plot(a.index, a.t_mean, color=c, marker="o", label=f"{pool_name} pool")
                if len(a) >= 3:  # indicative log-linear trend of the IC gap
                    slope, icpt = np.polyfit(np.log(d.n_train_tickers), d.ic_diff, 1)
                    notes.append(f"h={h}, test={test_name}, train={pool_name}: IC(HGB) − IC(ridge) ≈ {icpt:+.4f} {slope:+.4f}·ln(n) "
                                 f"(→ {icpt + slope * np.log(500):+.4f} at n=500, {icpt + slope * np.log(1000):+.4f} at n=1000; indicative only)")
            for ax, title in ((ax_ic, f"h={h}, test={test_name}: held-out IC"), (ax_t, f"h={h}, test={test_name}: NW t of HGB − ridge")):
                ax.set_xscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel("training tickers"); ax.legend(fontsize=7)
            ax_t.axhline(2, color="k", ls="--", lw=0.8); ax_t.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=130)
    show = agg.reset_index().astype({"n_train_tickers": str, "draws": str}).set_index("h")
    md = ["# Subset-size experiment", "",
          f"Train on n random tickers, score on fixed held-out tickers ({args.test_tickers} per universe), yearly purged walk-forward folds from "
          f"{args.first_test_year}, features re-ranked within each universe. {args.draws} random draws per size.", "",
          md_table(show, "{:+.4f}"), "", *[f"- {n}" for n in notes], "",
          "Read: if the HGB-minus-ridge gap and its t grow with n, more tickers help a flexible model; if flat, they don't. "
          "Held-out sets are small, so t-stats are noisy; compare trends rather than single points."]
    (out / "summary.md").write_text("\n".join(md))
    print(f"\n{out}\n" + "\n".join(md[4:]))


if __name__ == "__main__":
    main()
