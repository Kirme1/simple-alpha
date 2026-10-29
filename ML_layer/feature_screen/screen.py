"""Orchestrates the screen and writes tables, plots and summary.md."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.edgar.features import fundamentals_start
from data_pipeline.utils.config import load_tickers

from . import group_lasso as gl
from . import plots
from .common import (HORIZONS, REPO_ROOT, SEED, STAGE_SECONDS, evaluate, groups_for, ic_difference, load_screen_samples,
                     make_folds, nw_lags, stage, y_col)
from .importance import ridge_coefficient_summary
from .pipeline import run_horizon
from .univariate import average_rank_correlation, max_abs_correlation, univariate_screen

# Recommendation rule (stated in summary.md): LASSO selection frequency >= MIN_FREQ and |NW t| >= MIN_T,
# then greedily drop features with |rank corr| > MAX_CORR to an already-kept, higher-|t| feature.
MIN_FREQ, MIN_T, MAX_CORR = 0.6, 2.0, 0.7


def md_table(df: pd.DataFrame, floatfmt: str = "{:+.4f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    header = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    rule = "|" + "---|" * (df.shape[1] + 1)
    rows = ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([header, rule, *rows])


def recommend(table: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    candidates = table[(table["lasso_freq"] >= MIN_FREQ) & (table["ic_t"].abs() >= MIN_T)]
    kept: list[str] = []
    for f in candidates["ic_t"].abs().sort_values(ascending=False).index:
        if all(abs(corr.loc[f, k]) <= MAX_CORR for k in kept):
            kept.append(f)
    return kept


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature screener for cross-sectional return forecasting")
    p.add_argument("--data-directory", default=str(REPO_ROOT / "data_pipeline/local_data/raw_data"))
    p.add_argument("--ticker-file", default=str(REPO_ROOT / "data_pipeline/tickers_sp500.yaml"))
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs/feature_screen"))
    p.add_argument("--first-test-year", type=int, default=2016)
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--candidates", action="store_true", help="add r_12_7, log_mktcap, log_turnover (network on first run)")
    p.add_argument("--drop-features", nargs="+", default=[], help="base features to leave out of the screen")
    p.add_argument("--fundamentals", action="store_true", help="add the 7 EDGAR fundamentals (data_pipeline.edgar.build) as group 'fundamentals'")
    p.add_argument("--fundamentals-directory", default=str(REPO_ROOT / "data_pipeline/local_data/fundamentals"))
    p.add_argument("--min-date", default=None, help="first sample date; with --fundamentals defaults to when half the universe has them")
    p.add_argument("--no-purge", action="store_true", help="disable purging/embargo (leaky: overlapping targets in training)")
    p.add_argument("--embargo", type=int, default=None, help="embargo in trading days after the purge (default: h)")
    p.add_argument("--n-interior-knots", type=int, default=4)
    p.add_argument("--n-subsamples", type=int, default=50, help="block subsamples per fold for stability selection")
    p.add_argument("--train-stride", type=int, default=5, help="LASSO/composite fits use every n-th trading day")
    p.add_argument("--hgb-stride", type=int, default=2, help="HGB fits use every n-th trading day")
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--fast", action="store_true", help="last ~6 years of dates, last 3 test years, fewer subsamples/iterations")
    return p


def run(args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    out = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "tables").mkdir(parents=True)
    (out / "plots").mkdir()
    if args.fast:
        args.n_subsamples = min(args.n_subsamples, 8)
        args.train_stride = max(args.train_stride, 10)
        args.hgb_stride = max(args.hgb_stride, 4)
    print(f"Output: {out}", flush=True)

    with stage("load data + features", log=True):
        min_date = None
        probe = pd.read_parquet(Path(args.data_directory) / "AAPL_raw.parquet", columns=["date"])["date"]
        if args.fast:
            min_date = probe.max() - pd.DateOffset(years=6)
        elif args.min_date is not None:
            min_date = pd.Timestamp(args.min_date)
        elif args.fundamentals:
            tickers, _ = load_tickers(Path(args.ticker_file))
            min_date = fundamentals_start(args.fundamentals_directory, tickers)
            print(f"Common window starts {min_date.date()} (half the universe has gross_profitability)", flush=True)
        samples, features, notes = load_screen_samples(
            Path(args.data_directory), Path(args.ticker_file), args.candidates, REPO_ROOT / "outputs/feature_screen/_cache", min_date,
            drop_features=tuple(args.drop_features),
            fundamentals_directory=Path(args.fundamentals_directory) if args.fundamentals else None,
        )
        if args.fast:
            args.first_test_year = int(samples["year"].max()) - 2
    print(f"{len(samples)} samples, {samples['ticker'].nunique()} tickers, {samples['date'].min().date()} to "
          f"{samples['date'].max().date()}, {len(features)} features", flush=True)

    basis = gl.SplineBasis(args.n_interior_knots)
    with stage("spline expansion", log=True):
        B_all = basis.expand(samples[features].fillna(0.0).to_numpy())

    horizons = tuple(args.horizons)
    results, ranking, model_tables, ablations, yearly_all, recs, relaxed = {}, {}, {}, {}, {}, {}, {}
    folds_by_h = {h: make_folds(samples, h, args.first_test_year, purge=not args.no_purge, embargo=args.embargo) for h in horizons}
    test_index = np.concatenate([f.test for f in folds_by_h[horizons[0]]])
    test_df = samples.iloc[test_index]

    with stage("univariate screen: correlation", log=True):
        corr = average_rank_correlation(test_df, features)
        corr.to_csv(out / "tables/feature_rank_correlation.csv")
        plots.correlation_heatmap(corr, out / "plots/feature_correlation.png")
        redundancy = max_abs_correlation(corr)
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(features) for b in features[i + 1:] if abs(corr.loc[a, b]) > MAX_CORR]
        pd.DataFrame(pairs, columns=["feature_a", "feature_b", "rank_corr"]).to_csv(out / "tables/redundant_pairs.csv", index=False)

    ic_by_h = {}
    for h in horizons:
        print(f"=== horizon {h} ({len(folds_by_h[h])} folds, NW lags {nw_lags(h)}) ===", flush=True)
        with stage("univariate screen: IC", log=True):
            uni, yearly = univariate_screen(test_df, features, h)
            uni.to_csv(out / f"tables/univariate_h{h:02d}.csv")
            yearly.to_csv(out / f"tables/ic_by_year_h{h:02d}.csv")
            plots.ic_year_heatmap(yearly, h, out / f"plots/ic_by_year_h{h:02d}.png")
            ic_by_h[h] = uni["ic"]
            yearly_all[h] = yearly
        res = run_horizon(samples, features, h, folds_by_h[h], args, basis, B_all, args.seed)
        results[h] = res
        idx = res["test_index"]
        dates, y_raw = samples["forecast_date"].to_numpy()[idx], samples[y_col(h)].to_numpy()[idx]

        with stage("scoring + tables"):
            pd.DataFrame({"ticker": samples["ticker"].to_numpy()[idx], "date": samples["date"].to_numpy()[idx],
                          "forecast_date": dates, "y": y_raw, **res["preds"]}).to_parquet(out / f"tables/predictions_h{h:02d}.parquet", index=False)
            rows = {}
            for name, pred in res["preds"].items():
                row = evaluate(pred, y_raw, dates, h)
                row["ic_diff_vs_ridge"], row["ic_diff_t"] = (np.nan, np.nan) if name == "ridge" else ic_difference(
                    pred, res["preds"]["ridge"], y_raw, dates, h)
                rows[name] = row
            table = pd.DataFrame(rows).T
            model_tables[h] = table[~table.index.str.startswith("ridge_drop_")]
            table.to_csv(out / f"tables/model_comparison_h{h:02d}.csv")
            ab = {}
            for name, pred in res["preds"].items():
                if name.startswith("ridge_drop_"):
                    d, t = ic_difference(res["preds"]["ridge"], pred, y_raw, dates, h)
                    ab[name.removeprefix("ridge_drop_")] = {"ic_without_group": table.loc[name, "ic"],
                        "ic_full_ridge": table.loc["ridge", "ic"], "ic_loss_from_dropping": d, "nw_t": t,
                        "features": ", ".join(groups_for(features)[name.removeprefix("ridge_drop_")])}
            ablations[h] = pd.DataFrame(ab).T
            ablations[h].to_csv(out / f"tables/ablation_drop_group_h{h:02d}.csv")
            plots.model_comparison(model_tables[h], h, out / f"plots/model_ic_h{h:02d}.png")

            res["ridge_coefs"].to_csv(out / f"tables/ridge_coefs_by_fold_h{h:02d}.csv")
            ridge_sum = ridge_coefficient_summary(res["ridge_coefs"])
            ridge_sum.to_csv(out / f"tables/ridge_coef_summary_h{h:02d}.csv")
            res["perm"].to_csv(out / f"tables/hgb_permutation_importance_by_fold_h{h:02d}.csv")
            res["lasso_freq_by_fold"].to_csv(out / f"tables/lasso_selection_freq_by_fold_h{h:02d}.csv")
            res["lasso_selected_by_fold"].to_csv(out / f"tables/lasso_full_fit_selected_by_fold_h{h:02d}.csv")
            res["lasso_meta"].to_csv(out / f"tables/lasso_penalties_h{h:02d}.csv", index=False)
            freq = res["lasso_freq_by_fold"].mean()
            perm = res["perm"].mean()
            curve_rows = []
            for fold_year, curve in res["lasso_curves"].items():
                for f in features:
                    if res["lasso_selected_by_fold"].loc[fold_year, f]:
                        curve_rows.append(pd.DataFrame({"fold": fold_year, "feature": f, "rank": curve.index, "effect": curve[f].to_numpy()}))
            if curve_rows:
                pd.concat(curve_rows).to_csv(out / f"tables/partial_effects_h{h:02d}.csv", index=False)
            plots.partial_effect_grid(res["lasso_curves"], res["lasso_selected_by_fold"], h, out / f"plots/partial_effects_h{h:02d}.png")
            plots.bar_with_error(freq, None, f"Stability-selection frequency, h={h}", "share of (fold, subsample) fits selecting",
                                 out / f"plots/lasso_selection_freq_h{h:02d}.png")
            plots.bar_with_error(perm, res["perm"].std(ddof=1), f"HGB within-date permutation importance, h={h}",
                                 "drop in mean rank IC (± sd across folds)", out / f"plots/hgb_permutation_importance_h{h:02d}.png")
            plots.bar_with_error(ridge_sum["coef_mean"], ridge_sum["coef_std"], f"Ridge standardized coefficients, h={h}",
                                 "coefficient (± sd across folds)", out / f"plots/ridge_coefficients_h{h:02d}.png")

            rank = uni[["ic", "ic_t", "ic_ir", "pct_pos", "year_sign_share", "year_sign_flips"]].copy()
            rank["lasso_freq"], rank["perm_importance"] = freq, perm
            rank["perm_pos_folds"] = (res["perm"] > 0).mean()
            rank["ridge_sign_consistency"] = ridge_sum["sign_consistency"]
            rank["max_abs_corr"], rank["max_corr_with"] = redundancy["max_abs_corr"], redundancy["max_corr_with"]
            rank = rank.reindex(rank["ic_t"].abs().sort_values(ascending=False).index)
            rank.index.name = "feature"
            ranking[h] = rank
            rank.to_csv(out / f"tables/feature_ranking_h{h:02d}.csv")
            recs[h] = recommend(rank, corr)
            loose = rank[(rank["lasso_freq"] >= MIN_FREQ) | (rank["ic_t"].abs() >= MIN_T)]
            relaxed[h] = [f for f in loose["ic_t"].abs().sort_values(ascending=False).index]
            kept: list[str] = []
            for f in relaxed[h]:
                if all(abs(corr.loc[f, k]) <= MAX_CORR for k in kept):
                    kept.append(f)
            relaxed[h] = kept

    ic_h = pd.DataFrame(ic_by_h)
    ic_h.to_csv(out / "tables/ic_by_horizon.csv")
    plots.ic_by_horizon(ic_h, out / "plots/ic_by_horizon.png")
    pd.DataFrame({"seconds": STAGE_SECONDS}).to_csv(out / "tables/runtime_by_stage.csv")
    write_summary(out, args, features, notes, samples, ranking, model_tables, ablations, yearly_all, recs, relaxed, ic_h, pairs, horizons, folds_by_h)
    (out / "run_config.json").write_text(json.dumps({k: v for k, v in vars(args).items()}, indent=2, default=str))
    return out


def write_summary(out, args, features, notes, samples, ranking, model_tables, ablations, yearly_all, recs, relaxed, ic_h, pairs, horizons, folds_by_h):
    L = []
    purge = "off (leaky)" if args.no_purge else "on"
    L += ["# Feature screen", "",
          f"- Run: `{out.name}`, seed {args.seed}, `--fast`={args.fast}, candidates={args.candidates}",
          f"- Data: {len(samples):,} samples, {samples['ticker'].nunique()} tickers, {samples['date'].min().date()} to {samples['date'].max().date()}",
          f"- Folds: expanding window, one test year each from {args.first_test_year}; purge+embargo {purge}"
          f" (training target windows end at least h + embargo days before the first test day; embargo defaults to h).",
          f"- Target: cumulative excess log return over SPY, t to t+h. Features: cross-sectional percentile ranks per date. "
          f"Models fit the per-date rank of the target; IC/spread are scored against the raw excess return.",
          f"- IC t-stats: Newey-West (Bartlett) with max(h-1,1) lags = " + ", ".join(f"h{h}: {nw_lags(h)}" for h in horizons) + ".",
          f"- Excluded: `spy_ret_20d` (identical for all tickers on a date, so it has no cross-sectional IC) and earnings/FRED/calendar inputs.",
          f"- Spline group LASSO: quadratic B-splines, {args.n_interior_knots} interior knots, skglm GroupLasso, "
          f"{args.n_subsamples} block subsamples per fold, fits on every {args.train_stride}th trading day. HGB fits on every {args.hgb_stride}th.",
          ]
    L += [f"- {n}" for n in notes]
    L += ["", "## Go / no-go for a deep net", ""]
    verdicts = []
    for h in horizons:
        m = model_tables[h]
        row = {"ridge IC": m.loc["ridge", "ic"], "HGB IC": m.loc["hgb", "ic"], "HGB - ridge": m.loc["hgb", "ic_diff_vs_ridge"],
               "HGB - ridge NW t": m.loc["hgb", "ic_diff_t"], "adaptive LASSO IC": m.loc["adaptive_group_lasso", "ic"],
               "LASSO - ridge": m.loc["adaptive_group_lasso", "ic_diff_vs_ridge"], "LASSO - ridge NW t": m.loc["adaptive_group_lasso", "ic_diff_t"]}
        verdicts.append(pd.Series(row, name=f"h={h}"))
        beats = [n for n, t in (("HGB", row["HGB - ridge NW t"]), ("adaptive group LASSO", row["LASSO - ridge NW t"])) if t >= 2]
        L_note = f"h={h}: " + (f"{' and '.join(beats)} beat ridge at |t|>=2 -> some non-linear/sparse structure ridge misses."
                               if beats else "neither HGB nor the spline LASSO beats ridge at t>=2 -> no evidence of exploitable non-linearity.")
        L.append(L_note + "  ")
    L += ["", md_table(pd.DataFrame(verdicts), "{:+.4f}"), "",
          "Paired IC differences use the same dates and Newey-West errors. A deep net is only worth it if a flexible model beats ridge with t >= 2 "
          "at the horizon you care about; absent that, spend effort on features and data instead.", ""]
    for h in horizons:
        L += [f"## Horizon h={h}", "", "### Ranked features (by |NW t| of univariate IC)", ""]
        cols = ["ic", "ic_t", "ic_ir", "pct_pos", "lasso_freq", "perm_importance", "ridge_sign_consistency", "max_abs_corr", "max_corr_with",
                "year_sign_share", "year_sign_flips"]
        r = ranking[h][cols].copy()
        r["year_sign_flips"] = r["year_sign_flips"].astype(int)
        L += [md_table(r, "{:+.4f}"), ""]
        L += [f"**Recommended set (strict rule):** LASSO selection freq >= {MIN_FREQ} and |NW t| >= {MIN_T}, then drop features with "
              f"|rank corr| > {MAX_CORR} to a higher-|t| kept feature -> " + (", ".join(f"`{f}`" for f in recs[h]) if recs[h] else "**none qualify**"), ""]
        L += [f"**Relaxed set:** LASSO freq >= {MIN_FREQ} OR |NW t| >= {MIN_T}, same redundancy pruning -> "
              + (", ".join(f"`{f}`" for f in relaxed[h]) if relaxed[h] else "none"), "",
              "### Models (pooled out-of-sample)", "", md_table(model_tables[h].rename_axis("model"), "{:+.4f}"), "",
              "### Drop-one-group ridge ablation", "", md_table(ablations[h].astype({"features": str}).rename_axis("group_dropped"), "{:+.4f}"), ""]
    L += ["## IC across horizons", "", md_table(ic_h.rename(columns=lambda h: f"h={h}").rename_axis("feature"), "{:+.4f}"), ""]
    if pairs:
        L += ["## Redundant feature pairs (|rank corr| > 0.7)", ""] + [f"- `{a}` / `{b}`: {c:+.2f}" for a, b, c in pairs] + [""]
    L += ["## Runtime", "", md_table(pd.DataFrame({"seconds": STAGE_SECONDS}), "{:.1f}"), "",
          "Files: `tables/*.csv`, `plots/*.png`, `run_config.json`."]
    (out / "summary.md").write_text("\n".join(L))
