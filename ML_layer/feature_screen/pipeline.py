"""Walk-forward model stage: ridge, HGB, adaptive group LASSO, composites and ablations per fold."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import group_lasso as gl
from .baselines import TOP_K, composite_predictions
from .common import Fold, groups_for, stage, stride_rows, y_col, yr_col
from .importance import permutation_importance


MIN_TRAIN_COVERAGE = 0.10  # features observed on fewer training rows are excluded from that fold's LASSO


def make_ridge():
    # Same specification as walk_forward.make_model("ridge").
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def make_hgb(fast: bool):
    # Same specification as walk_forward.make_model("hgb"); fewer iterations under --fast.
    return HistGradientBoostingRegressor(
        max_iter=60 if fast else 200, learning_rate=0.05 if not fast else 0.1, max_leaf_nodes=15,
        min_samples_leaf=500, l2_regularization=1.0, random_state=0,
    )


def run_horizon(samples, features, h, folds: list[Fold], args, basis, B_all, seed: int):
    """Returns dict of OOS predictions and per-fold diagnostics for one horizon."""
    X = samples[features].fillna(0.0).to_numpy()  # missing rank -> neutral 0
    X_nan = samples[features].to_numpy()  # HGB handles missing values natively
    pos = samples["date_pos"].to_numpy()
    fdates = samples["forecast_date"].to_numpy()
    y_raw = samples[y_col(h)].to_numpy()
    y_rank = samples[yr_col(h)].to_numpy()
    groups = groups_for(features)
    test_index = np.concatenate([f.test for f in folds])
    preds: dict[str, list[np.ndarray]] = {}
    store = lambda name, p: preds.setdefault(name, []).append(np.asarray(p, float))

    ridge_coefs, perm_rows, lasso_freq, lasso_full, lasso_curves, lasso_meta = {}, {}, {}, {}, {}, []
    for fold in folds:
        rng = np.random.default_rng([seed, h, fold.year])
        tr_all = fold.train
        tr = stride_rows(pos, tr_all, args.train_stride)
        te = fold.test

        with stage("ridge + baselines"):
            ridge = make_ridge().fit(X[tr_all], y_rank[tr_all])
            store("ridge", ridge.predict(X[te]))
            ridge_coefs[fold.year] = pd.Series(ridge[-1].coef_, index=features)
            for name, members in groups.items():
                cols = [features.index(f) for f in features if f not in members]
                if not cols:
                    continue
                dropped = make_ridge().fit(X[tr_all][:, cols], y_rank[tr_all])
                store(f"ridge_drop_{name}", dropped.predict(X[te][:, cols]))
            comp, _ = composite_predictions(X[tr], y_raw[tr], fdates[tr], X[te])
            for k, p in comp.items():
                store(f"composite_top{k}", p)

        with stage("hgb + permutation importance"):
            tr_h = stride_rows(pos, tr_all, args.hgb_stride)
            hgb = make_hgb(args.fast).fit(X_nan[tr_h], y_rank[tr_h])
            store("hgb", hgb.predict(X_nan[te]))
            perm_rows[fold.year] = permutation_importance(
                hgb, X_nan[te], y_raw[te], fdates[te], features, rng, n_repeats=1 if args.fast else 2,
            )

        with stage("adaptive group LASSO"):
            usable = samples[features].iloc[tr].notna().mean().to_numpy() >= MIN_TRAIN_COVERAGE
            ortho = gl.Orthonormaliser().fit(B_all[tr], usable)
            column_mask = np.repeat(usable, basis.k)  # inert groups (too little data) are left out of the fit
            Z, Zte = ortho.transform(B_all[tr])[:, column_mask], ortho.transform(B_all[te])[:, column_mask]
            fit = gl.adaptive_group_lasso(Z, y_rank[tr], pos[tr], basis.k, h, (h if args.embargo is None else args.embargo))
            coef_full = np.zeros((len(features), basis.k))
            coef_full[usable] = fit["coef"]
            selected_full = np.zeros(len(features), dtype=bool)
            selected_full[usable] = fit["selected"]
            step1_full = np.zeros(len(features), dtype=bool)
            step1_full[usable] = fit["step1_selected"]
            store("adaptive_group_lasso", fit["intercept"] + Zte @ fit["coef"].ravel())
            lasso_full[fold.year] = pd.Series(selected_full, index=features)
            lasso_curves[fold.year] = pd.DataFrame(gl.partial_effects(basis, ortho, coef_full).T, index=gl.GRID, columns=features)
            lasso_meta.append({"fold": fold.year, "alpha1": fit["alpha1"], "alpha2": fit["alpha2"], "n_selected": int(selected_full.sum()),
                               "n_step1": int(step1_full.sum()), "inert_features": ", ".join(np.array(features)[~usable])})
        with stage("LASSO stability selection"):
            sel = gl.stability_selection(
                Z, y_rank[tr], pos[tr], basis.k, fit["alpha1"], fit["alpha2"], args.n_subsamples, rng, n_jobs=args.n_jobs,
            )
            freq = np.zeros(len(features))
            freq[usable] = sel.mean(axis=0)
            lasso_freq[fold.year] = pd.Series(freq, index=features)
        print(f"  h={h:>2} fold {fold.year}: train {len(tr_all):>6} (strided {len(tr):>6}) test {len(te):>6} "
              f"| LASSO keeps {selected_full.sum()} groups", flush=True)

    return {
        "test_index": test_index,
        "preds": {name: np.concatenate(parts) for name, parts in preds.items()},
        "ridge_coefs": pd.DataFrame(ridge_coefs).T,
        "perm": pd.DataFrame(perm_rows).T,
        "lasso_freq_by_fold": pd.DataFrame(lasso_freq).T,
        "lasso_selected_by_fold": pd.DataFrame(lasso_full).T,
        "lasso_curves": lasso_curves,
        "lasso_meta": pd.DataFrame(lasso_meta),
    }
