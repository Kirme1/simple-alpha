"""Adaptive group LASSO on quadratic B-spline expansions (Freyberger, Neuhierl & Weber, 2020).

Solver: skglm.GroupLasso (celer-style working-set coordinate descent). Chosen over
``group-lasso`` because it accepts per-group penalty weights natively
(needed for the adaptive step), it is much faster on tall matrices, and it has a
scikit-learn API. ``group-lasso`` has no per-group weight argument and is
FISTA-based (slow convergence at n ~ 1e5).

Each feature's spline columns form one group. Columns are centred and
orthonormalised within the group (Z_g'Z_g / n = I) so the penalty does not depend
on the basis parametrisation. The B-spline basis has a fixed uniform knot grid on
the rank scale [-0.5, 0.5] (ranks are uniform by construction, so uniform knots
are also quantile knots) and constant extrapolation.
"""

import numpy as np
from joblib import Parallel, delayed
from skglm import GroupLasso
from sklearn.preprocessing import SplineTransformer

GRID = np.linspace(-0.5, 0.5, 101)
N_ALPHAS = 10
ALPHA_RANGE = (0.95, 0.02)  # grid from 0.95 * alpha_max down to 0.02 * alpha_max
INNER_SPLITS = 3
ACTIVE_TOL = 1e-10


class SplineBasis:
    def __init__(self, n_interior_knots: int):
        self.transformer = SplineTransformer(
            degree=2, n_knots=n_interior_knots + 2, knots="uniform", extrapolation="constant", include_bias=True,
        ).fit(np.array([[-0.5], [0.5]]))
        self.n_basis = int(self.transformer.n_features_out_)  # interior knots + 3
        self.k = self.n_basis - 1  # group size after dropping the null direction

    def expand(self, X: np.ndarray) -> np.ndarray:
        """[n, p] ranks -> [n, p, n_basis] float32. NaN must already be filled."""
        out = np.empty((X.shape[0], X.shape[1], self.n_basis), dtype=np.float32)
        for j in range(X.shape[1]):
            out[:, j, :] = self.transformer.transform(X[:, [j]])
        return out


class Orthonormaliser:
    """Per-group centring and whitening fitted on training rows only."""

    def fit(self, B: np.ndarray, usable: np.ndarray | None = None) -> "Orthonormaliser":
        """``usable`` [p] bool marks features with enough non-missing training rows; others become inert (all-zero) groups."""
        n, p, nb = B.shape
        self.mu = B.mean(axis=0, dtype=np.float64)
        self.W = np.empty((p, nb, nb - 1))
        for j in range(p):
            C = B[:, j, :].astype(np.float64) - self.mu[j]
            if usable is not None and not usable[j]:
                self.W[j] = 0.0
                continue
            evals, V = np.linalg.eigh(C.T @ C / n)
            # Partition of unity makes exactly one direction null once centred: drop it.
            evals, V = evals[1:], V[:, 1:]
            # Directions with (numerically) no variance, e.g. basis functions never active, stay zero.
            scale = np.where(evals > 1e-10 * max(evals.max(), 1e-300), 1 / np.sqrt(np.maximum(evals, 1e-300)), 0.0)
            self.W[j] = V * scale
        return self

    def transform(self, B: np.ndarray) -> np.ndarray:
        n, p, _ = B.shape
        k = self.W.shape[2]
        Z = np.empty((n, p * k))
        for j in range(p):
            Z[:, j * k:(j + 1) * k] = (B[:, j, :] - self.mu[j]) @ self.W[j]
        return Z


def _fit(Z, y, k, alpha, weights=None):
    model = GroupLasso(groups=k, alpha=alpha, weights=weights, fit_intercept=True, tol=1e-4, max_iter=100)
    model.fit(Z, y)
    return model.coef_.reshape(-1, k), float(model.intercept_)


def _group_norms(coef):
    return np.linalg.norm(coef, axis=1)


def _alpha_grid(Z, y, k, weights):
    n = len(y)
    scores = np.linalg.norm(((Z.T @ (y - y.mean())) / n).reshape(-1, k), axis=1)
    if weights is not None:
        scores = scores / weights
    return scores.max() * np.geomspace(*ALPHA_RANGE, N_ALPHAS)


def _inner_splits(pos, h, embargo):
    edges = np.quantile(np.unique(pos), np.linspace(0, 1, INNER_SPLITS + 2))
    for j in range(1, INNER_SPLITS + 1):
        val = (pos >= edges[j]) & (pos < edges[j + 1] if j < INNER_SPLITS else pos <= edges[j + 1])
        train = pos < edges[j] - h - embargo
        if train.sum() > 100 and val.sum() > 100:
            yield train, val


def choose_alpha(Z, y, pos, k, h, embargo, weights=None):
    """Inner expanding-window time-series CV (purged) on validation MSE; never returns an empty model."""
    grid = _alpha_grid(Z, y, k, weights)
    mse = np.zeros(len(grid))
    splits = list(_inner_splits(pos, h, embargo))
    for train, val in splits:
        for i, alpha in enumerate(grid):
            coef, intercept = _fit(Z[train], y[train], k, alpha, weights)
            mse[i] += np.mean((y[val] - intercept - Z[val] @ coef.ravel()) ** 2)
    best = int(np.argmin(mse)) if splits else len(grid) // 2
    while best < len(grid) - 1 and not (_group_norms(_fit(Z, y, k, grid[best], weights)[0]) > ACTIVE_TOL).any():
        best += 1  # smaller alpha
    return float(grid[best])


def adaptive_group_lasso(Z, y, pos, k, h, embargo):
    """Two-step adaptive group LASSO. Returns dict with coef (zeros for dropped groups), intercept, alphas."""
    alpha1 = choose_alpha(Z, y, pos, k, h, embargo)
    coef1, _ = _fit(Z, y, k, alpha1)
    norms1 = _group_norms(coef1)
    active = norms1 > ACTIVE_TOL
    columns = np.repeat(active, k)
    weights = 1.0 / norms1[active]
    alpha2 = choose_alpha(Z[:, columns], y, pos, k, h, embargo, weights)
    coef2_active, intercept = _fit(Z[:, columns], y, k, alpha2, weights)
    coef2 = np.zeros_like(coef1)
    coef2[active] = coef2_active
    return {"coef": coef2, "intercept": intercept, "alpha1": alpha1, "alpha2": alpha2,
            "selected": _group_norms(coef2) > ACTIVE_TOL, "step1_selected": active}


def _subsample_fit(Z, y, mask, k, alpha1, alpha2):
    Zs, ys = Z[mask], y[mask]
    coef1, _ = _fit(Zs, ys, k, alpha1)
    norms1 = _group_norms(coef1)
    active = norms1 > ACTIVE_TOL
    selected = np.zeros(len(norms1), dtype=bool)
    if active.any():
        coef2, _ = _fit(Zs[:, np.repeat(active, k)], ys, k, alpha2, 1.0 / norms1[active])
        selected[active] = _group_norms(coef2) > ACTIVE_TOL
    return selected


def stability_selection(Z, y, pos, k, alpha1, alpha2, n_subsamples, rng, n_jobs=1, block_len=20, fraction=0.5):
    """Selection indicator [n_subsamples, n_groups] over block subsamples of training dates.

    Dates are cut into consecutive blocks of ``block_len`` (sampled) dates; each
    subsample keeps a random ``fraction`` of the blocks, preserving time order
    and the serial dependence inside blocks. Penalties are held fixed.
    """
    uniq = np.unique(pos)
    block_of_date = np.arange(len(uniq)) // block_len
    n_blocks = block_of_date.max() + 1
    row_block = block_of_date[np.searchsorted(uniq, pos)]
    masks = []
    for _ in range(n_subsamples):
        chosen = rng.choice(n_blocks, size=max(1, int(round(fraction * n_blocks))), replace=False)
        masks.append(np.isin(row_block, chosen))
    results = Parallel(n_jobs=n_jobs)(delayed(_subsample_fit)(Z, y, m, k, alpha1, alpha2) for m in masks)
    return np.array(results)


def partial_effects(basis: SplineBasis, ortho: Orthonormaliser, coef: np.ndarray) -> np.ndarray:
    """Fitted contribution of each feature on the rank grid: [p, len(GRID)]."""
    Bg = basis.transformer.transform(GRID[:, None])
    return np.array([(Bg - ortho.mu[j]) @ ortho.W[j] @ coef[j] for j in range(coef.shape[0])])
