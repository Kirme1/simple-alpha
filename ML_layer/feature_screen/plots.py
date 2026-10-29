"""PNG plots. Diverging maps are centred on zero; one colour per series where colour distinguishes models."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 130


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def ic_year_heatmap(yearly: pd.DataFrame, h: int, path):
    fig, ax = plt.subplots(figsize=(1 + 0.75 * yearly.shape[1], 0.35 * len(yearly) + 1.5))
    limit = max(0.02, float(np.nanpercentile(np.abs(yearly.to_numpy()), 98)))
    sns.heatmap(yearly, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".3f",
                annot_kws={"size": 7}, cbar_kws={"label": "mean daily rank IC"}, ax=ax)
    ax.set_title(f"Mean rank IC by calendar year, h={h}")
    _save(fig, path)


def ic_by_horizon(table: pd.DataFrame, path):
    """table: features x horizons of mean IC."""
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(table)), 4.5))
    width = 0.8 / table.shape[1]
    colours = ["#4c78a8", "#f58518", "#54a24b"]
    for i, h in enumerate(table.columns):
        ax.bar(np.arange(len(table)) + i * width, table[h], width, label=f"h={h}", color=colours[i % 3])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(table)) + width)
    ax.set_xticklabels(table.index, rotation=60, ha="right")
    ax.set_ylabel("mean rank IC")
    ax.set_title("Univariate IC by horizon")
    ax.legend()
    _save(fig, path)


def correlation_heatmap(corr: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(0.5 * len(corr) + 3, 0.5 * len(corr) + 2))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax)
    ax.set_title("Mean cross-sectional rank correlation between features")
    _save(fig, path)


def bar_with_error(mean: pd.Series, err: pd.Series, title: str, xlabel: str, path):
    order = mean.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(mean) + 1.5))
    ax.barh(order.index, order.values, xerr=None if err is None else err.reindex(order.index).values,
            color="#4c78a8", ecolor="#555555")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _save(fig, path)


def partial_effect_grid(curves: dict[int, pd.DataFrame], selected: pd.DataFrame, h: int, path):
    """curves: fold -> DataFrame(grid x feature); selected: folds x features booleans."""
    features = [f for f in selected.columns if selected[f].any()]
    if not features:
        return False
    ncols = min(4, len(features))
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows), squeeze=False, sharex=True)
    for ax, f in zip(axes.ravel(), features):
        folds = [y for y in curves if selected.loc[y, f]]
        stack = np.array([curves[y][f].to_numpy() for y in folds])
        grid = curves[folds[0]].index.to_numpy()
        for line in stack:
            ax.plot(grid, line, color="#4c78a8", alpha=0.25, lw=0.8)
        ax.plot(grid, stack.mean(axis=0), color="#d62728", lw=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{f} ({len(folds)}/{len(selected)} folds)", fontsize=8)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.supxlabel("cross-sectional rank of feature (centred)")
    fig.supylabel("partial effect on predicted rank return")
    fig.suptitle(f"Adaptive group LASSO partial effects, h={h} (blue: folds, red: mean)", fontsize=10)
    _save(fig, path)
    return True


def model_comparison(table: pd.DataFrame, h: int, path):
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(table) + 1.5))
    ax.barh(table.index, table["ic"], color="#4c78a8")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("pooled out-of-sample mean rank IC")
    ax.set_title(f"Model IC, h={h}")
    ax.invert_yaxis()
    _save(fig, path)
