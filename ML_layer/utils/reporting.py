"""Console tables and plots for evaluate.py: model comparison, matched signal metrics, permutation importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..framework.metrics import SIGNAL_HORIZONS, paired_ic_difference, signal_metrics

DISPLAY_METRICS = [
    ("test_loss",                 "MSE"),
    ("test_mae",                  "MAE"),
    ("test_directional_accuracy", "Dir. Accuracy"),
    ("test_sharpe",               "Sharpe (day+1)"),
]

RIDGE_CONFIG_NAME = "Ridge-cfg"  # ridge on the config's features, ranked like the TFT's inputs
TFT_NAME = "PTF-TFT"
MODEL_ORDER = ("Ridge", RIDGE_CONFIG_NAME, TFT_NAME)


def print_comparison_table(results: dict[str, dict], split: str = "test") -> None:
    col_width = 18
    model_names = list(results.keys())

    header = f"{'Metric':<28}" + "".join(f"{n:>{col_width}}" for n in model_names)
    sep = "=" * len(header)
    print(f"\n{sep}\n{split.upper()} SET COMPARISON\n{sep}")
    print(header)
    print("-" * len(header))

    for metric_key, display_name in DISPLAY_METRICS:
        row = f"{display_name:<28}"
        for name in model_names:
            val = results[name].get(metric_key, float("nan"))
            row += f"{val:>{col_width}.4f}"
        print(row)

    print(sep + "\n")


def save_prediction_distribution_plot(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> list[Path]:
    """Plot raw prediction and target distributions across all forecast horizons."""
    if not predictions:
        return []

    model_names = [name for name in MODEL_ORDER if name in predictions]
    if not model_names:
        return []

    reference_target = np.asarray(predictions[model_names[0]][1])
    horizon_count = min(
        [reference_target.shape[1]]
        + [np.asarray(predictions[name][0]).shape[1] for name in model_names]
        + [np.asarray(predictions[name][1]).shape[1] for name in model_names]
    )
    if horizon_count == 0:
        return []

    distributions = {"Actual": reference_target[:, :horizon_count].reshape(-1)}
    distributions.update({
        name: np.asarray(predictions[name][0])[:, :horizon_count].reshape(-1)
        for name in model_names
    })
    distributions = {
        name: values[np.isfinite(values)]
        for name, values in distributions.items()
    }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, (name, values) in enumerate(distributions.items()):
        if not values.size:
            continue
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.scatter(
            np.arange(values.size),
            values,
            s=8,
            alpha=0.5,
            color=colors[index % len(colors)],
            edgecolors="none",
        )
        axis.set_title(name)
        axis.set_xlabel("Sample / horizon index")
        axis.set_ylabel("Cumulative excess log return vs SPY")
        figure.tight_layout()
        image_path = output_path.with_name(
            f"{output_path.stem}_{name.lower().replace(' ', '_')}{output_path.suffix}"
        )
        figure.savefig(image_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(image_path)
    return saved_paths


SIGNAL_METRICS = [
    ("ic",           "IC (mean daily rank corr)", "{:.4f}"),
    ("ic_tstat",     "IC t-stat (Newey-West)",    "{:.2f}"),
    ("r2_oos",       "R² vs zero forecast",       "{:.4%}"),
    ("mse",          "MSE (raw units)",           "{:.3e}"),
    ("ls_spread",    "Top-bottom 20% spread",     "{:.4%}"),
    ("ls_sharpe",    "Long-short Sharpe (ann.)",  "{:.2f}"),
]


def matched_signal_comparison(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    keys: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    """Score models on the (ticker, first forecast date) pairs they all predicted.

    The baselines and the TFT use different data pipelines and ticker sets, so
    their raw test samples differ; matching makes the comparison like for like.
    Predictions must be cumulative excess log returns in raw units.
    """
    names = [name for name in MODEL_ORDER if name in predictions and name in keys]
    if not names:
        return {}
    frames = {}
    for name in names:
        pred, target = (np.asarray(a) for a in predictions[name])
        index = pd.MultiIndex.from_frame(keys[name][["ticker", "date"]].assign(date=lambda k: pd.to_datetime(k["date"])))
        frames[name] = (pd.DataFrame(pred, index=index), pd.DataFrame(target, index=index))
    common = frames[names[0]][0].index
    for name in names[1:]:
        common = common.intersection(frames[name][0].index)
    common = common.sort_values()
    # The matched metrics below use only the common samples; report how much of each model's set that is.
    for name in names:
        total = len(frames[name][0])
        share = len(common) / total
        print(f"{name}: {total} test samples, {len(common)} matched ({share:.1%})")
        if share < 0.99:
            print(f"Warning: {1 - share:.1%} of {name}'s test samples have no counterpart in the other model's set.")
    if common.empty:
        print("No overlapping (ticker, date) test samples between models; skipping signal metrics.")
        return {}

    target = frames[names[-1]][1].loc[common].to_numpy()
    for name in names[:-1]:
        mismatch = np.abs(frames[name][1].loc[common].to_numpy() - target).max()
        if mismatch > 1e-4:
            print(f"Warning: {name} targets differ from {names[-1]} by up to {mismatch:.2e} on matched samples.")
    dates = common.get_level_values("date").to_numpy()
    preds = {name: frames[name][0].loc[common].to_numpy() for name in names}
    results = {name: signal_metrics(preds[name], target, dates, SIGNAL_HORIZONS) for name in names}
    last = names[-1]
    for other in names[:-1]:
        paired = paired_ic_difference(preds[last], preds[other], target, dates, SIGNAL_HORIZONS)
        suffix = "" if len(names) == 2 else f"_vs_{other}"
        results[last].update({f"{k}{suffix}": v for k, v in paired.items()})

    n_tickers = common.get_level_values("ticker").nunique()
    n_dates = len(np.unique(dates))
    col_width = 14
    header = f"{'Metric':<28}" + "".join(f"{f'{n} h{h}':>{col_width}}" for h in SIGNAL_HORIZONS for n in names)
    sep = "=" * len(header)
    print(f"\n{sep}\nSIGNAL METRICS — {len(common)} matched samples, {n_tickers} tickers, {n_dates} dates\n{sep}")
    print(header)
    print("-" * len(header))
    for key, label, fmt in SIGNAL_METRICS:
        row = f"{label:<28}"
        for h in SIGNAL_HORIZONS:
            for name in names:
                row += f"{fmt.format(results[name][f'{key}_h{h:02d}']):>{col_width}}"
        print(row)
    if len(names) >= 2:
        print("-" * len(header))
        for other in names[:-1]:
            suffix = "" if len(names) == 2 else f"_vs_{other}"
            for h in SIGNAL_HORIZONS:
                diff, t = results[last][f"ic_diff_h{h:02d}{suffix}"], results[last][f"ic_diff_tstat_h{h:02d}{suffix}"]
                print(f"IC {last} − {other}, h{h}: {diff:+.4f} (Newey-West t = {t:.2f})")
    print(sep + "\n")
    return results


def print_permutation_importance(importance: pd.DataFrame) -> None:
    sort_key = f"ic_diff_h{SIGNAL_HORIZONS[len(SIGNAL_HORIZONS) // 2]:02d}"
    header = f"{'Feature':<28} {'Permuted':<15}" + "".join(
        f"{f'ΔIC h{h}':>10}{'t':>7}" for h in SIGNAL_HORIZONS
    )
    sep = "=" * len(header)
    print(f"\n{sep}\nTFT PERMUTATION IMPORTANCE — drop in test IC when the feature is shuffled")
    print(f"(positive = feature helps ranking; |t| > 2 is distinguishable from noise)\n{sep}")
    print(header)
    print("-" * len(header))
    for feature, row in importance.sort_values(sort_key, ascending=False).iterrows():
        line = f"{feature:<28} {row['mode']:<15}"
        for h in SIGNAL_HORIZONS:
            line += f"{row[f'ic_diff_h{h:02d}']:>+10.4f}{row[f'ic_diff_tstat_h{h:02d}']:>7.2f}"
        print(line)
    print(sep + "\n")
