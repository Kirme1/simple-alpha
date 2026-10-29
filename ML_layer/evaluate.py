"""
evaluate.py — Side-by-side comparison of TFT vs baseline models on the test set.

Usage:
    python -m ML_layer.evaluate [--config ML_layer/model-configs/ptf-config.yaml] [--ridge-alpha 1.0]

The config is loaded with its `test:` section merged in (see TrainingArgs), so the data, features and model are
the ones the checkpoint was trained with; set MLFLOW_RUN_ID to pick the checkpoint.

What it does:
  1. Builds a BasicLightningDataModule for baselines (same data dir and splits as the TFT config).
  2. Fits PersistenceBaseline and RidgeBaseline on the train split and scores them on the test split.
  3. Runs trainer.test() for the TFT with the checkpoint's weights.
  4. Prints the comparison tables (utils/reporting.py) and, on request, the TFT's permutation importance (utils/importance.py).

Baselines always use BasicLightningDataModule because PTFLightningDataModule batches are dicts
incompatible with sklearn. Both use the same raw data and chronological split.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from data_pipeline.interfaces import BasicLightningDataModule
from .framework.baselines import PersistenceBaseline, RidgeBaseline
from .utils.importance import permutation_importance
from .utils.reporting import (
    RIDGE_CONFIG_NAME,
    TFT_NAME,
    matched_signal_comparison,
    print_comparison_table,
    print_permutation_importance,
    save_prediction_distribution_plot,
)
from data_pipeline.utils.config import REPO_ROOT
from .train_model import DEFAULT_CONFIG, assemble_training_components, setup_training_environment
from .utils import TrainingArgs

PLOTS_DIR = REPO_ROOT / "ML_layer" / "plots"


def build_baseline_data_module(config, tft_inputs: bool = False, split: str = "test") -> BasicLightningDataModule:
    """Always use BasicLightningDataModule for baselines, even when testing a PTF model.

    Extracts data_directory and split_ratios from whatever dataset config was given. By default the
    baselines use every observed feature with per-window standardisation. With ``tft_inputs`` they see
    the config's features instead, ranked per date when the config says so and not window-standardised,
    which are the TFT's own inputs. ``split`` picks the split the baselines are scored on ("test" or "val").
    """
    dataset_args = dict(config.dataset["arguments"])
    tft_input_args = (
        dict(features=dataset_args["features"], xs_rank_features=dataset_args.get("xs_rank_features", False),
             standardize_windows=False)
        if tft_inputs else {}
    )
    return BasicLightningDataModule(
        **tft_input_args,
        data_directory=dataset_args["data_directory"],
        split_ratios=dataset_args["split_ratios"],
        split_seed=dataset_args.get("split_seed", 42),
        window_size=dataset_args.get("window_size", 60),
        stride=dataset_args.get("stride", 1),
        compute_stats=True,
        apply_normalization="standardize",
        include_val=False,
        stream_test=True,
        stream_split=split,
        start_date=dataset_args.get("start_date"),
        fundamentals_directory=dataset_args.get("fundamentals_directory"),
        pin_memory=False,
        ticker_file=dataset_args.get("ticker_file"),
        **dict(config.dataloader),
    )


def evaluate_baselines(config, split: str, ridge_alpha: float):
    """Score the persistence and both ridge baselines.

    Returns (metrics by model, (prediction, target) by model in raw units, forecast keys by model).
    """
    results: dict[str, dict] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    keys: dict[str, pd.DataFrame] = {}

    def score(name: str, baseline, data_module) -> None:
        metrics, pred, target = baseline.evaluate_with_predictions(data_module.iter_test_batches())
        std = float(data_module.target_std)
        results[name] = metrics
        predictions[name] = (pred * std, target * std)

    print("Setting up baseline data module...")
    baseline_dm = build_baseline_data_module(config, split=split)
    baseline_dm.setup(stage="test")

    print("Evaluating Persistence baseline...")
    persistence = PersistenceBaseline().fit(baseline_dm.train_dataloader())
    score(PersistenceBaseline.name, persistence, baseline_dm)

    print(f"Fitting Ridge baseline (alpha={ridge_alpha})...")
    ridge = RidgeBaseline(alpha=ridge_alpha).fit(baseline_dm.train_dataloader())
    print("Evaluating Ridge baseline...")
    score(RidgeBaseline.name, ridge, baseline_dm)
    keys[RidgeBaseline.name] = baseline_dm.test_keys

    # Ridge on the TFT's inputs (config features, ranked, no window standardisation). The first module's
    # training windows are released before this one is built to keep memory down.
    baseline_dm.train_dataset = None
    config_dm = build_baseline_data_module(config, tft_inputs=True, split=split)
    config_dm.setup(stage="test")
    print(f"Fitting {RIDGE_CONFIG_NAME} baseline on {len(config_dm.features)} config features "
          f"(ranked: {config_dm.xs_rank_features}, alpha={ridge_alpha})...")
    ridge_cfg = RidgeBaseline(alpha=ridge_alpha).fit(config_dm.train_dataloader())
    score(RIDGE_CONFIG_NAME, ridge_cfg, config_dm)
    keys[RIDGE_CONFIG_NAME] = config_dm.test_keys
    config_dm.train_dataset = None

    return results, predictions, keys


def evaluate_tft(config, split: str, permutation_repeats: int | None, importance_path: Path):
    """Run the TFT test pass. ``permutation_repeats`` is None to skip permutation importance.

    Returns (metrics, (prediction, target) in raw units or None, forecast keys or None).
    """
    print("Running PTF-TFT test pass...")
    config_env, logger = setup_training_environment(config)
    tft_dm, runner, pl_trainer = assemble_training_components(config_env, logger)

    ckpt_path = config_env.checkpoints["trainer"]
    if ckpt_path is not None:
        ckpt_path = mlflow.artifacts.download_artifacts(ckpt_path)
    pl_trainer.test(model=runner, datamodule=tft_dm, ckpt_path=ckpt_path)

    prediction, keys = None, None
    if runner.test_predictions is not None and runner.test_targets is not None:
        if tft_dm.xs_rank_target:
            # Trained on ranks: the loss, MSE/MAE and Sharpe in the table are on the rank scale; the
            # signal metrics use the realised returns like the baselines.
            print("TFT target mode is xs_rank; scoring its signal against realised excess returns.")
            time_idx = runner.test_index["date"].map({d: i for i, d in tft_dm.time_idx_dates.items()})
            raw_targets = tft_dm.raw_cumulative_targets(runner.test_index["ticker"], time_idx.to_numpy())
            ok = np.isfinite(raw_targets).all(axis=1)
            prediction = (runner.test_predictions.numpy()[ok], raw_targets[ok])
            keys = runner.test_index[ok].reset_index(drop=True)
        else:
            prediction = (runner.test_predictions.numpy(), runner.test_targets.numpy())
            keys = runner.test_index

    if permutation_repeats is not None:
        print("Computing TFT permutation importance...")
        importance = permutation_importance(runner, tft_dm, repeats=permutation_repeats)
        print_permutation_importance(importance)
        importance.to_csv(importance_path)
        print(f"Saved permutation importance to {importance_path}")

    # The comparison table is keyed by test_* names, so the validation metrics are listed under them.
    split_prefix = f"{split}_"
    metrics = {
        f"test_{k[len(split_prefix):]}": v.item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in pl_trainer.callback_metrics.items()
        if k.startswith(split_prefix)
    }
    return metrics, prediction, keys


def main():
    parser = argparse.ArgumentParser(description="Compare PTF-TFT against baselines on the test set")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to the YAML config, relative to the repo root (default: %(default)s)")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge regularization strength (default: 1.0)")
    parser.add_argument("--skip-tft", action="store_true",
                        help="Skip TFT test pass — only evaluate baselines")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable all MLflow logging during evaluation")
    parser.add_argument("--permutation-importance", action="store_true",
                        help="Also measure each TFT input's effect on test IC (one test pass per input)")
    parser.add_argument("--permutation-repeats", type=int, default=1,
                        help="Shuffles per input for permutation importance (default: 1)")
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    load_dotenv()

    config = TrainingArgs(args.config, test=True)
    if args.no_mlflow:
        config.mlflow = None
    # Split every model is scored on: "test" (default) or "val", the windows the TFT early-stops on.
    split = config["evaluation"].get("split", "test") if "evaluation" in config else "test"
    if split not in ("test", "val"):
        raise ValueError(f"evaluation.split must be 'test' or 'val', got {split!r}")
    if split == "val":
        print("Scoring on the VALIDATION split (evaluation.split: val); MLflow baseline logging and "
              "permutation importance are skipped.")
        if args.permutation_importance:
            raise ValueError("--permutation-importance only supports the test split")

    all_results, predictions, keys = evaluate_baselines(config, split, args.ridge_alpha)

    if not args.skip_tft:
        importance_path = PLOTS_DIR / f"{Path(args.config).stem}_permutation_importance.csv"
        repeats = args.permutation_repeats if args.permutation_importance else None
        tft_metrics, tft_prediction, tft_keys = evaluate_tft(config, split, repeats, importance_path)
        all_results[TFT_NAME] = tft_metrics
        if tft_prediction is not None:
            predictions[TFT_NAME], keys[TFT_NAME] = tft_prediction, tft_keys

    print_comparison_table(all_results, split)
    signal_results = matched_signal_comparison(predictions, keys)
    for name, metrics in signal_results.items():
        all_results[name].update({f"matched_{k}": v for k, v in metrics.items()})

    plot_path = PLOTS_DIR / f"{Path(args.config).stem}{'_val' if split == 'val' else ''}_prediction_distribution.png"
    for saved_plot_path in save_prediction_distribution_plot(predictions, plot_path):
        print(f"Saved prediction distribution plot to {saved_plot_path}")


if __name__ == "__main__":
    main()
