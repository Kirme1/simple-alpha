"""
walk_forward_tft.py — Train and test the TFT on yearly expanding-window folds.

For each test year Y in the config's ``walk_forward`` section, the TFT trains
on data through Y-2, early-stops on forecasts starting in Y-1, and predicts
every forecast starting in Y (see PTFLightningDataModule.fold_test_year). Ridge is
refitted on the same folds with year Y-1 held out too, and both are scored on
the pooled (ticker, date) samples they share.

Usage:
    python -m ML_layer.walk_forward_tft [--config ML_layer/model-configs/ptf-config.yaml] [--score-only]

Config:
    walk_forward:
      first_test_year: 2016
      last_test_year: null      # null = latest year in the data
      warm_start: false         # initialize fold Y from fold Y-1's best weights
      output_dir: "ML_layer/walk_forward_tft"

Each fold's predictions are saved as <output_dir>/fold_<Y>.parquet and folds
already saved are skipped, so an interrupted run resumes where it stopped.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_forecasting import TemporalFusionTransformer

from .framework.metrics import signal_metrics
from .train_model import assemble_training_components, setup_training_environment
from .train_model import DEFAULT_CONFIG
from .utils import TrainingArgs
from . import walk_forward as ridge_wf
from data_pipeline.utils.config import load_tickers, repo_path

MODEL_NAMES = ("Ridge", "PTF-TFT")  # paired IC differences are PTF-TFT minus Ridge
TARGET_MISMATCH_TOLERANCE = 1e-4


def _walk_forward_settings(config: TrainingArgs) -> dict:
    if "walk_forward" not in config or config.walk_forward is None:
        raise ValueError(
            f"{config.config_path} has no 'walk_forward' section; add first_test_year, "
            "last_test_year, warm_start and output_dir (see ML_layer/walk_forward_tft.py)."
        )
    settings = OmegaConf.to_container(config.walk_forward, resolve=True)
    settings["output_dir"] = repo_path(settings.get("output_dir") or "ML_layer/walk_forward_tft")
    settings.setdefault("warm_start", False)
    return settings


def _latest_data_year(config: TrainingArgs) -> int:
    arguments = config.dataset.arguments
    tickers, _ = load_tickers(arguments.ticker_file)
    paths = [Path(arguments.data_directory) / f"{t}_raw.parquet" for t in tickers]
    return max(pd.read_parquet(p, columns=["date"])["date"].max() for p in paths if p.exists()).year


def _fold_config(config_path: Path, year: int, fold_dir: Path) -> TrainingArgs:
    """The model config with this fold's split, run name, and a local best-checkpoint callback."""
    config = TrainingArgs(config_path)
    cfg = config._cfg
    OmegaConf.update(cfg, "dataset.arguments.fold_test_year", year)
    if cfg.get("mlflow"):
        OmegaConf.update(cfg, "mlflow.run_name", f"ptf-wf-{year}")
        OmegaConf.update(cfg, "mlflow.tags.walk_forward_fold", str(year))
        OmegaConf.update(cfg, "mlflow.tags.walk_forward_group", f"{config_path.stem}-{cfg.get('version', '')}")
    callbacks = OmegaConf.to_container(cfg.get("callbacks") or {}, resolve=True)
    # Uploading every improvement of every fold to DagsHub is slow; keep the
    # fold's best checkpoint locally instead.
    callbacks.pop("mlflowmodelcheckpoint", None)
    callbacks["modelcheckpoint"] = {
        "monitor": "val_loss", "mode": "min", "save_top_k": 1,
        "dirpath": str(fold_dir), "filename": "best", "enable_version_counter": False,
    }
    OmegaConf.update(cfg, "callbacks", callbacks, merge=False)
    return config


def _warm_start(runner, dm, checkpoint: Path) -> None:
    """Build the TFT for this fold and load the previous fold's best weights.

    Valid because every fold has the same inputs, categories, and architecture;
    only the scalers fitted on each fold's training data differ.
    """
    dm.setup("fit")
    runner.ptf_model = TemporalFusionTransformer.from_dataset(dm.train_dataset, **runner.ptf_kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    runner.load_state_dict(state)


def train_fold(config_path: Path, year: int, settings: dict) -> pd.DataFrame:
    fold_dir = settings["output_dir"] / f"fold_{year}"
    config = _fold_config(config_path, year, fold_dir)
    # A previous fold's run id would otherwise be reused by mlflow's fluent API.
    os.environ.pop("MLFLOW_RUN_ID", None)
    config_env, logger = setup_training_environment(config)
    dm, runner, pl_trainer = assemble_training_components(config_env, logger)

    previous = settings["output_dir"] / f"fold_{year - 1}" / "best.ckpt"
    if settings["warm_start"]:
        if previous.exists():
            print(f"  warm start from {previous}")
            _warm_start(runner, dm, previous)
        else:
            print(f"  no fold {year - 1} checkpoint; training fold {year} from scratch")

    pl_trainer.fit(model=runner, datamodule=dm)
    best = pl_trainer.checkpoint_callback.best_model_path
    if not best:
        raise RuntimeError(f"Fold {year} produced no checkpoint (no validation epoch completed?)")
    runner.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["state_dict"])
    runner.cpu()

    pred, target, dates, keys = runner.predict_dataset(dm, dm.test_dataset)
    fold = pd.DataFrame({"ticker": keys["ticker"].astype(str).to_numpy(), "date": pd.to_datetime(dates), "fold": year})
    for h in range(pred.shape[1]):
        fold[f"pred_h{h + 1:02d}"] = pred[:, h]
        fold[f"y_h{h + 1:02d}"] = target[:, h]

    metrics = signal_metrics(pred, target, fold["date"].to_numpy(), ridge_wf.HORIZONS)
    print(f"  fold {year}: {len(fold)} test samples, IC h05 {metrics['ic_h05']:+.4f}")
    if logger is not None:
        for key, value in metrics.items():
            if np.isfinite(value):
                logger.experiment.log_metric(logger.run_id, f"wf_test_{key}", value)
    return fold


def ridge_predictions(config: TrainingArgs, first_year: int) -> pd.DataFrame:
    """Ridge on the TFT's features and fold layout (year Y-1 held out, raw target)."""
    arguments = config.dataset.arguments
    samples = ridge_wf.load_samples(Path(arguments.data_directory), arguments.ticker_file)
    features = list(arguments.features)
    if arguments.get("xs_rank_features", False):
        samples = ridge_wf.cross_sectional_rank(samples, features)
    predictions = ridge_wf.walk_forward(samples, features, ["ridge"], first_year, "raw", holdout_years=1)
    return predictions["ridge"]


def score_folds(config: TrainingArgs, settings: dict) -> None:
    paths = sorted(settings["output_dir"].glob("fold_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No fold predictions in {settings['output_dir']}")
    tft = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    tft = tft[[c for c in tft.columns if c in {"ticker", "date", "fold"}
               or c.endswith(tuple(f"_h{h:02d}" for h in ridge_wf.HORIZONS))]]
    print(f"Fitting Ridge on the same folds ({tft['fold'].min()}–{tft['fold'].max()})...")
    ridge = ridge_predictions(config, int(tft["fold"].min()))

    key = ["ticker", "date"]
    matched = tft.merge(ridge, on=key, suffixes=("_tft", "_ridge"), validate="one_to_one")
    matched = matched.sort_values(key).reset_index(drop=True)
    y_columns = [f"y_h{h:02d}" for h in ridge_wf.HORIZONS]
    mismatch = np.nanmax(np.abs(
        matched[[f"{c}_tft" for c in y_columns]].to_numpy() - matched[[f"{c}_ridge" for c in y_columns]].to_numpy()
    ))
    if mismatch > TARGET_MISMATCH_TOLERANCE:
        print(f"Warning: TFT and Ridge targets differ by up to {mismatch:.2e} on matched samples.")
    print(f"Matched {len(matched)} of {len(tft)} TFT and {len(ridge)} Ridge samples.")

    predictions = {}
    for name, suffix in zip(MODEL_NAMES, ("_ridge", "_tft")):
        frame = matched[key].assign(fold=matched["fold_tft"])
        for h in ridge_wf.HORIZONS:
            frame[f"pred_h{h:02d}"] = matched[f"pred_h{h:02d}{suffix}"]
            frame[f"y_h{h:02d}"] = matched[f"y_h{h:02d}_tft"]
        predictions[name] = frame

    metrics = ridge_wf.score(predictions, "raw")
    per_year = ridge_wf.per_fold_ic(predictions, 5)
    title = (f"TFT WALK-FORWARD — {len(matched)} matched samples, {matched['date'].nunique()} dates, "
             f"test years {tft['fold'].min()}–{tft['fold'].max()}")
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(metrics.to_string())
        print(f"\nIC per test year, h05:\n{per_year.to_string()}")
    print(f"\nic_diff rows compare {MODEL_NAMES[1]} against {MODEL_NAMES[0]}.")
    metrics.to_csv(settings["output_dir"] / "summary.csv")
    per_year.to_csv(settings["output_dir"] / "per_year_ic_h05.csv")
    print(f"Saved summary to {settings['output_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward training and evaluation of the TFT")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to the YAML config, relative to the repo root (default: %(default)s)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip training and score the fold predictions already saved")
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")
    config_path = repo_path(args.config)
    config = TrainingArgs(config_path)
    settings = _walk_forward_settings(config)
    settings["output_dir"].mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        last_year = settings.get("last_test_year") or _latest_data_year(config)
        for year in range(int(settings["first_test_year"]), int(last_year) + 1):
            path = settings["output_dir"] / f"fold_{year}.parquet"
            if path.exists():
                print(f"Fold {year}: predictions exist, skipping")
                continue
            print(f"Fold {year}: training...")
            train_fold(config_path, year, settings).to_parquet(path, index=False)

    score_folds(config, settings)


if __name__ == "__main__":
    main()
