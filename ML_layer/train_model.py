"""Train (or test) the TFT from a yaml config: python -m ML_layer.train_model [--config ML_layer/model-configs/ptf-config.yaml]"""

import argparse
import os
import random
import warnings

import lightning.pytorch as pl
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from urllib3.exceptions import InsecureRequestWarning

DEFAULT_CONFIG = "ML_layer/model-configs/ptf-config.yaml"

from .utils import (
    dataset_factory,
    optimizer_factory,
    scheduler_factory,
    callback_factory,
    runner_factory,
    TrainingArgs,
)


def main():
    train(_parse_console_args())


def train(config: TrainingArgs) -> None:
    """Fit the model, or run its test pass when ``config.mode`` is not "train"."""
    torch.multiprocessing.set_sharing_strategy('file_system')

    config, logger = setup_training_environment(config)
    signal_data_module, runner, pl_trainer = assemble_training_components(config, logger)

    if config.checkpoints["trainer"] is not None:
        config.checkpoints["trainer"] = mlflow.artifacts.download_artifacts(
            config.checkpoints["trainer"]
        )

    run = pl_trainer.fit if config.mode == "train" else pl_trainer.test
    run(model=runner, datamodule=signal_data_module, ckpt_path=config.checkpoints.trainer)


def _init_mlflow_logger(
    mlflow_config: dict,
    train_checkpoint: str | None = None,
) -> MLFlowLogger:
    run_id = None
    if train_checkpoint is not None:
        if "mlflow-artifacts" in train_checkpoint:
            run_id = train_checkpoint.split("/")[2]
        elif "runs" in train_checkpoint:
            run_id = train_checkpoint.split("runs:/")[1].split("/")[0]

    logger = MLFlowLogger(log_model=False, run_id=run_id, save_dir="mlruns", **mlflow_config)

    warnings.simplefilter("ignore", InsecureRequestWarning)
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "True"

    if logger.run_id is not None:
        os.environ["MLFLOW_RUN_ID"] = logger.run_id
    if logger.experiment_id is not None:
        os.environ["MLFLOW_EXPERIMENT_ID"] = logger.experiment_id

    os.environ["MLFLOW_EXPERIMENT_NAME"] = logger._experiment_name
    os.environ["MLFLOW_TRACKING_URI"] = logger._tracking_uri
    return logger


def _parse_console_args():
    parser = argparse.ArgumentParser(description="Train configuration")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="path to the yaml config, relative to the repo root (default: %(default)s)",
    )
    return TrainingArgs(parser.parse_args().config)


def _log_config_params(logger: MLFlowLogger | None, config) -> None:
    """Log metadata-only run context as tags so it stays out of charts.

    MLflow charts are driven by metrics, while tags/params are metadata. We keep
    the run metadata separate from the metric stream to avoid chart clutter.
    """
    if logger is None:
        return

    dataset_args = getattr(config.dataset, "arguments", {})
    metadata = {
        "ticker_file": dataset_args.get("ticker_file", ""),
        "features": str(dataset_args.get("features", [])),
        "window_size": str(dataset_args.get("window_size", "")),
        "split_ratios": str(dataset_args.get("split_ratios", [])),
        "runner_name": str(config.runner.name),
        "batch_size": str(config.dataloader.get("batch_size", "")),
        "learning_rate": str(config.optimizer.parameters.get("lr", "")),
    }

    for key, value in metadata.items():
        if value == "":
            continue
        try:
            logger.experiment.set_tag(run_id=logger.run_id, key=key, value=value)
        except (MlflowException, OSError) as exc:
            # Tracking-server failures (requests errors subclass OSError) must not
            # fail training for run annotations, but they should not be silent.
            # Anything else is a bug and propagates.
            warnings.warn(f"Could not log MLflow tag '{key}': {exc}", RuntimeWarning)


def setup_training_environment(config: TrainingArgs):
    """Connect MLflow, seed every RNG and create the MLflow logger (None when ``mlflow`` is not configured).

    Returns (config, logger).
    """
    load_dotenv()

    # Configure the DagsHub-hosted MLflow server before Lightning creates its
    # logger so every run, including its checkpoint artifact, goes remote.
    if config.dagshub:
        _init_dagshub_mlflow(config.dagshub)

    # Set seeds for reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    if config.mlflow:
        logger: MLFlowLogger = _init_mlflow_logger(config.mlflow, config.checkpoints.trainer)
        logger.experiment.log_artifact(
            run_id=logger.run_id, local_path=config.config_path
        )
        _log_config_params(logger, config)
    else:
        logger = None

    return config, logger


def _init_dagshub_mlflow(dagshub_config: dict) -> None:
    """Connect MLflow to the DagsHub repository configured for this project."""
    repo_owner = dagshub_config.repo_owner
    repo_name = dagshub_config.repo_name
    token = dagshub_config.user_token
    if not repo_owner or not repo_name or not token:
        raise RuntimeError(
            "DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, and DAGSHUB_USER_TOKEN must "
            "be set to train with MLflow tracking enabled. Copy .env.example to "
            ".env and configure them."
        )

    os.environ["MLFLOW_TRACKING_URI"] = (
        f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    )
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token


def assemble_training_components(config, logger):
    """Create the datamodule, runner and Lightning trainer from the config. Returns (datamodule, runner, trainer)."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    signal_data_module = dataset_factory(
        **config.dataset, dataloader_arguments=config.dataloader
    )

    # The runner builds the TFT itself in setup(), once the datamodule exists
    runner_parameters = {
        "optimizer": optimizer_factory(**config.optimizer),
        "scheduler": (
            scheduler_factory(**config.scheduler)
            if config.scheduler is not None
            else None
        ),
    }
    if config.runner.parameters is not None:
        runner_parameters.update(config.runner.parameters)
    runner = runner_factory(name=config.runner.name, parameters=runner_parameters)

    pl_trainer = pl.Trainer(**{
        "devices": 1,
        "strategy": "auto",
        "logger": logger,
        "callbacks": callback_factory(config.callbacks),
        **config.trainer,
    })

    return signal_data_module, runner, pl_trainer


if __name__ == "__main__":
    main()
