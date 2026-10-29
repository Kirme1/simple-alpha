"""
Custom Lightning callbacks.

MLFlowModelCheckpointCallback:
  Uploads the best checkpoint to MLflow artifacts whenever val_loss improves.
  Embeds ``model_config`` (architecture + data settings) in every checkpoint it saves.
  No-ops when MLflow tracking is disabled (logger is not MLFlowLogger).
"""

import os
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger

from data_pipeline.utils.config import REPO_ROOT


def repo_relative(path: str | Path | None) -> str | None:
    """``path`` relative to the repo root, so checkpoints do not embed the training machine's absolute paths."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


class MLFlowModelCheckpointCallback(Callback):
    """Saves the best checkpoint to MLflow artifacts when the monitored metric improves."""

    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score: float = float("inf") if mode == "min" else float("-inf")

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        # Embed everything needed to rebuild the architecture, so the inference
        # API reads it from the checkpoint instead of keeping a second copy.
        dm = trainer.datamodule
        ptf_kwargs = getattr(pl_module, "ptf_kwargs", None)
        if dm is None or ptf_kwargs is None:
            return
        checkpoint["model_config"] = {
            "hidden_size": ptf_kwargs["hidden_size"],
            "attention_head_size": ptf_kwargs["attention_head_size"],
            "dropout": ptf_kwargs["dropout"],
            "loss": pl_module.loss_name,
            "features": list(dm.features),
            "ticker_file": dm.ticker_file,
            "window_size": dm.window_size,
            "split_ratios": list(dm.split_ratios),
            "stride": dm.stride,
            "xs_rank_features": dm.xs_rank_features,
            "xs_rank_target": dm.xs_rank_target,
            "calendar_inputs": dm.calendar_inputs,
            "fundamentals_directory": repo_relative(dm.fundamentals_directory),
            "start_date": dm.start_date,
            "forecast_horizon": dm.train_dataset.max_prediction_length,
        }

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        logged = trainer.callback_metrics.get(self.monitor)
        if logged is None:
            return

        current = logged.item() if hasattr(logged, "item") else float(logged)
        if not self._is_better(current):
            return

        self.best_score = current
        ckpt_path = os.path.join(trainer.default_root_dir, "best_mlflow_checkpoint.ckpt")
        trainer.save_checkpoint(ckpt_path)

        run_id = trainer.logger.run_id
        trainer.logger.experiment.log_artifact(
            run_id=run_id,
            local_path=ckpt_path,
            artifact_path="checkpoints",
        )
