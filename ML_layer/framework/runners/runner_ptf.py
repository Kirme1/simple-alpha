"""
PTFRunner: LightningModule adapter for pytorch-forecasting's TFT.

PTF's TemporalFusionTransformer is itself a LightningModule, so we hold it as
a plain sub-module (self.ptf_model) and drive training/val ourselves. This lets
us plug into the existing factory/runner pattern without forking train_model.py.

The PTF model is created lazily in setup() once the datamodule is available,
because TemporalFusionTransformer.from_dataset() requires a TimeSeriesDataSet.

Metrics logged:
    train_loss / val_loss / test_loss  — QuantileLoss (or MSE, see ``loss``) on the daily excess (vs SPY) log-return target
  val_directional_accuracy           — sign accuracy across all 20 horizons
  val_sharpe                         — annualised Sharpe on day-+1 horizon
  val_ic_h01 / _h05 / _h20 / val_ic  — mean daily cross-sectional rank IC of the cumulative forecast against the
                                       realised cumulative excess return (also in "xs_rank" mode, where the model
                                       itself is trained on ranks), and their mean; usable as an early-stopping /
                                       checkpoint monitor (mode "max")
"""

from typing import Callable

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE, QuantileLoss

from ..metrics import directional_accuracy, mean_daily_ic, point_forecast_metrics, sharpe_ratio, SIGNAL_HORIZONS


def make_loss(name: str):
    """Training loss of the TFT: "quantile" (median of 7 quantiles, the default) or "mse" (conditional mean)."""
    if name == "quantile":
        return QuantileLoss()
    if name == "mse":
        # RMSE with plain mean reduction is the mean squared error; it has a single point output.
        return RMSE(reduction="mean")
    raise ValueError(f"loss must be 'quantile' or 'mse', got {name!r}")


class PTFRunner(pl.LightningModule):
    def __init__(
        self,
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        loss: str = "quantile",
    ):
        super().__init__()
        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler
        self.loss_name = loss
        self.ptf_kwargs = dict(
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            loss=make_loss(loss),
            log_interval=-1,      # disable PTF's internal per-step logging
            log_val_interval=-1,
        )
        self.ptf_model: TemporalFusionTransformer | None = None

        # Per-batch daily point forecasts and targets of the running validation or test epoch
        self._pred_batches: list[torch.Tensor] = []
        self._target_batches: list[torch.Tensor] = []
        self._val_index_parts: list[pd.DataFrame] = []
        self.test_predictions: torch.Tensor | None = None
        self.test_targets: torch.Tensor | None = None
        # (ticker, first forecast date) per row of test_predictions
        self.test_index: pd.DataFrame | None = None
        self._test_index_parts: list[pd.DataFrame] = []

    def setup(self, stage: str | None = None) -> None:
        if self.ptf_model is not None:
            return
        train_dataset = self.trainer.datamodule.train_dataset
        self.ptf_model = TemporalFusionTransformer.from_dataset(
            train_dataset, **self.ptf_kwargs
        )

    def point_forecast(self, prediction: torch.Tensor) -> torch.Tensor:
        """Reduce the model output to a point forecast of the daily excess log return [B, H].

        That is the median quantile, or the single output of the MSE loss. TemporalFusionTransformer.forward()
        already applies transform_output, so ``prediction`` is on the original return scale.
        """
        return self.ptf_model.loss.to_prediction(prediction)

    @staticmethod
    def cumulative_returns(daily_log_returns: torch.Tensor) -> torch.Tensor:
        """Daily (excess) log returns [B, H] -> cumulative forward log returns [B, H]."""
        return daily_log_returns.cumsum(dim=1)

    @torch.no_grad()
    def predict_dataset(self, datamodule, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Cumulative excess-return forecasts, targets, first forecast dates and (ticker, time_idx) keys of ``dataset``.

        In xs_rank target mode the model target is a rank, so the targets returned are the realised returns,
        which is what the baselines are scored against; samples without a realised return are dropped.
        """
        preds, targets, keys = [], [], []
        self.ptf_model.eval()
        for x, (y, _) in dataset.to_dataloader(train=False, batch_size=datamodule.batch_size, num_workers=0):
            preds.append(self.cumulative_returns(self.point_forecast(self.ptf_model(x).prediction)).cpu())
            targets.append(self.cumulative_returns(y).cpu())
            keys.append(dataset.x_to_index(x))
        keys = pd.concat(keys, ignore_index=True)
        dates = keys["time_idx"].map(datamodule.time_idx_dates).to_numpy()
        preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        if datamodule.xs_rank_target:
            targets = datamodule.raw_cumulative_targets(keys["ticker"].astype(str), keys["time_idx"])
            ok = np.isfinite(targets).all(axis=1)
            preds, targets, dates, keys = preds[ok], targets[ok], dates[ok], keys[ok].reset_index(drop=True)
        return preds, targets, dates, keys

    def _eval_step(self, batch, loss_name: str, prog_bar: bool = False) -> dict[str, torch.Tensor]:
        """Log the loss and keep the point forecasts and targets of the batch for the epoch-end metrics."""
        x, (y, _weight) = batch
        output = self.ptf_model(x)
        loss = self.ptf_model.loss(output.prediction, y)
        self.log(loss_name, loss, on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=y.size(0))
        self._pred_batches.append(self.point_forecast(output.prediction).detach().cpu())
        self._target_batches.append(y.detach().cpu())
        return x

    def _drain_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative forecasts and targets [N, H] of the epoch's batches; clears them."""
        pred = self.cumulative_returns(torch.cat(self._pred_batches, dim=0))
        true = self.cumulative_returns(torch.cat(self._target_batches, dim=0))
        self._pred_batches.clear()
        self._target_batches.clear()
        return pred, true

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, (y, _weight) = batch
        output = self.ptf_model(x)
        loss = self.ptf_model.loss(output.prediction, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x = self._eval_step(batch, "val_loss", prog_bar=True)
        self._val_index_parts.append(self.trainer.datamodule.val_dataset.x_to_index(x)[["ticker", "time_idx"]])

    def on_validation_epoch_end(self) -> None:
        if not self._pred_batches:
            self._val_index_parts.clear()
            return

        pred_returns, true_returns = (t.numpy() for t in self._drain_epoch())  # [N, 20]
        n = len(pred_returns)

        self.log("val_directional_accuracy", directional_accuracy(pred_returns, true_returns), prog_bar=True, batch_size=n)
        self.log("val_sharpe", sharpe_ratio(pred_returns, true_returns, horizon_idx=0), batch_size=n)

        # IC is scored against the realised excess returns, as the baselines are, not the model target:
        # a rank-trained model can score well on summed ranks and still miss on returns.
        index = pd.concat(self._val_index_parts, ignore_index=True)
        self._val_index_parts.clear()
        raw = self.trainer.datamodule.raw_cumulative_targets(index["ticker"].astype(str), index["time_idx"].to_numpy())
        ok = np.isfinite(raw).all(axis=1)
        pred_ok, raw_ok, dates = pred_returns[ok], raw[ok], index["time_idx"].to_numpy()[ok]
        ics = [mean_daily_ic(pred_ok[:, h - 1], raw_ok[:, h - 1], dates) for h in SIGNAL_HORIZONS]
        for h, ic in zip(SIGNAL_HORIZONS, ics):
            self.log(f"val_ic_h{h:02d}", ic, prog_bar=h == SIGNAL_HORIZONS[-1], batch_size=n)
        self.log("val_ic", float(np.nanmean(ics)), prog_bar=True, batch_size=n)

    def test_step(self, batch, batch_idx: int) -> None:
        x = self._eval_step(batch, "test_loss")
        # x_to_index gives the group ids and first decoder time_idx of each sample.
        self._test_index_parts.append(self.trainer.datamodule.test_dataset.x_to_index(x))

    def on_test_epoch_end(self) -> None:
        if not self._pred_batches:
            return

        pred_returns, true_returns = self._drain_epoch()
        self.test_predictions, self.test_targets = pred_returns, true_returns
        index = pd.concat(self._test_index_parts, ignore_index=True)
        self._test_index_parts.clear()
        self.test_index = pd.DataFrame({
            "ticker": index["ticker"].astype(str),
            "date": index["time_idx"].map(self.trainer.datamodule.time_idx_dates).to_numpy(),
        })
        self.log_dict(
            point_forecast_metrics(pred_returns.numpy(), true_returns.numpy()), batch_size=len(pred_returns)
        )

    def configure_optimizers(self):
        if self._optimizer_factory is None:
            raise ValueError("optimizer factory not provided to PTFRunner")
        optimizer = self._optimizer_factory(self.parameters())
        if self._scheduler_factory is None:
            return optimizer
        scheduler_config = self._scheduler_factory(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
