"""
Sklearn-based baseline models for benchmarking TFT performance.

Both baselines operate in the same normalized feature and target space as the TFT:
  - Features are per-window standardized (done by _make_windows in basic_data_interface.py)
  - Targets are divided by target_std (done by BasicLightningDataModule.setup)

No denormalization is needed — all metrics are directly comparable to the TFT test
metrics logged by PTFRunner.on_test_epoch_end.

Ridge appears in three places in the repo, for different jobs:
  - RidgeBaseline (here): the benchmark in evaluate.py. Fits on streamed BasicLightningDataModule batches, all
    20 horizons, and is scored on the same samples as the TFT. Nothing from it is stored.
  - RidgeForecaster (ridge_forecaster.py): the companion forecast served by the API. Rebuilds the TFT's training
    windows from the checkpoint's model_config, fits only the horizons the API shows (RIDGE_HORIZONS) and stores
    its coefficients in the checkpoint.
  - make_model("ridge") (walk_forward.py): a tabular ridge on one feature row per ticker and date, with no
    window, refit each year of the walk-forward test.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge

from .metrics import point_forecast_metrics


FORECAST_HORIZON = 20


def _flattened_batches(batches: Iterable) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (X, y) numpy batches from a DataLoader or a streaming generator of tensor pairs.

    X is flattened from [n, window_size, num_features] to [n, window_size * num_features];
    y is [n, FORECAST_HORIZON].
    """
    for x_batch, y_batch in batches:
        if not isinstance(x_batch, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor batch, got {type(x_batch)}")
        yield x_batch.numpy().reshape(len(x_batch), -1), y_batch.numpy()


def _extract_arrays(dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Drain a DataLoader into (X_flat, y) arrays."""
    pairs = list(_flattened_batches(dataloader))
    if not pairs:
        raise ValueError("DataLoader yielded no samples; check split_ratios, window_size and stride")
    return np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs])


def _predict_in_batches(batches: Iterable, predict) -> tuple[np.ndarray, np.ndarray]:
    """Predict batch by batch so the flattened inputs of the whole split are never held at once.

    ``predict`` maps flattened [n, window_size * num_features] inputs to [n, FORECAST_HORIZON].
    """
    preds, targets = [], []
    for x, y in _flattened_batches(batches):
        preds.append(predict(x))
        targets.append(y)
    if not preds:
        raise ValueError("No test samples; check split_ratios, window_size and stride")
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """The metrics PTFRunner logs at the end of a test epoch, in numpy.

    All inputs are in normalized target space (divided by target_std); ``test_loss`` is the MSE.
    """
    return {"test_loss": float(np.mean((pred - target) ** 2)), **point_forecast_metrics(pred, target)}


class PersistenceBaseline:
    """Always predicts 0.0 for all horizons (efficient-market null hypothesis).

    In normalized target space, predicting 0 means predicting the training mean
    (returns are zero-mean after dividing by target_std). Directional accuracy
    converges to ~50% as N → ∞. This is the floor — TFT must beat this to be useful.
    """

    name = "Persistence"

    def fit(self, train_dataloader: DataLoader) -> "PersistenceBaseline":
        return self

    def evaluate_with_predictions(
        self, test_dataloader: DataLoader
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        pred, target = _predict_in_batches(test_dataloader, lambda x: np.zeros((len(x), FORECAST_HORIZON), dtype=np.float32))
        return _compute_metrics(pred, target), pred, target

    def evaluate(self, test_dataloader: DataLoader) -> dict[str, Any]:
        metrics, _, _ = self.evaluate_with_predictions(test_dataloader)
        return metrics


class RidgeBaseline:
    """Multi-output Ridge regression on flattened windows.

    Input:  [N, window_size * num_features] flattened features
    Output: [N, 20] normalized log returns (one per forecast horizon)

    Ridge's L2 regularization (alpha) keeps the fit stable in the high-dimensional input space.
    Any TFT improvement over this baseline is attributable to non-linearity
    or temporal structure that Ridge cannot capture.
    """

    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._is_fitted = False

    def fit(self, train_dataloader: DataLoader) -> "RidgeBaseline":
        x, y = _extract_arrays(train_dataloader)
        self._model.fit(x, y)
        self._is_fitted = True
        return self

    def evaluate_with_predictions(
        self, test_dataloader: DataLoader
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before evaluate()")
        pred, target = _predict_in_batches(test_dataloader, self._model.predict)  # [N, 20]
        return _compute_metrics(pred, target), pred, target

    def evaluate(self, test_dataloader: DataLoader) -> dict[str, Any]:
        metrics, _, _ = self.evaluate_with_predictions(test_dataloader)
        return metrics
