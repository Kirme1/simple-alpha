"""Pull a checkpoint from MLflow and embed everything the inference API needs in it.

Usage:
    python -m ML_layer.prepare_checkpoint [--run_id RUN_ID] [--config ML_layer/model-configs/ptf-config.yaml]
    python -m ML_layer.prepare_checkpoint --checkpoint path/to/local.ckpt     # skip the download

Paths are relative to the repo root.

The run id defaults to MLFLOW_RUN_ID. The checkpoint lands in
artifacts/checkpoints/best_mlflow_checkpoint.ckpt and gets, under these keys:

  model_config  the features, window, hidden size, ... the model was trained with. Taken from the
                checkpoint if it already has one, else from ``--config``, which must then be the yaml
                the run was trained with.
  serving       what the API needs besides the weights; it is rebuilt on every run, since it depends
                on the model and its training data:
      dataset_parameters  the encoders and scalers of the training dataset, to rebuild a dataset for one ticker
      rank_to_return      (xs_rank_target models only) quantiles of the training split's daily excess returns,
                          to map a predicted rank back to a return
      residual_std        per-horizon std of the TFT's validation residuals, for the q10/q90 bands
      ridge               the fitted ridge companion model (see ridge_forecaster.py)

Building ``serving`` reads the training data once (``dataset.arguments.data_directory`` in the config),
so run this where that data and enough memory are available. The API then serves from the checkpoint
alone. The API imports ``SERVING_VERSION`` and ``build_runner`` from here, so keep mlflow imported lazily.
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

from data_pipeline.interfaces import PTFLightningDataModule
from data_pipeline.utils.config import REPO_ROOT, repo_path
from .framework.callbacks import repo_relative
from .framework.ridge_forecaster import RidgeForecaster
from .framework.runners import PTFRunner

SERVING_VERSION = 1
DEFAULT_CONFIG = "ML_layer/model-configs/ptf-config.yaml"
CHECKPOINT_ARTIFACT = "checkpoints/best_mlflow_checkpoint.ckpt"
RANK_GRID_POINTS = 1001



def pull_checkpoint(run_id: str, output_dir: str | Path = REPO_ROOT / "artifacts") -> Path:
    """Download the run's best checkpoint from the DagsHub MLflow server into ``output_dir``."""
    import mlflow

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if not repo_owner or not repo_name:
        raise SystemExit("DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set (see .env.example).")
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    # Same credentials train_model.py uses; leave any the user already exported alone.
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", repo_owner)
    if os.getenv("DAGSHUB_USER_TOKEN"):
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ["DAGSHUB_USER_TOKEN"])
    client = mlflow.tracking.MlflowClient()
    return Path(client.download_artifacts(run_id, CHECKPOINT_ARTIFACT, output_dir))


def model_config_from_yaml(cfg: dict) -> dict:
    """The ``model_config`` a checkpoint trained from this yaml carries."""
    ds = cfg["dataset"]["arguments"]
    runner = cfg["runner"]["parameters"]
    return {
        "hidden_size": runner["hidden_size"],
        "attention_head_size": runner["attention_head_size"],
        "dropout": runner["dropout"],
        "loss": runner.get("loss", "quantile"),
        "features": ds["features"],
        "ticker_file": ds["ticker_file"],
        "window_size": ds["window_size"],
        "split_ratios": ds["split_ratios"],
        "stride": ds.get("stride", 1),
        # Defaults mirror PTFLightningDataModule's.
        "xs_rank_features": ds.get("xs_rank_features", False),
        "xs_rank_target": ds.get("xs_rank_target", False),
        "calendar_inputs": ds.get("calendar_inputs", True),
        "fundamentals_directory": ds.get("fundamentals_directory"),
        "start_date": ds.get("start_date"),
        "forecast_horizon": 20,  # PTFLightningDataModule hardcodes max_prediction_length=20
    }


def loss_name(cfg: dict, state_dict: dict) -> str:
    """The loss the checkpoint was trained with; it sets the output layer's size.

    Older checkpoints did not record it, so fall back to the output layer: 7 quantile
    outputs for "quantile", a single point output for "mse".
    """
    if "loss" in cfg:
        return cfg["loss"]
    return "mse" if state_dict["ptf_model.output_layer.bias"].shape[0] == 1 else "quantile"


def build_runner(cfg: dict, state_dict: dict, train_dataset) -> PTFRunner:
    """A PTFRunner with the checkpoint's weights, built from ``train_dataset`` (or a stand-in with its parameters)."""
    runner = PTFRunner(
        hidden_size=cfg["hidden_size"],
        attention_head_size=cfg["attention_head_size"],
        dropout=cfg["dropout"],
        loss=loss_name(cfg, state_dict),
    )
    # PTFRunner.setup() reads self.trainer.datamodule.train_dataset
    runner._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=train_dataset))
    runner.setup("predict")
    runner.load_state_dict(state_dict)
    runner.ptf_model.eval()
    return runner


def rank_quantiles(dm: PTFLightningDataModule) -> np.ndarray:
    """Quantiles (0, 1/(n-1), ..., 1) of the daily excess returns in the training split, to undo the rank target.

    A rank target is the return's percentile among tickers on the same date, centred on 0, so a predicted rank r
    maps back to the (r + 0.5) quantile of the cross-sectional return distribution. That distribution is
    approximated by pooling the training split's returns; later data is not used.
    """
    returns = np.concatenate([s[s.index <= dm.train_cutoff].to_numpy() for s in dm._raw_returns.values()])
    return np.quantile(returns, np.linspace(0.0, 1.0, RANK_GRID_POINTS))


def ranks_to_returns(ranks: torch.Tensor, quantiles: np.ndarray) -> torch.Tensor:
    """Map predicted daily ranks in [-0.5, 0.5] to daily excess returns by interpolating the quantile grid."""
    probs = np.linspace(0.0, 1.0, len(quantiles))
    returns = np.interp((ranks.detach().cpu().numpy() + 0.5).clip(0.0, 1.0), probs, quantiles)
    return torch.as_tensor(returns, dtype=ranks.dtype)


def residual_std(runner: PTFRunner, dm: PTFLightningDataModule, quantiles: np.ndarray | None = None) -> np.ndarray:
    """Per-horizon std of (predicted - realised) cumulative returns on the validation split. Shape [horizon].

    With ``quantiles`` (a rank-target model) predictions are converted to returns and scored against the
    realised returns, not the ranks, so the bands are on the scale the API reports.
    """
    residuals = []
    dataset = dm.val_dataset
    with torch.no_grad():
        for x, (y, _) in dm.val_dataloader():
            out = runner.ptf_model(x)
            pred_daily = runner.point_forecast(out.prediction.squeeze(-1))  # [B, 20]
            if quantiles is None:
                true_returns = runner.cumulative_returns(y).cpu().numpy()
            else:
                pred_daily = ranks_to_returns(pred_daily, quantiles)
                index = dataset.x_to_index(x)
                true_returns = dm.raw_cumulative_targets(index["ticker"].astype(str), index["time_idx"].to_numpy())
            diff = runner.cumulative_returns(pred_daily).cpu().numpy() - true_returns
            residuals.append(diff[np.isfinite(diff).all(axis=1)])
    return np.concatenate(residuals, axis=0).std(axis=0)


def build_bundle(ckpt: dict, data_directory: str, ridge_alpha: float = 1.0, ticker_file: str | None = None) -> dict:
    cfg = ckpt["model_config"]
    if ticker_file is not None:  # lets a smoke test use a few tickers
        cfg = {**cfg, "ticker_file": ticker_file}
    dm = PTFLightningDataModule(
        data_directory=data_directory,
        features=cfg["features"],
        ticker_file=cfg["ticker_file"],
        window_size=cfg["window_size"],
        split_ratios=cfg["split_ratios"],
        stride=cfg.get("stride", 1),
        xs_rank_features=cfg.get("xs_rank_features", False),
        xs_rank_target=cfg.get("xs_rank_target", False),
        calendar_inputs=cfg.get("calendar_inputs", True),
        fundamentals_directory=str(repo_path(cfg["fundamentals_directory"])) if cfg.get("fundamentals_directory") else None,
        start_date=cfg.get("start_date"),
        num_workers=0,
        load_test_split=False,
    )
    dm.setup()
    runner = build_runner(cfg, ckpt["state_dict"], dm.train_dataset)

    ridge = RidgeForecaster(alpha=ridge_alpha)
    ridge.fit(data_directory, cfg)

    quantiles = rank_quantiles(dm) if cfg.get("xs_rank_target") else None
    bundle = {
        "version": SERVING_VERSION,
        "dataset_parameters": dm.train_dataset.get_parameters(),
        "residual_std": residual_std(runner, dm, quantiles).tolist(),
        "ridge": ridge.to_state(),
    }
    if quantiles is not None:
        bundle["rank_to_return"] = quantiles.tolist()
    return bundle


def scrub_paths(ckpt: dict) -> None:
    """Remove the training machine's absolute paths (Lightning's checkpoint callback state, data directories)."""
    for state in ckpt.get("callbacks", {}).values():
        if isinstance(state, dict):
            for key in ("best_model_path", "dirpath", "last_model_path"):
                if key in state:
                    state[key] = ""
            state.get("best_k_models", {}).clear()  # keyed by checkpoint path
    cfg = ckpt.get("model_config", {})
    if cfg.get("fundamentals_directory"):
        cfg["fundamentals_directory"] = repo_relative(cfg["fundamentals_directory"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_id", help="MLflow run to download; defaults to MLFLOW_RUN_ID")
    parser.add_argument("--checkpoint", help="prepare this local checkpoint instead of downloading one")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    load_dotenv()
    with open(repo_path(args.config)) as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        path = repo_path(args.checkpoint)
    else:
        run_id = args.run_id or os.getenv("MLFLOW_RUN_ID")
        if not run_id:
            raise SystemExit("No --run_id given and MLFLOW_RUN_ID is not set.")
        path = pull_checkpoint(run_id)
        print(f"Pulled run {run_id} to {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" in ckpt:
        print("Keeping the model_config already in the checkpoint.")
    else:
        ckpt["model_config"] = model_config_from_yaml(config)
        print(f"Embedded model_config from {args.config}; it must be the config the run was trained with.")

    ridge_alpha = config.get("serving", {}).get("ridge_alpha", 1.0)
    data_directory = str(repo_path(config["dataset"]["arguments"]["data_directory"]))
    ckpt["serving"] = build_bundle(ckpt, data_directory, ridge_alpha)
    scrub_paths(ckpt)
    torch.save(ckpt, path)
    print(f"Wrote model_config and serving bundle to {path}")


if __name__ == "__main__":
    main()
