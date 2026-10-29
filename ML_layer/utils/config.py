"""TrainingArgs: loads a YAML config into an OmegaConf DictConfig."""

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from data_pipeline.utils.config import repo_path


class TrainingArgs:
    """Thin wrapper around OmegaConf that preserves the config path.

    Relative paths, the config's own and the data directories in it, are taken relative to the repo root.

    The yaml's ``test:`` section holds the overrides of a test run. It is merged over the rest when
    ``test=True`` and dropped otherwise, so training and testing share one set of dataset and model settings.
    """

    def __init__(self, config_path: str | Path, test: bool = False):
        config_path = repo_path(config_path)
        self.config_path = str(config_path)
        cfg = OmegaConf.load(config_path)
        overrides = OmegaConf.create(OmegaConf.to_container(cfg.pop("test", {}), resolve=False))
        if test:
            if not overrides:
                raise ValueError(f"{config_path} has no `test:` section")
            cfg = OmegaConf.merge(cfg, overrides)
        self._cfg: DictConfig = cfg
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        for key in ("data_directory", "fundamentals_directory"):
            directory = self._cfg.dataset.arguments.get(key)
            if directory:
                OmegaConf.update(self._cfg, f"dataset.arguments.{key}", str(repo_path(directory)))

    # Delegate attribute and item access to the underlying DictConfig so
    # callers can use config.trainer.max_epochs, config["mode"], etc.
    def __getattr__(self, name: str):
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            raise AttributeError(f"Config has no key '{name}'")

    def __getitem__(self, key: str):
        return self._cfg[key]

    def __contains__(self, key: str) -> bool:
        return key in self._cfg
