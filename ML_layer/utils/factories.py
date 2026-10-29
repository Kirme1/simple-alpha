from functools import partial
from typing import Union, Any

from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import LightningDataModule
from torch import optim

from data_pipeline.interfaces import PTFLightningDataModule

from ML_layer.framework import MLFlowModelCheckpointCallback
from ML_layer.framework.runners import PTFRunner


def dataset_factory(
    name: str, arguments: dict, dataloader_arguments: dict
) -> LightningDataModule:
    arguments = dict(arguments)

    if name == "ptf":
        return PTFLightningDataModule(
            **arguments,
            **dataloader_arguments,
        )
    else:
        raise AttributeError(f"Selected dataset type {name} is not supported, yet!")

def optimizer_factory(
    name: str,
    parameters: dict,
) -> partial[optim.Optimizer]:
    if name == "adam":
        return partial(optim.Adam, **parameters)
    elif name == "adamw":
        return partial(optim.AdamW, **parameters)
    else:
        raise AttributeError(f"Selected optimizer type {name} is not supported, yet!")


def scheduler_factory(
    function: dict[str, dict], trainer_parameters: dict[str, Any]
) -> partial[optim.lr_scheduler.LRScheduler]:
    scheduler_dict = dict(**trainer_parameters)

    if function["name"] == "cosine_annealing":
        scheduler_dict["scheduler"] = partial(
            optim.lr_scheduler.CosineAnnealingLR, **function["parameters"]
        )
    elif function["name"] == "cosine_annealing_warm_restarts":
        scheduler_dict["scheduler"] = partial(
            optim.lr_scheduler.CosineAnnealingWarmRestarts, **function["parameters"]
        )
    elif function["name"] == "reduce_on_plateau":
        scheduler_dict["scheduler"] = partial(
            optim.lr_scheduler.ReduceLROnPlateau, **function["parameters"]
        )
    else:
        raise AttributeError(
            f"Selected scheduler type {function['name']} is not supported, yet!"
        )

    def scheduler_init(optimizer, scheduler_dict_):
        scheduler_dict_["scheduler"] = scheduler_dict["scheduler"](optimizer=optimizer)
        return scheduler_dict_

    return partial(scheduler_init, scheduler_dict_=scheduler_dict)


def callback_factory(
    callbacks: Union[dict[str, dict[str, Any]], None],
) -> Union[list[pl_callbacks.Callback], None]:
    """Create PyTorch Lightning callbacks from configuration."""

    if callbacks is not None and len(callbacks) > 0:

        supported_callbacks = {
            "earlystopping": pl_callbacks.EarlyStopping,
            "modelcheckpoint": pl_callbacks.ModelCheckpoint,
            "mlflowmodelcheckpoint": MLFlowModelCheckpointCallback,
            "modelsummary": pl_callbacks.ModelSummary,
            "learningratemonitor": pl_callbacks.LearningRateMonitor,
        }

        callback_list = []
        for callback_name, callback_params in callbacks.items():
            if callback_name in supported_callbacks.keys():
                if callback_params is None:
                    callback_params = {}

                callback_list.append(
                    supported_callbacks[callback_name](**callback_params)
                )
            else:
                raise AttributeError(
                    f"Selected callback type {callback_name} is not supported, yet!"
                )
        return callback_list
    return None


def runner_factory(name: str, parameters: dict):
    """
    Create a runner with the given parameters.

    Args:
        name: The type of runner to create
        parameters: Configuration parameters for the runner

    Returns:
        Configured runner instance
    """
    if name == "ptf":
        return PTFRunner(**parameters)
    else:
        raise AttributeError(
            f"Selected runner type '{name}' is not supported. "
            f"Supported types: 'ptf'"
        )