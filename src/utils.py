import torch.optim.optimizer
from yacs.config import CfgNode as CN
from typing import Any, Dict, List, Tuple, Optional
from copy import deepcopy
from torch.utils.data import DataLoader
from torch import nn
from ray.tune import Stopper
from collections import defaultdict
import importlib
import math
import torch


class EarlyStopper(Stopper):
    def __init__(self, metric: str, patience: int, delta: float = 0):
        """Stops the training if the metric does not decrease by at least delta for patience epochs."""
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.counter: Dict[str, int] = defaultdict(lambda: 0)
        self.best_score: Dict[str, float] = defaultdict(lambda: math.inf)

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        score = result[self.metric]

        if score >= self.best_score[trial_id] - self.delta:
            self.counter[trial_id] += 1
        else:
            self.counter[trial_id] = 0

        self.best_score[trial_id] = min(self.best_score[trial_id], score)

        return self.patience <= self.counter[trial_id]

    def stop_all(self) -> bool:
        return False


def get_data_loaders(
    dataset_config: Dict[Any, Any], dataloader_config: Dict[Any, Any]
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    dataset_model = import_class_from_path(dataset_config["class_path"])

    dataset_kwargs = dataset_config["KWARGS"]

    def get_transform(config: Dict[Any, Any]) -> Any:
        if "TRANSFORM" not in config:
            return None
        transform_config = config["TRANSFORM"]
        transform_class = import_class_from_path(transform_config["class_path"])
        return transform_class(**transform_config.get("KWARGS", {}))

    transform_train = get_transform(dataset_config.get("TRAIN", {}))
    transform_val = get_transform(dataset_config.get("VAL", {}))
    transform_test = get_transform(dataset_config.get("TEST", {}))

    train_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["TRAIN"]["KWARGS"]}, transform=transform_train)
    val_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["VAL"]["KWARGS"]}, transform=transform_val)
    test_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["TEST"]["KWARGS"]}, transform=transform_test)

    train_dataloader = DataLoader(train_dataset, **dataloader_config)
    val_dataloader = DataLoader(val_dataset, **dataloader_config)
    test_dataloader = DataLoader(test_dataset, **dataloader_config)

    return train_dataloader, val_dataloader, test_dataloader


def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])


def load_model(config: CN) -> nn.Module:
    model_class = import_class_from_path(config.MODEL.class_path)
    return model_class(**config.MODEL.KWARGS)  # type: ignore


def merge_ray_config_with_config(config: CN, ray_config: CN) -> CN:
    config = deepcopy(config)
    ray_config = deepcopy(ray_config)
    model_kwargs = deepcopy(config)
    for k, v in ray_config.items():
        if k in model_kwargs:
            config.MODEL.KWARGS[k] = v
    return config


class CosineToConstantLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        eta_min_divisor: Optional[float] = None,
    ):  # noqa: D107
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_min_divisor = eta_min_divisor
        super().__init__(optimizer, -1)

    def get_lr(self) -> List[float]:
        """Retrieve the learning rate of each parameter group."""
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        def _get_lr(base_lr: float) -> float:
            min_lr = self.eta_min if self.eta_min_divisor is None else base_lr / self.eta_min_divisor
            if self._step_count - 1 >= self.T_max:
                return min_lr
            return min_lr + (base_lr - min_lr) * (1 + (math.cos(math.pi * (self._step_count - 1) / self.T_max))) / 2

        return [_get_lr(base_lr) for base_lr in self.base_lrs]
