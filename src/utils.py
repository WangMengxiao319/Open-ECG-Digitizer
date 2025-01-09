from yacs.config import CfgNode as CN
from typing import Any, Dict, Tuple
from copy import deepcopy
from torch.utils.data import DataLoader
from torch import nn
import importlib
import math
from ray.tune import Stopper


class EarlyStopper(Stopper):
    def __init__(self, metric: str, patience: int, delta: float = 0):
        """Stops the training if the metric does not decrease by at least delta for patience epochs."""
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = math.inf

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        score = result[self.metric]

        if score >= self.best_score - self.delta:
            self.counter += 1
        else:
            self.counter = 0

        self.best_score = min(self.best_score, score)

        return self.patience <= self.counter

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
