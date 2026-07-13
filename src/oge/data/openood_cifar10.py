"""Explicit loader factory for the OpenOOD-aligned CIFAR-10 protocol."""

from __future__ import annotations

from pathlib import Path

import yaml
from torch import Generator
from torch.utils.data import DataLoader

from .imglist_dataset import ImglistDataset
from .transforms import make_cifar10_eval_transform, make_cifar10_train_transform

DATASET_REGISTRY = {"imglist": ImglistDataset}


def load_dataset_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "datasets" not in config:
        raise ValueError("dataset config must contain a 'datasets' mapping")
    if not isinstance(config["datasets"], dict):
        raise ValueError("dataset config 'datasets' must be a mapping")
    return config


def build_openood_cifar10_loaders(
    config: dict,
    *,
    data_root: str | Path,
    id_max_samples: int | None = None,
    ood_max_samples: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    drop_last: bool | None = None,
    persistent_workers: bool | None = None,
    train_generator: Generator | None = None,
) -> dict[str, object]:
    dataset_class_name = config.get("dataset_class", "imglist")
    try:
        dataset_class = DATASET_REGISTRY[dataset_class_name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset_class: {dataset_class_name}") from exc

    batch_size = int(
        config.get("batch_size", 128) if batch_size is None else batch_size
    )
    num_workers = int(
        config.get("num_workers", 0) if num_workers is None else num_workers
    )
    pin_memory = bool(
        config.get("pin_memory", False) if pin_memory is None else pin_memory
    )
    drop_last = bool(config.get("drop_last", False) if drop_last is None else drop_last)
    persistent_workers = bool(
        config.get("persistent_workers", False)
        if persistent_workers is None
        else persistent_workers
    )
    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers=True requires num_workers > 0")

    root = Path(data_root)
    image_root = root / config.get("image_root", "")
    loaders: dict[str, object] = {"id": {}, "ood_validation": {}, "near": {}, "far": {}}

    for key, item in config["datasets"].items():
        required = {"dataset_name", "split", "is_id", "group", "imglist"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"dataset {key!r} is missing fields: {sorted(missing)}")
        group = item["group"]
        if group not in loaders:
            raise ValueError(f"dataset {key!r} has unknown group: {group}")
        is_train = bool(item["is_id"] and item["split"] == "train")
        transform = (
            make_cifar10_train_transform()
            if is_train
            else make_cifar10_eval_transform()
        )
        dataset = dataset_class(
            dataset_name=item["dataset_name"],
            split=item["split"],
            is_id=bool(item["is_id"]),
            imglist_path=root / item["imglist"],
            data_root=image_root,
            transform=transform,
        )
        max_samples = id_max_samples if item["is_id"] else ood_max_samples
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            from torch.utils.data import Subset

            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        loaders[group][key] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last if is_train else False,
            persistent_workers=persistent_workers,
            generator=train_generator if is_train else None,
        )
    return loaders
