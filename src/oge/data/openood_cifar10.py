"""Explicit loader factory for the OpenOOD-aligned CIFAR-10 protocol."""

from __future__ import annotations

from pathlib import Path

import yaml
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
) -> dict[str, object]:
    dataset_class_name = config.get("dataset_class", "imglist")
    try:
        dataset_class = DATASET_REGISTRY[dataset_class_name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset_class: {dataset_class_name}") from exc

    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 0))
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
        transform = (
            make_cifar10_train_transform()
            if item["is_id"] and item["split"] == "train"
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
            shuffle=bool(item["is_id"] and item["split"] == "train"),
            num_workers=num_workers,
        )
    return loaders
