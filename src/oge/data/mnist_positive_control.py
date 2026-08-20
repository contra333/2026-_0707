"""Pinned MNIST input path for the Zhao et al. ResNet9 positive control."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import MNIST

MNIST_POSITIVE_CONTROL_SIZE = 32
MNIST_POSITIVE_CONTROL_MEAN = (0.1307,)
MNIST_POSITIVE_CONTROL_STD = (0.3081,)


class IndexedClassificationDataset(Dataset):
    """Expose torchvision samples through the project's mapping batch API."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        image, label = self.dataset[index]
        return {
            "image": image,
            "class_label": int(label),
            "sample_index": int(index),
        }


class RecordingRandomSampler(Sampler[int]):
    """Match RandomSampler's randperm behavior and retain the epoch-order digest."""

    def __init__(self, dataset: Dataset, *, generator: torch.Generator) -> None:
        self.dataset = dataset
        self.generator = generator
        self.last_order_digest: str | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        payload = ",".join(str(index) for index in order).encode("ascii")
        self.last_order_digest = hashlib.sha256(payload).hexdigest()
        return iter(order)


def make_mnist_positive_control_transform() -> transforms.Compose:
    """Reproduce the pinned upstream Resize -> ToTensor -> Normalize recipe."""
    return transforms.Compose(
        [
            transforms.Resize(MNIST_POSITIVE_CONTROL_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                MNIST_POSITIVE_CONTROL_MEAN,
                MNIST_POSITIVE_CONTROL_STD,
            ),
        ]
    )


def build_mnist_positive_control_loaders(
    *,
    data_root: str | Path,
    batch_size: int,
    train_generator: torch.Generator,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, RecordingRandomSampler]:
    """Build stochastic training and deterministic train/test evaluation loaders."""
    transform = make_mnist_positive_control_transform()
    root = Path(data_root).expanduser().resolve()
    train_dataset = IndexedClassificationDataset(
        MNIST(root=root, train=True, transform=transform, download=download)
    )
    test_dataset = IndexedClassificationDataset(
        MNIST(root=root, train=False, transform=transform, download=download)
    )
    sampler = RecordingRandomSampler(train_dataset, generator=train_generator)
    shared = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=False, **shared)
    train_eval_loader = DataLoader(train_dataset, shuffle=False, drop_last=False, **shared)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **shared)
    return train_loader, train_eval_loader, test_loader, sampler
