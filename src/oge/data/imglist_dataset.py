"""Strict imglist dataset with stable project sample identities."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _normalized_relative_posix_path(raw_path: str) -> str:
    candidate = PurePosixPath(raw_path.replace("\\", "/"))
    if not raw_path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"imglist path must be a non-empty relative path: {raw_path!r}")
    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"imglist path must identify a file: {raw_path!r}")
    return normalized


def parse_imglist_entry(line: str, *, line_number: int) -> tuple[str, int]:
    """Parse one ``relative/path integer_label`` entry."""
    stripped = line.strip()
    tokens = stripped.split()
    if len(tokens) != 2:
        raise ValueError(
            f"malformed imglist entry at line {line_number}: "
            "expected '<relative_path> <integer_label>'"
        )
    relative_path = _normalized_relative_posix_path(tokens[0])
    try:
        class_label = int(tokens[1])
    except ValueError as exc:
        raise ValueError(
            f"malformed imglist entry at line {line_number}: label must be an integer"
        ) from exc
    return relative_path, class_label


class ImglistDataset(Dataset[dict[str, Tensor | int | str | bool]]):
    """Read images named by a strict, immutable imglist."""

    def __init__(
        self,
        *,
        dataset_name: str,
        split: str,
        is_id: bool,
        imglist_path: str | Path,
        data_root: str | Path,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if not dataset_name or ":" in dataset_name:
            raise ValueError("dataset_name must be non-empty and must not contain ':'")
        self.dataset_name = dataset_name
        self.split = split
        self.is_id = is_id
        self.imglist_path = Path(imglist_path)
        self.data_root = Path(data_root)
        self.transform = transform
        with self.imglist_path.open("r", encoding="utf-8") as handle:
            self.entries = [
                parse_imglist_entry(line, line_number=index)
                for index, line in enumerate(handle, start=1)
            ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Tensor | int | str | bool]:
        relative_path, class_label = self.entries[index]
        image_path = self.data_root / relative_path
        with Image.open(image_path) as image:
            tensor = self.transform(image)
        return {
            "image": tensor,
            "class_label": class_label,
            "sample_id": f"{self.dataset_name}:{relative_path}",
            "dataset_name": self.dataset_name,
            "split": self.split,
            "is_id": self.is_id,
        }
