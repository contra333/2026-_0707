"""Read-only validation of an imglist and its referenced images."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from .imglist_dataset import parse_imglist_entry


def inspect_imglist(
    *,
    imglist_path: str | Path,
    data_root: str | Path,
    dataset_name: str,
) -> dict[str, object]:
    path = Path(imglist_path)
    root = Path(data_root)
    content = path.read_bytes()
    lines = content.decode("utf-8").splitlines()
    entries = [
        parse_imglist_entry(line, line_number=index)
        for index, line in enumerate(lines, start=1)
    ]
    sample_ids = [f"{dataset_name}:{relative_path}" for relative_path, _ in entries]
    labels = [label for _, label in entries]
    missing = [relative_path for relative_path, _ in entries if not (root / relative_path).is_file()]
    counts = Counter(sample_ids)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    histogram = Counter(labels)
    return {
        "imglist_path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "line_count": len(entries),
        "missing_image_count": len(missing),
        "missing_images": missing,
        "duplicate_sample_id_count": len(duplicates),
        "duplicate_sample_ids": duplicates,
        "label_min": min(labels) if labels else None,
        "label_max": max(labels) if labels else None,
        "class_histogram": {str(label): histogram[label] for label in sorted(histogram)},
    }
