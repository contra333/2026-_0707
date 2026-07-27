#!/usr/bin/env python3
"""Verify frozen CIFAR-10 holdout membership against an OpenOOD data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from oge.data import generate_cifar10_holdout, inspect_imglist, load_dataset_config


FILENAMES = (
    "train_cifar10_45k.txt",
    "val_cifar10_5k.txt",
    "test_cifar10_10k.txt",
    "split_manifest.jsonl",
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=repository_root / "configs/datasets/oge_cifar10_holdout_v1.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.dataset_config.resolve()
    config = load_dataset_config(config_path)
    if config.get("protocol_name") != "oge_cifar10_holdout_v1":
        raise ValueError("dataset config must identify oge_cifar10_holdout_v1")
    source_root = args.data_root / "benchmark_imglist/cifar10"
    committed_root = config_path.parent / "oge_cifar10_holdout_v1"

    with tempfile.TemporaryDirectory(prefix="oge-cifar10-holdout-") as temporary:
        generated_root = Path(temporary)
        generated = generate_cifar10_holdout(
            official_train_imglist=source_root / "train_cifar10.txt",
            openood_id_validation_imglist=source_root / "val_cifar10.txt",
            openood_id_test_imglist=source_root / "test_cifar10.txt",
            output_dir=generated_root,
        )
        comparisons = {}
        for filename in FILENAMES:
            committed = (committed_root / filename).read_bytes()
            regenerated = (generated_root / filename).read_bytes()
            comparisons[filename] = {
                "byte_identical": committed == regenerated,
                "committed_sha256": hashlib.sha256(committed).hexdigest(),
                "regenerated_sha256": hashlib.sha256(regenerated).hexdigest(),
            }

    image_root = args.data_root / config.get("image_root", "")
    manifests = {}
    for role in ("id_train", "id_validation", "id_test"):
        item = config["datasets"][role]
        report = inspect_imglist(
            imglist_path=committed_root / Path(item["imglist"]).name,
            data_root=image_root,
            dataset_name=item["dataset_name"],
        )
        manifests[role] = report

    report = {
        "protocol_name": config["protocol_name"],
        "source_generation": generated,
        "committed_comparisons": comparisons,
        "image_manifests": manifests,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not all(item["byte_identical"] for item in comparisons.values()):
        raise RuntimeError("committed holdout differs from deterministic regeneration")
    for role, manifest in manifests.items():
        if manifest["missing_image_count"] or manifest["duplicate_sample_id_count"]:
            raise RuntimeError(f"{role} failed image-path or sample-ID validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

