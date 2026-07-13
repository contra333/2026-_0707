#!/usr/bin/env python3
"""Verify official OpenOOD imglists and referenced image paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.data import inspect_imglist, load_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10"], required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1]
        / "configs/datasets/openood_cifar10_v1_5.yaml",
    )
    parser.add_argument("--write-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_dataset_config(args.config)
    image_root = args.data_root / config.get("image_root", "")
    manifests: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for key, item in config["datasets"].items():
        manifest = inspect_imglist(
            imglist_path=args.data_root / item["imglist"],
            data_root=image_root,
            dataset_name=item["dataset_name"],
        )
        manifests[key] = manifest
        if manifest["missing_image_count"]:
            errors.append(f"{key}: missing images")
        if manifest["duplicate_sample_id_count"]:
            errors.append(f"{key}: duplicate sample IDs")
        if "expected_count" in item and manifest["line_count"] != item["expected_count"]:
            errors.append(
                f"{key}: expected {item['expected_count']} entries, "
                f"found {manifest['line_count']}"
            )
        if item["is_id"] and (
            manifest["label_min"] is None
            or manifest["label_min"] < 0
            or manifest["label_max"] > 9
        ):
            errors.append(f"{key}: ID label range is outside [0, 9]")

    report = {
        "protocol_name": config.get("protocol_name"),
        "data_root": str(args.data_root),
        "manifests": manifests,
        "errors": errors,
    }
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Wrote validation report to {args.write_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
