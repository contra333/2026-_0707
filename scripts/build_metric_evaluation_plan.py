#!/usr/bin/env python3
"""Build or verify the frozen Metric Contract v1.2 evaluation plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.evaluation.planning import (
    build_checkpoint_inventory,
    materialize_evaluation_plan,
    verify_materialized_evaluation_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE = (
    ROOT
    / "results/wrn28_10_optimizer_hpo_v1_2/followup_20260730/aggregate.json"
)
DEFAULT_DATASET_CONFIG = ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml"
DEFAULT_OUTPUT = ROOT / "configs/evaluation/wrn28_10_cifar10_metric_v1_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_checkpoint_inventory(
        aggregate_path=args.aggregate,
        aggregate_sha256_path=args.aggregate.with_name(
            f"{args.aggregate.name}.sha256"
        ),
        dataset_config_path=args.dataset_config,
    )
    if args.check:
        verify_materialized_evaluation_plan(args.output_dir, inventory)
        print(inventory["inventory_hash"])
        return 0
    hashes = materialize_evaluation_plan(inventory, args.output_dir)
    print(inventory["inventory_hash"])
    for name, digest in hashes.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
