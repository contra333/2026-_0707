#!/usr/bin/env python3
"""Aggregate verified Metric Contract v1.2 bundles across seeds and roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.evaluation.aggregation import aggregate_production_bundles
from oge.evaluation.production import load_frozen_inventory
from oge.studies.supervisor import upload_artifact_tree, verify_hf_preflight

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle-root", type=Path, action="append")
    source.add_argument("--operational-root", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-cli", type=Path)
    parser.add_argument("--upload-control-root", type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
        ),
    )
    args = parser.parse_args()
    output = aggregate_production_bundles(
        inventory=load_frozen_inventory(args.inventory),
        bundle_roots=args.bundle_root or [],
        output_dir=args.output_dir,
        operational_roots=args.operational_root or [],
    )
    if (args.hf_cli is None) != (args.upload_control_root is None):
        raise ValueError("--hf-cli and --upload-control-root must be provided together")
    if args.hf_cli is not None:
        verify_hf_preflight(
            args.hf_cli,
            account="contra333",
            bucket="contra333/ICLR_RUN",
        )
        summary = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
        destination = (
            "hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2/"
            f"_aggregates/{summary['inventory_hash']}/{summary['evaluation_git_sha']}"
        )
        upload_artifact_tree(
            output,
            hf_cli=args.hf_cli,
            bucket="contra333/ICLR_RUN",
            destination=destination,
            control_root=args.upload_control_root,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
