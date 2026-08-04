#!/usr/bin/env python3
"""Run one frozen protected Metric Contract v1.2 checkpoint job."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.evaluation.production import PROTECTED_AUTHORIZATION, run_production_job


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--protected-authorization", required=True)
    parser.add_argument("--expected-evaluation-git-sha")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=(
            root
            / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
        ),
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=root / "configs/datasets/oge_cifar10_holdout_v1.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.protected_authorization != PROTECTED_AUTHORIZATION:
        raise PermissionError("exact Issue #57 authorization string is required")
    output = run_production_job(
        inventory_path=args.inventory,
        job_id=args.job_id,
        host_id=args.host_id,
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        repository_root=Path(__file__).parents[1],
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        protected_authorization=args.protected_authorization,
        expected_evaluation_git_sha=args.expected_evaluation_git_sha,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
