#!/usr/bin/env python3
"""Run or inspect one source-local Metric Contract v1.2 host shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.evaluation.supervisor import execute_host_jobs, parse_source_roots

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--host-id", required=True)
    run.add_argument("--expected-git-sha", required=True)
    run.add_argument("--source-root", action="append", required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--state-root", type=Path, required=True)
    run.add_argument("--gpus", required=True)
    run.add_argument("--hf-cli", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--blas-threads", type=int, default=4)
    run.add_argument("--minimum-free-gb", type=float, default=100.0)
    run.add_argument(
        "--inventory",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
        ),
    )
    run.add_argument(
        "--execution",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/execution.yaml"
        ),
    )
    run.add_argument(
        "--dataset-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml",
    )
    status = subparsers.add_parser("status")
    status.add_argument("--state-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "status":
        state = args.state_root / "state.json"
        if not state.is_file():
            print(json.dumps({"status": "NOT_STARTED"}, indent=2))
            return 1
        print(state.read_text(encoding="utf-8"), end="")
        return 0
    ledger = execute_host_jobs(
        repository_root=REPOSITORY_ROOT,
        inventory_path=args.inventory,
        execution_path=args.execution,
        host_id=args.host_id,
        expected_git_sha=args.expected_git_sha,
        source_roots=parse_source_roots(args.source_root),
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        state_root=args.state_root,
        gpu_uuids=[item.strip() for item in args.gpus.split(",") if item.strip()],
        hf_cli=args.hf_cli,
        dataset_config_path=args.dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        blas_threads=args.blas_threads,
        minimum_free_gb=args.minimum_free_gb,
    )
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0 if ledger["status"] == "REMOTE_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
