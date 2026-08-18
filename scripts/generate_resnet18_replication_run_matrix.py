#!/usr/bin/env python3
"""Materialize the ignored ResNet-18 replication plan and pilot configs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from oge.training.resnet18_replication_plan import (
    build_resnet18_replication_training_config,
    generate_resnet18_approval_packet,
    generate_resnet18_execution_only_pilot_configs,
    generate_resnet18_replication_matrix,
    resnet18_replication_count_summary,
    validate_no_protected_references,
)
from oge.training.runner import load_training_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs/training/cifar10_resnet18_replication_v3.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--research-seeds", nargs=5, type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--execution-sha")
    parser.add_argument("--pilot-approval-reference")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    artifacts_root = (REPOSITORY_ROOT / "artifacts").resolve()
    if output_dir == artifacts_root or artifacts_root not in output_dir.parents:
        raise ValueError("replication launch bundles must be written below artifacts/")
    validate_no_protected_references(str(output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("replication output directory must be absent or empty")

    base_config = load_training_config(args.base_config)
    plan = generate_resnet18_replication_matrix(research_seeds=args.research_seeds)
    materialized_runs = []
    for run in plan["runs"]:
        config = build_resnet18_replication_training_config(base_config, run)
        relative = Path("runs") / f"{run['run_id']}.yaml"
        _write_yaml(output_dir / relative, config)
        row = dict(run)
        row["config_path"] = relative.as_posix()
        row["branch_neutral_config_sha256"] = config["resnet18_replication"][
            "branch_neutral_config_sha256"
        ]
        row["cross_lr_neutral_config_sha256"] = config["resnet18_replication"][
            "cross_lr_neutral_config_sha256"
        ]
        materialized_runs.append(row)
    plan = dict(plan)
    plan["runs"] = materialized_runs
    plan["count_summary"] = resnet18_replication_count_summary(plan)
    _write_json(output_dir / "run_matrix.json", plan)

    pilot_configs = generate_resnet18_execution_only_pilot_configs(
        base_config=base_config
    )
    for config in pilot_configs:
        run_id = config["resnet18_replication"]["run_id"]
        _write_yaml(output_dir / "pilot" / f"{run_id}.yaml", config)
    packet = generate_resnet18_approval_packet(
        execution_sha=args.execution_sha or _git_sha(),
        pilot_configs=pilot_configs,
        pilot_approval_reference=args.pilot_approval_reference,
    )
    _write_json(output_dir / "approval_packet.json", packet)
    print(json.dumps(plan["count_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
