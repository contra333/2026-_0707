#!/usr/bin/env python3
"""Materialize an ignored Task F launch bundle and deterministic approval packet."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from oge.training import (
    build_task_f_training_config,
    generate_approval_packet,
    generate_execution_only_pilot,
    generate_research_run_matrix,
    load_training_config,
    matrix_count_summary,
    validate_no_protected_ood_references,
)


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
        default=REPOSITORY_ROOT / "configs/training/cifar10_wrn28_10_holdout_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-seeds", nargs=5, type=int, required=True)
    parser.add_argument("--adam-factorial-seeds", nargs=3, type=int, required=True)
    parser.add_argument("--sgdm-seeds", nargs=3, type=int, required=True)
    parser.add_argument("--audit-steps", nargs="*", type=int, default=[])
    parser.add_argument("--pilot-seed", type=int)
    parser.add_argument("--pilot-max-epochs", type=int)
    parser.add_argument("--execution-sha", default=None)
    parser.add_argument("--expected-gpu-hours")
    parser.add_argument("--expected-wall-time")
    parser.add_argument("--expected-storage")
    parser.add_argument("--task-f-b-specification-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    artifacts_root = (REPOSITORY_ROOT / "artifacts").resolve()
    if output_dir == artifacts_root or artifacts_root not in output_dir.parents:
        raise ValueError("full Task F launch bundles must be written below ignored artifacts/")
    validate_no_protected_ood_references(str(output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Task F output directory must be absent or empty")

    base_config = load_training_config(args.base_config)
    plan = generate_research_run_matrix(
        anchor_seeds=args.anchor_seeds,
        adam_factorial_seeds=args.adam_factorial_seeds,
        sgdm_seeds=args.sgdm_seeds,
    )
    materialized_runs = []
    for run in plan["runs"]:
        config = build_task_f_training_config(
            base_config,
            run,
            audit_steps=args.audit_steps,
        )
        relative_path = Path("runs") / f"{run['run_id']}.yaml"
        _write_yaml(output_dir / relative_path, config)
        materialized = dict(run)
        materialized["config_path"] = relative_path.as_posix()
        materialized["branch_neutral_config_sha256"] = config["task_f"][
            "branch_neutral_config_sha256"
        ]
        materialized_runs.append(materialized)
    plan = dict(plan)
    plan["runs"] = materialized_runs
    plan["count_summary"] = matrix_count_summary(plan)
    _write_json(output_dir / "run_matrix.json", plan)

    if (args.pilot_seed is None) != (args.pilot_max_epochs is None):
        raise ValueError("pilot seed and max epochs must be supplied together")
    pilot = None
    if args.pilot_seed is not None:
        pilot = generate_execution_only_pilot(
            base_config=base_config,
            seed=args.pilot_seed,
            max_epochs=args.pilot_max_epochs,
            audit_steps=args.audit_steps,
        )
        _write_yaml(output_dir / "pilot/execution_only.yaml", pilot)

    calibration = {
        key: value
        for key, value in {
            "expected_gpu_hours": args.expected_gpu_hours,
            "expected_wall_time": args.expected_wall_time,
            "expected_storage": args.expected_storage,
        }.items()
        if value is not None
    }
    packet = generate_approval_packet(
        execution_sha=args.execution_sha or _git_sha(),
        pilot_config=pilot,
        calibration=calibration,
        task_f_b_specification_sha256=args.task_f_b_specification_sha256,
        audit_schedule_frozen=bool(args.audit_steps),
    )
    _write_json(output_dir / "approval_packet.json", packet)
    print(json.dumps(plan["count_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
