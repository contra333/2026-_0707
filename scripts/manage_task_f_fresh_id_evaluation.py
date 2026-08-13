#!/usr/bin/env python3
"""Plan and validate the Task F fresh ID-only evaluation bridge."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from oge.evaluation.task_f_fresh import (
    build_fresh_evaluation_plan,
    build_id_input,
    summarize_export_coverage,
    validate_bound_inventory,
)
from oge.studies.hashing import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_new_json(path: Path, value: object) -> None:
    artifacts = (REPOSITORY_ROOT / "artifacts").resolve()
    destination = path.resolve()
    if destination == artifacts or artifacts not in destination.parents:
        raise ValueError("Task F fresh evaluation outputs must be below ignored artifacts/")
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise FileExistsError(f"refusing to overwrite different output: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--run-matrix", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--observed-jobs", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--bindings", type=Path, required=True)
    validate.add_argument(
        "--dataset-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml",
    )
    validate.add_argument("--bridge-root", type=Path, required=True)
    validate.add_argument("--terminal", type=Path, required=True)

    input_parser = subparsers.add_parser("build-id-input")
    input_parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml",
    )
    input_parser.add_argument("--data-root", type=Path, required=True)
    input_parser.add_argument("--split", choices=("id_train", "id_validation"), required=True)
    input_parser.add_argument("--output", type=Path, required=True)
    input_parser.add_argument("--batch-size", type=int, default=512)
    input_parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        output = build_fresh_evaluation_plan(_load_json(args.run_matrix))
        if args.observed_jobs is not None:
            observed = _load_json(args.observed_jobs)
            output = dict(output)
            output["observed_coverage"] = summarize_export_coverage(
                output, observed["jobs"]
            )
        _write_new_json(args.output, output)
        print(json.dumps(output["counts"], sort_keys=True))
        return 0
    if args.command == "validate":
        terminal = validate_bound_inventory(
            plan=_load_json(args.plan),
            bindings=_load_json(args.bindings),
            dataset_config_path=args.dataset_config,
            output_root=args.bridge_root,
            terminal_path=args.terminal,
        )
        print(json.dumps({"status": terminal["status"], **terminal["counts"]}, sort_keys=True))
        return 0 if terminal["status"] == "PASS" else 2
    result = build_id_input(
        dataset_config_path=args.dataset_config,
        data_root=args.data_root,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
