#!/usr/bin/env python3
"""Materialize the pending evaluation plan or adjudicate an existing summary.

There is intentionally no command that opens protected data, exports features,
or computes detector results.  Those operations remain unavailable until a
separate owner-approved implementation exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from oge.evaluation.resnet18_replication import (
    adjudicate_resnet18_full_gate,
    build_pending_resnet18_evaluation_plan,
)
from oge.studies.hashing import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_new_json(path: Path, value: object) -> None:
    artifacts = (REPOSITORY_ROOT / "artifacts").resolve()
    destination = path.resolve()
    if destination == artifacts or artifacts not in destination.parents:
        raise ValueError("replication evaluation outputs must be below artifacts/")
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
    plan.add_argument("--planning-sha")

    gate = subparsers.add_parser("gate")
    gate.add_argument("--run-matrix", type=Path, required=True)
    gate.add_argument("--summary", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_plan = _load_json(args.run_matrix)
    expected_run_ids = [str(row["run_id"]) for row in run_plan["runs"]]
    if args.command == "plan":
        output = build_pending_resnet18_evaluation_plan(
            run_plan=run_plan,
            planning_git_sha=args.planning_sha or _git_sha(),
        )
        _write_new_json(args.output, output)
        print(
            json.dumps(
                {
                    "launch_authorization": output["launch_authorization"],
                    **output["expected_coverage"],
                },
                sort_keys=True,
            )
        )
        return 0

    summary = _load_json(args.summary)
    output = adjudicate_resnet18_full_gate(
        expected_run_ids=expected_run_ids,
        observed_run_ids=summary.get("observed_run_ids", []),
        seed_records=summary.get("seed_records", []),
        id_guardrail_by_cell=summary.get("id_guardrail_by_cell", {}),
    )
    _write_new_json(args.output, output)
    print(json.dumps({"verdict": output["verdict"]}, sort_keys=True))
    return 0 if output["verdict"] == "FULL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
