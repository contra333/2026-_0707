#!/usr/bin/env python3
"""Build the verified 42-row protocol-v1.2 follow-up aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.studies.aggregation import build_followup_aggregate
from oge.studies.artifacts import atomic_write_json, sha256_file
from oge.studies.execution import load_followup_execution_plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results/wrn28_10_optimizer_hpo_v1_2"
DEFAULT_FOLLOWUP_PLAN = RESULT_ROOT / "seed0_20260728/followup_plan.json"
DEFAULT_GRID_RESULTS = RESULT_ROOT / "seed0_20260728/grid_results.json"
DEFAULT_EXECUTION_PLAN = (
    REPOSITORY_ROOT
    / "configs/studies/wrn28_10_optimizer_hpo_v1_2/followup_execution.yaml"
)
DEFAULT_OUTPUT = RESULT_ROOT / "followup_20260730/aggregate.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--followup-plan", type=Path, default=DEFAULT_FOLLOWUP_PLAN)
    parser.add_argument("--grid-results", type=Path, default=DEFAULT_GRID_RESULTS)
    parser.add_argument("--execution-plan", type=Path, default=DEFAULT_EXECUTION_PLAN)
    parser.add_argument(
        "--host-artifact",
        action="append",
        required=True,
        metavar="HOST=PATH",
        help="Local copy containing state.json, artifact_validation.json, and trials/*/trial.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return value


def _host_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        host_id, separator, path = value.partition("=")
        if not separator or not host_id or not path or host_id in output:
            raise ValueError(f"invalid or duplicate --host-artifact value: {value!r}")
        output[host_id] = Path(path)
    return output


def _load_host_artifact(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    state = _load_json(path / "state.json")
    validation = _load_json(path / "artifact_validation.json")
    if validation != state.get("verification"):
        raise ValueError(f"{path} artifact validation differs from state.json")
    records: list[dict[str, object]] = []
    expected_checksums = validation.get("trial_json_sha256")
    if not isinstance(expected_checksums, dict):
        raise ValueError(f"{path} is missing trial checksums")
    candidates = sorted(path.glob("trials/*/trial.json"))
    if {candidate.parent.name for candidate in candidates} != set(expected_checksums):
        raise ValueError(f"{path} trial files differ from artifact validation")
    for candidate in candidates:
        trial_id = candidate.parent.name
        if sha256_file(candidate) != expected_checksums[trial_id]:
            raise ValueError(f"{candidate} checksum mismatch")
        records.append(_load_json(candidate))
    return records, state


def main() -> int:
    args = parse_args()
    followup_plan = _load_json(args.followup_plan)
    grid_results = _load_json(args.grid_results)
    execution_plan = load_followup_execution_plan(args.execution_plan, followup_plan)
    roots = _host_paths(args.host_artifact)
    if set(roots) != set(execution_plan["hosts"]):
        raise ValueError("host artifact arguments must match the execution plan hosts")
    host_records: dict[str, list[dict[str, object]]] = {}
    host_states: dict[str, dict[str, object]] = {}
    for host_id, path in roots.items():
        host_records[host_id], host_states[host_id] = _load_host_artifact(path)
    aggregate = build_followup_aggregate(
        followup_plan=followup_plan,
        execution_plan=execution_plan,
        grid_records=grid_results["records"],
        host_records=host_records,
        host_states=host_states,
    )
    atomic_write_json(args.output, aggregate)
    print(aggregate["aggregate_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
