#!/usr/bin/env python3
"""Run Task F fresh ID-only geometry or paired aggregation."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from oge.analysis.task_f_fresh_id import (
    aggregate_paired_records,
    analyze_bound_geometry,
    build_aggregation_contract,
    geometry_seed_records,
    write_aggregation_artifacts,
)
from oge.studies.hashing import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact_path(path: Path) -> Path:
    artifacts = (REPOSITORY_ROOT / "artifacts").resolve()
    resolved = path.resolve()
    if resolved == artifacts or artifacts not in resolved.parents:
        raise ValueError("Task F fresh ID analysis outputs must be below artifacts/")
    return resolved


def _write_new_json(path: Path, payload: object) -> None:
    destination = _artifact_path(path)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite different output: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--train-binding", type=Path, required=True)
    analyze.add_argument("--validation-binding", type=Path, required=True)
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--chunk-size", type=int, default=2048)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--geometry-inventory", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--records", type=Path, required=True)
    aggregate.add_argument("--evaluation-plan", type=Path, required=True)
    aggregate.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "analyze":
        output = analyze_bound_geometry(
            train_binding=_json(args.train_binding),
            validation_binding=_json(args.validation_binding),
            output_root=_artifact_path(args.output_root),
            chunk_size=args.chunk_size,
        )
        print(output)
        return 0
    if args.command == "collect":
        inventory = _json(args.geometry_inventory)
        records = geometry_seed_records(inventory["geometry_roots"])
        payload = {
            "schema_version": "task_f_fresh_id_seed_records_v1",
            "record_count": len(records),
            "records": records,
        }
        _write_new_json(args.output, payload)
        print(json.dumps({"record_count": len(records), "output": str(args.output)}))
        return 0
    records = _json(args.records)
    contract = build_aggregation_contract(_json(args.evaluation_plan))
    payload = aggregate_paired_records(
        records=records["records"],
        expected_seeds_by_cell=contract["expected_seeds_by_cell"],
        expected_contexts=contract["contexts"],
        protected_id_test_available=False,
    )
    destination = write_aggregation_artifacts(
        payload=payload,
        output_directory=_artifact_path(args.output_directory),
    )
    print(json.dumps({"status": payload["status"], "output": str(destination)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
