#!/usr/bin/env python3
"""Plan and, only with a separate exact authorization, run Task F protected evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.evaluation.task_f_protected import (
    aggregate_protected_scores,
    build_protected_plan,
    default_run_plan,
    export_protected_record,
    load_json_or_yaml,
    validate_protected_plan,
    verify_context_scores,
    verify_protected_feature_artifact,
    write_context_scores,
)
from oge.studies.hashing import canonical_json_bytes


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--id-pipeline", type=Path, required=True)
    plan.add_argument("--gate3-terminal", type=Path, required=True)
    plan.add_argument("--planning-git-sha", required=True)
    plan.add_argument("--output", type=Path, required=True)

    export = commands.add_parser("export-record")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--authorization", type=Path, required=True)
    export.add_argument("--record-id", required=True)
    export.add_argument("--execution-git-sha", required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--dataset-config", type=Path, required=True)
    export.add_argument("--data-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--device", required=True)
    export.add_argument("--batch-size", type=int, default=512)
    export.add_argument("--num-workers", type=int, default=4)

    score = commands.add_parser("score-context")
    score.add_argument("--geometry", type=Path, required=True)
    score.add_argument("--protected", action="append", required=True, metavar="SPLIT=PATH")
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--chunk-size", type=int, default=2048)

    collect = commands.add_parser("collect")
    collect.add_argument("--score-path", type=Path, action="append", required=True)
    collect.add_argument("--expected-contexts", type=int, default=360)
    collect.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--kind", choices=("plan", "feature", "scores"), required=True)
    verify.add_argument("--path", type=Path, required=True)
    return parser


def _split_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        split, separator, path = value.partition("=")
        if not separator or split in output:
            raise ValueError("--protected values must be unique SPLIT=PATH entries")
        output[split] = Path(path)
    return output


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        payload = build_protected_plan(
            run_plan=default_run_plan(),
            id_pipeline=load_json_or_yaml(args.id_pipeline),
            gate3_terminal=load_json_or_yaml(args.gate3_terminal),
            planning_git_sha=args.planning_git_sha,
        )
        _write_new(args.output, payload)
        print(json.dumps({"status": "PASS", "records": 2520, "plan_sha256": payload["plan_sha256"]}, sort_keys=True))
    elif args.command == "export-record":
        output = export_protected_record(
            plan=load_json_or_yaml(args.plan),
            authorization=load_json_or_yaml(args.authorization),
            record_id=args.record_id,
            execution_git_sha=args.execution_git_sha,
            checkpoint_path=args.checkpoint,
            dataset_config_path=args.dataset_config,
            data_root=args.data_root,
            output_root=args.output_root,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(output)
    elif args.command == "score-context":
        output = write_context_scores(
            geometry_path=args.geometry,
            protected_artifacts=_split_paths(args.protected),
            output_root=args.output_root,
            chunk_size=args.chunk_size,
        )
        print(output)
    elif args.command == "collect":
        payload = aggregate_protected_scores(
            score_paths=args.score_path, expected_contexts=args.expected_contexts
        )
        _write_new(args.output, payload)
        print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    else:
        if args.kind == "plan":
            payload = validate_protected_plan(load_json_or_yaml(args.path))
        elif args.kind == "feature":
            payload = verify_protected_feature_artifact(args.path)["manifest"]
        else:
            payload = verify_context_scores(args.path)["manifest"]
        print(json.dumps({"status": "PASS", "schema_version": payload["schema_version"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
