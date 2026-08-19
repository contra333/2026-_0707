#!/usr/bin/env python3
"""Plan, export, fit, score, and collect the approved ResNet replication."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from oge.evaluation.resnet18_replication_production import (
    build_production_plan,
    build_protected_authorization,
    collect_production_results,
    evaluate_paired_endpoint,
    export_checkpoint_phase,
    fit_id_artifact,
    recover_pair_artifact,
    validate_production_plan,
    validate_protected_authorization,
    verify_id_fit_artifact,
    verify_pair_artifact,
)
from oge.studies.hashing import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_new(path: Path, value: object) -> None:
    destination = path.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    if destination == artifacts or artifacts not in destination.parents:
        raise ValueError("production outputs must be below artifacts/")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-matrix", type=Path, required=True)
    plan.add_argument("--training-terminal", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--evaluation-git-sha")

    auth = commands.add_parser("authorize")
    auth.add_argument("--plan", type=Path, required=True)
    auth.add_argument("--approved-at", required=True)
    auth.add_argument("--output", type=Path, required=True)

    export = commands.add_parser("export")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--authorization", type=Path, required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--phase", choices=("id", "protected"), required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--dataset-config", type=Path, required=True)
    export.add_argument("--data-root", type=Path, required=True)
    export.add_argument("--artifact-root", type=Path, required=True)
    export.add_argument("--device", required=True)
    export.add_argument("--batch-size", type=int, default=512)
    export.add_argument("--num-workers", type=int, default=4)

    fit = commands.add_parser("fit-id")
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--id-artifact", type=Path, required=True)
    fit.add_argument("--output-root", type=Path, required=True)

    score = commands.add_parser("score-pair")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--authorization", type=Path, required=True)
    score.add_argument("--cell-id", required=True)
    score.add_argument("--training-seed", type=int, required=True)
    score.add_argument("--coupled-fit", type=Path, required=True)
    score.add_argument("--decoupled-fit", type=Path, required=True)
    score.add_argument("--coupled-protected", type=Path, required=True)
    score.add_argument("--decoupled-protected", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--chunk-size", type=int, default=2048)

    collect = commands.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--pair-artifact", type=Path, action="append", required=True)
    collect.add_argument("--output", type=Path, required=True)

    recover = commands.add_parser("recover-pair")
    recover.add_argument("--source", type=Path, required=True)
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--scoring-git-sha")

    verify = commands.add_parser("verify")
    verify.add_argument("--kind", choices=("plan", "authorization", "fit", "pair"), required=True)
    verify.add_argument("--path", type=Path, required=True)
    verify.add_argument("--plan", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        payload = build_production_plan(
            run_plan=_json(args.run_matrix),
            training_terminal=_json(args.training_terminal),
            training_terminal_path=args.training_terminal,
            evaluation_git_sha=args.evaluation_git_sha or _git_sha(),
        )
        _write_new(args.output, payload)
        print(json.dumps({"status": "PASS", "runs": 20, "plan_sha256": payload["plan_sha256"]}, sort_keys=True))
    elif args.command == "authorize":
        payload = build_protected_authorization(
            plan=_json(args.plan), evaluation_git_sha=_git_sha(), approved_at=args.approved_at
        )
        _write_new(args.output, payload)
        print(json.dumps({"status": "AUTHORIZED", "authorization_sha256": payload["authorization_sha256"]}, sort_keys=True))
    elif args.command == "export":
        output = export_checkpoint_phase(
            plan=_json(args.plan), authorization=_json(args.authorization),
            run_id=args.run_id, phase=args.phase, checkpoint_path=args.checkpoint,
            dataset_config_path=args.dataset_config, data_root=args.data_root,
            artifact_root=args.artifact_root, device=args.device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            repository_root=ROOT,
        )
        print(output)
    elif args.command == "fit-id":
        print(fit_id_artifact(plan=_json(args.plan), run_id=args.run_id, id_artifact=args.id_artifact, output_root=args.output_root))
    elif args.command == "score-pair":
        print(evaluate_paired_endpoint(
            plan=_json(args.plan), authorization=_json(args.authorization),
            cell_id=args.cell_id, training_seed=args.training_seed,
            coupled_fit=args.coupled_fit, decoupled_fit=args.decoupled_fit,
            coupled_protected=args.coupled_protected,
            decoupled_protected=args.decoupled_protected,
            output_root=args.output_root, chunk_size=args.chunk_size,
        ))
    elif args.command == "recover-pair":
        print(
            recover_pair_artifact(
                source_path=args.source,
                scoring_git_sha=args.scoring_git_sha or _git_sha(),
                output_root=args.output_root,
            )
        )
    elif args.command == "collect":
        payload = collect_production_results(plan=_json(args.plan), pair_artifacts=args.pair_artifact)
        _write_new(args.output, payload)
        print(json.dumps({"status": payload["status"], "scientific_verdict": payload["scientific_verdict"], "terminal_sha256": payload["terminal_sha256"]}, sort_keys=True))
    else:
        if args.kind == "plan":
            payload = validate_production_plan(_json(args.path))
        elif args.kind == "authorization":
            if args.plan is None:
                raise ValueError("authorization verification requires --plan")
            payload = validate_protected_authorization(_json(args.path), plan=_json(args.plan))
        elif args.kind == "fit":
            payload = verify_id_fit_artifact(args.path)["manifest"]
        else:
            payload = verify_pair_artifact(args.path)["record"]
        print(json.dumps({"status": "PASS", "schema_version": payload["schema_version"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
