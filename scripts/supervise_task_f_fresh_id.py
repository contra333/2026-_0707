#!/usr/bin/env python3
"""Plan, execute, inspect, and collect Task F fresh ID-only host shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from oge.analysis.task_f_fresh_id import (
    analyze_bound_alignment,
    analyze_bound_geometry,
)
from oge.evaluation.task_f_fresh import (
    build_fresh_evaluation_plan,
    build_id_input,
    verify_id_input,
    write_bridge_artifact,
)
from oge.evaluation.task_f_fresh_orchestration import (
    HOSTS,
    SOURCE_TRAINING_SHA,
    build_task_f_pipeline_manifest,
    collect_task_f_host_summaries,
    execute_task_f_host,
    load_completed_upload_evidence,
    wait_for_global_source_gate,
)
from oge.studies.artifacts import atomic_write_json
from oge.studies.hashing import canonical_json_bytes
from oge.studies.supervisor import build_artifact_manifest, upload_artifact_tree, verify_hf_preflight


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "contra333/ICLR_RUN"


def _json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_new_json(path: Path, payload: object) -> None:
    destination = path.resolve()
    artifacts = (REPOSITORY_ROOT / "artifacts").resolve()
    if destination == artifacts or artifacts not in destination.parents:
        raise ValueError("Task F orchestration outputs must be below ignored artifacts/")
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


def _worker(args: argparse.Namespace) -> int:
    spec = _json(args.job_spec)
    if args.command == "_bridge-worker":
        destination = write_bridge_artifact(
            record=spec["record"],
            checkpoint_path=spec["checkpoint_path"],
            feature_artifact_path=spec["feature_artifact_path"],
            dataset_config_path=spec["dataset_config_path"],
            output_root=spec["bridge_root"],
        )
    elif args.command == "_geometry-worker":
        destination = analyze_bound_geometry(
            train_binding=spec["train_binding"],
            validation_binding=spec["validation_binding"],
            output_root=spec["geometry_root"],
            chunk_size=int(spec["chunk_size"]),
        )
    else:
        destination = analyze_bound_alignment(
            left_train_binding=spec["left_train_binding"],
            left_validation_binding=spec["left_validation_binding"],
            right_train_binding=spec["right_train_binding"],
            right_validation_binding=spec["right_validation_binding"],
            pair_direction=spec["pair_direction"],
            output_root=spec["alignment_root"],
            chunk_size=int(spec["chunk_size"]),
        )
    print(json.dumps({"status": "PASS", "artifact_path": str(destination)}, sort_keys=True))
    return 0


def _download_host_shards(
    *, evaluation_git_sha: str, hf_cli: Path, download_root: Path
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for host_id in HOSTS:
        source = (
            f"hf://buckets/{BUCKET}/evaluations/task_f_fresh_id_v1/"
            f"{evaluation_git_sha}/{host_id}"
        )
        destination = download_root / host_id
        if destination.exists():
            if not (destination / "REMOTE_COMPLETE.json").is_file():
                raise ValueError(f"preserved host download is incomplete: {destination}")
            roots[host_id] = destination
            continue
        probe = download_root / f".{host_id}.marker-probe-{uuid.uuid4().hex}.json"
        try:
            subprocess.run(
                [
                    str(hf_cli),
                    "buckets",
                    "cp",
                    f"{source}/REMOTE_COMPLETE.json",
                    str(probe),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            marker = _json(probe)
            if marker.get("status") != "REMOTE_VERIFIED":
                raise ValueError(f"remote Task F host {host_id} is not terminal")
        finally:
            probe.unlink(missing_ok=True)
        attempt = download_root / f".{host_id}.attempt-{uuid.uuid4().hex}"
        attempt.mkdir(parents=True)
        dry = subprocess.run(
            [str(hf_cli), "buckets", "sync", source, str(attempt), "--dry-run", "--no-delete", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [json.loads(line) for line in dry.stdout.splitlines() if line.strip()]
        if not rows or int(rows[0].get("summary", {}).get("deletes", -1)) != 0:
            raise ValueError("Task F host download dry-run does not prove zero deletes")
        subprocess.run(
            [str(hf_cli), "buckets", "sync", source, str(attempt), "--no-delete"],
            check=True,
        )
        if not (attempt / "REMOTE_COMPLETE.json").is_file():
            raise ValueError(f"remote Task F host {host_id} is not terminal")
        os.replace(attempt, destination)
        roots[host_id] = destination
    return roots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--run-matrix", type=Path, required=True)
    plan.add_argument("--observed-export-jobs", type=Path, required=True)
    plan.add_argument("--host-assignment", type=Path, required=True)
    plan.add_argument("--gpu-queues", type=Path, required=True)
    plan.add_argument("--evaluation-git-sha", required=True)
    plan.add_argument("--evaluation-plan-output", type=Path, required=True)
    plan.add_argument("--pipeline-output", type=Path, required=True)

    run = subparsers.add_parser("run-host")
    run.add_argument("--host-id", choices=HOSTS, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--expected-evaluation-git-sha", required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--id-train-input", type=Path, required=True)
    run.add_argument("--id-validation-input", type=Path, required=True)
    run.add_argument("--dataset-config", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--state-root", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--hf-cli", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--blas-threads", type=int, default=4)
    run.add_argument("--minimum-free-gb", type=float, default=100.0)
    run.add_argument("--gate-poll-seconds", type=float, default=60.0)
    run.add_argument("--gate-timeout-hours", type=float, default=72.0)

    status = subparsers.add_parser("status")
    status.add_argument("--state-root", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--evaluation-git-sha", required=True)
    collect.add_argument("--evaluation-plan", type=Path, required=True)
    collect.add_argument("--download-root", type=Path, required=True)
    collect.add_argument("--output-directory", type=Path, required=True)
    collect.add_argument("--state-root", type=Path, required=True)
    collect.add_argument("--hf-cli", type=Path, required=True)
    collect.add_argument("--poll-seconds", type=float, default=60.0)
    collect.add_argument("--timeout-hours", type=float, default=72.0)

    for name in ("_bridge-worker", "_geometry-worker", "_alignment-worker"):
        worker = subparsers.add_parser(name)
        worker.add_argument("--job-spec", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command.startswith("_"):
        return _worker(args)
    if args.command == "plan":
        evaluation_plan = build_fresh_evaluation_plan(_json(args.run_matrix))
        pipeline = build_task_f_pipeline_manifest(
            evaluation_plan=evaluation_plan,
            observed_export_jobs=_json(args.observed_export_jobs)["jobs"],
            host_assignment=_json(args.host_assignment),
            gpu_queues=_json(args.gpu_queues),
            evaluation_git_sha=args.evaluation_git_sha,
        )
        _write_new_json(args.evaluation_plan_output, evaluation_plan)
        _write_new_json(args.pipeline_output, pipeline)
        print(json.dumps(pipeline["counts"], sort_keys=True))
        return 0
    if args.command == "status":
        paths = {
            "ledger": args.state_root / "ledger.json",
            "host_complete": args.state_root / "HOST_COMPLETE.json",
            "central_complete": args.state_root / "CENTRAL_COMPLETE.json",
        }
        output = {
            name: _json(path) if path.is_file() else {"status": "NOT_STARTED"}
            for name, path in paths.items()
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "run-host":
        verify_hf_preflight(args.hf_cli, account="contra333", bucket=BUCKET)
        gate = wait_for_global_source_gate(
            hf_cli=args.hf_cli,
            gate_root=args.state_root / "source_gate",
            poll_seconds=args.gate_poll_seconds,
            timeout_hours=args.gate_timeout_hours,
        )
        atomic_write_json(args.state_root / "SOURCE_GATE_PASS.json", gate)
        if args.id_validation_input.exists():
            verify_id_input(
                args.id_validation_input,
                dataset_config_path=args.dataset_config,
                split="id_validation",
            )
        else:
            build_id_input(
                dataset_config_path=args.dataset_config,
                data_root=args.data_root,
                split="id_validation",
                output_path=args.id_validation_input,
                batch_size=512,
                num_workers=0,
            )
        verify_id_input(
            args.id_train_input,
            dataset_config_path=args.dataset_config,
            split="id_train",
        )
        result = execute_task_f_host(
            repository_root=REPOSITORY_ROOT,
            manifest_path=args.manifest,
            host_id=args.host_id,
            expected_evaluation_git_sha=args.expected_evaluation_git_sha,
            source_root=args.source_root,
            data_root=args.data_root,
            id_train_input=args.id_train_input,
            id_validation_input=args.id_validation_input,
            dataset_config_path=args.dataset_config,
            artifact_root=args.artifact_root,
            state_root=args.state_root,
            python=args.python,
            hf_cli=args.hf_cli,
            batch_size=args.batch_size,
            blas_threads=args.blas_threads,
            minimum_free_gb=args.minimum_free_gb,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    verify_hf_preflight(args.hf_cli, account="contra333", bucket=BUCKET)
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    last_error = "not checked"
    host_roots = None
    while time.monotonic() < deadline:
        try:
            host_roots = _download_host_shards(
                evaluation_git_sha=args.evaluation_git_sha,
                hf_cli=args.hf_cli,
                download_root=args.download_root,
            )
            if all((root / "REMOTE_COMPLETE.json").is_file() for root in host_roots.values()):
                break
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            last_error = str(exc)
        time.sleep(args.poll_seconds)
    if host_roots is None or not all(
        (root / "REMOTE_COMPLETE.json").is_file() for root in host_roots.values()
    ):
        raise TimeoutError(f"Task F central collection timed out: {last_error}")
    central_path = args.output_directory / "CENTRAL_COMPLETE.json"
    manifest_path = args.output_directory / "artifact_manifest.json"
    if args.output_directory.exists():
        if not central_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("preserved Task F central output is incomplete")
        terminal = _json(central_path)
        if (
            terminal.get("status") != "PASS"
            or terminal.get("source_training_sha") != SOURCE_TRAINING_SHA
        ):
            raise ValueError("preserved Task F central terminal identity mismatch")
    else:
        terminal = collect_task_f_host_summaries(
            host_roots=host_roots,
            evaluation_plan=_json(args.evaluation_plan),
            output_directory=args.output_directory,
        )
        build_artifact_manifest(args.output_directory)
    destination = (
        f"hf://buckets/{BUCKET}/evaluations/task_f_fresh_id_v1/"
        f"{args.evaluation_git_sha}/_aggregate"
    )
    upload_control = args.state_root / "aggregate_upload"
    evidence = load_completed_upload_evidence(upload_control, destination)
    if evidence is None:
        evidence = upload_artifact_tree(
            args.output_directory,
            hf_cli=args.hf_cli,
            bucket=BUCKET,
            destination=destination,
            control_root=upload_control,
        )
    final = {**terminal, "remote": evidence}
    atomic_write_json(args.state_root / "CENTRAL_COMPLETE.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
