#!/usr/bin/env python3
"""Run one detached follow-up host shard and upload only verified artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from oge.studies.execution import load_followup_execution_plan
from oge.studies.supervisor import (
    SupervisorBlockedError,
    build_artifact_manifest,
    production_environment,
    update_supervisor_state,
    upload_artifact_tree,
    verify_clean_git,
    verify_followup_study_artifacts,
    verify_hf_preflight,
    wait_for_idle_gpus,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--host-id", required=True)
    run.add_argument("--execution-plan", type=Path, required=True)
    run.add_argument("--trial-plan", type=Path, required=True)
    run.add_argument("--expected-git-sha", required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--gpus", required=True)
    run.add_argument("--hf-cli", type=Path, required=True)
    run.add_argument("--gpu-wait-seconds", type=float, default=60.0)
    run.add_argument("--gpu-wait-timeout-hours", type=float, default=24.0)
    run.add_argument("--upload-retry-hours", type=float, default=24.0)
    status = subparsers.add_parser("status")
    status.add_argument("--state-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return value


def _study_root(artifact_root: Path) -> Path:
    matches = sorted(artifact_root.glob("*/study_summary.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one study under {artifact_root}, found {len(matches)}"
        )
    return matches[0].parent


def _run(args: argparse.Namespace) -> int:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "state.json"
    state = {
        "schema_version": "1.0",
        "host_id": args.host_id,
        "expected_git_sha": args.expected_git_sha,
        "artifact_root": str(args.artifact_root),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    state = update_supervisor_state(state_path, state, "STARTING")
    try:
        followup_plan = _load_json(args.trial_plan)
        execution_plan = load_followup_execution_plan(
            args.execution_plan,
            followup_plan,
        )
        if args.host_id not in execution_plan["hosts"]:
            raise SupervisorBlockedError(
                f"host {args.host_id!r} is absent from the execution plan"
            )
        verify_clean_git(REPOSITORY_ROOT, args.expected_git_sha)
        if not args.data_root.is_dir():
            raise SupervisorBlockedError(f"data root is absent: {args.data_root}")
        if args.artifact_root.exists() and any(args.artifact_root.iterdir()):
            raise SupervisorBlockedError(
                f"production artifact root is not empty: {args.artifact_root}"
            )
        verify_hf_preflight(
            args.hf_cli,
            account=execution_plan["hf_account"],
            bucket=execution_plan["hf_bucket"],
        )
        gpu_uuids = [value.strip() for value in args.gpus.split(",") if value.strip()]
        state = wait_for_idle_gpus(
            gpu_uuids,
            state_path=state_path,
            state=state,
            poll_seconds=args.gpu_wait_seconds,
            timeout_hours=args.gpu_wait_timeout_hours,
        )
        concurrency = int(execution_plan["hosts"][args.host_id]["concurrency"])
        state = update_supervisor_state(state_path, state, "RUNNING")
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_optimizer_study.py"),
            "--phase",
            "followup",
            "--trial-plan",
            str(args.trial_plan),
            "--execution-plan",
            str(args.execution_plan),
            "--host-id",
            args.host_id,
            "--data-root",
            str(args.data_root),
            "--artifact-root",
            str(args.artifact_root),
            "--gpus",
            ",".join(gpu_uuids),
            "--concurrency",
            str(concurrency),
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=production_environment(),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"follow-up runner exited with status {completed.returncode}"
            )
        study_root = _study_root(args.artifact_root)
        state = update_supervisor_state(
            state_path,
            state,
            "VALIDATING",
            study_root=str(study_root),
        )
        verification = verify_followup_study_artifacts(
            study_root,
            execution_plan=execution_plan,
            followup_plan=followup_plan,
            host_id=args.host_id,
            expected_git_sha=args.expected_git_sha,
        )
        build_artifact_manifest(study_root)
        destination = execution_plan["hf_destination_template"].format(
            host_id=args.host_id
        )
        state = update_supervisor_state(
            state_path,
            state,
            "UPLOADING",
            destination=destination,
            verification=verification,
        )
        deadline = datetime.now(timezone.utc) + timedelta(
            hours=args.upload_retry_hours
        )
        delay = 60.0
        while True:
            try:
                evidence = upload_artifact_tree(
                    study_root,
                    hf_cli=args.hf_cli,
                    bucket=execution_plan["hf_bucket"],
                    destination=destination,
                    control_root=args.state_dir,
                )
            except subprocess.CalledProcessError as exc:
                if datetime.now(timezone.utc) >= deadline:
                    raise SupervisorBlockedError(
                        "HF upload retry window expired"
                    ) from exc
                state = update_supervisor_state(
                    state_path,
                    state,
                    "UPLOAD_RETRY_WAIT",
                    last_upload_error=str(exc),
                    retry_delay_seconds=delay,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 1800.0)
            else:
                break
        update_supervisor_state(
            state_path,
            state,
            "REMOTE_VERIFIED",
            upload_evidence=evidence,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return 0
    except SupervisorBlockedError as exc:
        update_supervisor_state(
            state_path,
            state,
            "BLOCKED",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        print(str(exc), file=sys.stderr)
        return 2
    except BaseException as exc:
        update_supervisor_state(
            state_path,
            state,
            "FAILED",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        raise


def main() -> int:
    args = parse_args()
    if args.command == "status":
        state_path = args.state_dir / "state.json"
        if not state_path.is_file():
            print(json.dumps({"status": "NOT_STARTED"}, indent=2))
            return 1
        print(json.dumps(_load_json(state_path), indent=2, sort_keys=True))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
