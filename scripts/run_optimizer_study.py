#!/usr/bin/env python3
"""Validate and execute frozen optimizer-study trial plans."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from oge.studies.artifacts import atomic_write_json
from oge.studies.failures import classify_failure, make_retry_attempt
from oge.studies.hashing import provenance_identity_hash, scientific_config_hash
from oge.studies.hashing import canonical_sha256
from oge.studies.orchestration import (
    DEFAULT_MAX_CONCURRENCY,
    PhysicalGPU,
    discover_nvidia_gpus,
    prepare_retry_run_directory,
    run_independent_trials,
)
from oge.studies.preflight import (
    apply_numerical_policy,
    load_infrastructure_config,
    validate_host_shards,
    validate_runtime_environment,
)
from oge.studies.protocol import (
    PROTOCOL_VERSION,
    generate_grid_bundle,
    load_materialized_grid_bundle,
    load_study_config,
    verify_materialized_grid_bundle,
)
from oge.studies.schemas import (
    STUDY_SCHEMA_VERSION,
    TRIAL_SCHEMA_VERSION,
    validate_attempt_record,
    validate_study_record,
    validate_trial_record,
)
from oge.training import load_training_config, repository_git_state, resolve_training_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
)
DEFAULT_INFRASTRUCTURE = (
    REPOSITORY_ROOT
    / "configs/studies/wrn28_10_optimizer_hpo_v1_2/infrastructure.yaml"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument(
        "--phase",
        choices=["grid"],
        default="grid",
    )
    parser.add_argument(
        "--host-id",
        required=True,
        choices=["curie", "lise", "precision_medicine"],
    )
    parser.add_argument("--infrastructure", type=Path, default=DEFAULT_INFRASTRUCTURE)
    parser.add_argument("--preflight-freeze", type=Path, required=True)
    parser.add_argument("--host-manifest", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--throughput-decision", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated physical GPU UUIDs or parent-visible indices.",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-trials", type=int, default=1)
    parser.add_argument("--retry-trial-id")
    parser.add_argument("--retry-reason-code")
    return parser.parse_args()


def _select_gpus(inventory: list[PhysicalGPU], requested: str) -> list[PhysicalGPU]:
    tokens = [token.strip() for token in requested.split(",") if token.strip()]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("--gpus must contain unique explicit GPU identifiers")
    selected = []
    for token in tokens:
        matches = [
            gpu
            for gpu in inventory
            if token == gpu.uuid or token == str(gpu.parent_visible_index)
        ]
        if len(matches) != 1:
            raise ValueError(f"GPU selector {token!r} did not resolve uniquely")
        selected.append(matches[0])
    return selected


def _trial_record(
    row: dict[str, Any],
    *,
    study_id: str,
    phase: str,
    git_sha: str,
    dataset_membership_hashes: dict[str, str],
) -> dict[str, Any]:
    record = {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "trial_id": row["trial_id"],
        "study_id": study_id,
        "phase": phase,
        "optimizer_family": row["optimizer_family"],
        "assigned_slot": row["assigned_slot"],
        "lr_rank": row.get("lr_rank"),
        "weight_decay_rank": row.get("weight_decay_rank"),
        "scientific_config": copy.deepcopy(row["scientific_config"]),
        "config_hash": row["config_hash"],
        "training_seed": row["training_seed"],
        "git_sha": git_sha,
        "dataset_membership_hashes": dataset_membership_hashes,
        "provenance_hash": provenance_identity_hash(
            git_sha=git_sha,
            dataset_membership_hashes=dataset_membership_hashes,
        ),
        "status": "assigned",
        "failure": None,
        "attempt_ids": [],
        "completed_epoch": None,
        "best_validation": None,
        "final_validation": None,
        "ranking": None,
        "checkpoints": {},
        "artifact_references": {},
    }
    validate_trial_record(record)
    return record


def _study_record(
    *,
    study_id: str,
    study_config: dict[str, Any],
    manifest: dict[str, Any],
    git_sha: str,
    git_dirty: bool,
    dataset_membership_hashes: dict[str, str],
    artifact_root: Path,
    smoke_only: bool,
    assigned_execution_slots: int,
    host_id: str,
    preflight_freeze_hash: str,
    host_manifest_hash: str,
    throughput_decision_hash: str,
) -> dict[str, Any]:
    record = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "study_id": study_id,
        "host_id": host_id,
        "preflight_freeze_hash": preflight_freeze_hash,
        "host_manifest_hash": host_manifest_hash,
        "throughput_decision_hash": throughput_decision_hash,
        "grid_version": manifest["grid_version"],
        "grid_hash": manifest["grid_hash"],
        "grid_manifest_hash": manifest["manifest_hash"],
        "ordered_table_hashes": manifest["table_hashes"],
        "optimizer_families": study_config["optimizer_order"],
        "assigned_budget": {"per_optimizer": 12, "total": 36},
        "seed_roles": manifest["seed_roles"],
        "objective": study_config["objective"],
        "checkpoint_policy": study_config["checkpoint_policy"],
        "selection_declarations": study_config["selection_declarations"],
        "git": {"sha": git_sha, "dirty": git_dirty},
        "dataset_membership_hashes": dataset_membership_hashes,
        "artifact_root": str(artifact_root),
        "artifact_schema_version": "1.0",
        "created_at": None,
        "finished_at": None,
        "status": "smoke_only_planned" if smoke_only else "planned",
        "gpu_hours": {"total": 0.0, "per_optimizer": {}},
        "terminal_slot_accounting": {
            "assigned": 0 if smoke_only else assigned_execution_slots,
            "terminal": 0,
            "smoke_only": smoke_only,
        },
    }
    validate_study_record(record)
    return record


def _command_builder(data_root: Path):
    def build(
        trial: dict[str, Any],
        attempt_dir: Path,
        checkpoint_decision: dict[str, Any],
    ) -> list[str]:
        config_path = attempt_dir / "training_config.yaml"
        config_path.write_text(
            yaml.safe_dump(trial["scientific_config"], sort_keys=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/train_cifar10.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--run-dir",
            str(attempt_dir / "run"),
            "--device",
            "cuda:0",
            "--defer-id-test",
        ]
        if checkpoint_decision.get("action") == "resume_epoch_boundary":
            local_checkpoint = prepare_retry_run_directory(
                checkpoint_decision,
                attempt_dir,
            )
            command.extend(["--resume", str(local_checkpoint)])
        return command

    return build


def main() -> int:
    args = parse_args()
    study_config = load_study_config(args.study_config)
    infrastructure = load_infrastructure_config(args.infrastructure)
    preflight_freeze = json.loads(
        args.preflight_freeze.read_text(encoding="utf-8")
    )
    expected_preflight_hash = canonical_sha256(
        {
            key: value
            for key, value in preflight_freeze.items()
            if key != "freeze_hash"
        }
    )
    if (
        preflight_freeze.get("status") != "ready"
        or preflight_freeze.get("freeze_hash") != expected_preflight_hash
    ):
        raise ValueError("study execution requires a valid ready preflight freeze")
    if args.host_id not in preflight_freeze.get("host_order", []):
        raise ValueError("host-id is not present in the preflight freeze")
    apply_numerical_policy(
        preflight_freeze["common_environment_identity"]["numerical_policy"]
    )
    validate_runtime_environment(
        preflight_freeze,
        repository_root=REPOSITORY_ROOT,
        dependency_lock=args.dependency_lock,
    )
    base_config_path = REPOSITORY_ROOT / study_config["base_training_config"]
    base_config = load_training_config(base_config_path)
    resolved = resolve_training_config(
        base_config,
        data_root=args.data_root,
        device="cuda:0",
    )
    generated = generate_grid_bundle(study_config, resolved)
    frozen_dir = REPOSITORY_ROOT / study_config["frozen_grid_dir"]
    verify_materialized_grid_bundle(frozen_dir, generated)
    frozen = load_materialized_grid_bundle(frozen_dir)
    throughput_decision = json.loads(
        args.throughput_decision.read_text(encoding="utf-8")
    )
    expected_throughput_hash = canonical_sha256(
        {
            key: value
            for key, value in throughput_decision.items()
            if key != "decision_hash"
        }
    )
    if (
        throughput_decision.get("decision_hash") != expected_throughput_hash
        or throughput_decision.get("status") != "workers_zero_frozen"
        or throughput_decision.get("selected_num_workers") != 0
    ):
        raise ValueError(
            "current runner requires the frozen num_workers=0 throughput decision"
        )
    host_manifest = json.loads(args.host_manifest.read_text(encoding="utf-8"))
    validate_host_shards(
        host_manifest,
        frozen,
        preflight_freeze,
        throughput_decision,
    )

    git_sha, git_dirty = repository_git_state()
    if git_dirty:
        raise ValueError("study execution requires a clean Git checkout")
    dataset_hashes = {
        role: str(value["sha256"])
        for role, value in resolved["dataset"]["membership"].items()
    }
    dataset_hashes["membership_manifest"] = str(
        resolved["dataset"]["membership_manifest"]["sha256"]
    )
    study_id = str(study_config["study_id"])
    assigned_trial_ids = {
        row["trial_id"]
        for row in host_manifest["shards"][args.host_id]["rows"]
    }
    rows = [
        row
        for name in study_config["optimizer_order"]
        for row in frozen["tables"][name]["rows"]
        if row["trial_id"] in assigned_trial_ids
    ]
    trials = [
        _trial_record(
            row,
            study_id=study_id,
            phase="grid",
            git_sha=git_sha,
            dataset_membership_hashes=dataset_hashes,
        )
        for row in rows
    ]

    assigned_execution_slots = len(trials)
    if args.smoke_only:
        if args.smoke_trials <= 0:
            raise ValueError("--smoke-trials must be positive")
        study_id = f"{study_id}__smoke_only__{git_sha[:12]}"
        trials = copy.deepcopy(trials[: args.smoke_trials])
        for index, trial in enumerate(trials):
            trial["trial_id"] = f"smoke-{index:02d}-{trial['optimizer_family']}"
            trial["study_id"] = study_id
            trial["assigned_slot"] = index
            trial["scientific_config"]["training"]["max_epochs"] = 1
            trial["scientific_config"]["checkpoint"]["snapshot_epochs"] = []
            trial["config_hash"] = scientific_config_hash(trial["scientific_config"])
            validate_trial_record(trial)

    attempt_templates = None
    if args.retry_trial_id or args.retry_reason_code:
        if not args.retry_trial_id or not args.retry_reason_code:
            raise ValueError("retry requires both --retry-trial-id and --retry-reason-code")
        matches = [trial for trial in trials if trial["trial_id"] == args.retry_trial_id]
        if len(matches) != 1:
            raise ValueError("retry trial must resolve exactly once in the frozen plan")
        frozen_trial = matches[0]
        trial_root = args.artifact_root / study_id / "trials" / frozen_trial["trial_id"]
        saved_trial_path = trial_root / "trial.json"
        if not saved_trial_path.is_file():
            raise ValueError("retry requires the preserved external trial record")
        trial = json.loads(saved_trial_path.read_text(encoding="utf-8"))
        validate_trial_record(trial)
        for key in (
            "trial_id",
            "study_id",
            "phase",
            "config_hash",
            "training_seed",
            "git_sha",
            "dataset_membership_hashes",
            "provenance_hash",
        ):
            if trial[key] != frozen_trial[key]:
                raise ValueError(f"preserved retry identity differs at {key!r}")
        attempts_root = trial_root / "attempts"
        prior_attempts = []
        if attempts_root.is_dir():
            for path in sorted(attempts_root.glob("*/attempt.json")):
                attempt = json.loads(path.read_text(encoding="utf-8"))
                validate_attempt_record(attempt)
                prior_attempts.append(attempt)
        if not prior_attempts:
            raise ValueError("retry requires a preserved prior attempt")
        failure_class = classify_failure(args.retry_reason_code)
        atomic_write_json(
            attempts_root / prior_attempts[-1]["attempt_id"] / "failure_classification.json",
            {
                "attempt_id": prior_attempts[-1]["attempt_id"],
                "failure_class": failure_class,
                "reason_code": args.retry_reason_code,
                "classification_precedes_retry": True,
            },
        )
        checkpoint = attempts_root / prior_attempts[-1]["attempt_id"] / "run/checkpoints/last.pt"
        template = make_retry_attempt(
            trial,
            prior_attempts,
            reason_code=args.retry_reason_code,
            checkpoint_path=checkpoint,
        )
        attempt_templates = {trial["trial_id"]: template}
        trials = [trial]

    study_record = _study_record(
        study_id=study_id,
        study_config=study_config,
        manifest=frozen["manifest"],
        git_sha=git_sha,
        git_dirty=git_dirty,
        dataset_membership_hashes=dataset_hashes,
        artifact_root=args.artifact_root,
        smoke_only=args.smoke_only,
        assigned_execution_slots=assigned_execution_slots,
        host_id=args.host_id,
        preflight_freeze_hash=preflight_freeze["freeze_hash"],
        host_manifest_hash=host_manifest["manifest_hash"],
        throughput_decision_hash=throughput_decision["decision_hash"],
    )
    study_manifest_path = args.artifact_root / study_id / "study_manifest.json"
    study_summary_path = args.artifact_root / study_id / "study_summary.json"
    study_record["created_at"] = _utc_now()
    study_record["status"] = "smoke_only_running" if args.smoke_only else "running"
    validate_study_record(study_record)
    atomic_write_json(study_manifest_path, study_record)

    inventory = discover_nvidia_gpus()
    gpus = _select_gpus(inventory, args.gpus)
    approved_uuids = set(
        host_manifest["shards"][args.host_id]["approved_gpu_uuids"]
    )
    if {gpu.uuid for gpu in gpus} != approved_uuids:
        raise ValueError(
            "selected GPU UUIDs differ from the immutable approved host shard"
        )
    try:
        result = run_independent_trials(
            study_id=study_id,
            trials=trials,
            gpus=gpus,
            artifact_root=args.artifact_root,
            repository_root=REPOSITORY_ROOT,
            command_builder=_command_builder(args.data_root),
            concurrency=args.concurrency,
            dry_run=args.dry_run,
            attempt_templates=attempt_templates,
        )
    except BaseException as exc:
        study_record["finished_at"] = _utc_now()
        study_record["status"] = "smoke_only_failed" if args.smoke_only else "failed"
        study_record["failure"] = {
            "class": "implementation_or_infrastructure",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        validate_study_record(study_record)
        atomic_write_json(study_manifest_path, study_record)
        raise

    study_record["finished_at"] = _utc_now()
    if args.dry_run:
        study_record["status"] = (
            "smoke_only_dry_run_completed" if args.smoke_only else "dry_run_completed"
        )
        study_record["terminal_slot_accounting"] = {
            "assigned": 0 if args.smoke_only else assigned_execution_slots,
            "planned": len(trials),
            "terminal": 0,
            "completed": 0,
            "failed_pending_classification": 0,
            "smoke_only": args.smoke_only,
        }
        exit_code = 0
    else:
        accounting = result["terminal_slot_accounting"]
        all_completed = accounting["completed"] == accounting["planned"]
        if all_completed:
            study_record["status"] = "smoke_only_completed" if args.smoke_only else "completed"
        else:
            study_record["status"] = "smoke_only_failed" if args.smoke_only else "failed"
        study_record["gpu_hours"] = result["gpu_hours"]
        study_record["terminal_slot_accounting"] = {
            "assigned": 0 if args.smoke_only else assigned_execution_slots,
            "planned": accounting["planned"],
            "terminal": accounting["completed"] + accounting["failed_pending_classification"],
            "completed": accounting["completed"],
            "failed_pending_classification": accounting["failed_pending_classification"],
            "smoke_only": args.smoke_only,
        }
        exit_code = 0 if all_completed else 1
    validate_study_record(study_record)
    atomic_write_json(study_manifest_path, study_record)
    result.update(
        created_at=study_record["created_at"],
        finished_at=study_record["finished_at"],
        status=study_record["status"],
        terminal_slot_accounting=study_record["terminal_slot_accounting"],
        gpu_hours=study_record["gpu_hours"],
    )
    atomic_write_json(study_summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
