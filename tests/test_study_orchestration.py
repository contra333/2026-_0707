import json
import threading
import time
from pathlib import Path

import pytest

from oge.studies.artifacts import atomic_write_json, sha256_file
from oge.studies.failures import checkpoint_resume_decision, classify_failure, make_retry_attempt
from oge.studies.orchestration import (
    PhysicalGPU,
    child_gpu_environment,
    effective_concurrency,
    ordered_trial_plan,
    prepare_retry_run_directory,
    run_independent_trials,
    validate_gpu_inventory,
)
from oge.studies.protocol import PROTOCOL_VERSION
from oge.studies.schemas import (
    ATTEMPT_SCHEMA_VERSION,
    STUDY_SCHEMA_VERSION,
    TRIAL_SCHEMA_VERSION,
    validate_attempt_record,
    validate_study_record,
    validate_trial_record,
)
from oge.training import atomic_torch_save


def _trial(index):
    record = {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "trial_id": f"discovery-sgd-{index:02d}",
        "study_id": "study",
        "phase": "discovery",
        "optimizer_family": "sgd",
        "assigned_slot": index,
        "scientific_config": {"optimizer": {"name": "sgd", "lr": 0.1}},
        "config_hash": f"hash-{index}",
        "training_seed": 0,
        "git_sha": "git-sha",
        "dataset_membership_hashes": {"train": "train", "validation": "val", "test": "test"},
        "provenance_hash": "provenance",
        "status": "assigned",
        "failure": None,
        "attempt_ids": [],
        "best_validation": None,
        "ranking": None,
        "checkpoints": {},
        "artifact_references": {},
    }
    validate_trial_record(record)
    return record


def _gpus(count=3):
    return [PhysicalGPU(index, f"GPU-{index}", "Fake A5000") for index in range(count)]


def test_versioned_schema_validators_reject_wrong_versions_and_require_identity():
    trial = _trial(0)
    wrong = dict(trial, schema_version="2.0")
    with pytest.raises(ValueError, match="schema_version"):
        validate_trial_record(wrong)

    study = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "study_id": "study",
        "search_space_version": "v1.1",
        "search_space_hash": "hash",
        "sampler": {"name": "numpy_generator_pcg64", "seed": 0},
        "ordered_table_hashes": {},
        "optimizer_families": ["sgd", "sgdw", "adam", "adamw"],
        "assigned_budget": {"per_optimizer": 16, "total": 64},
        "seed_roles": {},
        "objective": {},
        "checkpoint_policy": {},
        "selection_declarations": {"id_test": "forbidden", "ood": "forbidden", "geometry_nc": "forbidden", "detector_metrics": "forbidden"},
        "git": {"sha": "sha", "dirty": False},
        "dataset_membership_hashes": {},
        "artifact_root": "/external",
        "artifact_schema_version": "1.0",
        "created_at": None,
        "finished_at": None,
        "status": "planned",
        "gpu_hours": {},
        "terminal_slot_accounting": {"assigned": 64, "terminal": 0},
    }
    validate_study_record(study)
    with pytest.raises(ValueError, match="schema_version"):
        validate_study_record(dict(study, schema_version="0"))


def test_gpu_mapping_cap_and_duplicate_physical_assignment_rejection():
    gpus = _gpus(5)
    assert effective_concurrency(gpus, requested=9) == 4
    plan = ordered_trial_plan([_trial(index) for index in range(6)], gpus, concurrency=3)
    assert {row["physical_gpu_uuid"] for row in plan} == {"GPU-0", "GPU-1", "GPU-2"}
    assert all(row["child_local_device"] == "cuda:0" for row in plan)
    environment = child_gpu_environment(gpus[2], base={"PATH": "fixture"})
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["OGE_CHILD_DEVICE"] == "cuda:0"
    with pytest.raises(ValueError, match="duplicate physical GPU UUID"):
        validate_gpu_inventory([PhysicalGPU(0, "same", "A"), PhysicalGPU(1, "same", "A")])


def test_dry_run_starts_no_subprocess_and_writes_ordered_plan(tmp_path):
    calls = []

    def forbidden_spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry-run started a subprocess")

    result = run_independent_trials(
        study_id="study",
        trials=[_trial(0), _trial(1)],
        gpus=_gpus(2),
        artifact_root=tmp_path,
        repository_root=tmp_path,
        command_builder=lambda trial, attempt_dir, decision: ["train", trial["trial_id"]],
        dry_run=True,
        spawn=forbidden_spawn,
    )
    assert calls == []
    assert result["dry_run"] is True
    assert len(result["ordered_trials"]) == 2
    assert json.loads((tmp_path / "study/ordered_plan.json").read_text()) == result


def test_fake_subprocess_uses_one_visible_gpu_and_never_overlaps_same_gpu(tmp_path):
    lock = threading.Lock()
    active_by_gpu = {}
    maximum_by_gpu = {}
    observed = []

    def fake_spawn(command, *, env, cwd):
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with lock:
            active_by_gpu[gpu] = active_by_gpu.get(gpu, 0) + 1
            maximum_by_gpu[gpu] = max(maximum_by_gpu.get(gpu, 0), active_by_gpu[gpu])
            observed.append((tuple(command), gpu, env["OGE_CHILD_DEVICE"], Path(cwd)))
        time.sleep(0.01)
        with lock:
            active_by_gpu[gpu] -= 1
        return 0

    result = run_independent_trials(
        study_id="study",
        trials=[_trial(index) for index in range(6)],
        gpus=_gpus(2),
        artifact_root=tmp_path,
        repository_root=tmp_path,
        command_builder=lambda trial, attempt_dir, decision: ["train", "--device", "cuda:0", trial["trial_id"]],
        concurrency=4,
        spawn=fake_spawn,
    )
    assert maximum_by_gpu == {"0": 1, "1": 1}
    assert len(observed) == 6
    assert all(item[2] == "cuda:0" for item in observed)
    assert all(attempt["status"] == "completed" for attempt in result["attempts"])
    assert result["terminal_slot_accounting"] == {
        "planned": 6,
        "completed": 6,
        "failed_pending_classification": 0,
    }
    assert len(list(tmp_path.glob("study/trials/*/attempts/*/attempt.json"))) == 6
    assert len(list(tmp_path.glob("study/trials/*/attempts/*/attempt.json.sha256"))) == 6
    assert (tmp_path / "study/study_summary.json.sha256").is_file()


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("oom_assigned_config", "scientific"),
        ("non_finite_loss_or_model", "scientific"),
        ("invalid_scientific_config", "scientific"),
        ("unrelated_gpu_contention", "infrastructure"),
        ("external_interruption", "infrastructure"),
        ("external_data_or_mount", "infrastructure"),
        ("dataset_manifest_mismatch", "infrastructure"),
        ("checkpoint_corruption", "infrastructure"),
        ("gpu_reset", "infrastructure"),
        ("driver_runtime_infrastructure", "infrastructure"),
    ],
)
def test_failure_taxonomy(reason, expected):
    assert classify_failure(reason) == expected


@pytest.mark.parametrize(
    "reason",
    ["low_validation_accuracy", "unfavorable_result", "ranking_changed", "selective_replacement"],
)
def test_result_driven_retry_is_rejected(reason):
    with pytest.raises(ValueError, match="forbidden"):
        classify_failure(reason)


def _failed_attempt(trial):
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "attempt_id": f"{trial['trial_id']}-attempt-001",
        "attempt_number": 1,
        "trial_id": trial["trial_id"],
        "origin": "fresh",
        "prior_attempt_ids": [],
        "identity": {
            "config_hash": trial["config_hash"],
            "study_id": trial["study_id"],
            "assigned_slot": trial["assigned_slot"],
            "training_seed": trial["training_seed"],
            "phase": trial["phase"],
            "git_sha": trial["git_sha"],
            "dataset_membership_hashes": trial["dataset_membership_hashes"],
            "provenance_hash": trial["provenance_hash"],
        },
        "gpu": {"physical_uuid": "GPU-0", "model": "Fake", "parent_visible_index": 0, "cuda_visible_devices": "0", "child_local_index": 0, "child_local_device": "cuda:0", "concurrent_study_trial_cap": 1},
        "command": ["train"],
        "environment": {},
        "started_at": "start",
        "finished_at": "finish",
        "elapsed_seconds": 1.0,
        "exit_status": 1,
        "checkpoint_decision": {"action": "fresh"},
        "logs": {"console": "console.log"},
        "output_paths": {"run": "run"},
        "status": "failed",
        "failure": {"class": "infrastructure", "reason_code": "external_interruption"},
    }
    validate_attempt_record(attempt)
    return attempt


def test_infrastructure_retry_preserves_trial_identity_and_creates_new_attempt(tmp_path):
    trial = _trial(0)
    first = _failed_attempt(trial)
    retry = make_retry_attempt(
        trial,
        [first],
        reason_code="external_interruption",
        checkpoint_path=tmp_path / "missing-last.pt",
    )
    assert retry["trial_id"] == trial["trial_id"]
    assert retry["attempt_id"].endswith("attempt-002")
    assert retry["prior_attempt_ids"] == [first["attempt_id"]]
    assert retry["identity"]["config_hash"] == trial["config_hash"]
    assert retry["checkpoint_decision"]["action"] == "restart_epoch_0"
    with pytest.raises(ValueError, match="only infrastructure"):
        make_retry_attempt(trial, [first], reason_code="oom_assigned_config", checkpoint_path=None)


def test_checkpoint_integrity_prefers_atomic_last_boundary_and_detects_corruption(tmp_path):
    checkpoint = tmp_path / "last.pt"
    atomic_torch_save({"checkpoint_type": "last", "completed_epoch": 7}, checkpoint)
    decision = checkpoint_resume_decision(checkpoint)
    assert decision["action"] == "resume_epoch_boundary"
    assert decision["completed_epoch"] == 7
    assert decision["sha256"] == sha256_file(checkpoint)
    checkpoint.write_bytes(b"corrupt")
    corrupt = checkpoint_resume_decision(checkpoint)
    assert corrupt["action"] == "restart_epoch_0"
    assert corrupt["integrity"] == "invalid"


def test_retry_resume_copies_prior_run_into_new_attempt_boundary(tmp_path):
    prior_run = tmp_path / "prior/attempt/run"
    checkpoint = prior_run / "checkpoints/last.pt"
    atomic_torch_save({"checkpoint_type": "last", "completed_epoch": 7}, checkpoint)
    (prior_run / "history.jsonl").write_text('{"epoch":7}\n', encoding="utf-8")
    decision = checkpoint_resume_decision(checkpoint)
    new_attempt = tmp_path / "new/attempt"
    new_attempt.mkdir(parents=True)
    copied = prepare_retry_run_directory(decision, new_attempt)
    assert copied == new_attempt / "run/checkpoints/last.pt"
    assert copied.read_bytes() == checkpoint.read_bytes()
    assert (new_attempt / "run/history.jsonl").read_text() == '{"epoch":7}\n'
    assert checkpoint.is_file()


def test_atomic_json_checksum_and_attempt_preservation(tmp_path):
    path = tmp_path / "record.json"
    atomic_write_json(path, {"value": 1})
    first_hash = sha256_file(path)
    assert path.with_name("record.json.sha256").is_file()
    atomic_write_json(path, {"value": 1})
    assert sha256_file(path) == first_hash

    kwargs = dict(
        study_id="study",
        trials=[_trial(0)],
        gpus=_gpus(1),
        artifact_root=tmp_path,
        repository_root=tmp_path,
        command_builder=lambda trial, attempt_dir, decision: ["train"],
        spawn=lambda *args, **kwargs: 0,
    )
    run_independent_trials(**kwargs)
    with pytest.raises(ValueError, match="existing trial record differs"):
        run_independent_trials(**kwargs)


def test_retry_execution_preserves_first_attempt_and_appends_second_attempt(tmp_path):
    trial = _trial(0)
    first = _failed_attempt(trial)
    trial["attempt_ids"] = [first["attempt_id"]]
    trial["status"] = "failed_pending_classification"
    trial["failure"] = first["failure"]
    trial_root = tmp_path / "study/trials" / trial["trial_id"]
    first_dir = trial_root / "attempts" / first["attempt_id"]
    first_dir.mkdir(parents=True)
    atomic_write_json(trial_root / "trial.json", trial)
    atomic_write_json(first_dir / "attempt.json", first)
    retry = make_retry_attempt(
        trial,
        [first],
        reason_code="external_interruption",
        checkpoint_path=None,
    )
    result = run_independent_trials(
        study_id="study",
        trials=[trial],
        gpus=_gpus(1),
        artifact_root=tmp_path,
        repository_root=tmp_path,
        command_builder=lambda trial, attempt_dir, decision: ["train"],
        spawn=lambda *args, **kwargs: 0,
        attempt_templates={trial["trial_id"]: retry},
    )
    assert result["attempts"][0]["attempt_id"].endswith("attempt-002")
    assert (first_dir / "attempt.json").is_file()
    persisted = json.loads((trial_root / "trial.json").read_text())
    assert persisted["attempt_ids"] == [first["attempt_id"], retry["attempt_id"]]
    assert persisted["status"] == "completed"
