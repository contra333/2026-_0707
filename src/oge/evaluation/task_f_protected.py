"""Frozen Task F protected one-shot planning, scoring, and aggregation.

Planning and fixture validation do not open protected data.  Runtime export
requires a separate checksummed authorization manifest bound to the merged
execution SHA and exact 2,520-record plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t

from oge.analysis.discriminant_residual_preflight import (
    GeometryFit,
    score_discriminant_components,
)
from oge.analysis.fixed_readout_component_attribution import (
    pair_outcome_summary,
    pair_transition_summary,
    paired_component_attribution,
)
from oge.analysis.task_f_fresh_id import (
    adjudicate_id_equivalence,
    classify_alpha_interior,
    paired_t_interval,
    verify_geometry_artifact,
)
from oge.evaluation.classification import (
    expected_calibration_error,
    negative_log_likelihood,
    top1_accuracy,
)
from oge.evaluation.extraction import build_extraction_loader, load_dataset_config
from oge.evaluation.metrics import compute_ood_metrics, macro_average_ood_metrics
from oge.evaluation.task_f_fresh import EXPECTED_SPECIFICATION_SHA256
from oge.evaluation.task_f_fresh_orchestration import (
    SOURCE_TRAINING_SHA,
    validate_task_f_pipeline_manifest,
)
from oge.feature_export import collect_runtime_provenance
from oge.feature_export.task_f import load_task_f_checkpoint
from oge.studies.hashing import canonical_json_bytes, canonical_sha256
from oge.training import generate_research_run_matrix, validate_research_run_matrix


PLAN_SCHEMA_VERSION = "task_f_protected_one_shot_plan_v1"
BUNDLE_PLAN_SCHEMA_VERSION = "task_f_protected_checkpoint_bundle_plan_v1"
AUTHORIZATION_SCHEMA_VERSION = "task_f_protected_one_shot_authorization_v1"
FEATURE_SCHEMA_VERSION = "task_f_protected_feature_record_v1"
SCORE_SCHEMA_VERSION = "task_f_protected_context_scores_v1"
TERMINAL_SCHEMA_VERSION = "task_f_protected_terminal_v1"
SOURCE_ID_EVALUATION_SHA = "2a22a651001e6466d067493e0966656c79219081"
RTMD_GATE3_SPECIFICATION_SHA256 = (
    "30e7f212c6e91b84885a7d06568820caa15c48fdcbe924af28818d07c428d270"
)
PROTECTED_SPLITS = (
    "id_test",
    "cifar100",
    "tin",
    "mnist",
    "svhn",
    "texture",
    "places365",
)
OOD_SPLITS = PROTECTED_SPLITS[1:]
NEAR_SPLITS = ("cifar100", "tin")
FAR_SPLITS = ("mnist", "svhn", "texture", "places365")
TRAJECTORY_EPOCHS = (10, 60, 120, 160, 200)
TRANSFORMS = ("raw", "l2")
DETECTORS = ("md", "marginal", "rmd")
PRIMARY_ANCHOR_CELL = "adam_lr1e-3_wd1e-4_anchor"
HOSTS = ("curie", "lise", "precision_medicine")
EXPECTED_HOST_RECORDS = {"curie": 1008, "lise": 630, "precision_medicine": 882}
EXPECTED_HOST_BUNDLES = {"curie": 120, "lise": 66, "precision_medicine": 114}
EXPECTED_HOST_DATASET_PASSES = {"curie": 840, "lise": 462, "precision_medicine": 798}
_FORBIDDEN_SPLITS = {"id_test_openood", "ood_validation_tin"}
_ARRAY_NAMES = ("features", "class_labels", "is_id", "sample_ids")
_BUNDLE_TAP_ORDER = ("penultimate", "stage1", "stage2", "stage3")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_ordered_sample_id_sha256(sample_ids: Sequence[str] | np.ndarray) -> str:
    """Hash authorized protected IDs without applying the ID-only name ban."""

    digest = hashlib.sha256()
    observed: set[str] = set()
    for index, sample_id in enumerate(sample_ids):
        value = str(sample_id)
        if not value or "\0" in value:
            raise ValueError(f"sample_ids[{index}] must be non-empty and contain no NUL")
        if value in observed:
            raise ValueError("sample_ids must be unique and ordered deterministically")
        observed.add(value)
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    if not observed:
        raise ValueError("sample_ids must not be empty")
    return digest.hexdigest()


def _canonical_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _record_id(
    run_id: str, role: str, epoch: int | None, tap: str, split: str
) -> str:
    rendered_epoch = "from-checkpoint" if epoch is None else f"{epoch:04d}"
    return "__".join((run_id, role, rendered_epoch, tap, split))


def _checkpoint_relative_path(role: str, epoch: int | None) -> str:
    if role == "best_val":
        return "checkpoints/best_val.pt"
    if role == "last":
        return "checkpoints/last.pt"
    if epoch is None:
        raise ValueError("snapshot checkpoint requires a frozen epoch")
    return f"checkpoints/snapshots/epoch_{epoch:04d}.pt"


def _run_locations(pipeline: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validated = validate_task_f_pipeline_manifest(pipeline)
    locations: dict[str, dict[str, Any]] = {}
    for job in validated["jobs"]:
        if job["stage"] != "feature_export":
            continue
        run_id = str(job["record"]["run_id"])
        location = {
            "host_id": str(job["host_id"]),
            "gpu_index": int(job["gpu_index"]),
            "gpu_uuid": str(job["gpu_uuid"]),
        }
        previous = locations.setdefault(run_id, location)
        if previous != location:
            raise ValueError(f"run {run_id!r} has inconsistent source placement")
    if len(locations) != 50:
        raise ValueError("ID evaluation pipeline does not place exactly 50 runs")
    return locations


def _gate3_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "task_f_rtmd_gate3_terminal_v1":
        raise ValueError("Gate 3 terminal schema mismatch")
    required = {
        "status": "PASS",
        "protected_data_access": False,
        "record_count": 10,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "rtmd_gate3_specification_sha256": RTMD_GATE3_SPECIFICATION_SHA256,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": SOURCE_ID_EVALUATION_SHA,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"Gate 3 terminal {key} mismatch")
    verdict = value.get("gate3_verdict")
    if not isinstance(verdict, Mapping):
        raise ValueError("Gate 3 terminal lacks a verdict")
    activated = bool(verdict.get("activated"))
    if bool(value.get("rtmd_included_in_protected_plan")) != activated:
        raise ValueError("Gate 3 RtMD activation fields disagree")
    if activated or verdict.get("status") != "FAILED_INAPPLICABLE":
        raise ValueError("this execution plan is bound to the actual closed RtMD slot")
    return {
        "status": str(verdict["status"]),
        "activated": activated,
        "terminal_sha256": canonical_sha256(value),
    }


def _logical_record(
    run: Mapping[str, Any], location: Mapping[str, Any], *, role: str,
    epoch: int | None, tap: str, split: str, purpose: str,
) -> dict[str, Any]:
    sibling_role = str(run["task_f_b_sibling_role"])
    sibling = run["task_f_b_sibling_members"][sibling_role]
    return {
        "record_id": _record_id(str(run["run_id"]), role, epoch, tap, split),
        "run_id": str(run["run_id"]),
        "family": str(run["family"]),
        "cell_id": str(run["cell_id"]),
        "training_seed": int(run["training_seed"]),
        "branch_policy": str(sibling["branch_policy"]),
        "sibling_group_id": str(run["sibling_group_id"]),
        "sibling_role": sibling_role,
        "checkpoint_role": role,
        "checkpoint_epoch": epoch,
        "checkpoint_relative_path": _checkpoint_relative_path(role, epoch),
        "depth_tap": tap,
        "dataset_split": split,
        "dataset_group": (
            "id" if split == "id_test" else "near" if split in NEAR_SPLITS else "far"
        ),
        "purpose": purpose,
        "classifier_evaluation": "REQUIRED" if tap == "penultimate" else "NOT_APPLICABLE",
        **dict(location),
    }


def build_protected_plan(
    *, run_plan: Mapping[str, Any], id_pipeline: Mapping[str, Any],
    gate3_terminal: Mapping[str, Any], planning_git_sha: str,
) -> dict[str, Any]:
    """Build the exact frozen 2,520-record plan without opening protected data."""

    validate_research_run_matrix(run_plan)
    if len(planning_git_sha) != 40:
        raise ValueError("planning Git SHA must be a full commit")
    runs = list(run_plan["runs"])
    if len(runs) != 50 or Counter(run["family"] for run in runs) != Counter(
        {"adam": 41, "sgdm": 9}
    ):
        raise ValueError("protected plan requires exactly 50 research runs")
    if any(bool(run.get("execution_only")) for run in runs):
        raise ValueError("execution-only pilot is forbidden from the protected plan")
    locations = _run_locations(id_pipeline)
    gate3 = _gate3_state(gate3_terminal)
    records: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: str(item["run_id"])):
        location = locations[str(run["run_id"])]
        for split in PROTECTED_SPLITS:
            for epoch in TRAJECTORY_EPOCHS:
                role = "last" if epoch == 200 else "snapshot"
                records.append(
                    _logical_record(
                        run, location, role=role, epoch=epoch, tap="penultimate",
                        split=split, purpose="trajectory",
                    )
                )
            if run["cell_id"] == PRIMARY_ANCHOR_CELL:
                for tap in ("stage1", "stage2", "stage3"):
                    records.append(
                        _logical_record(
                            run, location, role="last", epoch=200, tap=tap,
                            split=split, purpose="epoch_200_depth",
                        )
                    )
            records.append(
                _logical_record(
                    run, location, role="best_val", epoch=None, tap="penultimate",
                    split=split, purpose="id_selected_control",
                )
            )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "planning_git_sha": planning_git_sha,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "source_id_evaluation_sha": SOURCE_ID_EVALUATION_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "protected_data_access": False,
        "launch_authorization": "PENDING_EXPLICIT_OWNER_APPROVAL",
        "rtmd": gate3,
        "score_panel": {
            "detectors": list(DETECTORS),
            "transforms": list(TRANSFORMS),
            "id_metrics": ["accuracy", "nll", "ece"],
            "ood_metrics": ["auroc", "fpr95_id_tpr"],
            "pair_metrics": ["gain", "loss", "pair_order_churn", "delta_auroc"],
        },
        "counts": {
            "research_runs": 50,
            "checkpoint_depth_contexts_per_split": 360,
            "records_total": 2520,
            "records_by_split": {split: 360 for split in PROTECTED_SPLITS},
            "records_by_host": dict(EXPECTED_HOST_RECORDS),
        },
        "records": records,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return validate_protected_plan(payload)


def validate_protected_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    observed_hash = payload.pop("plan_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise ValueError("protected plan hash mismatch")
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported protected plan schema")
    if payload.get("source_training_sha") != SOURCE_TRAINING_SHA:
        raise ValueError("protected plan training SHA mismatch")
    if payload.get("source_id_evaluation_sha") != SOURCE_ID_EVALUATION_SHA:
        raise ValueError("protected plan ID-evaluation SHA mismatch")
    if payload.get("task_f_specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("Task F specification identity changed")
    if payload.get("protected_data_access") is not False:
        raise ValueError("planning must not claim protected access")
    if payload.get("launch_authorization") != "PENDING_EXPLICIT_OWNER_APPROVAL":
        raise ValueError("committed protected plan must remain pending approval")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2520:
        raise ValueError("protected plan must contain exactly 2,520 records")
    ids = [str(record.get("record_id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("protected plan contains duplicate records")
    if Counter(str(row.get("dataset_split")) for row in records) != Counter(
        {split: 360 for split in PROTECTED_SPLITS}
    ):
        raise ValueError("protected split coverage mismatch")
    if any(row.get("dataset_split") in _FORBIDDEN_SPLITS for row in records):
        raise ValueError("compatibility-only split entered the protected plan")
    host_counts = Counter(str(row.get("host_id")) for row in records)
    if host_counts != Counter(EXPECTED_HOST_RECORDS):
        raise ValueError("protected host placement changed")
    by_run = defaultdict(set)
    by_sibling = defaultdict(set)
    for row in records:
        location = (row["host_id"], int(row["gpu_index"]), row["gpu_uuid"])
        by_run[row["run_id"]].add(location)
        by_sibling[row["sibling_group_id"]].add(row["host_id"])
        if row["depth_tap"] != "penultimate" and not (
            row["checkpoint_role"] == "last" and row["checkpoint_epoch"] == 200
        ):
            raise ValueError("stage taps are restricted to epoch-200 last")
    if len(by_run) != 50 or any(len(value) != 1 for value in by_run.values()):
        raise ValueError("run placement is incomplete or unstable")
    if any(len(value) != 1 for value in by_sibling.values()):
        raise ValueError("a sibling group crosses hosts")
    payload["plan_sha256"] = observed_hash
    return json.loads(canonical_json_bytes(payload))


def build_protected_checkpoint_bundle_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive 300 runtime bundles from the frozen 2,520 logical records."""

    validated = validate_protected_plan(plan)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in validated["records"]:
        key = (
            record["run_id"],
            record["checkpoint_role"],
            record["checkpoint_epoch"],
            record["checkpoint_relative_path"],
            record["host_id"],
            int(record["gpu_index"]),
            record["gpu_uuid"],
        )
        grouped[key].append(record)

    bundles: list[dict[str, Any]] = []
    for key, records in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        (
            run_id,
            checkpoint_role,
            checkpoint_epoch,
            checkpoint_relative_path,
            host_id,
            gpu_index,
            gpu_uuid,
        ) = key
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_split[str(record["dataset_split"])].append(record)
        if set(by_split) != set(PROTECTED_SPLITS):
            raise ValueError("checkpoint bundle does not cover all protected splits")
        dataset_passes = []
        for split in PROTECTED_SPLITS:
            split_records = by_split[split]
            by_tap = {str(record["depth_tap"]): record for record in split_records}
            if len(by_tap) != len(split_records):
                raise ValueError("checkpoint bundle repeats a split/depth record")
            taps = [tap for tap in _BUNDLE_TAP_ORDER if tap in by_tap]
            dataset_passes.append(
                {
                    "dataset_split": split,
                    "depth_taps": taps,
                    "record_ids": [str(by_tap[tap]["record_id"]) for tap in taps],
                }
            )
        identity = {
            "plan_sha256": validated["plan_sha256"],
            "run_id": run_id,
            "checkpoint_role": checkpoint_role,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_relative_path": checkpoint_relative_path,
            "host_id": host_id,
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
        }
        bundles.append(
            {
                **identity,
                "bundle_id": canonical_sha256(identity),
                "dataset_passes": dataset_passes,
                "dataset_pass_count": len(dataset_passes),
                "logical_record_count": len(records),
            }
        )

    bundle_counts = Counter(str(bundle["host_id"]) for bundle in bundles)
    pass_counts = Counter()
    record_counts = Counter()
    for bundle in bundles:
        host = str(bundle["host_id"])
        pass_counts[host] += int(bundle["dataset_pass_count"])
        record_counts[host] += int(bundle["logical_record_count"])
    payload: dict[str, Any] = {
        "schema_version": BUNDLE_PLAN_SCHEMA_VERSION,
        "plan_sha256": validated["plan_sha256"],
        "protected_data_access": False,
        "counts": {
            "checkpoint_bundles": len(bundles),
            "dataset_passes": sum(pass_counts.values()),
            "logical_records": sum(record_counts.values()),
            "bundles_by_host": dict(sorted(bundle_counts.items())),
            "dataset_passes_by_host": dict(sorted(pass_counts.items())),
            "logical_records_by_host": dict(sorted(record_counts.items())),
        },
        "bundles": bundles,
    }
    payload["bundle_plan_sha256"] = canonical_sha256(payload)
    return validate_protected_checkpoint_bundle_plan(payload, plan=validated)


def validate_protected_checkpoint_bundle_plan(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validated_plan = validate_protected_plan(plan)
    payload = dict(value)
    observed_hash = payload.pop("bundle_plan_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise ValueError("protected checkpoint bundle plan hash mismatch")
    if payload.get("schema_version") != BUNDLE_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported protected checkpoint bundle plan schema")
    if payload.get("plan_sha256") != validated_plan["plan_sha256"]:
        raise ValueError("checkpoint bundle plan is bound to a different protected plan")
    if payload.get("protected_data_access") is not False:
        raise ValueError("checkpoint bundle planning must not claim protected access")
    bundles = payload.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 300:
        raise ValueError("protected execution requires exactly 300 checkpoint bundles")
    expected_counts = {
        "checkpoint_bundles": 300,
        "dataset_passes": 2100,
        "logical_records": 2520,
        "bundles_by_host": EXPECTED_HOST_BUNDLES,
        "dataset_passes_by_host": EXPECTED_HOST_DATASET_PASSES,
        "logical_records_by_host": EXPECTED_HOST_RECORDS,
    }
    if payload.get("counts") != expected_counts:
        raise ValueError("protected checkpoint bundle counts changed")
    bundle_ids = [str(bundle.get("bundle_id")) for bundle in bundles]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("protected checkpoint bundle plan contains duplicate bundles")
    records_by_id = {
        str(record["record_id"]): record for record in validated_plan["records"]
    }
    observed_records = []
    for bundle in bundles:
        identity = {
            "plan_sha256": validated_plan["plan_sha256"],
            "run_id": bundle.get("run_id"),
            "checkpoint_role": bundle.get("checkpoint_role"),
            "checkpoint_epoch": bundle.get("checkpoint_epoch"),
            "checkpoint_relative_path": bundle.get("checkpoint_relative_path"),
            "host_id": bundle.get("host_id"),
            "gpu_index": bundle.get("gpu_index"),
            "gpu_uuid": bundle.get("gpu_uuid"),
        }
        if bundle.get("bundle_id") != canonical_sha256(identity):
            raise ValueError("checkpoint bundle identity changed")
        passes = bundle.get("dataset_passes")
        if not isinstance(passes, list) or len(passes) != len(PROTECTED_SPLITS):
            raise ValueError("checkpoint bundle must contain exactly seven dataset passes")
        if [item.get("dataset_split") for item in passes] != list(PROTECTED_SPLITS):
            raise ValueError("checkpoint bundle split order changed")
        if int(bundle.get("dataset_pass_count", -1)) != 7:
            raise ValueError("checkpoint bundle dataset-pass count changed")
        record_ids = []
        for dataset_pass in passes:
            split = str(dataset_pass["dataset_split"])
            split_record_ids = [str(value) for value in dataset_pass.get("record_ids", [])]
            split_records = [records_by_id.get(record_id) for record_id in split_record_ids]
            if any(record is None for record in split_records):
                raise ValueError("checkpoint bundle references an unknown logical record")
            expected_taps = [str(record["depth_tap"]) for record in split_records]
            if dataset_pass.get("depth_taps") != expected_taps:
                raise ValueError("checkpoint bundle depth taps differ from its logical records")
            for record in split_records:
                assert record is not None
                if record["dataset_split"] != split:
                    raise ValueError("checkpoint bundle mixes protected split identities")
                record_identity = (
                    record["run_id"],
                    record["checkpoint_role"],
                    record["checkpoint_epoch"],
                    record["checkpoint_relative_path"],
                    record["host_id"],
                    int(record["gpu_index"]),
                    record["gpu_uuid"],
                )
                bundle_identity = (
                    bundle["run_id"],
                    bundle["checkpoint_role"],
                    bundle["checkpoint_epoch"],
                    bundle["checkpoint_relative_path"],
                    bundle["host_id"],
                    int(bundle["gpu_index"]),
                    bundle["gpu_uuid"],
                )
                if record_identity != bundle_identity:
                    raise ValueError("checkpoint bundle mixes checkpoint or placement identities")
            record_ids.extend(split_record_ids)
        if int(bundle.get("logical_record_count", -1)) != len(record_ids):
            raise ValueError("checkpoint bundle logical-record count changed")
        observed_records.extend(record_ids)
    expected_records = [str(record["record_id"]) for record in validated_plan["records"]]
    if Counter(observed_records) != Counter(expected_records):
        raise ValueError("checkpoint bundles do not conserve the 2,520 logical records")
    payload["bundle_plan_sha256"] = observed_hash
    return json.loads(canonical_json_bytes(payload))


def validate_authorization(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], execution_git_sha: str
) -> dict[str, Any]:
    validated_plan = validate_protected_plan(plan)
    expected = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "APPROVED",
        "one_shot": True,
        "execution_git_sha": execution_git_sha,
        "plan_sha256": validated_plan["plan_sha256"],
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "splits": list(PROTECTED_SPLITS),
        "record_count": 2520,
        "rtmd_included": bool(validated_plan["rtmd"]["activated"]),
    }
    if dict(value) != expected:
        raise PermissionError("protected authorization does not match the exact one-shot plan")
    if len(execution_git_sha) != 40:
        raise PermissionError("protected execution requires a full Git SHA")
    return dict(value)


def _write_checksums(root: Path, names: Sequence[str]) -> None:
    rows = [f"{sha256_file(root / name)}  {name}" for name in sorted(names)]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_checksums(root: Path, expected: set[str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = row.split("  ", 1)
        if relative in observed or relative not in expected:
            raise ValueError("protected artifact checksum catalog is invalid")
        actual = sha256_file(root / relative)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {relative}")
        observed[relative] = actual
    if set(observed) != expected:
        raise ValueError("protected artifact checksum catalog is incomplete")
    return observed


def write_protected_feature_artifact(
    *, record: Mapping[str, Any], plan: Mapping[str, Any], authorization: Mapping[str, Any],
    execution_git_sha: str, checkpoint_sha256: str, checkpoint_epoch: int,
    features: Any, logits: Any | None, class_labels: Any, is_id: Any,
    sample_ids: Any, runtime: Mapping[str, Any], output_root: str | Path,
) -> Path:
    validated_plan = validate_protected_plan(plan)
    validate_authorization(authorization, plan=validated_plan, execution_git_sha=execution_git_sha)
    return _write_protected_feature_artifact_validated(
        record=record,
        validated_plan=validated_plan,
        execution_git_sha=execution_git_sha,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=checkpoint_epoch,
        features=features,
        logits=logits,
        class_labels=class_labels,
        is_id=is_id,
        sample_ids=sample_ids,
        runtime=runtime,
        output_root=output_root,
        reuse_verified=False,
    )


def _write_protected_feature_artifact_validated(
    *, record: Mapping[str, Any], validated_plan: Mapping[str, Any],
    execution_git_sha: str, checkpoint_sha256: str, checkpoint_epoch: int,
    features: Any, logits: Any | None, class_labels: Any, is_id: Any,
    sample_ids: Any, runtime: Mapping[str, Any], output_root: str | Path,
    reuse_verified: bool,
) -> Path:
    expected_record = {row["record_id"]: row for row in validated_plan["records"]}.get(
        record.get("record_id")
    )
    if expected_record != dict(record):
        raise ValueError("feature record differs from the frozen protected plan")
    feature_array = np.asarray(features, dtype=np.float32)
    labels = np.asarray(class_labels, dtype=np.int64)
    flags = np.asarray(is_id, dtype=np.bool_)
    ids = np.asarray(sample_ids).astype(str)
    if feature_array.ndim != 2 or len(feature_array) == 0 or not np.isfinite(feature_array).all():
        raise ValueError("protected features must be a finite non-empty matrix")
    width = {"stage1": 160, "stage2": 320, "stage3": 640, "penultimate": 640}[
        str(record["depth_tap"])
    ]
    if feature_array.shape[1] != width:
        raise ValueError("protected feature width differs from the tap contract")
    if labels.shape != (len(feature_array),) or flags.shape != labels.shape or ids.shape != labels.shape:
        raise ValueError("protected feature sample axes differ")
    split = str(record["dataset_split"])
    if split == "id_test":
        if not flags.all() or np.any((labels < 0) | (labels > 9)):
            raise ValueError("id_test labels or is_id flags are invalid")
    elif flags.any() or np.any(labels != -1):
        raise ValueError("OOD labels or is_id flags are invalid")
    ordered_digest = _protected_ordered_sample_id_sha256(ids)
    arrays = {
        "features": feature_array,
        "class_labels": labels,
        "is_id": flags,
        "sample_ids": ids,
    }
    if record["depth_tap"] == "penultimate":
        logit_array = np.asarray(logits, dtype=np.float32)
        if logit_array.shape != (len(feature_array), 10) or not np.isfinite(logit_array).all():
            raise ValueError("penultimate protected record requires finite [N,10] logits")
        arrays["logits"] = logit_array
    elif logits is not None:
        raise ValueError("stage records must not store classifier logits")
    identity = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "record_id": record["record_id"],
        "plan_sha256": validated_plan["plan_sha256"],
        "execution_git_sha": execution_git_sha,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "dataset_split": split,
        "depth_tap": record["depth_tap"],
        "ordered_sample_id_sha256": ordered_digest,
        "runtime": dict(runtime),
    }
    output_identity = canonical_sha256(identity)
    destination = Path(output_root) / output_identity
    if destination.exists():
        verify_protected_feature_artifact(destination)
        if reuse_verified:
            return destination
        raise FileExistsError(f"protected feature artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        manifest = {
            **identity,
            "output_identity_sha256": output_identity,
            "protected_data_access": True,
            "record": dict(record),
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "array_sha256": _array_sha256(array),
                }
                for name, array in sorted(arrays.items())
            },
        }
        _canonical_write(temporary / "manifest.json", manifest)
        names = [f"{name}.npy" for name in arrays] + ["manifest.json"]
        _write_checksums(temporary, names)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_protected_feature_artifact(destination)
    return destination


def verify_protected_feature_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (root / "manifest.json").read_bytes() != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("protected feature manifest is not canonical JSON")
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported protected feature schema")
    if manifest.get("protected_data_access") is not True:
        raise ValueError("protected feature artifact lacks protected scope")
    if root.name != manifest.get("output_identity_sha256"):
        raise ValueError("protected feature directory identity mismatch")
    expected = {f"{name}.npy" for name in manifest["arrays"]} | {"manifest.json"}
    if {item.name for item in root.iterdir()} != expected | {"checksums.sha256"}:
        raise ValueError("protected feature artifact contains unexpected files")
    verified = _verify_checksums(root, expected)
    for name, metadata in manifest["arrays"].items():
        value = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
            raise ValueError(f"protected {name} shape or dtype mismatch")
        if _array_sha256(value) != metadata["array_sha256"]:
            raise ValueError(f"protected {name} array identity mismatch")
    ids = np.load(root / "sample_ids.npy", allow_pickle=False)
    if _protected_ordered_sample_id_sha256(ids.astype(str)) != manifest["ordered_sample_id_sha256"]:
        raise ValueError("protected sample order digest mismatch")
    return {"manifest": manifest, "verified_files": verified}


def export_protected_record(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any], record_id: str,
    execution_git_sha: str, checkpoint_path: str | Path,
    dataset_config_path: str | Path, data_root: str | Path,
    output_root: str | Path, device: str, batch_size: int = 512,
    num_workers: int = 4,
) -> Path:
    """Run one authorized source-local GPU record after the approval stop."""

    validated = validate_protected_plan(plan)
    validate_authorization(authorization, plan=validated, execution_git_sha=execution_git_sha)
    records = {row["record_id"]: row for row in validated["records"]}
    if record_id not in records:
        raise ValueError("record is outside the frozen protected plan")
    record = records[record_id]
    if not str(device).startswith("cuda:"):
        raise ValueError("production protected export requires an explicit CUDA device")
    model, provenance, checkpoint_sha256 = load_task_f_checkpoint(checkpoint_path, device=device)
    if provenance["run_id"] != record["run_id"] or provenance["checkpoint_role"] != record["checkpoint_role"]:
        raise ValueError("protected checkpoint identity differs from the planned record")
    actual_epoch = int(provenance["checkpoint_epoch"])
    if record["checkpoint_epoch"] is not None and actual_epoch != int(record["checkpoint_epoch"]):
        raise ValueError("protected checkpoint epoch differs from the frozen record")
    config = load_dataset_config(dataset_config_path)
    loader, expected_ids, _ = build_extraction_loader(
        config,
        dataset_key=record["dataset_split"],
        data_root=data_root,
        config_root=Path(dataset_config_path).parent,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        protected_authorization=canonical_sha256(authorization),
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    sample_ids: list[str] = []
    parity_checked = False
    target = torch.device(device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(target, non_blocking=True)
            batch_logits, taps = model(images, return_feature_taps=True)
            if not parity_checked:
                torch.testing.assert_close(batch_logits, model(images), rtol=0.0, atol=0.0)
                parity_checked = True
            selected = taps[str(record["depth_tap"])]
            if not torch.isfinite(selected).all() or not torch.isfinite(batch_logits).all():
                raise ValueError("protected extraction produced non-finite output")
            features.append(selected.detach().cpu().to(torch.float32).numpy())
            if record["depth_tap"] == "penultimate":
                logits.append(batch_logits.detach().cpu().to(torch.float32).numpy())
            labels.append(batch["class_label"].cpu().numpy())
            flags.append(batch["is_id"].cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    if not parity_checked or sample_ids != expected_ids:
        raise ValueError("protected extraction sample order is incomplete or changed")
    return write_protected_feature_artifact(
        record=record,
        plan=validated,
        authorization=authorization,
        execution_git_sha=execution_git_sha,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=actual_epoch,
        features=np.concatenate(features),
        logits=np.concatenate(logits) if logits else None,
        class_labels=np.concatenate(labels),
        is_id=np.concatenate(flags),
        sample_ids=np.asarray(sample_ids, dtype=str),
        runtime=collect_runtime_provenance(device),
        output_root=output_root,
    )


def export_protected_checkpoint_bundle(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any], bundle_id: str,
    execution_git_sha: str, checkpoint_path: str | Path,
    dataset_config_path: str | Path, data_root: str | Path,
    output_root: str | Path, device: str, batch_size: int = 512,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Export one checkpoint's seven splits and all required taps in one load."""

    validated = validate_protected_plan(plan)
    validate_authorization(authorization, plan=validated, execution_git_sha=execution_git_sha)
    bundle_plan = build_protected_checkpoint_bundle_plan(validated)
    bundles = {str(bundle["bundle_id"]): bundle for bundle in bundle_plan["bundles"]}
    if bundle_id not in bundles:
        raise ValueError("bundle is outside the protected checkpoint bundle plan")
    bundle = bundles[bundle_id]
    if not str(device).startswith("cuda:"):
        raise ValueError("production protected export requires an explicit CUDA device")
    model, provenance, checkpoint_sha256 = load_task_f_checkpoint(checkpoint_path, device=device)
    if (
        provenance["run_id"] != bundle["run_id"]
        or provenance["checkpoint_role"] != bundle["checkpoint_role"]
    ):
        raise ValueError("protected checkpoint identity differs from the planned bundle")
    actual_epoch = int(provenance["checkpoint_epoch"])
    if bundle["checkpoint_epoch"] is not None and actual_epoch != int(bundle["checkpoint_epoch"]):
        raise ValueError("protected checkpoint epoch differs from the frozen bundle")

    config = load_dataset_config(dataset_config_path)
    runtime = collect_runtime_provenance(device)
    accelerator = runtime.get("accelerator")
    if not isinstance(accelerator, Mapping) or (
        accelerator.get("device_uuid") != bundle["gpu_uuid"]
    ):
        raise ValueError("protected runtime GPU UUID differs from the planned bundle")
    authorization_sha256 = canonical_sha256(authorization)
    records = {str(record["record_id"]): record for record in validated["records"]}
    target = torch.device(device)
    model.eval()
    parity_checked = False
    artifact_paths: list[str] = []
    with torch.inference_mode():
        for dataset_pass in bundle["dataset_passes"]:
            split = str(dataset_pass["dataset_split"])
            taps = [str(tap) for tap in dataset_pass["depth_taps"]]
            loader, expected_ids, _ = build_extraction_loader(
                config,
                dataset_key=split,
                data_root=data_root,
                config_root=Path(dataset_config_path).parent,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                protected_authorization=authorization_sha256,
            )
            feature_parts: dict[str, list[np.ndarray]] = {tap: [] for tap in taps}
            logits: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            flags: list[np.ndarray] = []
            sample_ids: list[str] = []
            for batch in loader:
                images = batch["image"].to(target, non_blocking=True)
                batch_logits, batch_taps = model(images, return_feature_taps=True)
                if not parity_checked:
                    torch.testing.assert_close(batch_logits, model(images), rtol=0.0, atol=0.0)
                    parity_checked = True
                if not torch.isfinite(batch_logits).all():
                    raise ValueError("protected extraction produced non-finite logits")
                for tap in taps:
                    selected = batch_taps[tap]
                    if not torch.isfinite(selected).all():
                        raise ValueError("protected extraction produced non-finite features")
                    feature_parts[tap].append(
                        selected.detach().cpu().to(torch.float32).numpy()
                    )
                if "penultimate" in feature_parts:
                    logits.append(batch_logits.detach().cpu().to(torch.float32).numpy())
                labels.append(batch["class_label"].cpu().numpy())
                flags.append(batch["is_id"].cpu().numpy())
                sample_ids.extend(batch["sample_id"])
            if sample_ids != expected_ids:
                raise ValueError("protected extraction sample order is incomplete or changed")
            label_array = np.concatenate(labels)
            flag_array = np.concatenate(flags)
            sample_id_array = np.asarray(sample_ids, dtype=str)
            logit_array = np.concatenate(logits) if logits else None
            for tap, record_id in zip(
                dataset_pass["depth_taps"], dataset_pass["record_ids"], strict=True
            ):
                record = records[str(record_id)]
                path = _write_protected_feature_artifact_validated(
                    record=record,
                    validated_plan=validated,
                    execution_git_sha=execution_git_sha,
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_epoch=actual_epoch,
                    features=np.concatenate(feature_parts[str(tap)]),
                    logits=logit_array if tap == "penultimate" else None,
                    class_labels=label_array,
                    is_id=flag_array,
                    sample_ids=sample_id_array,
                    runtime=runtime,
                    output_root=output_root,
                    reuse_verified=True,
                )
                artifact_paths.append(str(path))
    if not parity_checked:
        raise ValueError("protected checkpoint bundle produced no batches")
    if len(artifact_paths) != int(bundle["logical_record_count"]):
        raise ValueError("protected checkpoint bundle artifact count changed")
    return {
        "status": "PASS",
        "protected_data_access": True,
        "bundle_id": bundle_id,
        "bundle_plan_sha256": bundle_plan["bundle_plan_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": actual_epoch,
        "dataset_pass_count": len(bundle["dataset_passes"]),
        "logical_record_count": len(artifact_paths),
        "artifact_paths": artifact_paths,
    }


def _transform(values: Any, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("query features must be a finite matrix")
    if transform == "raw":
        return array
    if transform != "l2":
        raise ValueError("unsupported protected transform")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("L2 query transform encountered a zero or non-finite norm")
    return array / norms[:, None]


def _geometry_fit(root: Path, manifest: Mapping[str, Any], transform: str) -> GeometryFit:
    with np.load(root / "fit_state.npz", allow_pickle=False) as state:
        arrays = {
            name: np.asarray(state[f"{transform}__{name}"])
            for name in (
                "mean", "class_means", "class_counts", "within_covariance",
                "between_covariance", "total_covariance", "within_precision",
                "global_precision", "within_sqrt", "within_invsqrt",
                "subspace_basis", "transformed_class_means", "parallel_global_precision",
            )
        }
    summary = manifest["summary"]["transforms"][transform]
    fit = summary["fit"]
    condition = fit["condition_number"]
    if condition == "infinity":
        condition = math.inf
    return GeometryFit(
        **arrays,
        dim=int(fit["dim_s"]),
        residual_dim=int(fit["dim_s_perp"]),
        condition_number=float(condition),
        tau_alg=float(fit["tau_alg"]),
        ridge=float(fit["ridge"]),
        applicable=bool(fit["applicable"]),
        numerical=dict(fit["numerical"]),
    )


def _score_chunks(fit: GeometryFit, values: np.ndarray, chunk_size: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    checks: list[Mapping[str, bool]] = []
    for start in range(0, len(values), chunk_size):
        record, arrays = score_discriminant_components(fit, values[start : start + chunk_size])
        checks.append(record["checks"])
        for name, array in arrays.items():
            chunks[name].append(np.asarray(array))
    combined = {name: np.concatenate(values) for name, values in chunks.items()}
    if not all(all(item.values()) for item in checks):
        raise ValueError("protected component reconstruction check failed")
    return {"chunk_count": len(checks), "checks_pass": True}, combined


def evaluate_context_arrays(
    *, geometry_path: str | Path, protected_artifacts: Mapping[str, str | Path],
    chunk_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score one checkpoint/depth context against all seven protected splits."""

    geometry_root = Path(geometry_path)
    geometry = verify_geometry_artifact(geometry_root)["manifest"]
    if set(protected_artifacts) != set(PROTECTED_SPLITS):
        raise ValueError("protected context requires exactly seven split artifacts")
    verified = {
        split: verify_protected_feature_artifact(path)["manifest"]
        for split, path in protected_artifacts.items()
    }
    identity_fields = ("run_id", "checkpoint_role", "depth_tap")
    for split, manifest in verified.items():
        record = manifest["record"]
        if manifest["dataset_split"] != split:
            raise ValueError("protected artifact split key mismatch")
        if any(record[field] != geometry[field] for field in identity_fields):
            raise ValueError("protected and ID geometry context identities differ")
        if manifest["checkpoint_sha256"] != geometry["checkpoint_sha256"]:
            raise ValueError("protected and ID geometry checkpoint SHA differ")
        if manifest["task_f_specification_sha256"] != EXPECTED_SPECIFICATION_SHA256:
            raise ValueError("protected feature specification identity changed")
        if record["checkpoint_epoch"] is not None and int(record["checkpoint_epoch"]) != int(geometry["checkpoint_epoch"]):
            raise ValueError("protected and ID geometry epochs differ")
    arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    per_dataset_metrics: dict[str, Any] = {}
    for transform in TRANSFORMS:
        fit = _geometry_fit(geometry_root, geometry, transform)
        diagnostics[transform] = {"fit_applicable": fit.applicable, "splits": {}}
        for split in PROTECTED_SPLITS:
            root = Path(protected_artifacts[split])
            values = _transform(np.load(root / "features.npy", mmap_mode="r"), transform)
            record, scored = _score_chunks(fit, values, chunk_size)
            diagnostics[transform]["splits"][split] = record
            for name, value in scored.items():
                arrays[f"{transform}__{split}__{name}"] = value
        detector_metrics = {}
        for detector in DETECTORS:
            id_scores = arrays[f"{transform}__id_test__{detector}"]
            per_ood = {
                split: compute_ood_metrics(
                    id_scores, arrays[f"{transform}__{split}__{detector}"]
                )
                for split in OOD_SPLITS
            }
            detector_metrics[detector] = {
                "per_dataset": per_ood,
                "macro": macro_average_ood_metrics(per_ood),
            }
        per_dataset_metrics[transform] = detector_metrics
    id_utility: dict[str, Any]
    if geometry["depth_tap"] == "penultimate":
        id_root = Path(protected_artifacts["id_test"])
        logits = np.load(id_root / "logits.npy", mmap_mode="r")
        labels = np.load(id_root / "class_labels.npy", mmap_mode="r")
        id_utility = {
            "status": "PASS",
            "accuracy": float(top1_accuracy(logits, labels)["metric"]["value"]),
            "nll": float(negative_log_likelihood(logits, labels)["metric"]["value"]),
            "ece": float(expected_calibration_error(logits, labels)["metric"]["value"]),
        }
    else:
        id_utility = {
            "status": "NOT_APPLICABLE",
            "reason": "classifier_reads_penultimate_only",
        }
    summary = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "PASS",
        "protected_data_access": True,
        "run_id": geometry["run_id"],
        "family": geometry["family"],
        "cell_id": geometry["cell_id"],
        "training_seed": int(geometry["training_seed"]),
        "branch_policy": geometry["branch_policy"],
        "sibling_group_id": geometry["sibling_group_id"],
        "sibling_role": geometry["sibling_role"],
        "initialization_sha256": geometry["initialization_sha256"],
        "data_stream_sha256": geometry["data_stream_sha256"],
        "checkpoint_role": geometry["checkpoint_role"],
        "checkpoint_epoch": int(geometry["checkpoint_epoch"]),
        "checkpoint_sha256": geometry["checkpoint_sha256"],
        "depth_tap": geometry["depth_tap"],
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "rtmd": "EXCLUDED_BY_FAILED_GATE3",
        "id_utility": id_utility,
        "ood_metrics": per_dataset_metrics,
        "component_diagnostics": diagnostics,
        "sample_order_sha256": {
            split: verified[split]["ordered_sample_id_sha256"] for split in PROTECTED_SPLITS
        },
    }
    return summary, arrays


def write_context_scores(
    *, geometry_path: str | Path, protected_artifacts: Mapping[str, str | Path],
    output_root: str | Path, chunk_size: int = 2048,
) -> Path:
    summary, arrays = evaluate_context_arrays(
        geometry_path=geometry_path, protected_artifacts=protected_artifacts, chunk_size=chunk_size
    )
    identity = {
        key: summary[key]
        for key in (
            "run_id", "checkpoint_role", "checkpoint_epoch", "checkpoint_sha256", "depth_tap",
            "task_f_specification_sha256",
        )
    }
    identity["protected_feature_identities"] = {
        split: verify_protected_feature_artifact(path)["manifest"]["output_identity_sha256"]
        for split, path in protected_artifacts.items()
    }
    output_identity = canonical_sha256(identity)
    destination = Path(output_root) / output_identity
    if destination.exists():
        verify_context_scores(destination)
        raise FileExistsError(f"protected context scores already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        np.savez(temporary / "scores.npz", **arrays)
        manifest = {
            **summary,
            "output_identity_sha256": output_identity,
            "identity": identity,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype), "array_sha256": _array_sha256(value)}
                for name, value in sorted(arrays.items())
            },
        }
        _canonical_write(temporary / "manifest.json", manifest)
        _write_checksums(temporary, ("manifest.json", "scores.npz"))
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_context_scores(destination)
    return destination


def verify_context_scores(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (root / "manifest.json").read_bytes() != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("protected score manifest is not canonical JSON")
    if manifest.get("schema_version") != SCORE_SCHEMA_VERSION or manifest.get("status") != "PASS":
        raise ValueError("protected score artifact is not terminal PASS")
    if root.name != manifest.get("output_identity_sha256"):
        raise ValueError("protected score directory identity mismatch")
    verified = _verify_checksums(root, {"manifest.json", "scores.npz"})
    with np.load(root / "scores.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(manifest["arrays"]):
            raise ValueError("protected score array catalog mismatch")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            metadata = manifest["arrays"][name]
            if list(value.shape) != metadata["shape"] or _array_sha256(value) != metadata["array_sha256"]:
                raise ValueError(f"protected score array identity mismatch: {name}")
    return {"manifest": manifest, "verified_files": verified}


def transition_rates(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    utility = {"incorrect": 0.0, "tie": 0.5, "correct": 1.0}
    count = int(summary["pair_count"])
    gain = 0.0
    loss = 0.0
    for source, targets in summary["transitions"].items():
        for target, number in targets.items():
            delta = utility[target] - utility[source]
            gain += max(delta, 0.0) * int(number)
            loss += max(-delta, 0.0) * int(number)
    return gain / count, loss / count, (gain + loss) / count


def compare_score_arrays(
    *, left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray],
    split: str, transform: str, detector: str,
) -> dict[str, Any]:
    id_key = f"{transform}__id_test__{detector}"
    ood_key = f"{transform}__{split}__{detector}"
    # Transition direction is decoupled/right -> coupled/left so Gain and Loss
    # follow Card 13's a_C - a_D convention.
    transition = pair_transition_summary(right[id_key], right[ood_key], left[id_key], left[ood_key])
    gain, loss, churn = transition_rates(transition)
    left_outcome = pair_outcome_summary(left[id_key], left[ood_key])
    right_outcome = pair_outcome_summary(right[id_key], right[ood_key])
    delta = float(left_outcome["auroc_id_positive"] - right_outcome["auroc_id_positive"])
    result = {
        "gain": gain,
        "loss": loss,
        "pair_order_churn": churn,
        "delta_auroc": delta,
        "balance_residual": delta - (gain - loss),
        "pair_count": transition["pair_count"],
        "transitions": transition["transitions"],
    }
    if detector == "md":
        result["component_attribution"] = paired_component_attribution(
            {
                "id": {name: right[f"{transform}__id_test__{name}"] for name in DETECTORS},
                "ood": {name: right[f"{transform}__{split}__{name}"] for name in DETECTORS},
            },
            {
                "id": {name: left[f"{transform}__id_test__{name}"] for name in DETECTORS},
                "ood": {name: left[f"{transform}__{split}__{name}"] for name in DETECTORS},
            },
        )
    return result


def holm_adjust(p_values: Mapping[str, float], *, alpha: float = 0.10) -> dict[str, Any]:
    if not p_values or any(not 0.0 <= float(value) <= 1.0 for value in p_values.values()):
        raise ValueError("Holm adjustment requires finite p-values in [0,1]")
    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), key))
    output: dict[str, Any] = {}
    still_rejecting = True
    count = len(ordered)
    for index, key in enumerate(ordered):
        threshold = alpha / (count - index)
        rejected = still_rejecting and float(p_values[key]) <= threshold
        still_rejecting = rejected
        output[key] = {
            "p_value": float(p_values[key]),
            "holm_threshold": threshold,
            "reject": rejected,
        }
    return output


def paired_t_p_value(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired t test requires at least two finite deltas")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    if sd == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    statistic = mean / (sd / math.sqrt(len(array)))
    return float(2.0 * student_t.sf(abs(statistic), len(array) - 1))


def simultaneous_sign_flip_band(seed_by_epoch: Any, *, confidence: float = 0.90) -> dict[str, Any]:
    values = np.asarray(seed_by_epoch, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("trajectory band requires a finite [seed,epoch] matrix")
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    if np.any(standard_error == 0.0):
        raise ValueError("trajectory band is undefined with zero seed standard error")
    maxima = []
    for signs in product((-1.0, 1.0), repeat=values.shape[0]):
        flipped = values * np.asarray(signs)[:, None]
        flipped_mean = np.mean(flipped, axis=0)
        flipped_se = np.std(flipped, axis=0, ddof=1) / math.sqrt(values.shape[0])
        flipped_t = np.divide(
            np.abs(flipped_mean),
            flipped_se,
            out=np.where(np.abs(flipped_mean) == 0.0, 0.0, np.inf),
            where=flipped_se > 0.0,
        )
        maxima.append(float(np.max(flipped_t)))
    critical = float(np.quantile(maxima, confidence, method="higher"))
    mean = np.mean(values, axis=0)
    return {
        "method": "all_exact_paired_seed_sign_flips_max_absolute_t",
        "confidence": confidence,
        "sign_flip_count": len(maxima),
        "critical_value": critical,
        "mean": mean.tolist(),
        "lower": (mean - critical * standard_error).tolist(),
        "upper": (mean + critical * standard_error).tolist(),
        "max_t_distribution": sorted(maxima),
    }


def _role(record: Mapping[str, Any]) -> str:
    sibling = str(record["sibling_role"])
    branch = str(record["branch_policy"])
    if sibling == "zero" or branch == "zero_decay":
        return "zero"
    if sibling == "alpha_0_5" or branch == "adam_mixed_alpha_0_5":
        return "alpha_0_5"
    if sibling == "alpha_0" or branch == "adamw_alpha_0":
        return "decoupled"
    if sibling == "alpha_1" or branch == "adam_alpha_1":
        return "coupled"
    if sibling.endswith(("_adam_coupled", "_sgdm_coupled")):
        return "coupled"
    if sibling.endswith(("_adamw_decoupled", "_sgdw_decoupled")):
        return "decoupled"
    raise ValueError("unrecognized Task F sibling role")


def _context_epoch(record: Mapping[str, Any]) -> int | str:
    if record["checkpoint_role"] == "best_val":
        return "selected_by_id_validation"
    return int(record["checkpoint_epoch"])


def _load_score_arrays(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = verify_context_scores(path)["manifest"]
    with np.load(path / "scores.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    return manifest, arrays


def aggregate_protected_scores(
    *, score_paths: Sequence[str | Path], expected_contexts: int = 360,
) -> dict[str, Any]:
    """Create complete seed, paired, ID-equivalence, and confirmatory records."""

    loaded = [_load_score_arrays(Path(path)) for path in score_paths]
    if len(loaded) != expected_contexts:
        return {
            "schema_version": TERMINAL_SCHEMA_VERSION,
            "status": "INCOMPLETE",
            "expected_contexts": expected_contexts,
            "observed_contexts": len(loaded),
            "missing_or_failed_records_are_excluded": False,
        }
    keyed: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for manifest, arrays in loaded:
        key = (
            manifest["cell_id"], int(manifest["training_seed"]),
            manifest["checkpoint_role"], _context_epoch(manifest),
            manifest["depth_tap"], _role(manifest),
        )
        if key in keyed:
            raise ValueError("protected score contexts contain a duplicate role")
        keyed[key] = (manifest, arrays)
    if expected_contexts == 360:
        manifests = [manifest for manifest, _ in loaded]
        if len({manifest["run_id"] for manifest in manifests}) != 50:
            raise ValueError("protected terminal does not cover exactly 50 runs")
        trajectory = [
            manifest for manifest in manifests
            if manifest["depth_tap"] == "penultimate"
            and manifest["checkpoint_role"] in {"snapshot", "last"}
            and int(manifest["checkpoint_epoch"]) in TRAJECTORY_EPOCHS
        ]
        best = [
            manifest for manifest in manifests
            if manifest["depth_tap"] == "penultimate"
            and manifest["checkpoint_role"] == "best_val"
        ]
        depth = [
            manifest for manifest in manifests
            if manifest["depth_tap"] in {"stage1", "stage2", "stage3"}
            and manifest["checkpoint_role"] == "last"
            and int(manifest["checkpoint_epoch"]) == 200
            and manifest["cell_id"] == PRIMARY_ANCHOR_CELL
        ]
        if (len(trajectory), len(best), len(depth)) != (250, 50, 60):
            raise ValueError("protected context coverage differs from 250/50/60")
    directions = (
        ("coupled_minus_decoupled", "coupled", "decoupled"),
        ("coupled_minus_zero", "coupled", "zero"),
        ("decoupled_minus_zero", "decoupled", "zero"),
    )
    zero_index = {
        (
            manifest["sibling_group_id"], int(manifest["training_seed"]),
            manifest["checkpoint_role"], _context_epoch(manifest),
            manifest["depth_tap"],
        ): (manifest, arrays)
        for manifest, arrays in loaded
        if _role(manifest) == "zero"
    }
    seed_records: list[dict[str, Any]] = []
    context_stems = sorted({key[:-1] for key in keyed}, key=str)
    for stem in context_stems:
        cell, seed, checkpoint_role, epoch, depth = stem
        for direction, left_role, right_role in directions:
            left = keyed.get((*stem, left_role))
            right = keyed.get((*stem, right_role))
            if right_role == "zero" and left is not None:
                right = zero_index.get(
                    (
                        left[0]["sibling_group_id"], seed, checkpoint_role,
                        epoch, depth,
                    )
                )
            if left is None or right is None:
                continue
            left_manifest, left_arrays = left
            right_manifest, right_arrays = right
            for field in ("sibling_group_id", "initialization_sha256", "data_stream_sha256"):
                if left_manifest[field] != right_manifest[field]:
                    raise ValueError("paired protected records have different sibling identity")
            if left_manifest["sample_order_sha256"] != right_manifest["sample_order_sha256"]:
                raise ValueError("paired protected records have different sample order")
            for transform in TRANSFORMS:
                for detector in DETECTORS:
                    dataset_rows = {}
                    for split in OOD_SPLITS:
                        dataset_rows[split] = compare_score_arrays(
                            left=left_arrays, right=right_arrays, split=split,
                            transform=transform, detector=detector,
                        )
                    for metric in ("gain", "loss", "pair_order_churn", "delta_auroc"):
                        dataset_rows[f"near_{metric}"] = float(np.mean([dataset_rows[name][metric] for name in NEAR_SPLITS]))
                        dataset_rows[f"far_{metric}"] = float(np.mean([dataset_rows[name][metric] for name in FAR_SPLITS]))
                    seed_records.append(
                        {
                            "cell_id": cell,
                            "training_seed": seed,
                            "checkpoint_role": checkpoint_role,
                            "checkpoint_epoch": epoch,
                            "depth_tap": depth,
                            "direction": direction,
                            "transform": transform,
                            "detector": detector,
                            "left_run_id": left_manifest["run_id"],
                            "right_run_id": right_manifest["run_id"],
                            "datasets": dataset_rows,
                        }
                    )
    aggregates: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (row["cell_id"], row["checkpoint_role"], row["checkpoint_epoch"], row["depth_tap"], row["direction"], row["transform"], row["detector"])
            for row in seed_records
        },
        key=str,
    )
    for key in group_keys:
        rows = [row for row in seed_records if tuple(row[name] for name in (
            "cell_id", "checkpoint_role", "checkpoint_epoch", "depth_tap", "direction", "transform", "detector"
        )) == key]
        metric_names = [f"{group}_{metric}" for group in ("near", "far") for metric in ("gain", "loss", "pair_order_churn", "delta_auroc")]
        summaries = {
            name: paired_t_interval([float(row["datasets"][name]) for row in rows])
            for name in metric_names
        }
        r_churn = {}
        if key[4] == "coupled_minus_decoupled":
            cell, checkpoint_role, epoch, depth, _, transform, detector = key
            seeds = sorted(int(row["training_seed"]) for row in rows)
            for split in OOD_SPLITS:
                same_policy = []
                for role in ("coupled", "decoupled"):
                    role_items = [
                        keyed[(cell, seed, checkpoint_role, epoch, depth, role)]
                        for seed in seeds
                        if (cell, seed, checkpoint_role, epoch, depth, role) in keyed
                    ]
                    for first, second in combinations(role_items, 2):
                        if first[0]["sample_order_sha256"] != second[0]["sample_order_sha256"]:
                            raise ValueError("same-policy protected records have different sample order")
                        first_arrays = first[1]
                        second_arrays = second[1]
                        transition = pair_transition_summary(
                            first_arrays[f"{transform}__id_test__{detector}"],
                            first_arrays[f"{transform}__{split}__{detector}"],
                            second_arrays[f"{transform}__id_test__{detector}"],
                            second_arrays[f"{transform}__{split}__{detector}"],
                        )
                        same_policy.append(transition_rates(transition)[2])
                numerator = float(
                    np.median([row["datasets"][split]["pair_order_churn"] for row in rows])
                )
                denominator = float(np.median(same_policy))
                pair_count = int(rows[0]["datasets"][split]["pair_count"])
                floor = max(1.0e-4, 10.0 / pair_count)
                r_churn[split] = {
                    "cross_policy_median": numerator,
                    "same_policy_median": denominator,
                    "same_policy_pair_count": len(same_policy),
                    "denominator_floor": floor,
                    "value": None if denominator < floor else numerator / denominator,
                    "status": (
                        "UNDEFINED_SMALL_DENOMINATOR" if denominator < floor else "PASS"
                    ),
                }
        aggregates.append({
            "cell_id": key[0], "checkpoint_role": key[1], "checkpoint_epoch": key[2],
            "depth_tap": key[3], "direction": key[4], "transform": key[5],
            "detector": key[6], "seeds": [row["training_seed"] for row in rows],
            "summaries": summaries,
            "r_churn": r_churn,
        })
    id_equivalence = {}
    for cell in sorted({manifest["cell_id"] for manifest, _ in loaded}):
        deltas: dict[str, list[float]] = defaultdict(list)
        seeds = sorted({int(manifest["training_seed"]) for manifest, _ in loaded if manifest["cell_id"] == cell})
        for seed in seeds:
            stem = (cell, seed, "last", 200, "penultimate")
            left = keyed.get((*stem, "coupled"))
            right = keyed.get((*stem, "decoupled"))
            if left is None or right is None:
                continue
            for metric in ("accuracy", "nll", "ece"):
                deltas[metric].append(float(left[0]["id_utility"][metric]) - float(right[0]["id_utility"][metric]))
        if set(deltas) == {"accuracy", "nll", "ece"} and all(len(values) >= 2 for values in deltas.values()):
            id_equivalence[cell] = adjudicate_id_equivalence(
                deltas, evidence_scope="protected_id_test", protected_id_test_available=True
            )
    alpha_records = []
    alpha_contexts = sorted(
        {
            (manifest["checkpoint_role"], _context_epoch(manifest), manifest["depth_tap"])
            for manifest, _ in loaded
            if manifest["cell_id"] == PRIMARY_ANCHOR_CELL
        },
        key=str,
    )
    for context in alpha_contexts:
        for transform in TRANSFORMS:
            for detector in DETECTORS:
                for group, names in (("near", NEAR_SPLITS), ("far", FAR_SPLITS)):
                    for metric in ("auroc", "fpr95_id_tpr"):
                        role_means = {}
                        for alpha_name, role in (
                            ("alpha_0", "decoupled"),
                            ("alpha_0_5", "alpha_0_5"),
                            ("alpha_1", "coupled"),
                        ):
                            values = [
                                float(np.mean([
                                    manifest["ood_metrics"][transform][detector]["per_dataset"][name][metric]
                                    for name in names
                                ]))
                                for manifest, _ in loaded
                                if manifest["cell_id"] == PRIMARY_ANCHOR_CELL
                                and _role(manifest) == role
                                and (
                                    manifest["checkpoint_role"], _context_epoch(manifest),
                                    manifest["depth_tap"],
                                ) == context
                            ]
                            if values:
                                role_means[alpha_name] = float(np.mean(values))
                        if len(role_means) == 3:
                            alpha_records.append(
                                {
                                    "context": context,
                                    "transform": transform,
                                    "detector": detector,
                                    "dataset_group": group,
                                    "metric": metric,
                                    **classify_alpha_interior(
                                        alpha_0_mean=role_means["alpha_0"],
                                        alpha_0_5_mean=role_means["alpha_0_5"],
                                        alpha_1_mean=role_means["alpha_1"],
                                    ),
                                }
                            )
    primary = [
        row for row in seed_records
        if row["cell_id"] == PRIMARY_ANCHOR_CELL
        and row["checkpoint_role"] == "last" and row["checkpoint_epoch"] == 200
        and row["depth_tap"] == "penultimate"
        and row["direction"] == "coupled_minus_decoupled"
        and row["transform"] == "raw" and row["detector"] == "md"
    ]
    if expected_contexts == 360 and len(primary) != 5:
        raise ValueError("primary endpoint does not contain exactly five paired seeds")
    p_values = {
        f"{group}_{metric}": paired_t_p_value([row["datasets"][f"{group}_{metric}"] for row in primary])
        for group in ("near", "far") for metric in ("delta_auroc", "pair_order_churn")
    }
    trajectory_bands = {}
    trajectory_rows = [
        row for row in seed_records
        if row["cell_id"] == PRIMARY_ANCHOR_CELL
        and row["checkpoint_role"] in {"snapshot", "last"}
        and row["checkpoint_epoch"] in TRAJECTORY_EPOCHS
        and row["depth_tap"] == "penultimate"
        and row["direction"] == "coupled_minus_decoupled"
        and row["transform"] == "raw" and row["detector"] == "md"
    ]
    trajectory_seeds = sorted({int(row["training_seed"]) for row in trajectory_rows})
    if trajectory_seeds and len(trajectory_rows) == len(trajectory_seeds) * len(TRAJECTORY_EPOCHS):
        for group in ("near", "far"):
            for metric in ("delta_auroc", "pair_order_churn"):
                matrix = np.asarray(
                    [
                        [
                            next(
                                float(row["datasets"][f"{group}_{metric}"])
                                for row in trajectory_rows
                                if int(row["training_seed"]) == seed
                                and int(row["checkpoint_epoch"]) == epoch
                            )
                            for epoch in TRAJECTORY_EPOCHS
                        ]
                        for seed in trajectory_seeds
                    ],
                    dtype=np.float64,
                )
                try:
                    band = simultaneous_sign_flip_band(matrix)
                    band_status = "PASS"
                except ValueError as exc:
                    band = {"reason": str(exc), "values": matrix.tolist()}
                    band_status = "DEGENERATE"
                trapezoid = getattr(np, "trapezoid", np.trapz)
                area = trapezoid(matrix, np.asarray(TRAJECTORY_EPOCHS), axis=1) / (
                    TRAJECTORY_EPOCHS[-1] - TRAJECTORY_EPOCHS[0]
                )
                slope = (matrix[:, 1] - matrix[:, 0]) / 50.0
                trajectory_bands[f"{group}_{metric}"] = {
                    "status": band_status,
                    "epochs": list(TRAJECTORY_EPOCHS),
                    "simultaneous_90_percent_band": band,
                    "normalized_trapezoidal_area": paired_t_interval(area),
                    "early_slope_10_to_60": paired_t_interval(slope),
                }
    terminal = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": "PASS",
        "protected_data_access": True,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "rtmd": "EXCLUDED_BY_FAILED_GATE3",
        "counts": {
            "contexts": len(loaded),
            "seed_pair_records": len(seed_records),
            "paired_aggregates": len(aggregates),
        },
        "seed_records": seed_records,
        "paired_aggregates": aggregates,
        "id_equivalence": id_equivalence,
        "alpha_classification": alpha_records,
        "primary_holm_alpha_0_10": holm_adjust(p_values),
        "primary_trajectory": trajectory_bands,
        "missing_or_failed_records_are_excluded": False,
    }
    terminal["terminal_sha256"] = canonical_sha256(terminal)
    return terminal


def default_run_plan() -> dict[str, Any]:
    return generate_research_run_matrix(
        anchor_seeds=(0, 1, 2, 3, 4),
        adam_factorial_seeds=(0, 1, 2),
        sgdm_seeds=(0, 1, 2),
    )


def load_json_or_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a mapping")
    return value
