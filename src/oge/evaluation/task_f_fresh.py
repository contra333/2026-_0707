"""Task F fresh ID-only evaluation planning and artifact bridging.

This module deliberately stops at ID-only feature and classifier artifacts.  It
does not accept the protected ID-test split, OOD datasets, or detector results.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from oge.data.imglist_dataset import ImglistDataset, parse_imglist_entry
from oge.data.openood_cifar10 import resolve_imglist_path
from oge.data.transforms import make_cifar10_eval_transform
from oge.feature_export import (
    ordered_sample_id_sha256,
    specification_payload,
    validate_task_f_checkpoint_payload,
    verify_task_f_artifact,
)
from oge.studies.hashing import canonical_json_bytes, canonical_sha256
from oge.training import TASK_F_SNAPSHOT_EPOCHS, validate_research_run_matrix
from oge.training.checkpoint import load_torch_artifact


PLAN_SCHEMA_VERSION = "task_f_fresh_id_evaluation_plan_v1"
BINDINGS_SCHEMA_VERSION = "task_f_fresh_id_evaluation_bindings_v1"
BRIDGE_SCHEMA_VERSION = "task_f_fresh_id_bridge_artifact_v1"
TERMINAL_SCHEMA_VERSION = "task_f_fresh_id_bridge_terminal_v1"
ID_INPUT_SCHEMA_VERSION = "task_f_fresh_id_input_v1"
ALLOWED_SPLITS = ("id_train", "id_validation")
PRIMARY_ANCHOR_CELL = "adam_lr1e-3_wd1e-4_anchor"
EXPECTED_SPECIFICATION_SHA256 = (
    "0ac3101e6d6aaed1a5a0d4891792d4700540bac5247cca8f7d6e67d664ffe9ba"
)
_EXPECTED_FAMILY_COUNTS = {"adam": 41, "sgdm": 9}
_PROTECTED_TOKENS = (
    "id_test",
    "openood",
    "protected_ood",
    "cifar100",
    "tinyimagenet",
    "mnist",
    "svhn",
    "texture",
    "places365",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_protected_reference(value: Any, label: str) -> None:
    normalized = str(value).lower().replace("-", "_")
    if any(token in normalized for token in _PROTECTED_TOKENS):
        raise ValueError(f"{label} contains a protected split reference")


def _canonical_write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _record_id(
    *, run_id: str, checkpoint_role: str, checkpoint_epoch: int | None,
    depth_tap: str, dataset_split: str,
) -> str:
    epoch = "from-checkpoint" if checkpoint_epoch is None else f"{checkpoint_epoch:04d}"
    return "__".join((run_id, checkpoint_role, epoch, depth_tap, dataset_split))


def _checkpoint_relative_path(role: str, epoch: int | None) -> str:
    if role == "best_val":
        return "checkpoints/best_val.pt"
    if role == "last":
        return "checkpoints/last.pt"
    assert epoch is not None
    return f"checkpoints/snapshots/epoch_{epoch:04d}.pt"


def _logical_record(
    run: Mapping[str, Any], *, checkpoint_role: str,
    checkpoint_epoch: int | None, depth_tap: str, dataset_split: str,
    purpose: str,
) -> dict[str, Any]:
    sibling_role = str(run["task_f_b_sibling_role"])
    sibling = run["task_f_b_sibling_members"][sibling_role]
    record_id = _record_id(
        run_id=str(run["run_id"]),
        checkpoint_role=checkpoint_role,
        checkpoint_epoch=checkpoint_epoch,
        depth_tap=depth_tap,
        dataset_split=dataset_split,
    )
    return {
        "record_id": record_id,
        "run_id": str(run["run_id"]),
        "family": str(run["family"]),
        "cell_id": str(run["cell_id"]),
        "training_seed": int(run["training_seed"]),
        "branch_policy": str(sibling["branch_policy"]),
        "sibling_group_id": str(run["sibling_group_id"]),
        "sibling_role": sibling_role,
        "checkpoint_role": checkpoint_role,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_epoch_source": (
            "checkpoint_payload" if checkpoint_epoch is None else "frozen_plan"
        ),
        "checkpoint_relative_path": _checkpoint_relative_path(
            checkpoint_role, checkpoint_epoch
        ),
        "depth_tap": depth_tap,
        "dataset_split": dataset_split,
        "purpose": purpose,
        "classifier_evaluation": (
            "REQUIRED" if depth_tap == "penultimate" else "NOT_APPLICABLE"
        ),
    }


def build_fresh_evaluation_plan(run_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact 1,320-record Task F fresh ID-only logical plan."""

    validate_research_run_matrix(run_plan)
    runs = list(run_plan["runs"])
    if len(runs) != 50 or Counter(run["family"] for run in runs) != Counter(
        _EXPECTED_FAMILY_COUNTS
    ):
        raise ValueError("fresh evaluation requires exactly 50 research runs")
    if any(bool(run.get("execution_only")) for run in runs):
        raise ValueError("execution-only runs are forbidden from fresh evaluation")
    specification_sha256 = canonical_sha256(specification_payload())
    if specification_sha256 != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("Task F feature-export specification identity changed")

    records: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda value: str(value["run_id"])):
        for split in ALLOWED_SPLITS:
            for epoch in TASK_F_SNAPSHOT_EPOCHS:
                role = "last" if epoch == 200 else "snapshot"
                records.append(
                    _logical_record(
                        run,
                        checkpoint_role=role,
                        checkpoint_epoch=epoch,
                        depth_tap="penultimate",
                        dataset_split=split,
                        purpose="trajectory",
                    )
                )
            if run["cell_id"] == PRIMARY_ANCHOR_CELL:
                for tap in ("stage1", "stage2", "stage3"):
                    records.append(
                        _logical_record(
                            run,
                            checkpoint_role="last",
                            checkpoint_epoch=200,
                            depth_tap=tap,
                            dataset_split=split,
                            purpose="epoch_200_depth",
                        )
                    )
            records.append(
                _logical_record(
                    run,
                    checkpoint_role="best_val",
                    checkpoint_epoch=None,
                    depth_tap="penultimate",
                    dataset_split=split,
                    purpose="id_selected_control",
                )
            )
    if len(records) != 1320 or len({row["record_id"] for row in records}) != 1320:
        raise AssertionError("Task F fresh evaluation logical coverage is not 1,320 unique records")
    split_counts = Counter(row["dataset_split"] for row in records)
    if split_counts != Counter({"id_train": 660, "id_validation": 660}):
        raise AssertionError("Task F fresh evaluation split coverage is invalid")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": run_plan["protocol_id"],
        "study_id": run_plan["study_id"],
        "source_run_plan_schema_version": run_plan["schema_version"],
        "task_f_specification_sha256": specification_sha256,
        "protected_data_access": False,
        "counts": {
            "research_runs": 50,
            "adam_runs": 41,
            "sgdm_runs": 9,
            "records_total": 1320,
            "records_by_split": {"id_train": 660, "id_validation": 660},
            "trajectory_per_split": 550,
            "depth_per_split": 60,
            "best_val_per_split": 50,
        },
        "records": records,
    }


def _observed_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    split = str(value["dataset_split"])
    _reject_protected_reference(split, "observed dataset_split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported observed ID split: {split}")
    epoch = value.get("checkpoint_epoch")
    if epoch is not None:
        epoch = int(epoch)
    return (
        str(value["run_id"]),
        str(value["checkpoint_role"]),
        epoch,
        str(value["depth_tap"]),
        split,
    )


def summarize_export_coverage(
    plan: Mapping[str, Any], observed_jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare an operational job list with the complete logical plan."""

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported fresh evaluation plan")
    expected = {
        _observed_key(record): str(record["record_id"])
        for record in plan["records"]
        if record["checkpoint_epoch"] is not None
    }
    observed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    duplicates: list[tuple[Any, ...]] = []
    unexpected: list[tuple[Any, ...]] = []
    for job in observed_jobs:
        key = _observed_key(job)
        if key in observed:
            duplicates.append(key)
            continue
        observed[key] = job
        if key not in expected:
            unexpected.append(key)
    covered = sorted(expected[key] for key in observed if key in expected)
    missing = sorted(record_id for key, record_id in expected.items() if key not in observed)
    best_val = sorted(
        str(row["record_id"])
        for row in plan["records"]
        if row["checkpoint_role"] == "best_val"
    )
    return {
        "status": "PASS" if not duplicates and not unexpected else "FAILED",
        "observed_job_count": len(observed_jobs),
        "covered_record_count": len(covered),
        "supplemental_record_count": len(missing) + len(best_val),
        "covered_record_ids": covered,
        "missing_fixed_epoch_record_ids": missing,
        "best_val_record_ids": best_val,
        "duplicate_keys": [list(key) for key in sorted(duplicates)],
        "unexpected_keys": [list(key) for key in sorted(unexpected)],
    }


def _load_dataset_policy(dataset_config_path: str | Path, split: str) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("fresh ID bridge permits only id_train and id_validation")
    config_path = Path(dataset_config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    item = config["datasets"][split]
    if not bool(item["is_id"]):
        raise ValueError("fresh ID bridge dataset policy is not ID-only")
    imglist_path = resolve_imglist_path(
        item, data_root=config_path.parent, config_root=config_path.parent
    )
    if sha256_file(imglist_path) != item["expected_sha256"]:
        raise ValueError("frozen imglist SHA-256 mismatch")
    membership = config["membership_manifest"]
    membership_path = config_path.parent / membership["path"]
    if sha256_file(membership_path) != membership["sha256"]:
        raise ValueError("frozen membership manifest SHA-256 mismatch")
    mapping: dict[str, tuple[int, str]] = {}
    with membership_path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("record_type") != "header":
            raise ValueError("membership manifest lacks its header")
        for line in handle:
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in mapping:
                raise ValueError("membership manifest contains duplicate sample IDs")
            mapping[sample_id] = (int(row["label"]), str(row["role"]))
    sample_ids: list[str] = []
    labels: list[int] = []
    with imglist_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            relative, parsed_label = parse_imglist_entry(line, line_number=line_number)
            sample_id = f"{item['dataset_name']}:{relative}"
            if sample_id not in mapping:
                raise ValueError("imglist sample is absent from membership manifest")
            label, role = mapping[sample_id]
            if label != parsed_label or role != split:
                raise ValueError("membership label or role differs from frozen imglist")
            sample_ids.append(sample_id)
            labels.append(label)
    expected_count = int(item["expected_count"])
    if len(sample_ids) != expected_count or len(set(sample_ids)) != expected_count:
        raise ValueError("frozen ID split count or uniqueness mismatch")
    sample_array = np.asarray(sample_ids, dtype=str)
    return {
        "split": split,
        "sample_ids": sample_array,
        "labels": np.asarray(labels, dtype=np.int64),
        "ordered_sample_id_sha256": ordered_sample_id_sha256(sample_array),
        "imglist_sha256": item["expected_sha256"],
        "membership_manifest_sha256": membership["sha256"],
    }


def _array_sha256(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _write_checksums(root: Path, names: Sequence[str]) -> None:
    rows = [f"{sha256_file(root / name)}  {name}" for name in sorted(names)]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_checksums(root: Path, names: set[str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in observed or name not in names or sha256_file(root / name) != digest:
            raise ValueError("bridge checksum catalog is invalid")
        observed[name] = digest
    if set(observed) != names:
        raise ValueError("bridge checksum catalog is incomplete")
    return observed


def _classifier_arrays(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    state = payload["model_state"]
    if "classifier.weight" not in state or "classifier.bias" not in state:
        raise ValueError("checkpoint lacks the WRN classifier weight or bias")
    weight = state["classifier.weight"].detach().cpu().numpy().astype(np.float32)
    bias = state["classifier.bias"].detach().cpu().numpy().astype(np.float32)
    if weight.shape != (10, 640) or bias.shape != (10,):
        raise ValueError("checkpoint classifier shape is not WRN-28-10/CIFAR-10")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("checkpoint classifier contains non-finite values")
    return weight, bias


def write_bridge_artifact(
    *, record: Mapping[str, Any], checkpoint_path: str | Path,
    feature_artifact_path: str | Path, dataset_config_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Validate and atomically materialize one ID-only bridge bundle."""

    split = str(record["dataset_split"])
    policy = _load_dataset_policy(dataset_config_path, split)
    verified = verify_task_f_artifact(feature_artifact_path)
    feature_manifest = verified["manifest"]
    checkpoint = load_torch_artifact(checkpoint_path, map_location="cpu")
    provenance = validate_task_f_checkpoint_payload(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    fixed_epoch = record["checkpoint_epoch"]
    actual_epoch = int(provenance["checkpoint_epoch"])
    checks = {
        "run_id": provenance["run_id"] == record["run_id"] == feature_manifest["run_id"],
        "training_seed": int(provenance["training_seed"]) == int(record["training_seed"]),
        "branch_policy": provenance["branch_policy"] == record["branch_policy"],
        "sibling_group": provenance["sibling_group_id"] == record["sibling_group_id"],
        "sibling_role": provenance["sibling_role"] == record["sibling_role"],
        "checkpoint_role": provenance["checkpoint_role"] == record["checkpoint_role"],
        "checkpoint_epoch": fixed_epoch is None or actual_epoch == int(fixed_epoch),
        "feature_epoch": int(feature_manifest["checkpoint_epoch"]) == actual_epoch,
        "feature_role": feature_manifest["checkpoint_role"] == record["checkpoint_role"],
        "checkpoint_sha256": feature_manifest["checkpoint_sha256"] == checkpoint_sha256,
        "depth_tap": feature_manifest["depth_tap"] == record["depth_tap"],
        "dataset_split": feature_manifest["dataset_split"] == split,
        "sample_order": feature_manifest["ordered_sample_id_sha256"]
        == policy["ordered_sample_id_sha256"],
        "initialization": feature_manifest["initialization_sha256"]
        == provenance["initialization_sha256"],
        "data_stream": feature_manifest["data_stream_sha256"]
        == provenance["data_stream_sha256"],
        "specification": feature_manifest["specification_sha256"]
        == EXPECTED_SPECIFICATION_SHA256,
        "not_execution_only": not bool(feature_manifest["execution_only"]),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"fresh bridge identity checks failed: {failed}")
    sample_ids = np.load(Path(feature_artifact_path) / "sample_ids.npy", allow_pickle=False)
    if not np.array_equal(sample_ids.astype(str), policy["sample_ids"]):
        raise ValueError("feature sample order differs from frozen membership order")
    features = np.load(Path(feature_artifact_path) / "features.npy", mmap_mode="r")
    labels = policy["labels"]
    if len(features) != len(labels):
        raise ValueError("feature and frozen label counts differ")

    output = Path(output_root) / str(record["record_id"])
    if output.exists():
        existing = verify_bridge_artifact(output)["manifest"]
        expected_existing = {
            "record_id": record["record_id"],
            "checkpoint_sha256": checkpoint_sha256,
            "feature_output_identity_sha256": feature_manifest[
                "output_identity_sha256"
            ],
            "ordered_sample_id_sha256": policy["ordered_sample_id_sha256"],
            "initialization_sha256": provenance["initialization_sha256"],
            "data_stream_sha256": provenance["data_stream_sha256"],
        }
        mismatches = sorted(
            key
            for key, expected_value in expected_existing.items()
            if existing.get(key) != expected_value
        )
        if mismatches:
            raise FileExistsError(
                "refusing to overwrite an existing bridge with different identity: "
                f"{mismatches}"
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        np.save(temporary / "labels.npy", labels, allow_pickle=False)
        names = ["labels.npy", "manifest.json"]
        classifier: dict[str, Any]
        if record["depth_tap"] == "penultimate":
            weight, bias = _classifier_arrays(checkpoint)
            logits = np.asarray(features, dtype=np.float32) @ weight.T + bias
            if not np.isfinite(logits).all():
                raise ValueError("classifier logits contain non-finite values")
            np.save(temporary / "classifier_weight.npy", weight, allow_pickle=False)
            np.save(temporary / "classifier_bias.npy", bias, allow_pickle=False)
            np.save(temporary / "logits.npy", logits.astype(np.float32), allow_pickle=False)
            names.extend(("classifier_weight.npy", "classifier_bias.npy", "logits.npy"))
            classifier = {
                "status": "READY",
                "weight_shape": list(weight.shape),
                "bias_shape": list(bias.shape),
                "logits_shape": list(logits.shape),
                "weight_array_sha256": _array_sha256(weight),
                "bias_array_sha256": _array_sha256(bias),
                "logits_array_sha256": _array_sha256(logits.astype(np.float32)),
            }
        else:
            classifier = {
                "status": "NOT_APPLICABLE",
                "reason": "classifier_reads_penultimate_only",
            }
        manifest = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "protected_data_access": False,
            "record_id": record["record_id"],
            "run_id": record["run_id"],
            "family": record["family"],
            "cell_id": record["cell_id"],
            "training_seed": int(record["training_seed"]),
            "branch_policy": record["branch_policy"],
            "sibling_group_id": record["sibling_group_id"],
            "sibling_role": record["sibling_role"],
            "checkpoint_role": record["checkpoint_role"],
            "checkpoint_epoch": actual_epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "feature_output_identity_sha256": feature_manifest["output_identity_sha256"],
            "feature_shape": feature_manifest["feature_shape"],
            "feature_dtype": feature_manifest["feature_dtype"],
            "depth_tap": record["depth_tap"],
            "dataset_split": split,
            "ordered_sample_id_sha256": policy["ordered_sample_id_sha256"],
            "labels_shape": list(labels.shape),
            "labels_dtype": str(labels.dtype),
            "labels_array_sha256": _array_sha256(labels),
            "membership_manifest_sha256": policy["membership_manifest_sha256"],
            "imglist_sha256": policy["imglist_sha256"],
            "initialization_sha256": provenance["initialization_sha256"],
            "data_stream_sha256": provenance["data_stream_sha256"],
            "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
            "identity_checks": checks,
            "classifier": classifier,
        }
        _canonical_write(temporary / "manifest.json", manifest)
        _write_checksums(temporary, names)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_bridge_artifact(output)
    return output


def verify_bridge_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (root / "manifest.json").read_bytes() != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("bridge manifest is not canonical JSON")
    if manifest.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        raise ValueError("unsupported bridge artifact schema")
    if manifest.get("protected_data_access") is not False:
        raise ValueError("bridge artifact is not ID-only")
    _reject_protected_reference(manifest["dataset_split"], "bridge dataset_split")
    expected = {"labels.npy", "manifest.json"}
    if manifest["classifier"]["status"] == "READY":
        expected.update(("classifier_weight.npy", "classifier_bias.npy", "logits.npy"))
    expected.add("checksums.sha256")
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("bridge artifact contains unexpected files")
    verified = _verify_checksums(root, expected - {"checksums.sha256"})
    labels = np.load(root / "labels.npy", allow_pickle=False)
    if list(labels.shape) != manifest["labels_shape"] or str(labels.dtype) != manifest["labels_dtype"]:
        raise ValueError("bridge label shape or dtype mismatch")
    if _array_sha256(labels) != manifest["labels_array_sha256"]:
        raise ValueError("bridge label array identity mismatch")
    if manifest["classifier"]["status"] == "READY":
        weight = np.load(root / "classifier_weight.npy", allow_pickle=False)
        bias = np.load(root / "classifier_bias.npy", allow_pickle=False)
        logits = np.load(root / "logits.npy", allow_pickle=False)
        if list(weight.shape) != manifest["classifier"]["weight_shape"]:
            raise ValueError("bridge classifier weight shape mismatch")
        if list(bias.shape) != manifest["classifier"]["bias_shape"]:
            raise ValueError("bridge classifier bias shape mismatch")
        if list(logits.shape) != manifest["classifier"]["logits_shape"]:
            raise ValueError("bridge logits shape mismatch")
        for name, value in (("weight", weight), ("bias", bias), ("logits", logits)):
            if not np.isfinite(value).all() or _array_sha256(value) != manifest["classifier"][f"{name}_array_sha256"]:
                raise ValueError(f"bridge classifier {name} identity mismatch")
    return {"manifest": manifest, "verified_files": verified}


def validate_bound_inventory(
    *, plan: Mapping[str, Any], bindings: Mapping[str, Any],
    dataset_config_path: str | Path, output_root: str | Path,
    terminal_path: str | Path,
) -> dict[str, Any]:
    """Validate all bindings while preserving every missing or failed record."""

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported fresh evaluation plan")
    if bindings.get("schema_version") != BINDINGS_SCHEMA_VERSION:
        raise ValueError("unsupported fresh evaluation bindings")
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: list[str] = []
    for binding in bindings.get("records", []):
        record_id = str(binding["record_id"])
        if record_id in by_id:
            duplicate_ids.append(record_id)
        else:
            by_id[record_id] = binding
    expected_ids = {str(row["record_id"]) for row in plan["records"]}
    unexpected = sorted(set(by_id) - expected_ids)
    results: list[dict[str, Any]] = []
    for record in plan["records"]:
        record_id = str(record["record_id"])
        binding = by_id.get(record_id)
        if binding is None:
            results.append({"record_id": record_id, "status": "MISSING", "reason": "binding_absent"})
            continue
        try:
            path = write_bridge_artifact(
                record=record,
                checkpoint_path=binding["checkpoint_path"],
                feature_artifact_path=binding["feature_artifact_path"],
                dataset_config_path=dataset_config_path,
                output_root=output_root,
            )
            manifest = verify_bridge_artifact(path)["manifest"]
            results.append(
                {
                    "record_id": record_id,
                    "status": "PASS",
                    "bridge_path": str(path),
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "feature_output_identity_sha256": manifest[
                        "feature_output_identity_sha256"
                    ],
                }
            )
        except Exception as error:  # Terminal validation must retain all failures.
            results.append(
                {
                    "record_id": record_id,
                    "status": "FAILED",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    counts = Counter(row["status"] for row in results)
    status = (
        "FAILED" if duplicate_ids or unexpected or counts["FAILED"] else
        "INCOMPLETE" if counts["MISSING"] else "PASS"
    )
    terminal = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": status,
        "protected_data_access": False,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "expected_record_count": 1320,
        "counts": dict(sorted(counts.items())),
        "duplicate_binding_record_ids": sorted(set(duplicate_ids)),
        "unexpected_binding_record_ids": unexpected,
        "records": results,
    }
    destination = Path(terminal_path)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != terminal:
            raise FileExistsError("terminal manifest exists with different content")
        return terminal
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    _canonical_write(temporary, terminal)
    os.replace(temporary, destination)
    return terminal


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return output.getvalue()


def verify_id_input(
    path: str | Path, *, dataset_config_path: str | Path, split: str
) -> dict[str, Any]:
    policy = _load_dataset_policy(dataset_config_path, split)
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"images", "is_id", "sample_ids"}:
            raise ValueError("Task F ID input has unexpected arrays")
        images = np.asarray(payload["images"])
        is_id = np.asarray(payload["is_id"])
        sample_ids = np.asarray(payload["sample_ids"])
    if images.shape != (len(policy["sample_ids"]), 3, 32, 32) or images.dtype != np.float32:
        raise ValueError("Task F ID input image shape or dtype mismatch")
    if not np.isfinite(images).all():
        raise ValueError("Task F ID input images contain non-finite values")
    if is_id.shape != (len(images),) or is_id.dtype != np.bool_ or not np.all(is_id):
        raise ValueError("Task F ID input is not explicitly all-ID")
    if not np.array_equal(sample_ids.astype(str), policy["sample_ids"]):
        raise ValueError("Task F ID input sample order differs from frozen membership")
    return {
        "schema_version": ID_INPUT_SCHEMA_VERSION,
        "dataset_split": split,
        "sample_count": len(images),
        "ordered_sample_id_sha256": policy["ordered_sample_id_sha256"],
        "npz_sha256": sha256_file(path),
    }


def build_id_input(
    *, dataset_config_path: str | Path, data_root: str | Path, split: str,
    output_path: str | Path, batch_size: int = 512, num_workers: int = 0,
) -> dict[str, Any]:
    """Create a deterministic evaluation-transformed ID input without overwrite."""

    if split not in ALLOWED_SPLITS:
        raise ValueError("Task F ID input permits only id_train and id_validation")
    destination = Path(output_path)
    _reject_protected_reference(destination, "ID input output path")
    if destination.exists():
        return verify_id_input(destination, dataset_config_path=dataset_config_path, split=split)
    config_path = Path(dataset_config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    item = config["datasets"][split]
    imglist = resolve_imglist_path(item, data_root=data_root, config_root=config_path.parent)
    dataset = ImglistDataset(
        dataset_name=item["dataset_name"],
        split=item["split"],
        is_id=True,
        imglist_path=imglist,
        data_root=Path(data_root) / config.get("image_root", ""),
        transform=make_cifar10_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    images: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        images.append(batch["image"].cpu().numpy().astype(np.float32, copy=False))
        sample_ids.extend(str(value) for value in batch["sample_id"])
    arrays = {
        "images": np.concatenate(images),
        "is_id": np.ones(len(sample_ids), dtype=np.bool_),
        "sample_ids": np.asarray(sample_ids, dtype=str),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_deterministic_npz_bytes(arrays))
        verify_id_input(temporary, dataset_config_path=dataset_config_path, split=split)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_id_input(destination, dataset_config_path=dataset_config_path, split=split)
