"""Owner-authorized production evaluation for the ResNet-18 replication.

The module keeps ID fitting separate from protected queries.  ID-train and
ID-validation caches are exported and fitted first; protected ID-test/OOD
features are opened only with an authorization bound to the exact production
plan and evaluation Git SHA.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oge.analysis.discriminant_residual_preflight import (
    GeometryFit,
    fit_discriminant_geometry,
    score_discriminant_components,
)
from oge.analysis.fixed_readout_component_attribution import (
    mahalanobis_score_components,
)
from oge.analysis.task_f_fresh_id import adjudicate_id_equivalence, paired_t_interval
from oge.evaluation.classification import (
    expected_calibration_error,
    negative_log_likelihood,
    top1_accuracy,
)
from oge.evaluation.extraction import (
    extract_checkpoint_artifact,
    sha256_file,
    verify_raw_feature_artifact,
)
from oge.evaluation.metrics import compute_ood_metrics
from oge.evaluation.resnet18_replication import (
    DETECTORS,
    EVALUATION_TARGETS,
    EXPECTED_PROTECTED_SPLITS,
    LARGE_CELL,
    SMALL_CELL,
    adjudicate_resnet18_full_gate,
)
from oge.evaluation.task_f_protected import (
    FAR_SPLITS,
    NEAR_SPLITS,
    OOD_SPLITS,
    _transform as feature_transform,
    compare_score_arrays,
)
from oge.studies.hashing import canonical_json_bytes, canonical_sha256
from oge.training import load_torch_artifact
from oge.training.resnet18_replication_plan import (
    RESNET18_REPLICATION_STUDY_ID,
    validate_resnet18_replication_matrix,
)
from oge.training.resnet18_replication_provenance import (
    validate_resnet18_replication_checkpoint_payload,
)


PLAN_SCHEMA_VERSION = "resnet18_cifar10_replication_production_plan_v3"
AUTHORIZATION_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_protected_authorization_v3"
)
FIT_SCHEMA_VERSION = "resnet18_cifar10_replication_id_fit_v3"
PAIR_SCHEMA_VERSION = "resnet18_cifar10_replication_pair_scores_v3"
PAIR_RECOVERY_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_pair_score_recovery_v1"
)
TERMINAL_SCHEMA_VERSION = "resnet18_cifar10_replication_terminal_v3"
SOURCE_TRAINING_SHA = "e2f6845e88b22bc0783c5fda58186f9930083ef7"
SOURCE_TRAINING_TERMINAL_SHA256 = (
    "780dcf602a955c8936d3c901a2d473fe2c510145f5520886887b0a5dd52b99b5"
)
OWNER_APPROVAL_ISSUE = 136
ID_SPLITS = ("id_train", "id_validation")
PROTECTED_SPLITS = tuple(EXPECTED_PROTECTED_SPLITS)
TRANSFORMS = ("raw", "l2")
_FIT_ARRAY_NAMES = (
    "mean",
    "class_means",
    "class_counts",
    "within_covariance",
    "between_covariance",
    "total_covariance",
    "within_precision",
    "global_precision",
    "within_sqrt",
    "within_invsqrt",
    "subspace_basis",
    "transformed_class_means",
    "parallel_global_precision",
)
_HOST_BY_SEED = {
    0: "curie",
    1: "precision_medicine",
    2: "lise",
    3: "curie",
    4: "precision_medicine",
}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _write_atomic_directory(
    destination: Path, *, files: Mapping[str, bytes]
) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, payload in files.items():
            path = temporary / name
            path.write_bytes(payload)
        checksum_names = sorted(files)
        (temporary / "checksums.sha256").write_text(
            "".join(
                f"{sha256_file(temporary / name)}  {name}\n"
                for name in checksum_names
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _verify_checksums(root: Path, expected: set[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for row in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = row.split("  ", 1)
        if name in rows or name not in expected or sha256_file(root / name) != digest:
            raise ValueError("production artifact checksum mismatch")
        rows[name] = digest
    if set(rows) != expected:
        raise ValueError("production artifact checksum coverage mismatch")
    return rows


def build_production_plan(
    *,
    run_plan: Mapping[str, Any],
    training_terminal: Mapping[str, Any],
    training_terminal_path: str | Path,
    evaluation_git_sha: str,
) -> dict[str, Any]:
    validate_resnet18_replication_matrix(run_plan)
    if len(evaluation_git_sha) != 40:
        raise ValueError("evaluation_git_sha must be a full Git SHA")
    terminal = copy.deepcopy(dict(training_terminal))
    if (
        terminal.get("status") != "PASS"
        or terminal.get("execution_sha") != SOURCE_TRAINING_SHA
        or terminal.get("run_count") != 20
        or terminal.get("seed_count") != 5
    ):
        raise ValueError("source training terminal is not the frozen 20-run PASS")
    terminal_path = Path(training_terminal_path)
    if sha256_file(terminal_path) != SOURCE_TRAINING_TERMINAL_SHA256:
        raise ValueError("source training terminal SHA256 changed")
    planned = {str(row["run_id"]): row for row in run_plan["runs"]}
    observed = {str(row["run_id"]): row for row in terminal["runs"]}
    if set(planned) != set(observed) or len(observed) != 20:
        raise ValueError("training terminal run coverage differs from the matrix")
    records = []
    for run_id in sorted(planned):
        plan_row = planned[run_id]
        source = observed[run_id]
        checkpoints = [
            row
            for row in source["checkpoints"]
            if row["checkpoint_role"] == "last" and row["checkpoint_epoch"] == 200
        ]
        if len(checkpoints) != 1:
            raise ValueError("each source run must have exactly one epoch-200 last.pt")
        checkpoint = checkpoints[0]
        seed = int(plan_row["training_seed"])
        for field in (
            "branch_policy",
            "sibling_group_id",
            "cross_lr_pairing_block_id",
        ):
            if source[field] != plan_row[field]:
                raise ValueError(f"source training identity mismatch: {field}")
        records.append(
            {
                "run_id": run_id,
                "cell_id": str(plan_row["cell_id"]),
                "training_seed": seed,
                "branch_policy": str(plan_row["branch_policy"]),
                "sibling_group_id": str(plan_row["sibling_group_id"]),
                "cross_lr_pairing_block_id": str(
                    plan_row["cross_lr_pairing_block_id"]
                ),
                "initialization_sha256": str(source["initialization_sha256"]),
                "data_stream_id": str(source["data_stream_id"]),
                "initial_dataloader_rng_sha256": str(
                    source["initial_dataloader_rng_sha256"]
                ),
                "first_minibatch_ordered_sample_id_sha256": str(
                    source["first_minibatch_ordered_sample_id_sha256"]
                ),
                "first_minibatch_transformed_image_sha256": str(
                    source["first_minibatch_transformed_image_sha256"]
                ),
                "host_id": _HOST_BY_SEED[seed],
                "checkpoint_role": "last",
                "checkpoint_epoch": 200,
                "checkpoint_path": str(checkpoint["path"]),
                "checkpoint_sha256": str(checkpoint["sha256"]),
                "id_splits": list(ID_SPLITS),
                "protected_splits": list(PROTECTED_SPLITS),
            }
        )
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "evaluation_git_sha": evaluation_git_sha,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "source_training_terminal_sha256": SOURCE_TRAINING_TERMINAL_SHA256,
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "feature_dim": 512,
        "id_fit_split": "id_train",
        "id_validation_split": "id_validation",
        "protected_splits": list(PROTECTED_SPLITS),
        "score_targets": ["raw_md", "raw_rmd", "raw_marginal", "l2_md"],
        "pair_direction": "coupled_minus_decoupled",
        "records": records,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return validate_production_plan(payload)


def validate_production_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    digest = payload.pop("plan_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("production plan hash mismatch")
    if (
        payload.get("schema_version") != PLAN_SCHEMA_VERSION
        or payload.get("study_id") != RESNET18_REPLICATION_STUDY_ID
        or payload.get("source_training_sha") != SOURCE_TRAINING_SHA
        or payload.get("source_training_terminal_sha256")
        != SOURCE_TRAINING_TERMINAL_SHA256
        or payload.get("checkpoint_role") != "last"
        or payload.get("checkpoint_epoch") != 200
        or payload.get("depth_tap") != "penultimate"
        or tuple(payload.get("protected_splits", ())) != PROTECTED_SPLITS
    ):
        raise ValueError("production plan contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("production plan requires exactly 20 records")
    if len({row["run_id"] for row in records}) != 20:
        raise ValueError("production plan contains duplicate runs")
    payload["plan_sha256"] = digest
    return payload


def build_protected_authorization(
    *, plan: Mapping[str, Any], evaluation_git_sha: str, approved_at: str
) -> dict[str, Any]:
    validated = validate_production_plan(plan)
    if evaluation_git_sha != validated["evaluation_git_sha"]:
        raise ValueError("authorization Git SHA differs from production plan")
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "issue_number": OWNER_APPROVAL_ISSUE,
        "approved_at": str(approved_at),
        "owner_instruction": (
            "Checkpoint inference, ID feature export, ID-test guardrail, "
            "Mahalanobis fitting and protected OOD evaluation approved"
        ),
        "evaluation_git_sha": evaluation_git_sha,
        "plan_sha256": validated["plan_sha256"],
        "source_training_terminal_sha256": SOURCE_TRAINING_TERMINAL_SHA256,
        "protected_splits": list(PROTECTED_SPLITS),
        "one_shot": True,
        "selection_or_tuning": False,
        "rescue_grid": False,
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return validate_protected_authorization(payload, plan=validated)


def validate_protected_authorization(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validated_plan = validate_production_plan(plan)
    payload = copy.deepcopy(dict(value))
    digest = payload.pop("authorization_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("protected authorization hash mismatch")
    checks = {
        "schema": payload.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION,
        "issue": payload.get("issue_number") == OWNER_APPROVAL_ISSUE,
        "git": payload.get("evaluation_git_sha")
        == validated_plan["evaluation_git_sha"],
        "plan": payload.get("plan_sha256") == validated_plan["plan_sha256"],
        "terminal": payload.get("source_training_terminal_sha256")
        == SOURCE_TRAINING_TERMINAL_SHA256,
        "splits": tuple(payload.get("protected_splits", ())) == PROTECTED_SPLITS,
        "one_shot": payload.get("one_shot") is True,
        "no_tuning": payload.get("selection_or_tuning") is False,
        "no_rescue": payload.get("rescue_grid") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"protected authorization checks failed: {failed}")
    payload["authorization_sha256"] = digest
    return payload


def _plan_record(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    validated = validate_production_plan(plan)
    records = {str(row["run_id"]): row for row in validated["records"]}
    if run_id not in records:
        raise ValueError("run_id is outside the production plan")
    return dict(records[run_id])


def export_checkpoint_phase(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    run_id: str,
    phase: str,
    checkpoint_path: str | Path,
    dataset_config_path: str | Path,
    data_root: str | Path,
    artifact_root: str | Path,
    device: str,
    batch_size: int = 512,
    num_workers: int = 4,
    repository_root: str | Path = ".",
) -> Path:
    validated_plan = validate_production_plan(plan)
    authorized = validate_protected_authorization(authorization, plan=validated_plan)
    record = _plan_record(validated_plan, run_id)
    path = Path(checkpoint_path)
    if str(path) != record["checkpoint_path"]:
        raise ValueError("checkpoint path differs from the frozen production plan")
    if sha256_file(path) != record["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 differs from the frozen production plan")
    checkpoint = load_torch_artifact(path, map_location="cpu")
    provenance = validate_resnet18_replication_checkpoint_payload(checkpoint)
    if (
        provenance["run_id"] != run_id
        or provenance["checkpoint_role"] != "last"
        or int(provenance["checkpoint_epoch"]) != 200
        or provenance["branch_policy"] != record["branch_policy"]
    ):
        raise ValueError("checkpoint provenance differs from the production record")
    if phase == "id":
        keys = list(ID_SPLITS)
        protected_token = None
    elif phase == "protected":
        keys = list(PROTECTED_SPLITS)
        protected_token = authorized["authorization_sha256"]
    else:
        raise ValueError("phase must be id or protected")
    output = extract_checkpoint_artifact(
        checkpoint_path=path,
        dataset_config_path=dataset_config_path,
        data_root=data_root,
        artifact_root=artifact_root,
        dataset_keys=keys,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        protected_authorization=protected_token,
        smoke_only=False,
        extraction_command=f"resnet18-production-{phase}",
        repository_root=repository_root,
    )
    manifest = verify_raw_feature_artifact(output)["manifest"]
    if (
        manifest["checkpoint"]["training_run_id"] != run_id
        or manifest["checkpoint"]["sha256"] != record["checkpoint_sha256"]
        or manifest["checkpoint"]["role"] != "last"
        or int(manifest["checkpoint"]["completed_epoch"]) != 200
        or set(manifest["dataset"]["splits"]) != set(keys)
        or manifest["model"]["feature_dim"] != 512
        or manifest["model"]["class_count"] != 10
    ):
        raise ValueError("exported cache differs from the frozen endpoint contract")
    return output


def _load_split_arrays(
    root: Path, manifest: Mapping[str, Any], split: str
) -> dict[str, np.ndarray]:
    split_record = manifest["dataset"]["splits"].get(split)
    if not isinstance(split_record, Mapping):
        raise ValueError(f"raw feature artifact lacks split {split}")
    directory = root / str(split_record["relative_directory"])
    return {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "features",
            "logits",
            "class_labels",
            "predictions",
            "is_id",
            "sample_ids",
        )
    }


def _split_arrays(
    artifact_root: str | Path, split: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(artifact_root)
    manifest = verify_raw_feature_artifact(root)["manifest"]
    return manifest, _load_split_arrays(root, manifest, split)


def fit_id_artifact(
    *, plan: Mapping[str, Any], run_id: str, id_artifact: str | Path,
    output_root: str | Path,
) -> Path:
    record = _plan_record(plan, run_id)
    id_root = Path(id_artifact)
    train_manifest = verify_raw_feature_artifact(id_root)["manifest"]
    train = _load_split_arrays(id_root, train_manifest, "id_train")
    validation = _load_split_arrays(id_root, train_manifest, "id_validation")
    if (
        train_manifest["checkpoint"]["training_run_id"] != run_id
        or train_manifest["checkpoint"]["sha256"] != record["checkpoint_sha256"]
    ):
        raise ValueError("ID artifact differs from the production source record")
    fit_arrays: dict[str, np.ndarray] = {}
    fit_summary: dict[str, Any] = {}
    validation_summary: dict[str, Any] = {}
    for transform in TRANSFORMS:
        train_features = feature_transform(train["features"], transform)
        validation_features = feature_transform(validation["features"], transform)
        fit = fit_discriminant_geometry(train_features, train["class_labels"])
        for name in _FIT_ARRAY_NAMES:
            fit_arrays[f"{transform}__{name}"] = np.asarray(getattr(fit, name))
        fit_summary[transform] = {
            "applicable": bool(fit.applicable),
            "dim": int(fit.dim),
            "residual_dim": int(fit.residual_dim),
            "condition_number": (
                float(fit.condition_number)
                if math.isfinite(fit.condition_number)
                else "infinity"
            ),
            "ridge": float(fit.ridge),
            "tau_alg": float(fit.tau_alg),
            "numerical": copy.deepcopy(dict(fit.numerical)),
        }
        scored = _score_fit(fit, validation_features, chunk_size=2048)
        validation_summary[transform] = {
            name: {
                "count": int(len(value)),
                "finite": bool(np.isfinite(value).all()),
                "array_sha256": _array_sha256(value),
            }
            for name, value in scored.items()
        }
    identity = {
        "schema_version": FIT_SCHEMA_VERSION,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "run_id": run_id,
        "cell_id": record["cell_id"],
        "training_seed": record["training_seed"],
        "branch_policy": record["branch_policy"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "id_artifact_sha256": canonical_sha256(train_manifest),
        "id_fit_split": "id_train",
        "protected_data_access": False,
    }
    output_identity = canonical_sha256(identity)
    manifest = {
        **identity,
        "output_identity_sha256": output_identity,
        "fit_summary": fit_summary,
        "id_validation_score_checks": validation_summary,
        "fit_arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "array_sha256": _array_sha256(value),
            }
            for name, value in sorted(fit_arrays.items())
        },
    }
    temporary_npz = Path(output_root) / f".{output_identity}.{uuid.uuid4().hex}.npz"
    temporary_npz.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(temporary_npz, **fit_arrays)
        path = _write_atomic_directory(
            Path(output_root) / output_identity,
            files={
                "fit_state.npz": temporary_npz.read_bytes(),
                "manifest.json": canonical_json_bytes(manifest) + b"\n",
            },
        )
    finally:
        temporary_npz.unlink(missing_ok=True)
    verify_id_fit_artifact(path)
    return path


def verify_id_fit_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != FIT_SCHEMA_VERSION
        or manifest.get("protected_data_access") is not False
        or root.name != manifest.get("output_identity_sha256")
    ):
        raise ValueError("ID fit artifact identity mismatch")
    _verify_checksums(root, {"fit_state.npz", "manifest.json"})
    with np.load(root / "fit_state.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(manifest["fit_arrays"]):
            raise ValueError("ID fit array catalog mismatch")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            meta = manifest["fit_arrays"][name]
            if list(value.shape) != meta["shape"] or _array_sha256(value) != meta["array_sha256"]:
                raise ValueError("ID fit array identity mismatch")
    return {"manifest": manifest}


def _load_fit(path: str | Path, transform: str) -> tuple[dict[str, Any], GeometryFit]:
    root = Path(path)
    manifest = verify_id_fit_artifact(root)["manifest"]
    with np.load(root / "fit_state.npz", allow_pickle=False) as state:
        arrays = {
            name: np.asarray(state[f"{transform}__{name}"])
            for name in _FIT_ARRAY_NAMES
        }
    summary = manifest["fit_summary"][transform]
    condition = summary["condition_number"]
    return manifest, GeometryFit(
        **arrays,
        dim=int(summary["dim"]),
        residual_dim=int(summary["residual_dim"]),
        condition_number=(math.inf if condition == "infinity" else float(condition)),
        tau_alg=float(summary["tau_alg"]),
        ridge=float(summary["ridge"]),
        applicable=bool(summary["applicable"]),
        numerical=copy.deepcopy(dict(summary["numerical"])),
    )


def _direct_scores(fit: GeometryFit, values: np.ndarray) -> dict[str, np.ndarray]:
    queries = np.asarray(values, dtype=np.float64)
    delta = queries[:, None, :] - fit.class_means[None, :, :]
    class_distances = np.einsum(
        "ncd,de,nce->nc", delta, fit.within_precision, delta, optimize=True
    )
    centered = queries - fit.mean
    global_distances = np.einsum(
        "nd,de,ne->n", centered, fit.global_precision, centered, optimize=True
    )
    result = mahalanobis_score_components(class_distances, global_distances)
    return {name: np.asarray(result[name], dtype=np.float64) for name in DETECTORS}


def _score_fit(
    fit: GeometryFit, values: np.ndarray, *, chunk_size: int
) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {name: [] for name in DETECTORS}
    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(values[start : start + chunk_size], dtype=np.float64)
        if fit.applicable:
            record, scored = score_discriminant_components(fit, chunk)
            if not all(record["checks"].values()):
                raise ValueError("ID-fit component reconstruction check failed")
        else:
            scored = _direct_scores(fit, chunk)
        for name in DETECTORS:
            output[name].append(np.asarray(scored[name], dtype=np.float64))
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _score_query_panel(
    *, fit_path: str | Path, protected_artifact: str | Path, chunk_size: int
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    artifact_root = Path(protected_artifact)
    artifact_manifest = verify_raw_feature_artifact(artifact_root)["manifest"]
    arrays: dict[str, np.ndarray] = {}
    fit_diagnostics: dict[str, Any] = {}
    for transform in TRANSFORMS:
        fit_manifest, fit = _load_fit(fit_path, transform)
        if fit_manifest["checkpoint_sha256"] != artifact_manifest["checkpoint"]["sha256"]:
            raise ValueError("ID fit and protected feature checkpoint SHA differ")
        fit_diagnostics[transform] = copy.deepcopy(fit_manifest["fit_summary"][transform])
        for split in PROTECTED_SPLITS:
            query = _load_split_arrays(artifact_root, artifact_manifest, split)
            values = feature_transform(query["features"], transform)
            scored = _score_fit(fit, values, chunk_size=chunk_size)
            for detector, value in scored.items():
                arrays[f"{transform}__{split}__{detector}"] = value
    id_test = _load_split_arrays(artifact_root, artifact_manifest, "id_test")
    utility = {
        "accuracy": float(top1_accuracy(id_test["logits"], id_test["class_labels"])["metric"]["value"]),
        "nll": float(negative_log_likelihood(id_test["logits"], id_test["class_labels"])["metric"]["value"]),
        "ece": float(expected_calibration_error(id_test["logits"], id_test["class_labels"])["metric"]["value"]),
    }
    return artifact_manifest, arrays, {"id_utility": utility, "fit": fit_diagnostics}


def evaluate_paired_endpoint(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    cell_id: str,
    training_seed: int,
    coupled_fit: str | Path,
    decoupled_fit: str | Path,
    coupled_protected: str | Path,
    decoupled_protected: str | Path,
    output_root: str | Path,
    chunk_size: int = 2048,
) -> Path:
    validated = validate_production_plan(plan)
    authorized = validate_protected_authorization(authorization, plan=validated)
    matching = [
        row
        for row in validated["records"]
        if row["cell_id"] == cell_id and int(row["training_seed"]) == training_seed
    ]
    by_role = {row["branch_policy"]: row for row in matching}
    if set(by_role) != {"adam_coupled", "adamw_decoupled"}:
        raise ValueError("paired endpoint requires one coupled and one decoupled sibling")
    coupled_manifest, coupled_scores, coupled_summary = _score_query_panel(
        fit_path=coupled_fit,
        protected_artifact=coupled_protected,
        chunk_size=chunk_size,
    )
    decoupled_manifest, decoupled_scores, decoupled_summary = _score_query_panel(
        fit_path=decoupled_fit,
        protected_artifact=decoupled_protected,
        chunk_size=chunk_size,
    )
    source_manifests = {
        "coupled": coupled_manifest,
        "decoupled": decoupled_manifest,
    }
    for role, expected in (
        ("coupled", by_role["adam_coupled"]),
        ("decoupled", by_role["adamw_decoupled"]),
    ):
        observed = source_manifests[role]
        if (
            observed["checkpoint"]["training_run_id"] != expected["run_id"]
            or observed["checkpoint"]["sha256"] != expected["checkpoint_sha256"]
        ):
            raise ValueError("protected artifact source differs from paired plan")
    for split in PROTECTED_SPLITS:
        left = coupled_manifest["dataset"]["splits"][split]
        right = decoupled_manifest["dataset"]["splits"][split]
        if (
            left["ordered_sample_id_sha256"] != right["ordered_sample_id_sha256"]
            or left["sample_count"] != right["sample_count"]
        ):
            raise ValueError("paired protected sample order differs")
    datasets: dict[str, Any] = {}
    for split in OOD_SPLITS:
        datasets[split] = {}
        for target in EVALUATION_TARGETS:
            transform, detector = (
                ("l2", "md") if target == "l2_md" else ("raw", target)
            )
            comparison = compare_score_arrays(
                left=coupled_scores,
                right=decoupled_scores,
                split=split,
                transform=transform,
                detector=detector,
            )
            coupled_metrics = compute_ood_metrics(
                coupled_scores[f"{transform}__id_test__{detector}"],
                coupled_scores[f"{transform}__{split}__{detector}"],
            )
            decoupled_metrics = compute_ood_metrics(
                decoupled_scores[f"{transform}__id_test__{detector}"],
                decoupled_scores[f"{transform}__{split}__{detector}"],
            )
            datasets[split][target] = {
                "coupled_auroc": float(coupled_metrics["auroc"]),
                "decoupled_auroc": float(decoupled_metrics["auroc"]),
                "coupled_fpr95": float(coupled_metrics["fpr95_id_tpr"]),
                "decoupled_fpr95": float(decoupled_metrics["fpr95_id_tpr"]),
                **comparison,
            }
    macro: dict[str, Any] = {}
    for group, splits in (("near", NEAR_SPLITS), ("far", FAR_SPLITS)):
        macro[group] = {}
        for target in EVALUATION_TARGETS:
            rows = [datasets[split][target] for split in splits]
            macro[group][target] = {
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in (
                    "coupled_auroc",
                    "decoupled_auroc",
                    "coupled_fpr95",
                    "decoupled_fpr95",
                    "gain",
                    "loss",
                    "pair_order_churn",
                    "delta_auroc",
                    "balance_residual",
                )
            }
        attributions = [
            datasets[split]["md"]["component_attribution"] for split in splits
        ]
        macro[group]["md"]["phi_rmd"] = float(
            np.mean(
                [row["component_auroc_attribution"]["rmd"] for row in attributions]
            )
        )
        macro[group]["md"]["phi_marginal"] = float(
            np.mean(
                [
                    row["component_auroc_attribution"]["marginal"]
                    for row in attributions
                ]
            )
        )
    record = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "status": "PASS",
        "research_evidence": True,
        "protected_data_access": True,
        "authorization_sha256": authorized["authorization_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "cell_id": cell_id,
        "training_seed": int(training_seed),
        "direction": "coupled_minus_decoupled",
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "source_run_ids": {
            "coupled": by_role["adam_coupled"]["run_id"],
            "decoupled": by_role["adamw_decoupled"]["run_id"],
        },
        "source_checkpoint_sha256": {
            "coupled": by_role["adam_coupled"]["checkpoint_sha256"],
            "decoupled": by_role["adamw_decoupled"]["checkpoint_sha256"],
        },
        "sample_order_sha256": {
            split: coupled_manifest["dataset"]["splits"][split][
                "ordered_sample_id_sha256"
            ]
            for split in PROTECTED_SPLITS
        },
        "id_utility": {
            "coupled": coupled_summary["id_utility"],
            "decoupled": decoupled_summary["id_utility"],
        },
        "fit_diagnostics": {
            "coupled": coupled_summary["fit"],
            "decoupled": decoupled_summary["fit"],
        },
        "datasets": datasets,
        "macro": macro,
    }
    identity = {
        key: record[key]
        for key in (
            "schema_version",
            "cell_id",
            "training_seed",
            "source_run_ids",
            "source_checkpoint_sha256",
            "authorization_sha256",
        )
    }
    output_identity = canonical_sha256(identity)
    record["output_identity_sha256"] = output_identity
    score_arrays = {
        f"coupled__{name}": value for name, value in coupled_scores.items()
    }
    score_arrays.update(
        {f"decoupled__{name}": value for name, value in decoupled_scores.items()}
    )
    record["score_arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "array_sha256": _array_sha256(value),
        }
        for name, value in sorted(score_arrays.items())
    }
    temporary_npz = Path(output_root) / f".{output_identity}.{uuid.uuid4().hex}.npz"
    temporary_npz.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(temporary_npz, **score_arrays)
        path = _write_atomic_directory(
            Path(output_root) / output_identity,
            files={
                "record.json": canonical_json_bytes(record) + b"\n",
                "scores.npz": temporary_npz.read_bytes(),
            },
        )
    finally:
        temporary_npz.unlink(missing_ok=True)
    verify_pair_artifact(path)
    return path


def _load_pair_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(path)
    record = json.loads((root / "record.json").read_text(encoding="utf-8"))
    if (
        record.get("schema_version")
        not in {PAIR_SCHEMA_VERSION, PAIR_RECOVERY_SCHEMA_VERSION}
        or record.get("status") != "PASS"
        or record.get("research_evidence") is not True
        or record.get("protected_data_access") is not True
        or root.name != record.get("output_identity_sha256")
    ):
        raise ValueError("pair score artifact identity mismatch")
    _verify_checksums(root, {"record.json", "scores.npz"})
    loaded: dict[str, np.ndarray] = {}
    with np.load(root / "scores.npz", allow_pickle=False) as source:
        if set(source.files) != set(record["score_arrays"]):
            raise ValueError("pair score array catalog mismatch")
        for name in source.files:
            value = np.asarray(source[name])
            meta = record["score_arrays"][name]
            if list(value.shape) != meta["shape"] or _array_sha256(value) != meta["array_sha256"]:
                raise ValueError("pair score array identity mismatch")
            loaded[name] = value
    return record, loaded


def verify_pair_artifact(path: str | Path) -> dict[str, Any]:
    record, _ = _load_pair_artifact(path)
    for split in OOD_SPLITS:
        for target in EVALUATION_TARGETS:
            row = record["datasets"][split][target]
            if abs(row["delta_auroc"] - (row["gain"] - row["loss"])) > 1e-12:
                raise ValueError("DeltaAUROC does not reconstruct from Gain-Loss")
            if abs(row["pair_order_churn"] - (row["gain"] + row["loss"])) > 1e-12:
                raise ValueError("Churn does not reconstruct from Gain+Loss")
        attribution = record["datasets"][split]["md"]["component_attribution"]
        if not attribution["pass"]:
            raise ValueError("MD component attribution reconstruction failed")
    return {"record": record}


def recover_pair_artifact(
    *, source_path: str | Path, scoring_git_sha: str,
    output_root: str | Path,
) -> Path:
    """Revalidate stored score arrays after the score-scale tolerance fix.

    No checkpoint, feature cache, or protected dataset is opened here.  The
    exact stored score arrays are retained byte-for-byte in the new artifact.
    """

    if not isinstance(scoring_git_sha, str) or len(scoring_git_sha) != 40:
        raise ValueError("scoring_git_sha must be a full Git SHA")
    source_record, stored_arrays = _load_pair_artifact(source_path)
    coupled_scores = {
        name.removeprefix("coupled__"): value
        for name, value in stored_arrays.items()
        if name.startswith("coupled__")
    }
    decoupled_scores = {
        name.removeprefix("decoupled__"): value
        for name, value in stored_arrays.items()
        if name.startswith("decoupled__")
    }
    record = copy.deepcopy(source_record)
    record["schema_version"] = PAIR_RECOVERY_SCHEMA_VERSION
    for split in OOD_SPLITS:
        for target in EVALUATION_TARGETS:
            transform, detector = (
                ("l2", "md") if target == "l2_md" else ("raw", target)
            )
            preserved_metrics = {
                name: record["datasets"][split][target][name]
                for name in (
                    "coupled_auroc",
                    "decoupled_auroc",
                    "coupled_fpr95",
                    "decoupled_fpr95",
                )
            }
            comparison = compare_score_arrays(
                left=coupled_scores,
                right=decoupled_scores,
                split=split,
                transform=transform,
                detector=detector,
            )
            record["datasets"][split][target] = {
                **preserved_metrics,
                **comparison,
            }
    for group, splits in (("near", NEAR_SPLITS), ("far", FAR_SPLITS)):
        for target in EVALUATION_TARGETS:
            rows = [record["datasets"][split][target] for split in splits]
            record["macro"][group][target] = {
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in (
                    "coupled_auroc",
                    "decoupled_auroc",
                    "coupled_fpr95",
                    "decoupled_fpr95",
                    "gain",
                    "loss",
                    "pair_order_churn",
                    "delta_auroc",
                    "balance_residual",
                )
            }
        attributions = [
            record["datasets"][split]["md"]["component_attribution"]
            for split in splits
        ]
        record["macro"][group]["md"]["phi_rmd"] = float(
            np.mean(
                [row["component_auroc_attribution"]["rmd"] for row in attributions]
            )
        )
        record["macro"][group]["md"]["phi_marginal"] = float(
            np.mean(
                [
                    row["component_auroc_attribution"]["marginal"]
                    for row in attributions
                ]
            )
        )
    source_identity = source_record["output_identity_sha256"]
    record["recovery"] = {
        "reason": "issue_138_score_scale_tolerance_fix",
        "source_output_identity_sha256": source_identity,
        "scoring_git_sha": scoring_git_sha,
        "protected_checkpoint_inference_rerun": False,
        "detector_refit": False,
        "score_arrays_changed": False,
    }
    identity = {
        "schema_version": PAIR_RECOVERY_SCHEMA_VERSION,
        "source_output_identity_sha256": source_identity,
        "scoring_git_sha": scoring_git_sha,
        "cell_id": record["cell_id"],
        "training_seed": record["training_seed"],
        "source_run_ids": record["source_run_ids"],
        "source_checkpoint_sha256": record["source_checkpoint_sha256"],
        "authorization_sha256": record["authorization_sha256"],
    }
    output_identity = canonical_sha256(identity)
    record["output_identity_sha256"] = output_identity
    record["score_arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "array_sha256": _array_sha256(value),
        }
        for name, value in sorted(stored_arrays.items())
    }
    temporary_npz = Path(output_root) / f".{output_identity}.{uuid.uuid4().hex}.npz"
    temporary_npz.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(temporary_npz, **stored_arrays)
        destination = _write_atomic_directory(
            Path(output_root) / output_identity,
            files={
                "record.json": canonical_json_bytes(record) + b"\n",
                "scores.npz": temporary_npz.read_bytes(),
            },
        )
    finally:
        temporary_npz.unlink(missing_ok=True)
    verified = verify_pair_artifact(destination)["record"]
    if any(
        verified["score_arrays"][name]["array_sha256"]
        != source_record["score_arrays"][name]["array_sha256"]
        for name in verified["score_arrays"]
    ):
        raise ValueError("pair recovery changed a stored score array")
    return destination


def collect_production_results(
    *, plan: Mapping[str, Any], pair_artifacts: Sequence[str | Path]
) -> dict[str, Any]:
    validated = validate_production_plan(plan)
    records = [verify_pair_artifact(path)["record"] for path in pair_artifacts]
    if len(records) != 10:
        raise ValueError("production collection requires exactly ten seed-pair records")
    identities = {(row["cell_id"], int(row["training_seed"])) for row in records}
    expected_identities = {
        (cell, seed)
        for cell in (LARGE_CELL, SMALL_CELL)
        for seed in range(5)
    }
    if identities != expected_identities:
        raise ValueError("production seed-pair coverage mismatch")
    observed_run_ids = sorted(
        run_id for row in records for run_id in row["source_run_ids"].values()
    )
    expected_run_ids = sorted(str(row["run_id"]) for row in validated["records"])
    if observed_run_ids != expected_run_ids:
        raise ValueError("production run coverage mismatch")
    guardrails: dict[str, Any] = {}
    guardrail_status: dict[str, str] = {}
    for cell in (LARGE_CELL, SMALL_CELL):
        cell_rows = sorted(
            (row for row in records if row["cell_id"] == cell),
            key=lambda row: int(row["training_seed"]),
        )
        deltas = {
            metric: [
                float(row["id_utility"]["coupled"][metric])
                - float(row["id_utility"]["decoupled"][metric])
                for row in cell_rows
            ]
            for metric in ("accuracy", "nll", "ece")
        }
        verdict = adjudicate_id_equivalence(
            deltas,
            evidence_scope="protected_id_test",
            protected_id_test_available=True,
        )
        guardrails[cell] = {"paired_deltas": deltas, "verdict": verdict}
        guardrail_status[cell] = "PASS" if verdict["status"] == "PASS" else "FAILED"
    gate = adjudicate_resnet18_full_gate(
        expected_run_ids=expected_run_ids,
        observed_run_ids=observed_run_ids,
        seed_records=records,
        id_guardrail_by_cell=guardrail_status,
    )
    seed_summaries: dict[str, Any] = {}
    for cell in (LARGE_CELL, SMALL_CELL):
        seed_summaries[cell] = {}
        cell_rows = [row for row in records if row["cell_id"] == cell]
        for group in ("near", "far"):
            seed_summaries[cell][group] = {}
            for target in EVALUATION_TARGETS:
                seed_summaries[cell][group][target] = {}
                metric_names = (
                    "coupled_auroc",
                    "decoupled_auroc",
                    "gain",
                    "loss",
                    "pair_order_churn",
                    "delta_auroc",
                )
                for metric in metric_names:
                    seed_summaries[cell][group][target][metric] = paired_t_interval(
                        [float(row["macro"][group][target][metric]) for row in cell_rows]
                    )
            for metric in ("phi_rmd", "phi_marginal"):
                seed_summaries[cell][group]["md"][metric] = paired_t_interval(
                    [float(row["macro"][group]["md"][metric]) for row in cell_rows]
                )
    payload = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_verdict": gate["verdict"],
        "plan_sha256": validated["plan_sha256"],
        "source_training_terminal_sha256": SOURCE_TRAINING_TERMINAL_SHA256,
        "observed_run_ids": observed_run_ids,
        "pair_record_count": len(records),
        "seed_records": records,
        "id_guardrails": guardrails,
        "id_guardrail_by_cell": guardrail_status,
        "seed_first_summaries": seed_summaries,
        "gate": gate,
    }
    payload["terminal_sha256"] = canonical_sha256(payload)
    return payload
