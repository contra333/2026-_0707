"""Endpoint-only ID feature artifacts for the ResNet-18 replication."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from oge.models import ResNet18, make_model
from oge.studies.artifacts import sha256_file
from oge.studies.hashing import canonical_json_bytes, canonical_sha256
from oge.training import load_torch_artifact
from oge.training.resnet18_replication_provenance import (
    validate_resnet18_replication_checkpoint_payload,
    validate_resnet18_replication_checkpoint_provenance,
)
from oge.training.resnet18_replication_plan import RESNET18_REPLICATION_STUDY_ID

from .task_f import collect_runtime_provenance, ordered_sample_id_sha256


RESNET18_REPLICATION_ARTIFACT_SCHEMA_VERSION = (
    "resnet18_cifar10_id_feature_artifact_v2"
)
RESNET18_REPLICATION_ARTIFACT_SPECIFICATION_VERSION = (
    "resnet18_cifar10_id_feature_specification_v2"
)
RESNET18_REPLICATION_ID_SPLITS = (
    "id_train",
    "id_validation",
    "synthetic_id_fixture",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN = re.compile(
    r"(?:^|[/_.-])(?:ood|openood|id[_-]?test|nearood|farood)(?:$|[/_.-])",
    re.IGNORECASE,
)
_FILES = (
    "classifier_bias.npy",
    "classifier_weight.npy",
    "features.npy",
    "logits.npy",
    "manifest.json",
    "sample_ids.npy",
)


def _reject_protected(value: str | Path, label: str) -> None:
    if _FORBIDDEN.search(str(value).replace("\\", "/")):
        raise ValueError(f"{label} contains a protected reference")


def _validate_split(value: str, *, execution_only: bool) -> str:
    if value not in RESNET18_REPLICATION_ID_SPLITS:
        raise ValueError("replication export split must be ID-train or ID-validation")
    if value == "synthetic_id_fixture" and not execution_only:
        raise ValueError("synthetic_id_fixture requires execution_only=true")
    return value


def resnet18_replication_specification_payload() -> dict[str, Any]:
    return {
        "specification_version": RESNET18_REPLICATION_ARTIFACT_SPECIFICATION_VERSION,
        "artifact_schema_version": RESNET18_REPLICATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_namespace": RESNET18_REPLICATION_STUDY_ID,
        "model": {
            "name": "resnet18",
            "variant": "cifar",
            "num_classes": 10,
            "feature_dim": 512,
            "classifier_weight_shape": [10, 512],
        },
        "depth_tap": "penultimate",
        "allowed_splits": list(RESNET18_REPLICATION_ID_SPLITS),
        "files": list(_FILES),
        "feature_dtype": "float32",
        "logit_dtype": "float32",
        "publication": "temporary-directory-then-rename-no-overwrite",
    }


def load_resnet18_replication_id_input(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    _reject_protected(path, "input NPZ path")
    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) != {"images", "sample_ids", "is_id"}:
            raise ValueError("replication input NPZ fields are invalid")
        images = np.asarray(payload["images"])
        sample_ids = np.asarray(payload["sample_ids"])
        is_id = np.asarray(payload["is_id"])
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("replication images must be float32 [N,3,H,W]")
    if images.shape[0] == 0 or not np.isfinite(images).all():
        raise ValueError("replication images must be non-empty and finite")
    if is_id.dtype != np.bool_ or is_id.shape != (images.shape[0],) or not is_id.all():
        raise ValueError("replication input must explicitly mark every sample as ID")
    if sample_ids.shape != (images.shape[0],) or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("replication sample_ids must be a rank-1 string array")
    sample_ids = sample_ids.astype(str, copy=False)
    ordered_sample_id_sha256(sample_ids)
    return images, sample_ids


def load_resnet18_replication_checkpoint(
    path: str | Path, *, device: str | torch.device
) -> tuple[ResNet18, dict[str, Any], str]:
    _reject_protected(path, "checkpoint path")
    payload = load_torch_artifact(path, map_location="cpu")
    provenance = validate_resnet18_replication_checkpoint_payload(payload)
    model = make_model(provenance["model_config"])
    if not isinstance(model, ResNet18):
        raise ValueError("replication checkpoint must construct ResNet18")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, provenance, sha256_file(path)


def extract_resnet18_replication_outputs(
    model: ResNet18,
    images: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if model.feature_dim != 512 or tuple(model.classifier.weight.shape) != (10, 512):
        raise ValueError("replication model must expose 512D features and a (10,512) head")
    target = torch.device(device)
    feature_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(target)
            logits, features = model(batch, return_features=True)
            ordinary_logits = model(batch)
            torch.testing.assert_close(logits, ordinary_logits, rtol=0.0, atol=0.0)
            if features.ndim != 2 or features.shape[1] != 512:
                raise ValueError("replication model returned a non-512D penultimate feature")
            if logits.ndim != 2 or logits.shape[1] != 10:
                raise ValueError("replication model returned invalid logits")
            feature_chunks.append(features.detach().cpu().to(torch.float32).numpy())
            logit_chunks.append(logits.detach().cpu().to(torch.float32).numpy())
            if progress_callback is not None:
                progress_callback(min(start + batch_size, images.shape[0]), images.shape[0])
    features = np.concatenate(feature_chunks).astype(np.float32, copy=False)
    logits = np.concatenate(logit_chunks).astype(np.float32, copy=False)
    weight = model.classifier.weight.detach().cpu().to(torch.float32).numpy().copy()
    bias = model.classifier.bias.detach().cpu().to(torch.float32).numpy().copy()
    return features, logits, weight, bias


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def write_resnet18_replication_artifact(
    *,
    artifact_root: str | Path,
    features: np.ndarray,
    logits: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    sample_ids: Sequence[str] | np.ndarray,
    checkpoint_sha256: str,
    checkpoint_provenance: Mapping[str, Any],
    dataset_split: str,
    runtime: Mapping[str, Any],
) -> Path:
    provenance = validate_resnet18_replication_checkpoint_provenance(
        checkpoint_provenance
    )
    if _SHA256.fullmatch(checkpoint_sha256) is None:
        raise ValueError("checkpoint_sha256 is invalid")
    dataset_split = _validate_split(
        dataset_split, execution_only=bool(provenance["execution_only"])
    )
    arrays = {
        "features.npy": np.asarray(features),
        "logits.npy": np.asarray(logits),
        "classifier_weight.npy": np.asarray(classifier_weight),
        "classifier_bias.npy": np.asarray(classifier_bias),
        "sample_ids.npy": np.asarray(sample_ids).astype(str),
    }
    n = arrays["features.npy"].shape[0]
    expected_shapes = {
        "features.npy": (n, 512),
        "logits.npy": (n, 10),
        "classifier_weight.npy": (10, 512),
        "classifier_bias.npy": (10,),
        "sample_ids.npy": (n,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} shape must be {expected}")
    for name in ("features.npy", "logits.npy", "classifier_weight.npy", "classifier_bias.npy"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite float32")
    sample_digest = ordered_sample_id_sha256(arrays["sample_ids.npy"])

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".resnet18-rep-{uuid.uuid4().hex}.tmp"
    temporary.mkdir(exist_ok=False)
    try:
        for name, array in arrays.items():
            _write_npy(temporary / name, array)
        serialization = {
            name: {
                "sha256": sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in arrays
        }
        manifest = {
            "schema_version": RESNET18_REPLICATION_ARTIFACT_SCHEMA_VERSION,
            "artifact_namespace": RESNET18_REPLICATION_STUDY_ID,
            "numerical_policy_id": provenance["numerical_policy_id"],
            "run_id": provenance["run_id"],
            "training_seed": provenance["training_seed"],
            "branch_policy": provenance["branch_policy"],
            "initial_lr": provenance["initial_lr"],
            "weight_decay": provenance["weight_decay"],
            "checkpoint_epoch": provenance["checkpoint_epoch"],
            "checkpoint_role": provenance["checkpoint_role"],
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_split": dataset_split,
            "depth_tap": "penultimate",
            "feature_shape": [n, 512],
            "logit_shape": [n, 10],
            "classifier_weight_shape": [10, 512],
            "classifier_bias_shape": [10],
            "ordered_sample_id_sha256": sample_digest,
            "initialization_sha256": provenance["initialization_sha256"],
            "data_stream_sha256": provenance["data_stream_sha256"],
            "sibling_group_id": provenance["sibling_group_id"],
            "cross_lr_pairing_block_id": provenance["cross_lr_pairing_block_id"],
            "specification_sha256": canonical_sha256(
                resnet18_replication_specification_payload()
            ),
            "serialization": serialization,
            "runtime": dict(runtime),
            "output_identity_sha256": "0" * 64,
        }
        identity_payload = dict(manifest)
        identity_payload.pop("output_identity_sha256")
        manifest["output_identity_sha256"] = canonical_sha256(identity_payload)
        _write_bytes(
            temporary / "manifest.json", canonical_json_bytes(manifest) + b"\n"
        )
        checksums = "".join(
            f"{sha256_file(temporary / name)}  {name}\n" for name in _FILES
        )
        _write_bytes(temporary / "checksums.sha256", checksums.encode("utf-8"))
        destination = root / manifest["output_identity_sha256"]
        if destination.exists():
            raise FileExistsError(f"replication artifact already exists: {destination}")
        os.rename(temporary, destination)
        verify_resnet18_replication_artifact(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_resnet18_replication_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != RESNET18_REPLICATION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported replication artifact schema")
    identity_payload = dict(manifest)
    output_identity = identity_payload.pop("output_identity_sha256")
    if canonical_sha256(identity_payload) != output_identity or root.name != output_identity:
        raise ValueError("replication artifact output identity mismatch")
    if manifest["specification_sha256"] != canonical_sha256(
        resnet18_replication_specification_payload()
    ):
        raise ValueError("replication artifact specification mismatch")
    checksum_rows = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    observed = {}
    for row in checksum_rows:
        digest, name = row.split("  ", 1)
        if name in observed or name not in _FILES or sha256_file(root / name) != digest:
            raise ValueError("replication artifact checksum mismatch")
        observed[name] = digest
    if set(observed) != set(_FILES):
        raise ValueError("replication artifact checksum coverage mismatch")
    arrays = {name: np.load(root / name, allow_pickle=False) for name in _FILES if name.endswith(".npy")}
    n = int(manifest["feature_shape"][0])
    if arrays["features.npy"].shape != (n, 512) or arrays["logits.npy"].shape != (n, 10):
        raise ValueError("replication artifact feature/logit shape mismatch")
    if arrays["classifier_weight.npy"].shape != (10, 512) or arrays["classifier_bias.npy"].shape != (10,):
        raise ValueError("replication artifact classifier shape mismatch")
    if ordered_sample_id_sha256(arrays["sample_ids.npy"]) != manifest[
        "ordered_sample_id_sha256"
    ]:
        raise ValueError("replication artifact sample order mismatch")
    for name, details in manifest["serialization"].items():
        if sha256_file(root / name) != details["sha256"] or (root / name).stat().st_size != details["bytes"]:
            raise ValueError("replication artifact serialization mismatch")
    return {"manifest": manifest, "verified_files": observed}


def export_resnet18_replication_from_files(
    *,
    checkpoint_path: str | Path,
    input_npz_path: str | Path,
    artifact_root: str | Path,
    dataset_split: str,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    _validate_split(dataset_split, execution_only=True)
    _reject_protected(checkpoint_path, "checkpoint path")
    _reject_protected(input_npz_path, "input NPZ path")
    runtime = collect_runtime_provenance(device)
    images, sample_ids = load_resnet18_replication_id_input(input_npz_path)
    model, provenance, checkpoint_sha256 = load_resnet18_replication_checkpoint(
        checkpoint_path, device=device
    )
    _validate_split(dataset_split, execution_only=bool(provenance["execution_only"]))
    features, logits, weight, bias = extract_resnet18_replication_outputs(
        model,
        images,
        device=device,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    return write_resnet18_replication_artifact(
        artifact_root=artifact_root,
        features=features,
        logits=logits,
        classifier_weight=weight,
        classifier_bias=bias,
        sample_ids=sample_ids,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_provenance=provenance,
        dataset_split=dataset_split,
        runtime=runtime,
    )
