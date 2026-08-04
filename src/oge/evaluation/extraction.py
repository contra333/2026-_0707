"""Deterministic checkpoint-to-raw-feature artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from oge.data.imglist_dataset import ImglistDataset
from oge.data.openood_cifar10 import load_dataset_config, resolve_imglist_path
from oge.data.transforms import make_cifar10_eval_transform
from oge.models import make_model
from oge.studies.hashing import scientific_config_hash
from oge.training import load_torch_artifact

from .common import DEFINITION_VERSION

RAW_FEATURE_SCHEMA_VERSION = "1.0"
PROTECTED_SPLITS = {"id_test", "cifar100", "tin", "mnist", "svhn", "texture", "places365"}
FORBIDDEN_SPLITS = {"id_test_openood", "ood_validation_tin"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repository_root: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def ordered_sample_id_digest(sample_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids.astype(str):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    expected_role: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any], str]:
    path = Path(checkpoint_path)
    checkpoint_sha = sha256_file(path)
    payload = load_torch_artifact(path, map_location="cpu")
    required = {
        "checkpoint_type",
        "completed_epoch",
        "model_state",
        "resolved_config",
        "oge_git_sha",
        "run_id",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"evaluation checkpoint is missing fields: {sorted(missing)}")
    role = str(payload["checkpoint_type"])
    if role not in {"last", "best_val"}:
        raise ValueError("Metric Contract v1.2 evaluates only last or best_val checkpoints")
    if expected_role is not None and role != expected_role:
        raise ValueError(f"checkpoint role is {role!r}, expected {expected_role!r}")
    resolved_config = payload["resolved_config"]
    if not isinstance(resolved_config, dict):
        raise ValueError("checkpoint resolved_config must be a mapping")
    model = make_model(resolved_config["model"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, payload, checkpoint_sha


def build_extraction_loader(
    dataset_config: Mapping[str, Any],
    *,
    dataset_key: str,
    data_root: str | Path,
    config_root: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_samples: int | None = None,
    smoke_samples_per_class: int | None = None,
    protected_authorization: str | None = None,
) -> tuple[DataLoader, list[str], Mapping[str, Any]]:
    if dataset_key in FORBIDDEN_SPLITS:
        raise ValueError(f"{dataset_key} is compatibility-only and cannot be extracted")
    if dataset_key in PROTECTED_SPLITS and not protected_authorization:
        raise PermissionError(f"{dataset_key} requires explicit protected-split authorization")
    datasets = dataset_config.get("datasets")
    if not isinstance(datasets, dict) or dataset_key not in datasets:
        raise ValueError(f"dataset config does not define {dataset_key!r}")
    item = datasets[dataset_key]
    root = Path(data_root)
    imglist_path = resolve_imglist_path(item, data_root=root, config_root=config_root)
    expected_sha256 = item.get("expected_sha256")
    if expected_sha256 is not None and sha256_file(imglist_path) != expected_sha256:
        raise ValueError(f"{dataset_key} imglist SHA256 does not match the frozen config")
    dataset = ImglistDataset(
        dataset_name=item["dataset_name"],
        split=item["split"],
        is_id=bool(item["is_id"]),
        imglist_path=imglist_path,
        data_root=root / dataset_config.get("image_root", ""),
        transform=make_cifar10_eval_transform(),
    )
    expected_count = item.get("expected_count")
    if expected_count is not None and len(dataset) != int(expected_count):
        raise ValueError(f"{dataset_key} imglist count does not match the frozen config")
    if max_samples is not None and smoke_samples_per_class is not None:
        raise ValueError("max_samples and smoke_samples_per_class are mutually exclusive")
    smoke_selection: dict[str, Any] | None = None
    if smoke_samples_per_class is not None:
        if smoke_samples_per_class <= 0 or not bool(item["is_id"]):
            raise ValueError("smoke_samples_per_class requires a positive ID-only selection")
        by_class: dict[int, list[int]] = {}
        for index, (_, label) in enumerate(dataset.entries):
            by_class.setdefault(label, []).append(index)
        expected_classes = list(range(10))
        if sorted(by_class) != expected_classes:
            raise ValueError("class-balanced smoke requires exactly CIFAR-10 labels 0..9")
        if any(len(by_class[label]) < smoke_samples_per_class for label in expected_classes):
            raise ValueError("class-balanced smoke request exceeds an available class count")
        selected_indices = sorted(
            index
            for label in expected_classes
            for index in by_class[label][:smoke_samples_per_class]
        )
        dataset = Subset(dataset, selected_indices)
        smoke_selection = {
            "policy": "first_n_per_class_then_original_imglist_order",
            "samples_per_class": smoke_samples_per_class,
            "class_count": 10,
            "sample_count": len(selected_indices),
        }
    elif max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        smoke_selection = {
            "policy": "first_n_in_original_imglist_order",
            "sample_count": len(dataset),
        }
    expected_ids = []
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base_dataset))
    for index in indices:
        relative_path, _ = base_dataset.entries[index]
        expected_ids.append(f"{base_dataset.dataset_name}:{relative_path}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    item_record = dict(item)
    item_record["smoke_selection"] = smoke_selection
    item_record["observed_imglist_sha256"] = sha256_file(imglist_path)
    item_record["observed_imglist_count"] = len(base_dataset)
    return loader, expected_ids, item_record


def extract_raw_arrays(
    model: nn.Module,
    loader: DataLoader,
    *,
    expected_sample_ids: list[str],
    device: str | torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_device = torch.device(device)
    model.eval()
    output: dict[str, list[Any]] = {
        "features": [],
        "logits": [],
        "class_labels": [],
        "predictions": [],
        "is_id": [],
        "sample_ids": [],
    }
    logits_parity_checked = False
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(target_device)
            logits, features = model(images, return_features=True)
            if not logits_parity_checked:
                ordinary_logits = model(images)
                torch.testing.assert_close(logits, ordinary_logits, rtol=0.0, atol=0.0)
                logits_parity_checked = True
            if features.ndim != 2 or features.shape[1] != int(model.feature_dim):
                raise ValueError("extracted features do not match model.feature_dim")
            if model.classifier.in_features != features.shape[1]:
                raise ValueError("classifier input width does not match feature width")
            if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
                raise ValueError("extraction produced non-finite feature or logit")
            prediction = torch.argmax(logits, dim=1)
            output["features"].append(features.detach().cpu().to(torch.float32).numpy())
            output["logits"].append(logits.detach().cpu().to(torch.float32).numpy())
            output["class_labels"].append(batch["class_label"].cpu().numpy())
            output["predictions"].append(prediction.cpu().numpy())
            output["is_id"].append(batch["is_id"].cpu().numpy())
            output["sample_ids"].extend(batch["sample_id"])
    if not logits_parity_checked:
        raise ValueError("extraction loader was empty")
    arrays = {
        "features": np.concatenate(output["features"]).astype(np.float32, copy=False),
        "logits": np.concatenate(output["logits"]).astype(np.float32, copy=False),
        "class_labels": np.concatenate(output["class_labels"]).astype(np.int64, copy=False),
        "predictions": np.concatenate(output["predictions"]).astype(np.int64, copy=False),
        "is_id": np.concatenate(output["is_id"]).astype(np.bool_, copy=False),
        "sample_ids": np.asarray(output["sample_ids"], dtype=str),
    }
    if arrays["sample_ids"].tolist() != expected_sample_ids:
        raise ValueError("observed sample order does not match configured imglist order")
    if len(set(expected_sample_ids)) != len(expected_sample_ids):
        raise ValueError("configured extraction split contains duplicate sample IDs")
    return arrays, {
        "ordinary_forward_logits_parity_checked": True,
        "parity_scope": "first_nonempty_batch",
        "model_eval": True,
        "inference_mode": True,
        "shuffle": False,
        "drop_last": False,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    rows = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_raw_feature_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    observed: dict[str, str] = {}
    for row in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = row.split("  ", 1)
        target = root / relative
        actual = sha256_file(target)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {relative}")
        observed[relative] = actual
    return {"manifest": manifest, "verified_files": observed}


def write_raw_feature_artifact(
    *,
    artifact_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: str,
    model: nn.Module,
    split_arrays: Mapping[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any], Mapping[str, Any]]],
    dataset_config: Mapping[str, Any],
    dataset_config_path: str | Path,
    extraction_command: str,
    device: str,
    protected_authorization: str | None,
    smoke_only: bool,
    repository_root: str | Path,
) -> Path:
    destination = Path(artifact_root) / checkpoint_sha256
    if destination.exists():
        raise FileExistsError(f"completed checkpoint artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{checkpoint_sha256}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        classifier_weight = model.classifier.weight.detach().cpu().to(torch.float32).numpy()
        np.save(temporary / "classifier_weight.npy", classifier_weight, allow_pickle=False)
        classifier_bias = model.classifier.bias
        bias_present = classifier_bias is not None
        if bias_present:
            np.save(
                temporary / "classifier_bias.npy",
                classifier_bias.detach().cpu().to(torch.float32).numpy(),
                allow_pickle=False,
            )
        split_manifest: dict[str, Any] = {}
        for dataset_key, (arrays, extraction_diagnostics, dataset_item) in split_arrays.items():
            split_dir = temporary / str(dataset_item["dataset_name"]) / str(dataset_item["split"])
            split_dir.mkdir(parents=True)
            for array_name, array in arrays.items():
                np.save(split_dir / f"{array_name}.npy", array, allow_pickle=False)
            split_manifest[dataset_key] = {
                "dataset_name": dataset_item["dataset_name"],
                "physical_split": dataset_item["split"],
                "is_id": bool(dataset_item["is_id"]),
                "group": dataset_item["group"],
                "sample_count": int(len(arrays["sample_ids"])),
                "ordered_sample_id_sha256": ordered_sample_id_digest(arrays["sample_ids"]),
                "expected_imglist_sha256": dataset_item.get("expected_sha256"),
                "observed_imglist_sha256": dataset_item["observed_imglist_sha256"],
                "observed_imglist_count": int(dataset_item["observed_imglist_count"]),
                "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
                "relative_directory": split_dir.relative_to(temporary).as_posix(),
                "extraction_diagnostics": dict(extraction_diagnostics),
                "smoke_selection": dataset_item.get("smoke_selection"),
            }
        resolved_config = checkpoint_payload["resolved_config"]
        oge_git_sha, git_dirty = _git_state(Path(repository_root))
        manifest = {
            "schema_version": RAW_FEATURE_SCHEMA_VERSION,
            "definition_version": DEFINITION_VERSION,
            "artifact_role": "raw_checkpoint_feature_cache",
            "checkpoint": {
                "path": str(Path(checkpoint_path).resolve()),
                "sha256": checkpoint_sha256,
                "completed_epoch": int(checkpoint_payload["completed_epoch"]),
                "role": checkpoint_payload["checkpoint_type"],
                "training_run_id": checkpoint_payload["run_id"],
                "training_git_sha": checkpoint_payload["oge_git_sha"],
                "training_seed": int(resolved_config["training"]["seed"]),
                "config_hash": scientific_config_hash(resolved_config),
            },
            "model": {
                "name": resolved_config["model"]["name"],
                "feature_endpoint": "model(x, return_features=True)",
                "feature_dim": int(model.feature_dim),
                "class_count": int(model.classifier.out_features),
                "classifier_bias_present": bias_present,
            },
            "evaluation": {
                "oge_git_sha": oge_git_sha,
                "git_dirty": git_dirty,
                "device": device,
                "command": extraction_command,
                "extraction_dtype": "float32",
                "protected_split_authorization": protected_authorization,
                "smoke_only": bool(smoke_only),
            },
            "dataset": {
                "protocol": dataset_config["protocol_name"],
                "config_path": str(Path(dataset_config_path)),
                "membership_manifest_sha256": dataset_config["membership_manifest"]["sha256"],
                "splits": split_manifest,
            },
        }
        _json_dump(temporary / "manifest.json", manifest)
        _write_checksums(temporary)
        verify_raw_feature_artifact(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def extract_checkpoint_artifact(
    *,
    checkpoint_path: str | Path,
    dataset_config_path: str | Path,
    data_root: str | Path,
    artifact_root: str | Path,
    dataset_keys: list[str],
    device: str,
    batch_size: int = 128,
    num_workers: int = 0,
    max_samples: int | None = None,
    smoke_samples_per_class: int | None = None,
    protected_authorization: str | None = None,
    smoke_only: bool = False,
    extraction_command: str = "",
    repository_root: str | Path = ".",
) -> Path:
    model, checkpoint, checkpoint_sha = load_checkpoint_model(
        checkpoint_path, device=device
    )
    dataset_config = load_dataset_config(dataset_config_path)
    membership_manifest = dataset_config.get("membership_manifest")
    if not isinstance(membership_manifest, dict):
        raise ValueError("dataset config must define membership_manifest")
    if membership_manifest.get("location", "dataset_config") != "dataset_config":
        raise ValueError("metric-contract membership manifest must be config-owned")
    membership_path = Path(dataset_config_path).parent / membership_manifest["path"]
    if sha256_file(membership_path) != membership_manifest["sha256"]:
        raise ValueError("membership manifest SHA256 does not match the frozen config")
    with membership_path.open("r", encoding="utf-8") as handle:
        membership_rows = sum(1 for _ in handle)
    if membership_rows != int(membership_manifest["row_count"]):
        raise ValueError("membership manifest row count does not match the frozen config")
    split_arrays = {}
    for dataset_key in dataset_keys:
        loader, expected_ids, item = build_extraction_loader(
            dataset_config,
            dataset_key=dataset_key,
            data_root=data_root,
            config_root=Path(dataset_config_path).parent,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
            max_samples=max_samples,
            smoke_samples_per_class=smoke_samples_per_class,
            protected_authorization=protected_authorization,
        )
        arrays, diagnostics = extract_raw_arrays(
            model, loader, expected_sample_ids=expected_ids, device=device
        )
        split_arrays[dataset_key] = (arrays, diagnostics, item)
    return write_raw_feature_artifact(
        artifact_root=artifact_root,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        model=model,
        split_arrays=split_arrays,
        dataset_config=dataset_config,
        dataset_config_path=dataset_config_path,
        extraction_command=extraction_command,
        device=device,
        protected_authorization=protected_authorization,
        smoke_only=smoke_only,
        repository_root=repository_root,
    )
