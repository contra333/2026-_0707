import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from oge.evaluation.extraction import (
    build_extraction_loader,
    extract_checkpoint_artifact,
    load_checkpoint_model,
    ordered_sample_id_digest,
    sha256_file,
    verify_raw_feature_artifact,
)
from oge.evaluation.records import (
    ResultIdentity,
    metric_result_record,
    write_metric_bundle,
)
from oge.evaluation.registry import METRIC_REGISTRY, registry_by_artifact_key
from oge.models import make_model


def _resolved_config():
    membership = {
        "train": {"sha256": "train", "line_count": 2, "selection_key": "id_train"},
        "validation": {"sha256": "val", "line_count": 1, "selection_key": "id_validation"},
        "test": {"sha256": "test", "line_count": 1, "selection_key": "id_test"},
    }
    return {
        "optimizer": {"name": "sgd", "lr": 0.1},
        "model": {"name": "toy_cifar_cnn", "num_classes": 10, "feature_dim": 8},
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "dataset": {
            "protocol": "oge_cifar10_holdout_v1",
            "train_split": "id_train",
            "validation_split": "id_validation",
            "test_split": "id_test",
            "membership": membership,
            "membership_manifest": {"sha256": "manifest", "row_count": 4},
        },
        "scheduler": {"name": "none"},
        "training": {"seed": 3, "batch_size": 2, "max_epochs": 1},
    }


def _dataset_fixture(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    image_root = data_root / "images_classic"
    config_root = tmp_path / "configs"
    config_root.mkdir()
    entries = {
        "id_train": [("cifar10/train/a.png", 0), ("cifar10/train/b.png", 1)],
        "id_validation": [("cifar10/train/c.png", 0)],
        "id_test": [("cifar10/test/d.png", 1)],
        "id_test_openood": [("cifar10/test/d.png", 1)],
    }
    datasets = {}
    for key, rows in entries.items():
        image_list = config_root / f"{key}.txt"
        image_list.write_text("".join(f"{path} {label}\n" for path, label in rows))
        for relative, _ in rows:
            image_path = image_root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(len(relative), 10, 20)).save(image_path)
        datasets[key] = {
            "dataset_name": "cifar10",
            "split": key,
            "is_id": True,
            "group": "id",
            "imglist_location": "dataset_config",
            "imglist": image_list.name,
            "expected_count": len(rows),
            "expected_sha256": sha256_file(image_list),
        }
    datasets["id_test"]["role"] = "metric_contract_primary"
    datasets["id_test_openood"]["role"] = "compatibility_only"
    membership_path = config_root / "split_manifest.jsonl"
    membership_path.write_text('{"record_type":"header"}\n')
    config = {
        "protocol_name": "oge_cifar10_holdout_v1",
        "dataset_class": "imglist",
        "image_root": "images_classic",
        "membership_manifest": {
            "location": "dataset_config",
            "path": membership_path.name,
            "sha256": sha256_file(membership_path),
            "row_count": 1,
        },
        "datasets": datasets,
    }
    config_path = config_root / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_extraction_writes_card08_arrays_manifest_and_checksums(tmp_path):
    dataset_config = _dataset_fixture(tmp_path)
    model = make_model(_resolved_config()["model"])
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "checkpoint_type": "last",
            "completed_epoch": 1,
            "model_state": model.state_dict(),
            "resolved_config": _resolved_config(),
            "oge_git_sha": "training-sha",
            "run_id": "run-id",
        },
        checkpoint,
    )

    artifact = extract_checkpoint_artifact(
        checkpoint_path=checkpoint,
        dataset_config_path=dataset_config,
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        dataset_keys=["id_train", "id_validation"],
        device="cpu",
        batch_size=2,
        smoke_only=True,
        extraction_command="fixture",
        repository_root=Path.cwd(),
    )
    verified = verify_raw_feature_artifact(artifact)
    manifest = verified["manifest"]

    assert manifest["checkpoint"]["role"] == "last"
    assert manifest["checkpoint"]["training_seed"] == 3
    assert manifest["evaluation"]["smoke_only"] is True
    assert set(manifest["dataset"]["splits"]) == {"id_train", "id_validation"}
    train_dir = artifact / "cifar10/id_train"
    assert np.load(train_dir / "features.npy", allow_pickle=False).dtype == np.float32
    assert np.load(train_dir / "logits.npy", allow_pickle=False).dtype == np.float32
    assert np.load(train_dir / "class_labels.npy", allow_pickle=False).dtype == np.int64
    sample_ids = np.load(train_dir / "sample_ids.npy", allow_pickle=False)
    assert sample_ids.tolist() == ["cifar10:cifar10/train/a.png", "cifar10:cifar10/train/b.png"]
    assert manifest["dataset"]["splits"]["id_train"]["ordered_sample_id_sha256"] == ordered_sample_id_digest(sample_ids)
    assert (
        manifest["dataset"]["splits"]["id_train"]["observed_imglist_sha256"]
        == manifest["dataset"]["splits"]["id_train"]["expected_imglist_sha256"]
    )
    assert manifest["dataset"]["splits"]["id_train"]["observed_imglist_count"] == 2
    assert "manifest.json" in verified["verified_files"]


def test_last_and_best_val_have_separate_checkpoint_identity_and_role_validation(tmp_path):
    model = make_model(_resolved_config()["model"])
    common = {
        "completed_epoch": 1,
        "model_state": model.state_dict(),
        "resolved_config": _resolved_config(),
        "oge_git_sha": "training-sha",
        "run_id": "run-id",
    }
    last_path = tmp_path / "last.pt"
    best_path = tmp_path / "best_val.pt"
    torch.save({**common, "checkpoint_type": "last"}, last_path)
    torch.save({**common, "checkpoint_type": "best_val"}, best_path)

    _, last_payload, last_sha = load_checkpoint_model(last_path, expected_role="last")
    _, best_payload, best_sha = load_checkpoint_model(best_path, expected_role="best_val")

    assert last_payload["checkpoint_type"] == "last"
    assert best_payload["checkpoint_type"] == "best_val"
    assert last_sha != best_sha
    with pytest.raises(ValueError, match="expected"):
        load_checkpoint_model(best_path, expected_role="last")


def test_protected_and_compatibility_splits_are_refused_without_contract_authorization(tmp_path):
    dataset_config_path = _dataset_fixture(tmp_path)
    config = yaml.safe_load(dataset_config_path.read_text())

    with pytest.raises(PermissionError, match="authorization"):
        build_extraction_loader(
            config,
            dataset_key="id_test",
            data_root=tmp_path / "data",
            config_root=dataset_config_path.parent,
        )
    with pytest.raises(ValueError, match="compatibility-only"):
        build_extraction_loader(
            config,
            dataset_key="id_test_openood",
            data_root=tmp_path / "data",
            config_root=dataset_config_path.parent,
            protected_authorization="Issue #fixture",
        )


def test_class_balanced_smoke_selector_preserves_original_imglist_order(tmp_path):
    config_root = tmp_path / "configs"
    config_root.mkdir()
    image_list = config_root / "train.txt"
    rows = [
        (f"cifar10/train/class{label}_{copy}.png", label)
        for label in range(10)
        for copy in range(2)
    ]
    image_list.write_text("".join(f"{path} {label}\n" for path, label in rows))
    config = {
        "image_root": "images_classic",
        "datasets": {
            "id_train": {
                "dataset_name": "cifar10",
                "split": "train",
                "is_id": True,
                "group": "id",
                "imglist_location": "dataset_config",
                "imglist": image_list.name,
                "expected_count": 20,
                "expected_sha256": sha256_file(image_list),
            }
        },
    }

    loader, sample_ids, item = build_extraction_loader(
        config,
        dataset_key="id_train",
        data_root=tmp_path / "data",
        config_root=config_root,
        smoke_samples_per_class=1,
    )

    assert len(loader.dataset) == 10
    assert loader.dataset.indices == list(range(0, 20, 2))
    assert sample_ids == [f"cifar10:{rows[index][0]}" for index in range(0, 20, 2)]
    assert item["smoke_selection"] == {
        "policy": "first_n_per_class_then_original_imglist_order",
        "samples_per_class": 1,
        "class_count": 10,
        "sample_count": 10,
    }
def test_registry_has_unique_keys_frozen_formula_coverage_and_only_contract_tiers():
    by_key = registry_by_artifact_key()
    formula_ids = {item.formula_id for item in METRIC_REGISTRY}

    assert formula_ids == {
        "ID-1", "ID-2", "ID-3", "ID-4", "ID-5", "ID-6", "ID-7",
        "LOGIT-1", "LOGIT-2", "LOGIT-3", "LOGIT-X",
        "GAUSS-0", "GAUSS-1", "GAUSS-2", "GAUSS-3", "GAUSS-4", "GAUSS-5", "GAUSS-6",
        "SUB-1", "SUB-2", "SUB-3", "SUB-4", "SUB-5", "SUB-6",
        "OOD-1", "OOD-2", "OOD-3", "OOD-4", "OOD-5", "OOD-6", "OOD-7",
        "GEO-1", "GEO-2", "GEO-3", "GEO-4", "GEO-5", "GEO-6", "GEO-7", "GEO-8", "GEO-9",
        "GEO-10", "GEO-11", "GEO-12", "GEO-13", "GEO-14", "GEO-15", "GEO-16", "GEO-17",
    }
    assert len(by_key) == len(set(by_key))
    assert {item.reporting_tier for item in METRIC_REGISTRY} <= {
        "primary", "control", "appendix", "exploratory", "excluded"
    }


def test_metric_result_and_bundle_store_identity_status_tolerance_and_intermediate(tmp_path):
    definition = registry_by_artifact_key()["geometry/nc1_pinv"]
    identity = ResultIdentity(
        checkpoint_role="last",
        checkpoint_sha256="a" * 64,
        completed_epoch=200,
        training_seed=1,
        config_hash="b" * 64,
        dataset_protocol="oge_cifar10_holdout_v1",
        membership_manifest_sha256="c" * 64,
        evaluation_git_sha="d" * 40,
        smoke_only=True,
    )
    record = metric_result_record(
        definition=definition,
        identity=identity,
        value=0.25,
        status="success",
        reason_codes=[],
        fit_split="id_train",
        query_split="id_train",
        transform="raw",
        sample_count=6,
        class_counts=[2, 2, 2],
        source_commit="7cab4a59",
        runtime_diagnostics={"retained_rank": 2},
    )
    path = write_metric_bundle(
        output_root=tmp_path,
        identity=identity,
        records=[record],
        intermediates={"geometry/nc1_pinv": {"eigenvalues": np.array([2.0, 1.0])}},
    )

    payload = json.loads((path / "metrics.json").read_text())
    assert payload["records"][0]["definition_version"] == "metric_contract_v1.2"
    assert payload["records"][0]["checkpoint_role"] == "last"
    assert payload["records"][0]["tolerance"] == {"atol": 1e-12, "rtol": 1e-10}
    assert (path / "intermediates/geometry__nc1_pinv.npz").is_file()
    assert "metrics.json" in (path / "checksums.sha256").read_text()
