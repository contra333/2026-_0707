import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from oge.evaluation.task_f_fresh import (
    BINDINGS_SCHEMA_VERSION,
    EXPECTED_SPECIFICATION_SHA256,
    build_id_input,
    build_fresh_evaluation_plan,
    summarize_export_coverage,
    validate_bound_inventory,
    verify_bridge_artifact,
    verify_id_input,
    write_bridge_artifact,
)
from oge.feature_export import (
    TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
    write_task_f_artifact,
)
from oge.training import generate_research_run_matrix
from oge.training.checkpoint import atomic_torch_save


def _plan():
    return generate_research_run_matrix(
        anchor_seeds=(0, 1, 2, 3, 4),
        adam_factorial_seeds=(0, 1, 2),
        sgdm_seeds=(0, 1, 2),
    )


def test_fresh_plan_has_exact_1320_id_only_records_and_no_pilot():
    result = build_fresh_evaluation_plan(_plan())
    assert result["task_f_specification_sha256"] == EXPECTED_SPECIFICATION_SHA256
    assert result["counts"] == {
        "research_runs": 50,
        "adam_runs": 41,
        "sgdm_runs": 9,
        "records_total": 1320,
        "records_by_split": {"id_train": 660, "id_validation": 660},
        "trajectory_per_split": 550,
        "depth_per_split": 60,
        "best_val_per_split": 50,
    }
    records = result["records"]
    assert len({record["record_id"] for record in records}) == 1320
    assert {record["dataset_split"] for record in records} == {
        "id_train",
        "id_validation",
    }
    assert not any("pilot" in record["run_id"] for record in records)
    assert sum(record["checkpoint_role"] == "best_val" for record in records) == 100
    assert sum(record["depth_tap"] != "penultimate" for record in records) == 120


def test_current_310_job_shape_is_detected_as_subset_with_1010_supplemental():
    plan = build_fresh_evaluation_plan(_plan())
    observed = []
    for record in plan["records"]:
        if record["dataset_split"] != "id_train":
            continue
        if record["checkpoint_role"] == "best_val":
            continue
        if record["depth_tap"] == "penultimate" and record["checkpoint_epoch"] not in {
            10,
            60,
            120,
            160,
            200,
        }:
            continue
        observed.append(record)
    assert len(observed) == 310
    coverage = summarize_export_coverage(plan, observed)
    assert coverage["status"] == "PASS"
    assert coverage["covered_record_count"] == 310
    assert coverage["supplemental_record_count"] == 1010
    missing_epochs = {
        key.split("__")[2]
        for key in coverage["missing_fixed_epoch_record_ids"]
        if "__snapshot__" in key
    }
    assert {"0000", "0001", "0030", "0061", "0121", "0161"}.issubset(
        missing_epochs
    )


def test_export_coverage_rejects_duplicates_and_protected_splits():
    plan = build_fresh_evaluation_plan(_plan())
    first = plan["records"][0]
    duplicate = summarize_export_coverage(plan, [first, first])
    assert duplicate["status"] == "FAILED"
    protected = dict(first, dataset_split="id_test")
    with pytest.raises(ValueError, match="protected"):
        summarize_export_coverage(plan, [protected])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_config(tmp_path: Path) -> tuple[Path, np.ndarray]:
    sample_ids = np.asarray(["cifar10:a.png", "cifar10:b.png"])
    imglist = tmp_path / "train.txt"
    imglist.write_text("a.png 0\nb.png 1\n", encoding="utf-8")
    validation_imglist = tmp_path / "validation.txt"
    validation_imglist.write_text("c.png 2\nd.png 3\n", encoding="utf-8")
    membership = tmp_path / "membership.jsonl"
    rows = [
        {"record_type": "header"},
        {
            "record_type": "membership",
            "sample_id": sample_ids[0],
            "label": 0,
            "role": "id_train",
        },
        {
            "record_type": "membership",
            "sample_id": sample_ids[1],
            "label": 1,
            "role": "id_train",
        },
        {
            "record_type": "membership",
            "sample_id": "cifar10:c.png",
            "label": 2,
            "role": "id_validation",
        },
        {
            "record_type": "membership",
            "sample_id": "cifar10:d.png",
            "label": 3,
            "role": "id_validation",
        },
    ]
    membership.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "protocol_name": "fixture",
        "dataset_class": "imglist",
        "image_root": "images",
        "membership_manifest": {
            "location": "dataset_config",
            "path": membership.name,
            "sha256": _sha(membership),
            "row_count": 5,
        },
        "datasets": {
            "id_train": {
                "dataset_name": "cifar10",
                "split": "train",
                "is_id": True,
                "group": "id",
                "imglist_location": "dataset_config",
                "imglist": imglist.name,
                "expected_count": 2,
                "expected_sha256": _sha(imglist),
            },
            "id_validation": {
                "dataset_name": "cifar10",
                "split": "validation",
                "is_id": True,
                "group": "id",
                "imglist_location": "dataset_config",
                "imglist": validation_imglist.name,
                "expected_count": 2,
                "expected_sha256": _sha(validation_imglist),
            }
        },
    }
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, sample_ids


def _members(run: dict) -> dict:
    members = copy.deepcopy(run["task_f_b_sibling_members"])
    for member in members.values():
        member["initialization_sha256"] = "1" * 64
        member["data_stream_sha256"] = "2" * 64
    return members


def _provenance(run: dict, *, epoch: int = 200, role: str = "last") -> dict:
    members = _members(run)
    current = members[run["task_f_b_sibling_role"]]
    return {
        "schema_version": TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "training_seed": run["training_seed"],
        "branch_policy": current["branch_policy"],
        "total_weight_decay": current["total_weight_decay"],
        "coupled_ratio": current["coupled_ratio"],
        "checkpoint_epoch": epoch,
        "checkpoint_role": role,
        "oge_git_sha": "a" * 40,
        "execution_only": False,
        "initialization_sha256": "1" * 64,
        "data_stream_sha256": "2" * 64,
        "sibling_group_id": run["sibling_group_id"],
        "sibling_role": run["task_f_b_sibling_role"],
        "sibling_members": members,
        "model_config": {
            "name": "wrn28_10",
            "depth": 28,
            "widen_factor": 10,
            "dropout_rate": 0.0,
            "num_classes": 10,
            "init_policy": "msr_fan_in",
        },
    }


def _runtime() -> dict:
    return {
        "python_version": "fixture",
        "python_implementation": "CPython",
        "numpy_version": "fixture",
        "pytorch_version": "fixture",
        "device_type": "cpu",
        "device": "cpu",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "cpu_count": 1,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "thread_environment": {
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "VECLIB_MAXIMUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
        "numpy_blas": {"available": True, "blas": {"name": "fixture"}},
        "accelerator": {
            "backend": "cpu",
            "local_device_index": None,
            "device_name": None,
            "device_uuid": None,
            "total_memory_bytes": None,
            "cuda_runtime_version": None,
            "cudnn_version": None,
            "cuda_visible_devices": None,
        },
    }


def _bridge_fixture(tmp_path: Path, *, tap: str = "penultimate"):
    plan = build_fresh_evaluation_plan(_plan())
    record = next(
        row
        for row in plan["records"]
        if row["dataset_split"] == "id_train"
        and row["checkpoint_role"] == "last"
        and row["depth_tap"] == tap
    )
    run = next(row for row in _plan()["runs"] if row["run_id"] == record["run_id"])
    provenance = _provenance(run)
    checkpoint = tmp_path / f"{tap}.pt"
    payload = {
        "checkpoint_type": "last",
        "completed_epoch": 200,
        "model_state": {
            "classifier.weight": torch.arange(6400, dtype=torch.float32).reshape(10, 640)
            / 6400,
            "classifier.bias": torch.arange(10, dtype=torch.float32) / 10,
        },
        "oge_git_sha": "a" * 40,
        "run_id": run["run_id"],
        "task_f_provenance": provenance,
    }
    atomic_torch_save(payload, checkpoint)
    dataset_config, sample_ids = _dataset_config(tmp_path)
    width = {"stage1": 160, "stage2": 320, "stage3": 640, "penultimate": 640}[tap]
    features = np.arange(2 * width, dtype=np.float32).reshape(2, width) / width
    artifact = write_task_f_artifact(
        artifact_root=tmp_path / "features",
        features=features,
        sample_ids=sample_ids,
        checkpoint_sha256=_sha(checkpoint),
        checkpoint_provenance=provenance,
        depth_tap=tap,
        dataset_split="id_train",
        runtime=_runtime(),
    )
    return record, checkpoint, artifact, dataset_config


def test_bridge_links_membership_checkpoint_classifier_and_features(tmp_path):
    record, checkpoint, artifact, dataset_config = _bridge_fixture(tmp_path)
    output = write_bridge_artifact(
        record=record,
        checkpoint_path=checkpoint,
        feature_artifact_path=artifact,
        dataset_config_path=dataset_config,
        output_root=tmp_path / "bridge",
    )
    verified = verify_bridge_artifact(output)
    assert verified["manifest"]["classifier"]["status"] == "READY"
    assert verified["manifest"]["labels_shape"] == [2]
    logits = np.load(output / "logits.npy", allow_pickle=False)
    assert logits.shape == (2, 10)
    assert np.array_equal(np.load(output / "labels.npy"), np.asarray([0, 1]))


def test_stage_bridge_records_classifier_as_not_applicable(tmp_path):
    record, checkpoint, artifact, dataset_config = _bridge_fixture(tmp_path, tap="stage1")
    output = write_bridge_artifact(
        record=record,
        checkpoint_path=checkpoint,
        feature_artifact_path=artifact,
        dataset_config_path=dataset_config,
        output_root=tmp_path / "bridge",
    )
    manifest = verify_bridge_artifact(output)["manifest"]
    assert manifest["classifier"] == {
        "status": "NOT_APPLICABLE",
        "reason": "classifier_reads_penultimate_only",
    }
    assert not (output / "logits.npy").exists()


def test_bridge_rejects_sample_order_and_checkpoint_identity_drift(tmp_path):
    record, checkpoint, artifact, dataset_config = _bridge_fixture(tmp_path)
    reversed_artifact = write_task_f_artifact(
        artifact_root=tmp_path / "reversed",
        features=np.zeros((2, 640), dtype=np.float32),
        sample_ids=np.asarray(["cifar10:b.png", "cifar10:a.png"]),
        checkpoint_sha256=_sha(checkpoint),
        checkpoint_provenance=load_provenance(checkpoint),
        depth_tap="penultimate",
        dataset_split="id_train",
        runtime=_runtime(),
    )
    with pytest.raises(ValueError, match="sample"):
        write_bridge_artifact(
            record=record,
            checkpoint_path=checkpoint,
            feature_artifact_path=reversed_artifact,
            dataset_config_path=dataset_config,
            output_root=tmp_path / "bridge",
        )
    changed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    changed["model_state"]["classifier.bias"][0] += 1
    atomic_torch_save(changed, checkpoint)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        write_bridge_artifact(
            record=record,
            checkpoint_path=checkpoint,
            feature_artifact_path=artifact,
            dataset_config_path=dataset_config,
            output_root=tmp_path / "bridge-two",
        )


def test_bridge_rejects_membership_label_sibling_and_feature_drift(tmp_path):
    record, checkpoint, artifact, dataset_config = _bridge_fixture(tmp_path)
    train_imglist = tmp_path / "train.txt"
    train_imglist.write_text("a.png 2\nb.png 1\n", encoding="utf-8")
    config = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    config["datasets"]["id_train"]["expected_sha256"] = _sha(train_imglist)
    dataset_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="membership label"):
        write_bridge_artifact(
            record=record,
            checkpoint_path=checkpoint,
            feature_artifact_path=artifact,
            dataset_config_path=dataset_config,
            output_root=tmp_path / "label-drift",
        )

    dataset_config, _ = _dataset_config(tmp_path)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sibling = next(iter(checkpoint_payload["task_f_provenance"]["sibling_members"].values()))
    sibling["initialization_sha256"] = "f" * 64
    atomic_torch_save(checkpoint_payload, checkpoint)
    with pytest.raises(ValueError, match="siblings"):
        write_bridge_artifact(
            record=record,
            checkpoint_path=checkpoint,
            feature_artifact_path=artifact,
            dataset_config_path=dataset_config,
            output_root=tmp_path / "sibling-drift",
        )

    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    record, checkpoint, artifact, dataset_config = _bridge_fixture(corrupt_root)
    features_path = artifact / "features.npy"
    features_path.write_bytes(features_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum|size"):
        write_bridge_artifact(
            record=record,
            checkpoint_path=checkpoint,
            feature_artifact_path=artifact,
            dataset_config_path=dataset_config,
            output_root=tmp_path / "feature-drift",
        )


def test_bridge_reuses_only_the_same_verified_identity(tmp_path):
    record, checkpoint, artifact, dataset_config = _bridge_fixture(tmp_path)
    output_root = tmp_path / "bridge"
    output = write_bridge_artifact(
        record=record,
        checkpoint_path=checkpoint,
        feature_artifact_path=artifact,
        dataset_config_path=dataset_config,
        output_root=output_root,
    )
    assert write_bridge_artifact(
        record=record,
        checkpoint_path=checkpoint,
        feature_artifact_path=artifact,
        dataset_config_path=dataset_config,
        output_root=output_root,
    ) == output

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_id"] = "different-record"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|checksum"):
        write_bridge_artifact(
            record=record,
            checkpoint_path=checkpoint,
            feature_artifact_path=artifact,
            dataset_config_path=dataset_config,
            output_root=output_root,
        )


def load_provenance(checkpoint: Path) -> dict:
    return torch.load(checkpoint, map_location="cpu", weights_only=False)["task_f_provenance"]


def test_terminal_manifest_keeps_all_missing_records(tmp_path):
    plan = build_fresh_evaluation_plan(_plan())
    terminal = validate_bound_inventory(
        plan=plan,
        bindings={"schema_version": BINDINGS_SCHEMA_VERSION, "records": []},
        dataset_config_path=tmp_path / "unused.yaml",
        output_root=tmp_path / "bridge",
        terminal_path=tmp_path / "terminal.json",
    )
    assert terminal["status"] == "INCOMPLETE"
    assert terminal["counts"] == {"MISSING": 1320}
    assert len(terminal["records"]) == 1320


def test_id_validation_input_is_deterministic_and_membership_ordered(tmp_path):
    dataset_config, _ = _dataset_config(tmp_path)
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name, value in (("c.png", 32), ("d.png", 192)):
        pixels = np.full((32, 32, 3), value, dtype=np.uint8)
        Image.fromarray(pixels).save(image_root / name)
    first = tmp_path / "validation-one.npz"
    second = tmp_path / "validation-two.npz"
    first_result = build_id_input(
        dataset_config_path=dataset_config,
        data_root=tmp_path,
        split="id_validation",
        output_path=first,
        batch_size=2,
    )
    second_result = build_id_input(
        dataset_config_path=dataset_config,
        data_root=tmp_path,
        split="id_validation",
        output_path=second,
        batch_size=1,
    )
    assert first_result["npz_sha256"] == second_result["npz_sha256"]
    verified = verify_id_input(
        first,
        dataset_config_path=dataset_config,
        split="id_validation",
    )
    assert verified["sample_count"] == 2
    with np.load(first, allow_pickle=False) as payload:
        assert payload["sample_ids"].tolist() == ["cifar10:c.png", "cifar10:d.png"]
        assert payload["is_id"].tolist() == [True, True]
