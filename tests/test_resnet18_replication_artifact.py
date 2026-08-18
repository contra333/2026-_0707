import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from oge.feature_export.resnet18_replication import (
    extract_resnet18_replication_outputs,
    verify_resnet18_replication_artifact,
    write_resnet18_replication_artifact,
)
from oge.feature_export.task_f import collect_runtime_provenance
from oge.models import ResNet18
from oge.training.resnet18_replication_provenance import (
    RESNET18_REPLICATION_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
    validate_resnet18_replication_checkpoint_payload,
)
from oge.training.resnet18_replication_plan import (
    RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG,
    RESNET18_REPLICATION_NUMERICAL_POLICY_ID,
    RESNET18_REPLICATION_STUDY_ID,
)


def _sha(character: str) -> str:
    return character * 64


def _provenance() -> dict:
    common = {
        "training_seed": 7,
        "initial_lr": 1e-3,
        "weight_decay": 1e-4,
        "initialization_sha256": _sha("a"),
        "data_stream_sha256": _sha("b"),
    }
    members = {
        "adam_coupled": {
            **common,
            "run_id": "fixture-adam",
            "branch_policy": "adam_coupled",
        },
        "adamw_decoupled": {
            **common,
            "run_id": "fixture-adamw",
            "branch_policy": "adamw_decoupled",
        },
    }
    return {
        "schema_version": RESNET18_REPLICATION_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "numerical_policy_id": RESNET18_REPLICATION_NUMERICAL_POLICY_ID,
        "cublas_workspace_config": RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG,
        "run_id": "fixture-adam",
        "training_seed": 7,
        "branch_policy": "adam_coupled",
        "initial_lr": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint_epoch": 2,
        "checkpoint_role": "last",
        "oge_git_sha": "c" * 40,
        "execution_only": True,
        "initialization_sha256": _sha("a"),
        "data_stream_sha256": _sha("b"),
        "sibling_group_id": "fixture-lr1e-3-seed7",
        "cross_lr_pairing_block_id": "fixture-cross-lr-seed7",
        "branch_neutral_config_sha256": _sha("d"),
        "cross_lr_neutral_config_sha256": _sha("e"),
        "sibling_role": "adam_coupled",
        "sibling_members": members,
        "model_config": {
            "name": "resnet18",
            "variant": "cifar",
            "num_classes": 10,
        },
    }


def test_native_resnet18_endpoint_shapes_and_classifier_contract():
    model = ResNet18(num_classes=10, variant="cifar").eval()
    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    features, logits, weight, bias = extract_resnet18_replication_outputs(
        model, images, device="cpu", batch_size=1
    )
    assert features.shape == (2, 512)
    assert logits.shape == (2, 10)
    assert weight.shape == (10, 512)
    assert bias.shape == (10,)
    assert all(array.dtype == np.float32 for array in (features, logits, weight, bias))


def test_replication_artifact_checksums_shapes_and_checkpoint_payload(tmp_path):
    rng = np.random.default_rng(17)
    provenance = _provenance()
    checkpoint = {
        "checkpoint_type": "last",
        "completed_epoch": 2,
        "model_state": {},
        "oge_git_sha": "c" * 40,
        "run_id": "fixture-adam",
        "resnet18_replication_provenance": provenance,
    }
    assert validate_resnet18_replication_checkpoint_payload(checkpoint) == provenance
    artifact = write_resnet18_replication_artifact(
        artifact_root=tmp_path,
        features=rng.normal(size=(3, 512)).astype(np.float32),
        logits=rng.normal(size=(3, 10)).astype(np.float32),
        classifier_weight=rng.normal(size=(10, 512)).astype(np.float32),
        classifier_bias=rng.normal(size=(10,)).astype(np.float32),
        sample_ids=np.asarray(["fixture:0", "fixture:1", "fixture:2"]),
        checkpoint_sha256=_sha("f"),
        checkpoint_provenance=provenance,
        dataset_split="synthetic_id_fixture",
        runtime=collect_runtime_provenance("cpu"),
    )
    verified = verify_resnet18_replication_artifact(artifact)
    manifest = verified["manifest"]
    assert manifest["feature_shape"] == [3, 512]
    assert manifest["logit_shape"] == [3, 10]
    assert manifest["classifier_weight_shape"] == [10, 512]
    assert manifest["cross_lr_pairing_block_id"] == "fixture-cross-lr-seed7"

    features = np.load(artifact / "features.npy", allow_pickle=False)
    features[0, 0] += 1.0
    np.save(artifact / "features.npy", features, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum"):
        verify_resnet18_replication_artifact(artifact)


def test_checkpoint_provenance_rejects_classifier_or_sibling_drift():
    provenance = _provenance()
    broken = copy.deepcopy(provenance)
    broken["model_config"]["variant"] = "imagenet"
    checkpoint = {
        "checkpoint_type": "last",
        "completed_epoch": 2,
        "model_state": {},
        "oge_git_sha": "c" * 40,
        "run_id": "fixture-adam",
        "resnet18_replication_provenance": broken,
    }
    with pytest.raises(ValueError, match="ResNet-18"):
        validate_resnet18_replication_checkpoint_payload(checkpoint)
