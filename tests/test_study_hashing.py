import copy
import math

import pytest

from oge.studies.hashing import (
    canonical_json_bytes,
    provenance_identity_hash,
    scientific_config_hash,
)


def _config():
    return {
        "optimizer": {"name": "sgd", "lr": 0.1, "weight_decay": 0.0005},
        "model": {"name": "wrn28_10", "dropout_rate": 0.0, "init_policy": "msr_fan_in"},
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "dataset": {
            "protocol": "oge_cifar10_holdout_v1",
            "train_split": "id_train",
            "validation_split": "id_validation",
            "test_split": "id_test",
            "data_root": "/runtime/a",
            "membership": {
                "train": {"sha256": "train"},
                "validation": {"sha256": "validation"},
                "test": {"sha256": "test"},
            },
            "membership_manifest": {"sha256": "manifest"},
        },
        "scheduler": {"name": "multistep", "milestones": [60, 120, 160], "gamma": 0.2},
        "training": {"max_epochs": 200, "batch_size": 128, "seed": 0, "precision": "fp32"},
        "checkpoint": {"snapshot_epochs": []},
        "runtime": {"device": "cuda:0"},
        "study_phase": "grid",
    }


def test_canonical_json_is_key_order_invariant_and_compact_utf8():
    left = {"b": [2, {"z": "한글", "a": 1}], "a": 0}
    right = {"a": 0, "b": [2, {"a": 1, "z": "한글"}]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == '{"a":0,"b":[2,{"a":1,"z":"한글"}]}'.encode()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"nested": [value]})


def test_scientific_hash_included_fields_change_identity():
    config = _config()
    changed = copy.deepcopy(config)
    changed["optimizer"]["lr"] = 0.2
    assert scientific_config_hash(config) != scientific_config_hash(changed)


def test_scientific_hash_covers_model_initialization_policy():
    """Initialization changes penultimate geometry, so it must change identity.

    Weight initialization may change optimization and representation geometry,
    which are mechanism variables for this project. Two runs that differ only
    in initialization must not share a scientific config hash.
    """
    config = _config()
    changed = copy.deepcopy(config)
    changed["model"]["init_policy"] = "some_other_policy"
    assert scientific_config_hash(config) != scientific_config_hash(changed)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["training"].update(seed=99),
        lambda value: value["checkpoint"].update(snapshot_epochs=[0, 200]),
        lambda value: value["runtime"].update(device="cuda:3"),
        lambda value: value.update(study_phase="confirmation"),
        lambda value: value["dataset"].update(data_root="/runtime/b"),
    ],
)
def test_scientific_hash_excludes_phase_seed_runtime_and_snapshot(mutate):
    config = _config()
    changed = copy.deepcopy(config)
    mutate(changed)
    assert scientific_config_hash(config) == scientific_config_hash(changed)


def test_holdout_membership_and_manifest_are_part_of_scientific_identity():
    config = _config()
    changed_membership = copy.deepcopy(config)
    changed_membership["dataset"]["membership"]["train"]["sha256"] = "other"
    changed_manifest = copy.deepcopy(config)
    changed_manifest["dataset"]["membership_manifest"]["sha256"] = "other"
    assert scientific_config_hash(config) != scientific_config_hash(changed_membership)
    assert scientific_config_hash(config) != scientific_config_hash(changed_manifest)


def test_provenance_hash_is_separate_and_changes_with_git_or_dataset_identity():
    first = provenance_identity_hash(
        git_sha="a" * 40,
        dataset_membership_hashes={"train": "one", "validation": "two"},
    )
    changed_git = provenance_identity_hash(
        git_sha="b" * 40,
        dataset_membership_hashes={"train": "one", "validation": "two"},
    )
    changed_data = provenance_identity_hash(
        git_sha="a" * 40,
        dataset_membership_hashes={"train": "other", "validation": "two"},
    )
    assert len(first) == 64
    assert first != changed_git != changed_data
