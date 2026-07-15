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
        "model": {"name": "wrn28_10", "dropout_rate": 0.0},
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "dataset": {
            "protocol": "openood_v1_5_aligned_cifar10",
            "train_split": "id_train",
            "validation_split": "id_validation",
            "test_split": "id_test",
            "data_root": "/runtime/a",
            "membership": {"train": "separate-provenance"},
        },
        "scheduler": {"name": "multistep", "milestones": [60, 120, 160], "gamma": 0.2},
        "training": {"max_epochs": 200, "batch_size": 128, "seed": 0, "precision": "fp32"},
        "checkpoint": {"snapshot_epochs": []},
        "runtime": {"device": "cuda:0"},
        "study_phase": "discovery",
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["training"].update(seed=99),
        lambda value: value["checkpoint"].update(snapshot_epochs=[0, 200]),
        lambda value: value["runtime"].update(device="cuda:3"),
        lambda value: value.update(study_phase="confirmation"),
        lambda value: value["dataset"].update(data_root="/runtime/b"),
        lambda value: value["dataset"].update(membership={"train": "other"}),
    ],
)
def test_scientific_hash_excludes_phase_seed_runtime_snapshot_and_membership(mutate):
    config = _config()
    changed = copy.deepcopy(config)
    mutate(changed)
    assert scientific_config_hash(config) == scientific_config_hash(changed)


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
