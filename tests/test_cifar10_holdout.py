import copy
import hashlib
from collections import Counter
from pathlib import Path

from oge.data import (
    generate_cifar10_holdout,
    load_dataset_config,
    membership_digest,
    parse_imglist_entry,
    read_holdout_manifest,
)
from oge.studies.hashing import scientific_config, scientific_config_hash
from oge.training.runner import load_training_config, resolve_training_config


REPOSITORY_ROOT = Path(__file__).parents[1]
FROZEN_DIR = REPOSITORY_ROOT / "configs/datasets/oge_cifar10_holdout_v1"


def _write_source(path: Path, *, split: str, per_class: int, offset: int = 0) -> None:
    lines = [
        f"cifar10/{split}/class_{label}/{index + offset:04d}.png {label}\n"
        for label in range(10)
        for index in range(per_class)
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _read_ids(path: Path) -> list[tuple[str, int]]:
    return [
        parse_imglist_entry(line, line_number=index)
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
    ]


def test_membership_digest_matches_frozen_formula():
    sample_id = "cifar10:cifar10/train/cat/0001.png"
    expected = hashlib.sha256(
        b"oge-cifar10-holdout-v1\0"
        + b"2027\0"
        + sample_id.encode("utf-8")
    ).hexdigest()
    assert membership_digest(sample_id) == expected


def test_generator_is_byte_identical_and_satisfies_membership_contract(tmp_path):
    train_source = tmp_path / "train_cifar10.txt"
    validation_source = tmp_path / "val_cifar10.txt"
    test_source = tmp_path / "test_cifar10.txt"
    _write_source(train_source, split="train", per_class=5_000)
    _write_source(validation_source, split="test", per_class=100)
    _write_source(test_source, split="test", per_class=900, offset=100)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = generate_cifar10_holdout(
        official_train_imglist=train_source,
        openood_id_validation_imglist=validation_source,
        openood_id_test_imglist=test_source,
        output_dir=first,
    )
    second_result = generate_cifar10_holdout(
        official_train_imglist=train_source,
        openood_id_validation_imglist=validation_source,
        openood_id_test_imglist=test_source,
        output_dir=second,
    )

    for filename in (
        "train_cifar10_45k.txt",
        "val_cifar10_5k.txt",
        "test_cifar10_10k.txt",
        "split_manifest.jsonl",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert first_result == second_result

    train = _read_ids(first / "train_cifar10_45k.txt")
    validation = _read_ids(first / "val_cifar10_5k.txt")
    test = _read_ids(first / "test_cifar10_10k.txt")
    source = _read_ids(train_source)
    assert len(train) == 45_000
    assert len(validation) == 5_000
    assert len(test) == 10_000
    assert set(train).isdisjoint(validation)
    assert set(train).union(validation) == set(source)
    assert Counter(label for _, label in train) == {label: 4_500 for label in range(10)}
    assert Counter(label for _, label in validation) == {
        label: 500 for label in range(10)
    }
    assert Counter(label for _, label in test) == {label: 1_000 for label in range(10)}
    assert test == _read_ids(validation_source) + _read_ids(test_source)
    train_set = set(train)
    validation_set = set(validation)
    assert train == [entry for entry in source if entry in train_set]
    assert validation == [entry for entry in source if entry in validation_set]

    header, memberships = read_holdout_manifest(first / "split_manifest.jsonl")
    assert header["protocol_name"] == "oge_cifar10_holdout_v1"
    assert header["openood_test_union"]["intersection_count"] == 0
    assert header["openood_test_union"]["union_count"] == 10_000
    assert len(memberships) == 60_000
    assert Counter(row["role"] for row in memberships) == {
        "id_train": 45_000,
        "id_validation": 5_000,
        "id_test": 10_000,
    }


def test_committed_holdout_hashes_counts_and_resolved_identity(tmp_path):
    expected = {
        "train_cifar10_45k.txt": (
            45_000,
            "54fae08d35de5be7239b8fc35413531335a71858288f00c5583de44d759bfbc6",
        ),
        "val_cifar10_5k.txt": (
            5_000,
            "fe74684d1211a92eaccba00957ca308b43bff286f230cd62db89412fb753b488",
        ),
        "test_cifar10_10k.txt": (
            10_000,
            "4178c1a97951afab348d63734d426d7e3cb0d13ddf3c0aebae331467d723fd73",
        ),
    }
    for filename, (row_count, digest) in expected.items():
        content = (FROZEN_DIR / filename).read_bytes()
        assert len(content.decode("utf-8").splitlines()) == row_count
        assert hashlib.sha256(content).hexdigest() == digest

    manifest_content = (FROZEN_DIR / "split_manifest.jsonl").read_bytes()
    assert hashlib.sha256(manifest_content).hexdigest() == (
        "7101ea2a8e637d72385b2e5b1aadde1f4b99ee34d0ede4753169715ab8ed4056"
    )
    header, memberships = read_holdout_manifest(FROZEN_DIR / "split_manifest.jsonl")
    assert header["sources"]["openood_official_train_50k"]["sha256"] == (
        "9317e18980f1b11df74bba9d64223e57e69bd08c2943adbc75aec393a4976d1b"
    )
    assert header["sources"]["openood_id_validation_1k"]["sha256"] == (
        "6f64c4370bf3c61dab3308f7e8c5c94938b112b20d38cca77c1dfa0bcc5c1d89"
    )
    assert header["sources"]["openood_id_test_9k"]["sha256"] == (
        "84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13"
    )
    assert len(memberships) == 60_000

    training_config = load_training_config(
        REPOSITORY_ROOT / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )
    resolved = resolve_training_config(
        training_config,
        data_root=tmp_path,
        device="cpu",
    )
    assert resolved["dataset"]["protocol"] == "oge_cifar10_holdout_v1"
    assert resolved["dataset"]["membership"]["train"]["line_count"] == 45_000
    assert resolved["dataset"]["membership"]["validation"]["line_count"] == 5_000
    assert resolved["dataset"]["membership"]["test"]["line_count"] == 10_000
    assert resolved["dataset"]["membership_manifest"]["sha256"] == (
        "7101ea2a8e637d72385b2e5b1aadde1f4b99ee34d0ede4753169715ab8ed4056"
    )
    identity = scientific_config(resolved)
    assert identity["dataset"]["membership"] == resolved["dataset"]["membership"]
    assert identity["dataset"]["membership_manifest"] == (
        resolved["dataset"]["membership_manifest"]
    )
    changed = copy.deepcopy(resolved)
    changed["dataset"]["membership"]["validation"]["sha256"] = "0" * 64
    assert scientific_config_hash(changed) != scientific_config_hash(resolved)


def test_new_dataset_config_keeps_openood_ood_roles_separate():
    config = load_dataset_config(
        REPOSITORY_ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml"
    )
    assert config["protocol_name"] == "oge_cifar10_holdout_v1"
    assert config["datasets"]["id_train"]["expected_count"] == 45_000
    assert config["datasets"]["id_validation"]["expected_count"] == 5_000
    assert config["datasets"]["id_test"]["expected_count"] == 10_000
    assert config["datasets"]["id_test_openood"]["expected_count"] == 9_000
    assert (
        config["datasets"]["ood_validation_tin"]["role"] == "compatibility_only"
    )
