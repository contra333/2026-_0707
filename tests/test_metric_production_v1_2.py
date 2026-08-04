import json

import numpy as np
import pytest
import yaml

from oge.evaluation.extraction import sha256_file
from oge.evaluation.production import (
    EXECUTABLE_SPLITS,
    OOD_SPLITS,
    PROTECTED_AUTHORIZATION,
    evaluate_raw_feature_cache_production,
    validate_dataset_policy,
    verify_production_bundle,
)


def _write_split(root, relative, *, features, logits, labels, sample_prefix, is_id):
    directory = root / relative
    directory.mkdir(parents=True)
    arrays = {
        "features": features.astype(np.float32),
        "logits": logits.astype(np.float32),
        "class_labels": labels.astype(np.int64),
        "predictions": np.argmax(logits, axis=1).astype(np.int64),
        "is_id": np.full(len(features), is_id, dtype=np.bool_),
        "sample_ids": np.asarray(
            [f"{sample_prefix}-{index:04d}" for index in range(len(features))]
        ),
    }
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array, allow_pickle=False)
    return arrays


def _production_fixture(tmp_path):
    checkpoint_sha = "a" * 64
    root = tmp_path / checkpoint_sha
    root.mkdir()
    rng = np.random.default_rng(91)
    feature_dim = 100
    class_count = 10
    weight = rng.normal(size=(class_count, feature_dim))
    bias = rng.normal(size=class_count)
    offsets = rng.normal(scale=0.3, size=(class_count, feature_dim))
    split_sizes = {
        "id_train": 120,
        "id_validation": 20,
        "id_test": 20,
        "cifar100": 3,
        "tin": 3,
        "mnist": 3,
        "svhn": 3,
        "texture": 3,
        "places365": 3,
    }
    split_manifest = {}
    for split_index, (split_key, count) in enumerate(split_sizes.items()):
        if split_key.startswith("id_"):
            labels = np.resize(np.arange(class_count), count)
            features = rng.normal(size=(count, feature_dim)) + offsets[labels]
        else:
            labels = np.full(count, -1)
            features = rng.normal(loc=2.0 + split_index, size=(count, feature_dim))
        logits = features @ weight.T + bias
        relative = f"fixture/{split_key}"
        _write_split(
            root,
            relative,
            features=features,
            logits=logits,
            labels=labels,
            sample_prefix=split_key,
            is_id=split_key.startswith("id_"),
        )
        split_manifest[split_key] = {
            "relative_directory": relative,
            "sample_count": count,
        }
    np.save(root / "classifier_weight.npy", weight.astype(np.float32), allow_pickle=False)
    np.save(root / "classifier_bias.npy", bias.astype(np.float32), allow_pickle=False)
    evaluation_sha = "d" * 40
    job = {
        "job_id": "fixture-job-last",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": "last",
        "reporting_role": "confirmatory_primary",
        "completed_epoch": 200,
        "config_hash": "b" * 64,
        "training_seed": 1,
        "optimizer_family": "sgd",
        "labels": ["role:sgd:C1"],
    }
    manifest = {
        "checkpoint": {
            "sha256": checkpoint_sha,
            "role": "last",
            "completed_epoch": 200,
            "training_seed": 1,
            "config_hash": "b" * 64,
        },
        "evaluation": {
            "protected_split_authorization": PROTECTED_AUTHORIZATION,
            "oge_git_sha": evaluation_sha,
            "git_dirty": False,
        },
        "dataset": {"splits": split_manifest},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw_rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "checksums.sha256").write_text("\n".join(raw_rows) + "\n")
    id_splits = {
        key: {"count": split_sizes[key]}
        for key in ("id_train", "id_validation", "id_test")
    }
    ood_splits = {key: {"count": split_sizes[key]} for key in OOD_SPLITS}
    inventory = {
        "inventory_hash": "c" * 64,
        "dataset_policy_hash": "e" * 64,
        "dataset_policy": {"id_splits": id_splits, "ood_splits": ood_splits},
    }
    sources_lock = tmp_path / "sources.lock.yaml"
    sources_lock.write_text("sources: {}\n")
    return root, job, inventory, evaluation_sha, sources_lock


def test_production_cache_writes_complete_checkpoint_bundle_and_scalar_panel(tmp_path):
    root, job, inventory, evaluation_sha, sources_lock = _production_fixture(tmp_path)

    output = evaluate_raw_feature_cache_production(
        raw_artifact_path=root,
        job=job,
        inventory=inventory,
        evaluation_git_sha=evaluation_sha,
        sources_lock_path=sources_lock,
    )

    assert output == root
    verified = verify_production_bundle(root)
    assert verified["completion"]["split_counts"] == {
        "id_train": 120,
        "id_validation": 20,
        "id_test": 20,
        "cifar100": 3,
        "tin": 3,
        "mnist": 3,
        "svhn": 3,
        "texture": 3,
        "places365": 3,
    }
    assert verified["artifact_manifest"]["split_order"] == list(EXECUTABLE_SPLITS)
    records = [
        json.loads(row)
        for row in (root / "metrics/scalar_records.jsonl").read_text().splitlines()
    ]
    keys = {record["artifact_key"] for record in records}
    assert "id/accuracy" in keys
    assert "detector/vim_author_dim" not in keys
    assert "ood_metric/auroc_id_positive" in keys
    assert "geometry/nc1_pinv" in keys
    assert "geometry/lid_k50" in keys
    assert "analysis/restoration_gap_mahalanobis" in keys
    assert all(record["checkpoint_sha256"] == job["checkpoint_sha256"] for record in records)
    assert (root / "metrics/arrays/scores/id_test/detector__vim_author_dim.npy").is_file()

    resumed = evaluate_raw_feature_cache_production(
        raw_artifact_path=root,
        job=job,
        inventory=inventory,
        evaluation_git_sha=evaluation_sha,
        sources_lock_path=sources_lock,
    )
    assert resumed == root


def test_dataset_policy_preflight_reads_ood_imglist_count_and_hash_from_data_root(tmp_path):
    config_root = tmp_path / "configs"
    data_root = tmp_path / "data"
    config_root.mkdir()
    data_root.mkdir()
    membership = config_root / "membership.jsonl"
    membership.write_text('{"record_type":"header"}\n')
    datasets = {}
    id_keys = {"id_train", "id_validation", "id_test"}
    policy_id = {}
    policy_ood = {}
    for index, split_key in enumerate(EXECUTABLE_SPLITS, start=1):
        is_id = split_key in id_keys
        imglist_root = config_root if is_id else data_root
        imglist = imglist_root / f"{split_key}.txt"
        imglist.write_text("".join(f"image-{row}.png {row % 10}\n" for row in range(index)))
        dataset_name = "cifar10" if is_id else split_key
        item = {
            "dataset_name": dataset_name,
            "split": "test",
            "is_id": is_id,
            "group": "id" if is_id else ("near" if split_key in {"cifar100", "tin"} else "far"),
            "imglist": imglist.name,
        }
        if is_id:
            item["imglist_location"] = "dataset_config"
            item["expected_count"] = index
            item["expected_sha256"] = sha256_file(imglist)
            policy_id[split_key] = {
                "count": index,
                "imglist_sha256": sha256_file(imglist),
                "dataset_name": dataset_name,
            }
        else:
            policy_ood[split_key] = {
                "count": index,
                "imglist_sha256": sha256_file(imglist),
                "dataset_name": dataset_name,
                "group": item["group"],
            }
        datasets[split_key] = item
    config = {
        "membership_manifest": {
            "path": membership.name,
            "sha256": sha256_file(membership),
            "row_count": 1,
        },
        "datasets": datasets,
    }
    config_path = config_root / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    inventory = {
        "dataset_policy": {
            "dataset_config_sha256": sha256_file(config_path),
            "membership_manifest": {
                "sha256": sha256_file(membership),
                "row_count": 1,
            },
            "id_splits": policy_id,
            "ood_splits": policy_ood,
        }
    }

    validate_dataset_policy(
        inventory,
        dataset_config_path=config_path,
        data_root=data_root,
    )
    (data_root / "cifar100.txt").write_text("drift.png 0\n")
    with pytest.raises(ValueError, match="cifar100.*(count|imglist)"):
        validate_dataset_policy(
            inventory,
            dataset_config_path=config_path,
            data_root=data_root,
        )
