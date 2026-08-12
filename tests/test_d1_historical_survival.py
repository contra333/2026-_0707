import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oge.analysis.fixed_readout_component_attribution import (
    pair_outcome_summary,
    pair_transition_summary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "analyze_d1_historical_survival.py"
SPEC = importlib.util.spec_from_file_location("d1_historical_survival", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
D1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D1)


LOGICAL_PATHS = {
    "id_sample_ids": "cifar10/test/sample_ids.npy",
    "cifar100_sample_ids": "cifar100/test/sample_ids.npy",
    "mnist_sample_ids": "mnist/test/sample_ids.npy",
    "raw_md_id": "metrics/arrays/scores/id_test/detector__mahalanobis_raw.npy",
    "raw_rmd_id": (
        "metrics/arrays/scores/id_test/detector__relative_mahalanobis_raw.npy"
    ),
    "raw_md_cifar100": (
        "metrics/arrays/scores/cifar100/detector__mahalanobis_raw.npy"
    ),
    "raw_rmd_cifar100": (
        "metrics/arrays/scores/cifar100/detector__relative_mahalanobis_raw.npy"
    ),
    "raw_md_mnist": "metrics/arrays/scores/mnist/detector__mahalanobis_raw.npy",
    "raw_rmd_mnist": (
        "metrics/arrays/scores/mnist/detector__relative_mahalanobis_raw.npy"
    ),
}


def _save_array(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)


def _fixture_manifest(tmp_path):
    retrieval_root = tmp_path / "retrieval"
    groups = ("adam_c3", "adamw_c3", "adam_c4", "adamw_c4")
    templates = {
        "id_sample_ids": np.asarray(["id0", "id1", "id2"], dtype="<U8"),
        "cifar100_sample_ids": np.asarray(["c0", "c1"], dtype="<U8"),
        "mnist_sample_ids": np.asarray(["m0", "m1"], dtype="<U8"),
        "raw_md_id": np.asarray([3.0, 2.0, 1.0]),
        "raw_rmd_id": np.asarray([2.5, 1.5, 0.5]),
        "raw_md_cifar100": np.asarray([0.0, 1.0]),
        "raw_rmd_cifar100": np.asarray([-0.5, 0.5]),
        "raw_md_mnist": np.asarray([-1.0, 0.5]),
        "raw_rmd_mnist": np.asarray([-1.5, 0.0]),
    }
    bundles = []
    logical_files = {}
    for group_index, group in enumerate(groups):
        for seed in range(3):
            bundle_id = f"{group_index * 3 + seed + 1:064x}"
            files = {}
            for key, logical_path in LOGICAL_PATHS.items():
                path = retrieval_root / bundle_id / logical_path
                _save_array(path, templates[key])
                files[key] = D1.sha256(path)
                logical_files.setdefault(
                    key,
                    {"logical_path": logical_path, "size": path.stat().st_size},
                )
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "group": group,
                    "role": group.rsplit("_", 1)[1].upper(),
                    "optimizer": "AdamW" if group.startswith("adamw") else "Adam",
                    "seed": seed,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "weight_decay_policy": "weights_only_no_bias_norm",
                    "config_hash": f"{group_index + 100:064x}",
                    "files": files,
                }
            )
    manifest = {
        "schema_version": D1.SCHEMA_VERSION,
        "analysis_role": "historical_descriptive_discovery",
        "claim_boundary": "fixture only",
        "analysis": {
            "datasets": {
                "cifar100": {
                    "id_sample_ids": "id_sample_ids",
                    "ood_sample_ids": "cifar100_sample_ids",
                    "scores": {
                        "raw_md": {"id": "raw_md_id", "ood": "raw_md_cifar100"},
                        "raw_rmd": {
                            "id": "raw_rmd_id",
                            "ood": "raw_rmd_cifar100",
                        },
                    },
                },
                "mnist": {
                    "id_sample_ids": "id_sample_ids",
                    "ood_sample_ids": "mnist_sample_ids",
                    "scores": {
                        "raw_md": {"id": "raw_md_id", "ood": "raw_md_mnist"},
                        "raw_rmd": {
                            "id": "raw_rmd_id",
                            "ood": "raw_rmd_mnist",
                        },
                    },
                },
            },
            "cross_policy_pairs": 6,
            "same_policy_seed_pairs": 12,
        },
        "logical_files": logical_files,
        "bundles": bundles,
        "expected_output_identity": {
            "original_generator": {"filename": "original.py", "sha256": "0" * 64},
            "data_quality.json": "0" * 64,
            "pair_metrics.csv": "0" * 64,
            "summary.json": "0" * 64,
        },
    }
    return retrieval_root, manifest


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _replace_and_rehash(retrieval_root, manifest, bundle_index, key, values):
    bundle = manifest["bundles"][bundle_index]
    path = retrieval_root / bundle["bundle_id"] / LOGICAL_PATHS[key]
    _save_array(path, values)
    bundle["files"][key] = D1.sha256(path)
    return path


def test_tie_aware_gain_loss_churn_and_auroc_balance():
    id_0 = np.asarray([1.0, 2.0])
    ood_0 = np.asarray([1.0, 2.0])
    id_1 = np.asarray([2.0, 1.0])
    ood_1 = np.asarray([1.0, 2.0])

    transition = pair_transition_summary(id_0, ood_0, id_1, ood_1)
    gain, loss, churn = D1.transition_rates(transition)
    auroc_0 = pair_outcome_summary(id_0, ood_0)["auroc_id_positive"]
    auroc_1 = pair_outcome_summary(id_1, ood_1)["auroc_id_positive"]

    assert gain == pytest.approx(0.25)
    assert loss == pytest.approx(0.25)
    assert churn == pytest.approx(0.5)
    assert auroc_0 == pytest.approx(0.5)
    assert auroc_1 == pytest.approx(0.5)
    assert auroc_1 - auroc_0 == pytest.approx(gain - loss)


def test_corrupted_hash_exits_nonzero_without_output(tmp_path):
    retrieval_root, manifest = _fixture_manifest(tmp_path)
    manifest["bundles"][0]["files"]["raw_md_id"] = "0" * 64
    manifest_path = _write_manifest(tmp_path, manifest)
    output = tmp_path / "output"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--retrieval-root",
            str(retrieval_root),
            "--input-manifest",
            str(manifest_path),
            "--output-directory",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "hash mismatch" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("sample_ids", "sample-ID mismatch"),
        ("non_finite", "non-finite score values"),
        ("shape", "shape failure"),
    ],
)
def test_validation_failure_writes_no_output(tmp_path, failure, message):
    retrieval_root, manifest = _fixture_manifest(tmp_path)
    if failure == "sample_ids":
        _replace_and_rehash(
            retrieval_root,
            manifest,
            1,
            "cifar100_sample_ids",
            np.asarray(["c0", "DIFF"], dtype="<U8"),
        )
    elif failure == "non_finite":
        _replace_and_rehash(
            retrieval_root,
            manifest,
            0,
            "raw_md_id",
            np.asarray([np.nan, 2.0, 1.0]),
        )
    else:
        path = _replace_and_rehash(
            retrieval_root,
            manifest,
            0,
            "raw_md_id",
            np.asarray([[3.0, 2.0, 1.0]]),
        )
        manifest["logical_files"]["raw_md_id"]["size"] = path.stat().st_size
    manifest_path = _write_manifest(tmp_path, manifest)
    output = tmp_path / "output"

    with pytest.raises(D1.D1ValidationError, match=message):
        D1.reproduce(
            retrieval_root=retrieval_root,
            input_manifest=manifest_path,
            output_directory=output,
        )
    assert not output.exists()


def test_output_no_overwrite_preserves_existing_directory(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(D1.D1ValidationError, match="refusing overwrite"):
        D1.reproduce(
            retrieval_root=tmp_path / "missing-retrieval",
            input_manifest=tmp_path / "missing-input.json",
            output_directory=output,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_committed_manifest_freezes_required_scope():
    manifest = json.loads(
        (
            REPOSITORY_ROOT
            / "configs"
            / "analysis"
            / "d1_historical_survival_input_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == D1.SCHEMA_VERSION
    assert len(manifest["bundles"]) == 12
    assert {bundle["role"] for bundle in manifest["bundles"]} == {"C3", "C4"}
    assert {bundle["optimizer"] for bundle in manifest["bundles"]} == {
        "Adam",
        "AdamW",
    }
    assert {bundle["seed"] for bundle in manifest["bundles"]} == {0, 1, 2}
    assert set(manifest["analysis"]["datasets"]) == {"cifar100", "mnist"}
    assert set(manifest["logical_files"]) == set(LOGICAL_PATHS)
    assert all(
        set(bundle["files"]) == set(LOGICAL_PATHS)
        for bundle in manifest["bundles"]
    )
    score_uses = sum(
        2
        for _bundle in manifest["bundles"]
        for dataset in manifest["analysis"]["datasets"].values()
        for _detector in dataset["scores"].values()
    )
    assert score_uses == 96
