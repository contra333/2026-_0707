import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from oge.evaluation.classification import logit_ood_scores
from oge.evaluation.detectors import (
    exact_knn_l2_scores,
    fit_tied_gaussians,
    fit_vim,
    prototype_scores,
    score_tied_gaussians,
    score_vim,
)
from oge.evaluation.extraction import ordered_sample_id_digest, sha256_file


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/analyze_fixed_readout_stage2_mechanism.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_mechanism", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE2
SPEC.loader.exec_module(STAGE2)

ANALYZE_BUNDLE = STAGE2.analyze_bundle
LOAD_BUNDLE = STAGE2.load_selective_bundle
LOAD_CONTEXT = STAGE2.load_reuse_context
LOAD_PAIRS = STAGE2.load_pair_manifest
RUN_ANALYSIS = STAGE2.run_analysis
WRITE_REPORT = STAGE2.write_json_and_sidecar_no_overwrite

EVALUATION_SHA = "e" * 40
INVENTORY_HASH = "f" * 64
DATASET_POLICY_HASH = "9" * 64
PROTECTED_AUTHORIZATION = "issue-57-protected-metric-v1.2"
QUERY_KEYS = ("id_test", "near_ood", "far_ood")
SPLIT_COUNTS = {
    "id_train": 60,
    "id_test": 18,
    "near_ood": 13,
    "far_ood": 15,
}


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _identity(bundle_id: str, config_hash: str, seed: int) -> dict[str, Any]:
    return {
        "artifact_role": "checkpoint_metric_contract_v1.2_bundle",
        "checkpoint_role": "last",
        "checkpoint_sha256": bundle_id,
        "config_hash": config_hash,
        "dataset_policy_hash": DATASET_POLICY_HASH,
        "evaluation_git_sha": EVALUATION_SHA,
        "inventory_hash": INVENTORY_HASH,
        "protected_authorization": PROTECTED_AUTHORIZATION,
        "reporting_role": "confirmatory_primary",
        "status": "LOCAL_COMPLETE",
        "training_seed": seed,
    }


def _make_arrays(variant: float) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260810)
    feature_dim = 6
    class_count = 2
    weight = rng.normal(size=(class_count, feature_dim)).astype(np.float32)
    bias = rng.normal(size=class_count).astype(np.float32)
    centers = np.array(
        [
            [-1.2, 0.1, 0.5, -0.3, 0.4, -0.2],
            [1.1, -0.2, -0.4, 0.3, -0.5, 0.2],
        ],
        dtype=np.float64,
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for split_index, (split_name, count) in enumerate(SPLIT_COUNTS.items()):
        if split_name.startswith("id_"):
            labels = np.resize(np.arange(class_count), count)
            features = centers[labels] + rng.normal(
                scale=0.55, size=(count, feature_dim)
            )
        else:
            labels = None
            location = 0.35 if split_name == "near_ood" else 1.8
            features = rng.normal(
                loc=location, scale=1.0 + 0.03 * split_index, size=(count, feature_dim)
            )
        if variant:
            radial = 1.0 + variant * np.linspace(-0.35, 0.45, count)
            features = features * radial[:, None]
        features = features.astype(np.float32)
        logits = (features @ weight.T + bias).astype(np.float32)
        sample_ids = np.asarray(
            [f"{split_name}:sample-{index:04d}" for index in range(count)]
        )
        arrays = {
            "features": features,
            "logits": logits,
            "sample_ids": sample_ids,
        }
        if labels is not None:
            arrays["class_labels"] = labels.astype(np.int64)
        output[split_name] = arrays
    return output, weight, bias


def _reference(metrics_root: Path, relative_path: str, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    path = metrics_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {
        "array_path": relative_path,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _write_real_bundle(
    root: Path,
    *,
    identity: dict[str, Any],
    variant: float,
) -> None:
    root.mkdir(parents=True)
    splits, weight, bias = _make_arrays(variant)
    split_manifest = {}
    for split_name, arrays in splits.items():
        relative = Path("fixture") / split_name
        directory = root / relative
        directory.mkdir(parents=True)
        for array_name, array in arrays.items():
            np.save(directory / f"{array_name}.npy", array, allow_pickle=False)
        split_manifest[split_name] = {
            "dataset_name": "fixture",
            "physical_split": split_name,
            "is_id": split_name.startswith("id_"),
            "group": "id" if split_name.startswith("id_") else "ood",
            "sample_count": len(arrays["sample_ids"]),
            "ordered_sample_id_sha256": ordered_sample_id_digest(
                arrays["sample_ids"]
            ),
            "array_shapes": {
                name: list(array.shape) for name, array in arrays.items()
            },
            "array_dtypes": {
                name: str(array.dtype) for name, array in arrays.items()
            },
            "relative_directory": relative.as_posix(),
        }
    np.save(root / "classifier_weight.npy", weight, allow_pickle=False)
    np.save(root / "classifier_bias.npy", bias, allow_pickle=False)
    raw_manifest = {
        "schema_version": "1.0",
        "definition_version": "metric_contract_v1.2",
        "artifact_role": "raw_checkpoint_feature_cache",
        "checkpoint": {
            "sha256": identity["checkpoint_sha256"],
            "role": identity["checkpoint_role"],
            "completed_epoch": 200,
            "training_run_id": f"run-{identity['checkpoint_sha256'][:8]}",
            "training_git_sha": "1" * 40,
            "training_seed": identity["training_seed"],
            "config_hash": identity["config_hash"],
        },
        "model": {
            "name": "fixture_model",
            "feature_dim": weight.shape[1],
            "class_count": weight.shape[0],
            "classifier_bias_present": True,
        },
        "evaluation": {
            "oge_git_sha": identity["evaluation_git_sha"],
            "git_dirty": False,
            "protected_split_authorization": PROTECTED_AUTHORIZATION,
            "smoke_only": False,
        },
        "dataset": {"protocol": "fixture_protocol", "splits": split_manifest},
    }
    _write_json(root / "manifest.json", raw_manifest)

    train = splits["id_train"]
    raw_fit = fit_tied_gaussians(
        train["features"], train["class_labels"], num_classes=weight.shape[0]
    )
    normalized_fit = fit_tied_gaussians(
        train["features"],
        train["class_labels"],
        num_classes=weight.shape[0],
        normalize=True,
    )
    vim_fit = fit_vim(train["features"], train["logits"], weight, bias)
    metrics_root = root / "metrics"
    score_document = {}
    for split_name in QUERY_KEYS:
        query = splits[split_name]
        raw = score_tied_gaussians(raw_fit, query["features"])
        normalized = score_tied_gaussians(normalized_fit, query["features"])
        knn = exact_knn_l2_scores(
            train["features"],
            query["features"],
            fit_sample_ids=train["sample_ids"],
            k=50,
        )
        prototype = prototype_scores(
            train["features"],
            train["class_labels"],
            query["features"],
            classifier_weight=weight,
            num_classes=weight.shape[0],
        )
        vim = score_vim(vim_fit, query["features"], query["logits"])
        score_root = f"arrays/scores/{split_name}"
        split_scores = {
            "detector/mahalanobis_raw": _reference(
                metrics_root,
                f"{score_root}/detector__mahalanobis_raw.npy",
                raw["mahalanobis"],
            ),
            "detector/mahalanobis_pp": _reference(
                metrics_root,
                f"{score_root}/detector__mahalanobis_pp.npy",
                normalized["mahalanobis"],
            ),
            "detector/knn_l2_k50": _reference(
                metrics_root,
                f"{score_root}/detector__knn_l2_k50.npy",
                knn["score"],
            ),
            "detector/ctm_prototype_cosine": _reference(
                metrics_root,
                f"{score_root}/detector__ctm_prototype_cosine.npy",
                prototype["ctm_score"],
            ),
            "detector/vim_author_dim": _reference(
                metrics_root,
                f"{score_root}/detector__vim_author_dim.npy",
                vim["score"],
            ),
            "ood_score/energy_t1": _reference(
                metrics_root,
                f"{score_root}/ood_score__energy_t1.npy",
                logit_ood_scores(query["logits"])["energy_t1"],
            ),
            "diagnostics": {
                "raw_gaussian": {
                    "class_distances": _reference(
                        metrics_root,
                        f"{score_root}/diagnostics/raw_gaussian/class_distances.npy",
                        raw["class_distances"],
                    ),
                    "nearest_class": _reference(
                        metrics_root,
                        f"{score_root}/diagnostics/raw_gaussian/nearest_class.npy",
                        raw["nearest_class"],
                    ),
                },
                "normalized_gaussian": {
                    "class_distances": _reference(
                        metrics_root,
                        f"{score_root}/diagnostics/normalized_gaussian/class_distances.npy",
                        normalized["class_distances"],
                    )
                },
                "knn": {
                    "kth_squared_distance": _reference(
                        metrics_root,
                        f"{score_root}/diagnostics/knn/kth_squared_distance.npy",
                        knn["kth_squared_distance"],
                    ),
                    "kth_neighbor_sample_id": _reference(
                        metrics_root,
                        f"{score_root}/diagnostics/knn/kth_neighbor_sample_id.npy",
                        knn["kth_neighbor_sample_id"],
                    ),
                },
                "vim": {
                    "residual": _reference(
                        metrics_root,
                        f"{score_root}/diagnostics/vim/residual.npy",
                        vim["residual"],
                    )
                },
            },
            # A real result.json contains many references outside the analyzer's
            # required subset.  This deliberately unsafe unused reference proves
            # that the analyzer does not recursively traverse the whole result.
            "unused_reference": {
                "array_path": "../must-not-be-read.npy",
                "sha256": "0" * 64,
                "shape": [1],
                "dtype": "float64",
            },
        }
        score_document[split_name] = split_scores
    result = {
        "schema_version": "1.0",
        "definition_version": "metric_contract_v1.2",
        "protected_result": True,
        "smoke_only": False,
        "job": {
            "job_id": f"job-{identity['checkpoint_sha256'][:8]}",
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "checkpoint_role": identity["checkpoint_role"],
            "reporting_role": identity["reporting_role"],
            "completed_epoch": 200,
            "config_hash": identity["config_hash"],
            "training_seed": identity["training_seed"],
            "optimizer_family": "fixture",
            "labels": ["fixture"],
        },
        "inventory_hash": identity["inventory_hash"],
        "dataset_policy_hash": identity["dataset_policy_hash"],
        "evaluation_git_sha": identity["evaluation_git_sha"],
        "scores": score_document,
    }
    _write_json(metrics_root / "result.json", result)
    _write_json(root / "artifact_manifest.json", identity)
    (root / "bundle_checksums.sha256").write_text(
        "selective fixture does not consume the complete-bundle inventory\n",
        encoding="utf-8",
    )


def _write_placeholder_bundle(
    root: Path, *, identity: dict[str, Any], required_paths: tuple[str, ...]
) -> None:
    root.mkdir(parents=True)
    for logical_path in required_paths:
        path = root / logical_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if logical_path == "artifact_manifest.json":
            _write_json(path, identity)
        else:
            path.write_bytes(b"selective-placeholder\n")


def _integrity(root: Path, required_paths: tuple[str, ...]) -> dict[str, Any]:
    files = [
        {
            "logical_path": logical_path,
            "sha256": sha256_file(root / logical_path),
            "size": (root / logical_path).stat().st_size,
        }
        for logical_path in required_paths
    ]
    by_path = {row["logical_path"]: row for row in files}
    digest_payload = {
        "artifact_manifest_file_sha256": by_path["artifact_manifest.json"][
            "sha256"
        ],
        "bundle_checksums_file_sha256": by_path["bundle_checksums.sha256"][
            "sha256"
        ],
        "files": files,
    }
    return {
        **digest_payload,
        "allowed_payload_catalog_sha256": _canonical_sha256(digest_payload),
        "catalog_file_count": len(files),
        "selected_file_count": len(files),
        "selected_total_size": sum(row["size"] for row in files),
    }


@dataclass(frozen=True)
class SelectiveFixture:
    retrieval_root: Path
    manifest_path: Path
    pair_path: Path
    left_bundle_id: str
    right_bundle_id: str
    required_paths: tuple[str, ...]


def _build_selective_fixture(base: Path) -> SelectiveFixture:
    retrieval_root = base / "selective-retrieval"
    retrieval_root.mkdir(parents=True)
    specs = []
    for config_index in range(10):
        config_hash = hashlib.sha256(
            f"fixture-config-{config_index}".encode("utf-8")
        ).hexdigest()
        for seed in (0, 1, 2):
            bundle_id = hashlib.sha256(
                f"fixture-bundle-{config_index}-{seed}".encode("utf-8")
            ).hexdigest()
            specs.append(
                {
                    "bundle_id": bundle_id,
                    "config_hash": config_hash,
                    "training_seed": seed,
                    "identity": _identity(bundle_id, config_hash, seed),
                    "optimizer_family": (
                        "adam" if config_index == 0 else "adamw" if config_index == 1 else "fixture"
                    ),
                }
            )
    left = specs[0]
    right = specs[3]
    _write_real_bundle(
        retrieval_root / left["bundle_id"], identity=left["identity"], variant=0.0
    )
    _write_real_bundle(
        retrieval_root / right["bundle_id"], identity=right["identity"], variant=0.45
    )
    required_paths = tuple(
        sorted(
            path.relative_to(retrieval_root / left["bundle_id"]).as_posix()
            for path in (retrieval_root / left["bundle_id"]).rglob("*")
            if path.is_file()
        )
    )
    right_paths = tuple(
        sorted(
            path.relative_to(retrieval_root / right["bundle_id"]).as_posix()
            for path in (retrieval_root / right["bundle_id"]).rglob("*")
            if path.is_file()
        )
    )
    assert right_paths == required_paths
    for spec in specs:
        bundle_root = retrieval_root / spec["bundle_id"]
        if not bundle_root.exists():
            _write_placeholder_bundle(
                bundle_root,
                identity=spec["identity"],
                required_paths=required_paths,
            )
        spec["remote_integrity"] = _integrity(bundle_root, required_paths)
    manifest = {
        "schema_version": STAGE2.REUSE_SCHEMA_VERSION,
        "retrieval_ready": True,
        "analysis_policy": {
            name: {
                "id": policy_id,
                "path": relative_path,
                "sha256": sha256_file(Path(__file__).parents[1] / relative_path),
            }
            for name, (policy_id, relative_path) in STAGE2.ANALYSIS_POLICY_FILES.items()
        },
        "authorization": {
            "analysis_scope": "discovery_only",
            "bulk_fetch_allowed": False,
            "checkpoint_or_feature_recompute_allowed": False,
            "destination_overwrite_allowed": False,
            "mode": "reuse_only",
            "new_protected_split_traversal_allowed": False,
            "remote_delete_allowed": False,
            "remote_mutation_allowed": False,
            "remote_upload_allowed": False,
            "retrieval_policy": "catalog_verified_exact_allowlist_only",
            "source_artifact_authorization": PROTECTED_AUTHORIZATION,
        },
        "selection": {
            "checkpoint_role": "last",
            "configuration_count": 10,
            "job_count": 30,
            "reporting_role": "confirmatory_primary",
            "seeds": [0, 1, 2],
        },
        "remote_checksum_catalog": {
            "schema_version": STAGE2.REMOTE_CATALOG_SCHEMA_VERSION,
            "sha256": hashlib.sha256(b"fixture-remote-catalog").hexdigest(),
            "status": "complete",
        },
        "path_contract": {
            "query_keys": list(QUERY_KEYS),
            "required_logical_paths": list(required_paths),
        },
        "bundles": specs,
    }
    manifest_path = base / "reuse-manifest.json"
    _write_json(manifest_path, manifest)
    pair_path = base / "pairs.json"
    _write_json(
        pair_path,
        {
            "schema_version": STAGE2.PAIR_MANIFEST_SCHEMA_VERSION,
            "pairs": [
                {
                    "pair_id": "adam-vs-adamw-seed0",
                    "left": {"bundle_id": left["bundle_id"], "label": "adam"},
                    "right": {
                        "bundle_id": right["bundle_id"],
                        "label": "adamw",
                    },
                }
            ],
        },
    )
    return SelectiveFixture(
        retrieval_root=retrieval_root,
        manifest_path=manifest_path,
        pair_path=pair_path,
        left_bundle_id=left["bundle_id"],
        right_bundle_id=right["bundle_id"],
        required_paths=required_paths,
    )


def _reseal_bundle(fixture: SelectiveFixture, bundle_id: str) -> None:
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["bundles"] if item["bundle_id"] == bundle_id)
    row["remote_integrity"] = _integrity(
        fixture.retrieval_root / bundle_id, fixture.required_paths
    )
    _write_json(fixture.manifest_path, manifest)


@pytest.fixture
def selective_fixture(tmp_path: Path) -> SelectiveFixture:
    return _build_selective_fixture(tmp_path)


def test_selective_manifest_verifies_30_bundles_and_loads_only_required_arrays(
    selective_fixture: SelectiveFixture,
):
    context = LOAD_CONTEXT(
        selective_fixture.manifest_path, selective_fixture.retrieval_root
    )

    assert len(context.bundles) == 30
    with pytest.raises(PermissionError, match="outside selective allowlist"):
        context.accesses[selective_fixture.left_bundle_id].path(
            "metrics/arrays/not-allowlisted.npy"
        )
    bundle = LOAD_BUNDLE(context, selective_fixture.left_bundle_id)
    assert set(bundle.splits) == {"id_train", *QUERY_KEYS}
    assert "class_labels" in bundle.splits["id_train"]
    assert "class_labels" in bundle.splits["id_test"]
    assert "class_labels" not in bundle.splits["near_ood"]
    assert "class_labels" not in bundle.splits["far_ood"]
    assert set(bundle.scores["near_ood"]) == set(STAGE2.STORED_SCORE_KEYS)


def test_skip_raw_knn_is_explicit_not_run_and_pair_paths_are_forbidden(
    selective_fixture: SelectiveFixture,
):
    report = RUN_ANALYSIS(
        reuse_manifest_path=selective_fixture.manifest_path,
        retrieval_root=selective_fixture.retrieval_root,
        pair_manifest_path=selective_fixture.pair_path,
        skip_raw_knn=True,
    )

    assert report["analysis_role"] == "discovery_only"
    assert report["selection_performed"] is False
    assert report["extractor_completeness"]["status"] == "NOT_RUN"
    assert report["scientific_gate"]["status"] == "NOT_RUN"
    pair = report["pairs"][0]
    assert pair["extractor_completeness"]["status"] == "NOT_RUN"
    assert pair["scientific_gate"]["status"] == "NOT_RUN"
    assert "not causal evidence" in pair["claim_boundary"]
    endpoint = pair["per_ood"]["near_ood"]["endpoint_results"]["left"]
    assert endpoint["detectors"]["detector/knn_raw_k50"]["status"] == "NOT_RUN"
    assert set(endpoint["candidate_geometry_channels"]) == {
        "radial_feature_norm",
        "mahalanobis_precision_spectral_bands",
        "local_raw_vs_l2",
        "angular_ctm",
        "vim_residual_subspace",
        "frozen_logit_energy_null",
    }
    bundle_report = RUN_ANALYSIS(
        reuse_manifest_path=selective_fixture.manifest_path,
        retrieval_root=selective_fixture.retrieval_root,
        bundle_id=selective_fixture.left_bundle_id,
        skip_raw_knn=True,
    )
    assert bundle_report["raw_knn"]["status"] == "NOT_RUN"
    assert bundle_report["extractor_completeness"]["status"] == "NOT_RUN"
    assert bundle_report["scientific_gate"]["status"] == "NOT_RUN"

    context = LOAD_CONTEXT(
        selective_fixture.manifest_path, selective_fixture.retrieval_root
    )
    pair_payload = json.loads(selective_fixture.pair_path.read_text(encoding="utf-8"))
    pair_payload["pairs"][0]["left"]["path"] = "/tmp/arbitrary-bundle"
    invalid_pair_path = selective_fixture.pair_path.with_name("invalid-pairs.json")
    _write_json(invalid_pair_path, pair_payload)
    with pytest.raises(ValueError, match="arbitrary path is forbidden"):
        LOAD_PAIRS(invalid_pair_path, context)


def test_exact_raw_knn_and_bundle_reports_are_cacheable_without_overwrite(
    selective_fixture: SelectiveFixture, tmp_path: Path
):
    raw_cache = tmp_path / "raw-knn-cache"
    bundle_cache = tmp_path / "bundle-report-cache"
    kwargs = {
        "reuse_manifest_path": selective_fixture.manifest_path,
        "retrieval_root": selective_fixture.retrieval_root,
        "pair_manifest_path": selective_fixture.pair_path,
        "raw_knn_cache_root": raw_cache,
        "bundle_cache_root": bundle_cache,
    }

    first = RUN_ANALYSIS(**kwargs)
    cache_snapshot = {
        path.relative_to(raw_cache).as_posix(): path.read_bytes()
        for path in raw_cache.rglob("*")
        if path.is_file()
    }
    second = RUN_ANALYSIS(**kwargs)

    assert first == second
    assert first["extractor_completeness"]["status"] == "PASS"
    assert first["scientific_gate"]["status"] == "NOT_RUN"
    assert cache_snapshot == {
        path.relative_to(raw_cache).as_posix(): path.read_bytes()
        for path in raw_cache.rglob("*")
        if path.is_file()
    }
    for bundle_id in (
        selective_fixture.left_bundle_id,
        selective_fixture.right_bundle_id,
    ):
        assert (bundle_cache / f"{bundle_id}.json").is_file()
        assert (bundle_cache / f"{bundle_id}.json.sha256").is_file()
        for split_name, count in (
            ("id_test", 18),
            ("near_ood", 13),
            ("far_ood", 15),
        ):
            split_root = raw_cache / bundle_id / "splits" / split_name
            assert np.load(
                split_root / "neighbor_squared_distances.npy", allow_pickle=False
            ).shape == (count, 50)
            assert np.load(
                split_root / "neighbor_sample_ids.npy", allow_pickle=False
            ).shape == (count, 50)

    output = tmp_path / "stage2-report.json"
    digest = WRITE_REPORT(first, output)
    assert digest == sha256_file(output)
    assert output.with_suffix(".json.sha256").is_file()
    with pytest.raises(FileExistsError, match="no-overwrite pair"):
        WRITE_REPORT(first, output)

    orphan_output = tmp_path / "orphan-report.json"
    orphan_output.with_suffix(".json.sha256").write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="no-overwrite pair"):
        WRITE_REPORT(first, orphan_output)
    assert not orphan_output.exists()


def test_manifest_size_nan_shape_and_artifact_identity_fail_closed(tmp_path: Path):
    checksum_fixture = _build_selective_fixture(tmp_path / "checksum")
    checksum_target = (
        checksum_fixture.retrieval_root
        / checksum_fixture.left_bundle_id
        / "fixture/id_test/features.npy"
    )
    checksum_target.write_bytes(checksum_target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size mismatch"):
        LOAD_CONTEXT(checksum_fixture.manifest_path, checksum_fixture.retrieval_root)

    nan_fixture = _build_selective_fixture(tmp_path / "nan")
    nan_root = nan_fixture.retrieval_root / nan_fixture.left_bundle_id
    result_path = nan_root / "metrics/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ref = result["scores"]["id_test"]["detector/mahalanobis_raw"]
    array_path = nan_root / "metrics" / ref["array_path"]
    values = np.load(array_path, allow_pickle=False)
    values[0] = np.nan
    np.save(array_path, values, allow_pickle=False)
    ref["sha256"] = sha256_file(array_path)
    _write_json(result_path, result)
    _reseal_bundle(nan_fixture, nan_fixture.left_bundle_id)
    nan_context = LOAD_CONTEXT(nan_fixture.manifest_path, nan_fixture.retrieval_root)
    with pytest.raises(ValueError, match="NaN or infinity"):
        LOAD_BUNDLE(nan_context, nan_fixture.left_bundle_id)

    shape_fixture = _build_selective_fixture(tmp_path / "shape")
    shape_root = shape_fixture.retrieval_root / shape_fixture.left_bundle_id
    raw_manifest_path = shape_root / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_manifest["dataset"]["splits"]["id_test"]["array_shapes"]["features"] = [
        999,
        6,
    ]
    _write_json(raw_manifest_path, raw_manifest)
    _reseal_bundle(shape_fixture, shape_fixture.left_bundle_id)
    shape_context = LOAD_CONTEXT(
        shape_fixture.manifest_path, shape_fixture.retrieval_root
    )
    with pytest.raises(ValueError, match="shape differs from raw manifest"):
        LOAD_BUNDLE(shape_context, shape_fixture.left_bundle_id)

    identity_fixture = _build_selective_fixture(tmp_path / "identity")
    artifact_path = (
        identity_fixture.retrieval_root
        / identity_fixture.left_bundle_id
        / "artifact_manifest.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["config_hash"] = "0" * 64
    _write_json(artifact_path, artifact)
    _reseal_bundle(identity_fixture, identity_fixture.left_bundle_id)
    with pytest.raises(ValueError, match="retrieved artifact identity mismatch"):
        LOAD_CONTEXT(identity_fixture.manifest_path, identity_fixture.retrieval_root)


def test_required_result_reference_outside_allowlist_fails_before_io(
    selective_fixture: SelectiveFixture,
):
    root = selective_fixture.retrieval_root / selective_fixture.left_bundle_id
    result_path = root / "metrics/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["scores"]["id_test"]["detector/mahalanobis_raw"][
        "array_path"
    ] = "arrays/not-allowlisted.npy"
    _write_json(result_path, result)
    _reseal_bundle(selective_fixture, selective_fixture.left_bundle_id)
    context = LOAD_CONTEXT(
        selective_fixture.manifest_path, selective_fixture.retrieval_root
    )

    with pytest.raises(PermissionError, match="outside selective allowlist"):
        LOAD_BUNDLE(context, selective_fixture.left_bundle_id)


def test_finite_stored_score_formula_corruption_is_rejected(
    selective_fixture: SelectiveFixture,
):
    root = selective_fixture.retrieval_root / selective_fixture.right_bundle_id
    result_path = root / "metrics/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ref = result["scores"]["near_ood"]["detector/mahalanobis_raw"]
    array_path = root / "metrics" / ref["array_path"]
    values = np.load(array_path, allow_pickle=False)
    np.save(array_path, values + 0.25, allow_pickle=False)
    ref["sha256"] = sha256_file(array_path)
    _write_json(result_path, result)
    _reseal_bundle(selective_fixture, selective_fixture.right_bundle_id)

    with pytest.raises(AssertionError, match="stored/recomputed score mismatch"):
        RUN_ANALYSIS(
            reuse_manifest_path=selective_fixture.manifest_path,
            retrieval_root=selective_fixture.retrieval_root,
            bundle_id=selective_fixture.right_bundle_id,
            skip_raw_knn=True,
        )


def test_invariance_fits_are_once_per_bundle_not_once_per_ood(
    selective_fixture: SelectiveFixture, monkeypatch: pytest.MonkeyPatch
):
    context = LOAD_CONTEXT(
        selective_fixture.manifest_path, selective_fixture.retrieval_root
    )
    bundle = LOAD_BUNDLE(context, selective_fixture.left_bundle_id)
    original = STAGE2.fit_tied_gaussians
    calls = []

    def counted_fit(*args, **kwargs):
        calls.append(bool(kwargs.get("normalize", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(STAGE2, "fit_tied_gaussians", counted_fit)
    report = ANALYZE_BUNDLE(bundle, skip_raw_knn=True)

    assert len(bundle.ood_keys) == 2
    assert len(calls) == 4
    assert calls.count(False) == 2
    assert calls.count(True) == 2
    assert report["extractor_smoke_oracles"]["status"] == "PASS"
    assert report["extractor_smoke_oracles"]["scope"].startswith(
        "extractor_smoke_only"
    )


def test_exact_pairwise_misordering_and_mahalanobis_contribution_oracles():
    payload = STAGE2.pairwise_misordering([3.0, 2.0], [2.0, 1.0])
    assert payload["strict_id_below_ood_probability"] == pytest.approx(0.0)
    assert payload["tie_probability"] == pytest.approx(0.25)
    assert payload["pairwise_misordering_probability"] == pytest.approx(0.125)

    features = np.array(
        [[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    fit = fit_tied_gaussians(features, [0, 0, 1, 1], num_classes=2)
    contributions = STAGE2.mahalanobis_channel_contributions(
        fit, [[-1.5, 0.25], [1.5, -0.25]]
    )
    summed = sum(contributions["bands"].values(), np.zeros(2))
    np.testing.assert_allclose(
        summed, contributions["total_distance"], atol=1.0e-10, rtol=1.0e-9
    )
