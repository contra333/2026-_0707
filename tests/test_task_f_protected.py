import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

import oge.evaluation.task_f_protected as protected
from oge.analysis.discriminant_residual_preflight import fit_discriminant_geometry
from oge.evaluation.task_f_fresh import build_fresh_evaluation_plan
from oge.evaluation.task_f_fresh_orchestration import build_task_f_pipeline_manifest
from oge.training import generate_research_run_matrix


def _run_plan():
    return generate_research_run_matrix(
        anchor_seeds=(0, 1, 2, 3, 4),
        adam_factorial_seeds=(0, 1, 2),
        sgdm_seeds=(0, 1, 2),
    )


def _placement(run_plan):
    groups = {
        "curie": {
            "task-f-adam-lr1e-3-seed0",
            "task-f-adam-lr1e-3-seed1",
            "task-f-adam-lr3e-4-seed0",
            "task-f-sgdm-lr0.1-seed0",
        },
        "lise": {
            "task-f-adam-lr1e-3-seed3",
            "task-f-adam-lr1e-3-seed4",
            "task-f-sgdm-lr0.1-seed1",
        },
        "precision_medicine": {
            "task-f-adam-lr1e-3-seed2",
            "task-f-adam-lr3e-4-seed1",
            "task-f-adam-lr3e-4-seed2",
            "task-f-sgdm-lr0.1-seed2",
        },
    }
    concurrency = {"curie": 4, "lise": 2, "precision_medicine": 4}
    by_host = defaultdict(list)
    for run in run_plan["runs"]:
        host = next(host for host, members in groups.items() if run["sibling_group_id"] in members)
        by_host[host].append(run["run_id"])
    queues = {"execution_sha": protected.SOURCE_TRAINING_SHA, "hosts": {}}
    for host, run_ids in by_host.items():
        queues["hosts"][host] = {
            str(index): {"gpu_uuid": f"GPU-{host}-{index}", "run_ids": []}
            for index in range(concurrency[host])
        }
        for offset, run_id in enumerate(sorted(run_ids)):
            queues["hosts"][host][str(offset % concurrency[host])]["run_ids"].append(run_id)
    assignment = {
        "execution_sha": protected.SOURCE_TRAINING_SHA,
        "host_assignments": {
            host: {
                "expected_run_count": len(by_host[host]),
                "concurrency": concurrency[host],
                "sibling_group_ids": sorted(members),
            }
            for host, members in groups.items()
        },
    }
    locations = {}
    for host, host_queues in queues["hosts"].items():
        for index, queue in host_queues.items():
            for run_id in queue["run_ids"]:
                locations[run_id] = (host, int(index), queue["gpu_uuid"])
    return assignment, queues, locations


def _id_pipeline(run_plan):
    plan = build_fresh_evaluation_plan(run_plan)
    assignment, queues, locations = _placement(run_plan)
    observed = []
    for record in plan["records"]:
        if record["dataset_split"] != "id_train" or record["checkpoint_role"] == "best_val":
            continue
        if record["depth_tap"] == "penultimate" and record["checkpoint_epoch"] not in {10, 60, 120, 160, 200}:
            continue
        host, index, gpu_uuid = locations[record["run_id"]]
        observed.append({**record, "host_id": host, "gpu_index": index, "gpu_uuid": gpu_uuid})
    assert len(observed) == 310
    return build_task_f_pipeline_manifest(
        evaluation_plan=plan,
        observed_export_jobs=observed,
        host_assignment=assignment,
        gpu_queues=queues,
        evaluation_git_sha=protected.SOURCE_ID_EVALUATION_SHA,
    )


def _gate3():
    return {
        "schema_version": "task_f_rtmd_gate3_terminal_v1",
        "status": "PASS",
        "protected_data_access": False,
        "record_count": 10,
        "task_f_specification_sha256": protected.EXPECTED_SPECIFICATION_SHA256,
        "rtmd_gate3_specification_sha256": protected.RTMD_GATE3_SPECIFICATION_SHA256,
        "source_training_sha": protected.SOURCE_TRAINING_SHA,
        "evaluation_git_sha": protected.SOURCE_ID_EVALUATION_SHA,
        "rtmd_included_in_protected_plan": False,
        "gate3_verdict": {
            "status": "FAILED_INAPPLICABLE",
            "activated": False,
        },
    }


@pytest.fixture(scope="module")
def frozen_plan():
    run_plan = _run_plan()
    return protected.build_protected_plan(
        run_plan=run_plan,
        id_pipeline=_id_pipeline(run_plan),
        gate3_terminal=_gate3(),
        planning_git_sha="a" * 40,
    )


def _authorization(plan, sha="b" * 40):
    return {
        "schema_version": protected.AUTHORIZATION_SCHEMA_VERSION,
        "status": "APPROVED",
        "one_shot": True,
        "execution_git_sha": sha,
        "plan_sha256": plan["plan_sha256"],
        "source_training_sha": protected.SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": protected.EXPECTED_SPECIFICATION_SHA256,
        "splits": list(protected.PROTECTED_SPLITS),
        "record_count": 2520,
        "rtmd_included": False,
    }


def test_exact_2520_plan_preserves_host_placement_and_excludes_rtmd(frozen_plan):
    assert frozen_plan["counts"] == {
        "research_runs": 50,
        "checkpoint_depth_contexts_per_split": 360,
        "records_total": 2520,
        "records_by_split": {split: 360 for split in protected.PROTECTED_SPLITS},
        "records_by_host": {"curie": 1008, "lise": 630, "precision_medicine": 882},
    }
    assert frozen_plan["rtmd"]["status"] == "FAILED_INAPPLICABLE"
    assert frozen_plan["score_panel"]["detectors"] == ["md", "marginal", "rmd"]
    assert not any(row["dataset_split"] in protected._FORBIDDEN_SPLITS for row in frozen_plan["records"])
    assert sum(row["depth_tap"] != "penultimate" for row in frozen_plan["records"]) == 420
    assert sum(row["checkpoint_role"] == "best_val" for row in frozen_plan["records"]) == 350


def test_checkpoint_bundle_plan_conserves_records_and_reduces_execution_units(frozen_plan):
    bundle_plan = protected.build_protected_checkpoint_bundle_plan(frozen_plan)
    assert bundle_plan["counts"] == {
        "checkpoint_bundles": 300,
        "dataset_passes": 2100,
        "logical_records": 2520,
        "bundles_by_host": {"curie": 120, "lise": 66, "precision_medicine": 114},
        "dataset_passes_by_host": {"curie": 840, "lise": 462, "precision_medicine": 798},
        "logical_records_by_host": {"curie": 1008, "lise": 630, "precision_medicine": 882},
    }
    assert all(bundle["dataset_pass_count"] == 7 for bundle in bundle_plan["bundles"])
    assert sum(bundle["logical_record_count"] for bundle in bundle_plan["bundles"]) == 2520
    assert sum(bundle["logical_record_count"] == 28 for bundle in bundle_plan["bundles"]) == 20
    assert sum(bundle["logical_record_count"] == 7 for bundle in bundle_plan["bundles"]) == 280
    anchor = next(
        bundle
        for bundle in bundle_plan["bundles"]
        if bundle["logical_record_count"] == 28
    )
    assert all(
        dataset_pass["depth_taps"] == ["penultimate", "stage1", "stage2", "stage3"]
        for dataset_pass in anchor["dataset_passes"]
    )
    assert protected.validate_protected_checkpoint_bundle_plan(
        bundle_plan, plan=frozen_plan
    ) == bundle_plan


def test_checkpoint_bundle_loads_once_and_emits_all_taps_atomically(
    tmp_path, frozen_plan, monkeypatch
):
    bundle_plan = protected.build_protected_checkpoint_bundle_plan(frozen_plan)
    bundle = next(item for item in bundle_plan["bundles"] if item["logical_record_count"] == 28)
    calls = {"checkpoint_loads": 0, "dataset_loaders": 0}

    class FixtureModel:
        def __init__(self):
            self.forward_calls = 0
            self.tap_forward_calls = 0

        def eval(self):
            return self

        def __call__(self, images, return_feature_taps=False):
            self.forward_calls += 1
            logits = torch.zeros((len(images), 10), dtype=torch.float32)
            if not return_feature_taps:
                return logits
            self.tap_forward_calls += 1
            taps = {
                "stage1": torch.ones((len(images), 160), dtype=torch.float32),
                "stage2": torch.ones((len(images), 320), dtype=torch.float32),
                "stage3": torch.ones((len(images), 640), dtype=torch.float32),
                "penultimate": torch.ones((len(images), 640), dtype=torch.float32),
            }
            return logits, taps

    model = FixtureModel()

    def load_checkpoint(path, device):
        calls["checkpoint_loads"] += 1
        return (
            model,
            {
                "run_id": bundle["run_id"],
                "checkpoint_role": bundle["checkpoint_role"],
                "checkpoint_epoch": bundle["checkpoint_epoch"],
            },
            "c" * 64,
        )

    def build_loader(config, *, dataset_key, **kwargs):
        calls["dataset_loaders"] += 1
        is_id = dataset_key == "id_test"
        sample_ids = [f"{dataset_key}:0", f"{dataset_key}:1"]
        batch = {
            "image": torch.zeros((2, 3, 32, 32), dtype=torch.float32),
            "class_label": torch.asarray([0, 1] if is_id else [-1, -1]),
            "is_id": torch.asarray([is_id, is_id]),
            "sample_id": sample_ids,
        }
        return [batch], sample_ids, {}

    real_device = torch.device
    monkeypatch.setattr(protected, "load_task_f_checkpoint", load_checkpoint)
    monkeypatch.setattr(protected, "load_dataset_config", lambda path: {})
    monkeypatch.setattr(protected, "build_extraction_loader", build_loader)
    monkeypatch.setattr(
        protected,
        "collect_runtime_provenance",
        lambda device: {
            "device_type": "fixture",
            "accelerator": {
                "device_uuid": str(bundle["gpu_uuid"]).removeprefix("GPU-")
            },
        },
    )
    monkeypatch.setattr(protected.torch, "device", lambda value: real_device("cpu"))

    kwargs = {
        "plan": frozen_plan,
        "authorization": _authorization(frozen_plan),
        "bundle_id": bundle["bundle_id"],
        "execution_git_sha": "b" * 40,
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "dataset_config_path": tmp_path / "dataset.yaml",
        "data_root": tmp_path,
        "output_root": tmp_path / "exports",
        "device": "cuda:0",
        "batch_size": 512,
        "num_workers": 0,
    }
    first = protected.export_protected_checkpoint_bundle(**kwargs)
    assert first["status"] == "PASS"
    assert first["dataset_pass_count"] == 7
    assert first["logical_record_count"] == 28
    assert calls == {"checkpoint_loads": 1, "dataset_loaders": 7}
    assert model.tap_forward_calls == 7
    assert model.forward_calls == 8  # seven dataset passes plus one parity call
    assert len(set(first["artifact_paths"])) == 28
    assert all(Path(path).is_dir() for path in first["artifact_paths"])

    second = protected.export_protected_checkpoint_bundle(**kwargs)
    assert second["artifact_paths"] == first["artifact_paths"]
    assert len(list((tmp_path / "exports").iterdir())) == 28

    monkeypatch.setattr(
        protected,
        "collect_runtime_provenance",
        lambda device: {
            "device_type": "fixture",
            "accelerator": {"device_uuid": "GPU-wrong"},
        },
    )
    with pytest.raises(ValueError, match="runtime GPU UUID"):
        protected.export_protected_checkpoint_bundle(**kwargs)


def test_plan_and_runtime_authorization_fail_closed(frozen_plan):
    drift = copy.deepcopy(frozen_plan)
    drift["records"][0]["dataset_split"] = "id_test_openood"
    with pytest.raises(ValueError, match="hash mismatch"):
        protected.validate_protected_plan(drift)
    pending = _authorization(frozen_plan)
    pending["status"] = "PENDING"
    with pytest.raises(PermissionError, match="exact one-shot plan"):
        protected.validate_authorization(pending, plan=frozen_plan, execution_git_sha="b" * 40)


def test_feature_artifact_is_atomic_checksums_identity_and_no_overwrite(tmp_path, frozen_plan):
    record = next(
        row for row in frozen_plan["records"]
        if row["dataset_split"] == "id_test" and row["depth_tap"] == "penultimate"
    )
    kwargs = {
        "record": record,
        "plan": frozen_plan,
        "authorization": _authorization(frozen_plan),
        "execution_git_sha": "b" * 40,
        "checkpoint_sha256": "c" * 64,
        "checkpoint_epoch": int(record["checkpoint_epoch"]),
        "features": np.ones((2, 640), dtype=np.float32),
        "logits": np.asarray([[2.0] + [0.0] * 9, [0.0, 2.0] + [0.0] * 8], dtype=np.float32),
        "class_labels": np.asarray([0, 1]),
        "is_id": np.asarray([True, True]),
        "sample_ids": np.asarray(["cifar10:a", "cifar10:b"]),
        "runtime": {"device_type": "fixture"},
        "output_root": tmp_path,
    }
    path = protected.write_protected_feature_artifact(**kwargs)
    assert protected.verify_protected_feature_artifact(path)["manifest"]["record_id"] == record["record_id"]
    with pytest.raises(FileExistsError, match="already exists"):
        protected.write_protected_feature_artifact(**kwargs)
    with (path / "features.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"0")
    with pytest.raises(ValueError, match="checksum mismatch"):
        protected.verify_protected_feature_artifact(path)


def test_context_scoring_reuses_id_fit_and_checks_raw_l2_reconstruction(tmp_path, monkeypatch):
    train = np.asarray(
        [[-2.0, -0.2], [-1.8, 0.1], [-2.1, 0.3], [2.0, -0.3], [1.9, 0.2], [2.2, 0.1]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    fits = {
        "raw": fit_discriminant_geometry(train, labels),
        "l2": fit_discriminant_geometry(train / np.linalg.norm(train, axis=1)[:, None], labels),
    }
    geometry = {
        "run_id": "run-a",
        "family": "adam",
        "cell_id": protected.PRIMARY_ANCHOR_CELL,
        "training_seed": 0,
        "branch_policy": "adam_alpha_1",
        "sibling_group_id": "group-a",
        "sibling_role": "alpha_1",
        "initialization_sha256": "1" * 64,
        "data_stream_sha256": "2" * 64,
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "checkpoint_sha256": "3" * 64,
        "depth_tap": "penultimate",
    }
    monkeypatch.setattr(protected, "verify_geometry_artifact", lambda path: {"manifest": geometry})
    monkeypatch.setattr(protected, "_geometry_fit", lambda root, manifest, transform: fits[transform])
    artifact_manifests = {}
    paths = {}
    for index, split in enumerate(protected.PROTECTED_SPLITS):
        root = tmp_path / split
        root.mkdir()
        values = np.asarray([[-1.9, 0.0], [2.1, 0.0], [0.0, 2.5 + index * 0.1]], dtype=np.float32)
        if split != "id_test":
            values = values[:2]
        np.save(root / "features.npy", values, allow_pickle=False)
        if split == "id_test":
            np.save(root / "logits.npy", np.asarray([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32), allow_pickle=False)
            np.save(root / "class_labels.npy", np.asarray([0, 1, 0]), allow_pickle=False)
        record = {
            "run_id": "run-a",
            "checkpoint_role": "last",
            "checkpoint_epoch": 200,
            "depth_tap": "penultimate",
        }
        artifact_manifests[str(root)] = {
            "dataset_split": split,
            "ordered_sample_id_sha256": f"sha-{split}",
            "checkpoint_sha256": geometry["checkpoint_sha256"],
            "task_f_specification_sha256": protected.EXPECTED_SPECIFICATION_SHA256,
            "record": record,
        }
        paths[split] = root
    monkeypatch.setattr(
        protected,
        "verify_protected_feature_artifact",
        lambda path: {"manifest": artifact_manifests[str(path)]},
    )
    summary, arrays = protected.evaluate_context_arrays(
        geometry_path=tmp_path / "geometry",
        protected_artifacts=paths,
        chunk_size=3,
    )
    assert summary["status"] == "PASS"
    assert summary["id_utility"]["accuracy"] == pytest.approx(1.0)
    assert summary["component_diagnostics"]["raw"]["splits"]["id_test"]["checks_pass"]
    np.testing.assert_allclose(
        arrays["raw__id_test__md"],
        arrays["raw__id_test__marginal"] + arrays["raw__id_test__rmd"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_inapplicable_fit_preserves_direct_scores_without_component_claim(monkeypatch):
    train = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    fit = fit_discriminant_geometry(train, labels)
    assert not fit.applicable

    def unexpected_component_scoring(*_args, **_kwargs):
        raise AssertionError("component scoring must not run for an inapplicable fit")

    monkeypatch.setattr(
        protected, "score_discriminant_components", unexpected_component_scoring
    )
    queries = np.asarray([[-1.5, 0.2], [1.5, -0.2]], dtype=np.float64)

    record, arrays = protected._score_chunks(
        fit,
        queries,
        chunk_size=2,
    )

    assert record["status"] == "NOT_APPLICABLE"
    assert record["checks_required"] is False
    assert record["checks_pass"] is None
    assert record["component_arrays_emitted"] is False
    assert set(arrays) == set(protected.DETECTORS)
    assert all(np.isfinite(arrays[name]).all() for name in protected.DETECTORS)
    delta = queries[:, None, :] - fit.class_means[None, :, :]
    class_distances = np.einsum(
        "ncd,de,nce->nc", delta, fit.within_precision, delta, optimize=True
    )
    centered = queries - fit.mean
    global_distances = np.einsum(
        "nd,de,ne->n", centered, fit.global_precision, centered, optimize=True
    )
    expected = protected.mahalanobis_score_components(
        class_distances, global_distances
    )
    for detector in protected.DETECTORS:
        np.testing.assert_allclose(arrays[detector], expected[detector])


def test_applicable_fit_still_rejects_reconstruction_failure(monkeypatch):
    train = np.asarray(
        [[-2.0, -0.2], [-1.8, 0.1], [-2.1, 0.3], [2.0, -0.3], [1.9, 0.2], [2.2, 0.1]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    fit = fit_discriminant_geometry(train, labels)
    assert fit.applicable
    original = protected.score_discriminant_components

    def broken_components(fitted, queries):
        record, arrays = original(fitted, queries)
        record = copy.deepcopy(record)
        record["checks"]["score_reconstruction"] = False
        return record, arrays

    monkeypatch.setattr(protected, "score_discriminant_components", broken_components)
    with pytest.raises(ValueError, match="component reconstruction"):
        protected._score_chunks(
            fit,
            np.asarray([[-1.5, 0.2], [1.5, -0.2]], dtype=np.float64),
            chunk_size=2,
        )


def _score_arrays(offset):
    arrays = {}
    for transform in protected.TRANSFORMS:
        for split_index, split in enumerate(protected.PROTECTED_SPLITS):
            if split == "id_test":
                base = np.asarray([0.9, 0.7, 0.6])
            else:
                base = np.asarray([0.2, 0.4]) + split_index * 0.001
            md = base + offset
            arrays[f"{transform}__{split}__md"] = md
            arrays[f"{transform}__{split}__marginal"] = 0.6 * md
            arrays[f"{transform}__{split}__rmd"] = 0.4 * md
    return arrays


def _score_manifest(seed, role, offset):
    sibling_role = {
        "coupled": "alpha_1",
        "decoupled": "alpha_0",
    }[role]
    metrics = {
        transform: {
            detector: {
                "per_dataset": {
                    split: protected.compute_ood_metrics(
                        _score_arrays(offset)[f"{transform}__id_test__{detector}"],
                        _score_arrays(offset)[f"{transform}__{split}__{detector}"],
                    )
                    for split in protected.OOD_SPLITS
                }
            }
            for detector in protected.DETECTORS
        }
        for transform in protected.TRANSFORMS
    }
    return {
        "schema_version": protected.SCORE_SCHEMA_VERSION,
        "status": "PASS",
        "run_id": f"run-{seed}-{role}",
        "family": "adam",
        "cell_id": protected.PRIMARY_ANCHOR_CELL,
        "training_seed": seed,
        "branch_policy": "adam_alpha_1" if role == "coupled" else "adamw_alpha_0",
        "sibling_group_id": f"group-{seed}",
        "sibling_role": sibling_role,
        "initialization_sha256": f"{seed + 1:064x}",
        "data_stream_sha256": f"{seed + 11:064x}",
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "checkpoint_sha256": f"{seed + (1 if role == 'coupled' else 21):064x}",
        "depth_tap": "penultimate",
        "sample_order_sha256": {split: f"order-{split}" for split in protected.PROTECTED_SPLITS},
        "id_utility": {
            "status": "PASS",
            "accuracy": 0.90 + offset / 10,
            "nll": 0.30 - offset / 10,
            "ece": 0.04 - offset / 20,
        },
        "component_diagnostics": {
            transform: {"fit_applicable": True}
            for transform in protected.TRANSFORMS
        },
        "ood_metrics": metrics,
    }


def test_pair_accounting_holm_id_equivalence_and_incomplete_terminal(monkeypatch):
    fixtures = {}
    for seed in (0, 1):
        fixtures[f"{seed}-coupled"] = (_score_manifest(seed, "coupled", 0.03 + seed * 0.005), _score_arrays(0.03 + seed * 0.005))
        fixtures[f"{seed}-decoupled"] = (_score_manifest(seed, "decoupled", 0.0), _score_arrays(0.0))
    monkeypatch.setattr(protected, "_load_score_arrays", lambda path: fixtures[path.name])
    paths = [Path(name) for name in sorted(fixtures)]
    result = protected.aggregate_protected_scores(score_paths=paths, expected_contexts=4)
    assert result["status"] == "PASS"
    assert result["rtmd"] == "EXCLUDED_BY_FAILED_GATE3"
    assert result["id_equivalence"][protected.PRIMARY_ANCHOR_CELL]["comparable_id"]
    assert set(result["primary_holm_alpha_0_10"]) == {
        "near_delta_auroc",
        "near_pair_order_churn",
        "far_delta_auroc",
        "far_pair_order_churn",
    }
    primary = next(row for row in result["seed_records"] if row["detector"] == "md" and row["transform"] == "raw")
    assert abs(primary["datasets"]["cifar100"]["balance_residual"]) < 1e-12
    incomplete = protected.aggregate_protected_scores(score_paths=paths, expected_contexts=5)
    assert incomplete["status"] == "INCOMPLETE"
    assert not incomplete["missing_or_failed_records_are_excluded"]


def test_aggregate_marks_component_attribution_not_applicable(monkeypatch):
    fixtures = {}
    for seed in (0, 1):
        coupled_manifest = _score_manifest(seed, "coupled", 0.03 + seed * 0.005)
        coupled_manifest["component_diagnostics"] = {
            transform: {"fit_applicable": False}
            for transform in protected.TRANSFORMS
        }
        fixtures[f"{seed}-coupled"] = (
            coupled_manifest,
            _score_arrays(0.03 + seed * 0.005),
        )
        fixtures[f"{seed}-decoupled"] = (
            _score_manifest(seed, "decoupled", 0.0),
            _score_arrays(0.0),
        )
    monkeypatch.setattr(protected, "_load_score_arrays", lambda path: fixtures[path.name])

    result = protected.aggregate_protected_scores(
        score_paths=[Path(name) for name in sorted(fixtures)],
        expected_contexts=4,
    )

    assert result["status"] == "PASS"
    assert result["component_applicability"] == {
        "raw": {"applicable_contexts": 2, "inapplicable_contexts": 2},
        "l2": {"applicable_contexts": 2, "inapplicable_contexts": 2},
    }
    primary = next(
        row
        for row in result["seed_records"]
        if row["detector"] == "md" and row["transform"] == "raw"
    )
    for split in protected.OOD_SPLITS:
        assert primary["datasets"][split]["component_attribution_status"] == "NOT_APPLICABLE"
        assert "component_attribution" not in primary["datasets"][split]


def test_exact_sign_flip_band_and_holm_are_deterministic():
    values = np.asarray(
        [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.15, 0.25, 0.35]],
        dtype=np.float64,
    )
    first = protected.simultaneous_sign_flip_band(values)
    second = protected.simultaneous_sign_flip_band(values)
    assert first == second
    assert first["sign_flip_count"] == 8
    adjusted = protected.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20, "d": 0.30})
    assert adjusted["a"]["reject"]
    assert not adjusted["b"]["reject"]


def test_gain_loss_direction_is_coupled_minus_decoupled():
    left = {
        "raw__id_test__rmd": np.asarray([2.0, 0.0]),
        "raw__cifar100__rmd": np.asarray([1.0]),
    }
    right = {
        "raw__id_test__rmd": np.asarray([0.0, 0.0]),
        "raw__cifar100__rmd": np.asarray([1.0]),
    }
    result = protected.compare_score_arrays(
        left=left,
        right=right,
        split="cifar100",
        transform="raw",
        detector="rmd",
    )
    assert result["delta_auroc"] == pytest.approx(0.5)
    assert result["gain"] == pytest.approx(0.5)
    assert result["loss"] == pytest.approx(0.0)
    assert result["balance_residual"] == pytest.approx(0.0)
