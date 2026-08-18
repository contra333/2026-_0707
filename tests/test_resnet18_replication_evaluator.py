import copy

import numpy as np
import pytest

from oge.evaluation.resnet18_replication import (
    LARGE_CELL,
    SMALL_CELL,
    adjudicate_resnet18_full_gate,
    build_pending_resnet18_evaluation_plan,
    evaluate_synthetic_paired_endpoint_arrays,
    validate_endpoint_artifact_manifests,
    validate_pending_resnet18_evaluation_plan,
)
from oge.training.resnet18_replication_plan import (
    RESNET18_REPLICATION_STUDY_ID,
    generate_resnet18_replication_matrix,
)


def _run_ids(plan):
    return [str(row["run_id"]) for row in plan["runs"]]


def _endpoint_manifests(plan):
    return [
        {
            "artifact_namespace": RESNET18_REPLICATION_STUDY_ID,
            "run_id": row["run_id"],
            "checkpoint_role": "last",
            "checkpoint_epoch": 200,
            "depth_tap": "penultimate",
            "dataset_split": "id_train",
            "feature_shape": [45_000, 512],
            "classifier_weight_shape": [10, 512],
        }
        for row in plan["runs"]
    ]


def _synthetic_inputs():
    rng = np.random.default_rng(20260818)
    labels = np.repeat(np.arange(3), 12)
    centers = np.asarray(
        [
            [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    coupled_train = centers[labels] + rng.normal(scale=0.35, size=(36, 6))
    decoupled_train = coupled_train @ np.asarray(
        [
            [1.00, 0.05, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.95, 0.03, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.03, 0.02, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.97, 0.04, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.02, 0.01],
            [0.01, 0.00, 0.00, 0.00, 0.00, 0.98],
        ]
    ) + rng.normal(scale=0.03, size=(36, 6))

    coupled_queries = {
        "id_test": centers[np.arange(18) % 3]
        + rng.normal(scale=0.4, size=(18, 6))
    }
    decoupled_queries = {
        "id_test": coupled_queries["id_test"]
        + rng.normal(scale=0.04, size=(18, 6))
    }
    for index, split in enumerate(
        ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
    ):
        offset = np.zeros(6)
        offset[(index + 2) % 6] = 2.5 + 0.2 * index
        coupled = rng.normal(scale=0.9, size=(13, 6)) + offset
        coupled_queries[split] = coupled
        decoupled_queries[split] = coupled + rng.normal(scale=0.08, size=(13, 6))
    return labels, coupled_train, decoupled_train, coupled_queries, decoupled_queries


def _research_seed_records(plan):
    by_cell_seed = {}
    for run in plan["runs"]:
        by_cell_seed.setdefault((run["cell_id"], run["training_seed"]), {})[
            run["branch_policy"]
        ] = run["run_id"]

    records = []
    for (cell, seed), members in sorted(by_cell_seed.items()):
        if cell == LARGE_CELL:
            md_delta = -0.20
            rmd_delta = -0.04
            phi_marginal = -0.16
            churn = 0.20
        else:
            md_delta = -0.02
            rmd_delta = -0.01
            phi_marginal = -0.015
            churn = 0.06
        records.append(
            {
                "status": "PASS",
                "research_evidence": True,
                "cell_id": cell,
                "training_seed": seed,
                "source_run_ids": {
                    "coupled": members["adam_coupled"],
                    "decoupled": members["adamw_decoupled"],
                },
                "macro": {
                    group: {
                        "md": {
                            "delta_auroc": md_delta,
                            "pair_order_churn": churn,
                            "phi_marginal": phi_marginal,
                        },
                        "rmd": {"delta_auroc": rmd_delta},
                    }
                    for group in ("near", "far")
                },
            }
        )
    return records


def test_pending_plan_is_endpoint_only_and_cannot_bind_protected_paths():
    plan = generate_resnet18_replication_matrix()
    pending = build_pending_resnet18_evaluation_plan(
        run_plan=plan, planning_git_sha="a" * 40
    )
    assert pending["expected_coverage"] == {
        "runs": 20,
        "paired_seed_records": 10,
        "cells": 2,
        "seeds_per_cell": 5,
    }
    assert pending["protected_executor"] == "NOT_IMPLEMENTED_IN_PREAPPROVAL_SCAFFOLD"
    assert set(pending["protected_paths"].values()) == {None}
    assert {
        (row["checkpoint_role"], row["checkpoint_epoch"], row["depth_tap"])
        for row in pending["records"]
    } == {("last", 200, "penultimate")}

    broken = copy.deepcopy(pending)
    broken["protected_paths"]["id_test"] = "/protected/id-test"
    broken_without_hash = dict(broken)
    broken_without_hash.pop("plan_sha256")
    from oge.studies.hashing import canonical_sha256

    broken["plan_sha256"] = canonical_sha256(broken_without_hash)
    with pytest.raises(ValueError, match="protected paths"):
        validate_pending_resnet18_evaluation_plan(broken)


def test_endpoint_manifest_coverage_rejects_non_last_or_incomplete_sources():
    plan = generate_resnet18_replication_matrix()
    manifests = _endpoint_manifests(plan)
    assert len(
        validate_endpoint_artifact_manifests(
            manifests, expected_run_ids=_run_ids(plan)
        )
    ) == 20

    broken = copy.deepcopy(manifests)
    broken[0]["checkpoint_role"] = "best_val"
    with pytest.raises(ValueError, match="frozen endpoint"):
        validate_endpoint_artifact_manifests(
            broken, expected_run_ids=_run_ids(plan)
        )
    with pytest.raises(ValueError, match="exactly 20"):
        validate_endpoint_artifact_manifests(
            manifests[:-1], expected_run_ids=_run_ids(plan)
        )


def test_synthetic_evaluator_reuses_pair_and_shapley_accounting_contracts():
    labels, coupled_train, decoupled_train, coupled_queries, decoupled_queries = (
        _synthetic_inputs()
    )
    record = evaluate_synthetic_paired_endpoint_arrays(
        cell_id=LARGE_CELL,
        training_seed=0,
        coupled_run_id="coupled-fixture",
        decoupled_run_id="decoupled-fixture",
        coupled_train_features=coupled_train,
        decoupled_train_features=decoupled_train,
        train_labels=labels,
        coupled_queries=coupled_queries,
        decoupled_queries=decoupled_queries,
    )
    assert record["status"] == "SYNTHETIC_FIXTURE_ONLY"
    assert record["research_evidence"] is False
    for detectors in record["datasets"].values():
        assert set(detectors) == {"md", "rmd", "marginal", "l2_md"}
        for row in detectors.values():
            assert row["balance_residual"] == pytest.approx(0.0, abs=1e-12)
            assert row["delta_auroc"] == pytest.approx(
                row["gain"] - row["loss"], abs=1e-12
            )
            assert row["pair_order_churn"] == pytest.approx(
                row["gain"] + row["loss"], abs=1e-12
            )
        md = detectors["md"]
        attribution = md["component_attribution"]
        assert attribution["pass"] is True
        assert attribution["auroc_delta"] == pytest.approx(
            attribution["component_auroc_attribution"]["rmd"]
            + attribution["component_auroc_attribution"]["marginal"],
            abs=1e-12,
        )
    assert set(record["macro"]) == {"near", "far"}
    assert all("l2_md" in record["macro"][group] for group in ("near", "far"))


def test_full_gate_requires_exact_research_coverage_and_prespecified_pattern():
    plan = generate_resnet18_replication_matrix()
    run_ids = _run_ids(plan)
    records = _research_seed_records(plan)
    full = adjudicate_resnet18_full_gate(
        expected_run_ids=run_ids,
        observed_run_ids=run_ids,
        seed_records=records,
        id_guardrail_by_cell={SMALL_CELL: "PASS"},
    )
    assert full["verdict"] == "FULL"
    assert full["full_gate_evaluated"] is True
    assert all(full["checks"].values())

    incomplete = adjudicate_resnet18_full_gate(
        expected_run_ids=run_ids,
        observed_run_ids=run_ids[:-1],
        seed_records=records,
        id_guardrail_by_cell={SMALL_CELL: "PASS"},
    )
    assert incomplete["verdict"] == "BLOCKED"
    assert incomplete["full_gate_evaluated"] is False

    partial_records = copy.deepcopy(records)
    for record in partial_records:
        if record["cell_id"] == LARGE_CELL:
            record["macro"]["near"]["rmd"]["delta_auroc"] = -0.25
    partial = adjudicate_resnet18_full_gate(
        expected_run_ids=run_ids,
        observed_run_ids=run_ids,
        seed_records=partial_records,
        id_guardrail_by_cell={SMALL_CELL: "PASS"},
    )
    assert partial["verdict"] == "PARTIAL"

    failed_records = copy.deepcopy(records)
    for record in failed_records:
        if record["cell_id"] == LARGE_CELL:
            for group in ("near", "far"):
                record["macro"][group]["md"]["delta_auroc"] = 0.20
    failed = adjudicate_resnet18_full_gate(
        expected_run_ids=run_ids,
        observed_run_ids=run_ids,
        seed_records=failed_records,
        id_guardrail_by_cell={SMALL_CELL: "PASS"},
    )
    assert failed["verdict"] == "FAIL"

    synthetic = copy.deepcopy(records)
    synthetic[0]["status"] = "SYNTHETIC_FIXTURE_ONLY"
    synthetic[0]["research_evidence"] = False
    with pytest.raises(ValueError, match="research-evidence"):
        adjudicate_resnet18_full_gate(
            expected_run_ids=run_ids,
            observed_run_ids=run_ids,
            seed_records=synthetic,
            id_guardrail_by_cell={SMALL_CELL: "PASS"},
        )
