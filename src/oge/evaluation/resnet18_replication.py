"""Bounded evaluation scaffold for the ResNet-18/CIFAR-10 replication.

The committed CLI can only materialize a pending plan or adjudicate an
already-produced summary.  It has no protected feature loader or inference
entrypoint.  The array evaluator is explicitly synthetic-only and exists to
lock the reused Task F score and pair-accounting contract in CPU tests.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from oge.analysis.discriminant_residual_preflight import (
    fit_discriminant_geometry,
    score_discriminant_components,
)
from oge.analysis.fixed_readout_component_attribution import (
    pair_outcome_summary,
    reconstruction_record,
)
from oge.evaluation.task_f_protected import (
    FAR_SPLITS,
    NEAR_SPLITS,
    OOD_SPLITS,
    _transform as task_f_feature_transform,
    compare_score_arrays,
)
from oge.studies.hashing import canonical_sha256
from oge.training.resnet18_replication_plan import (
    RESNET18_REPLICATION_LRS,
    RESNET18_REPLICATION_ROLES,
    RESNET18_REPLICATION_STUDY_ID,
    validate_resnet18_replication_matrix,
)


RESNET18_REPLICATION_EVALUATION_PLAN_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_evaluation_plan_v3"
)
RESNET18_REPLICATION_SEED_RECORD_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_seed_record_v3"
)
RESNET18_REPLICATION_GATE_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_full_gate_v3"
)
EXPECTED_PROTECTED_SPLITS = ("id_test", *OOD_SPLITS)
LARGE_CELL = "resnet18_c10_lr1e-03_wd1e-4"
SMALL_CELL = "resnet18_c10_lr3e-04_wd1e-4"
DETECTORS = ("md", "rmd", "marginal")
EVALUATION_TARGETS = (*DETECTORS, "l2_md")


def build_pending_resnet18_evaluation_plan(
    *, run_plan: Mapping[str, Any], planning_git_sha: str
) -> dict[str, Any]:
    """Freeze coverage without opening or naming a protected data path."""

    validate_resnet18_replication_matrix(run_plan)
    if not isinstance(planning_git_sha, str) or len(planning_git_sha) != 40:
        raise ValueError("planning_git_sha must be a full Git SHA")
    records = []
    for run in sorted(run_plan["runs"], key=lambda row: str(row["run_id"])):
        records.append(
            {
                "run_id": str(run["run_id"]),
                "cell_id": str(run["cell_id"]),
                "training_seed": int(run["training_seed"]),
                "branch_policy": str(run["branch_policy"]),
                "sibling_group_id": str(run["sibling_group_id"]),
                "cross_lr_pairing_block_id": str(
                    run["cross_lr_pairing_block_id"]
                ),
                "checkpoint_role": "last",
                "checkpoint_epoch": 200,
                "depth_tap": "penultimate",
                "id_fit_splits": ["id_train"],
                "evaluation_splits": list(EXPECTED_PROTECTED_SPLITS),
            }
        )
    payload = {
        "schema_version": RESNET18_REPLICATION_EVALUATION_PLAN_SCHEMA_VERSION,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "planning_git_sha": planning_git_sha,
        "protected_data_access": False,
        "launch_authorization": "PENDING_EXPLICIT_OWNER_APPROVAL",
        "protected_executor": "NOT_IMPLEMENTED_IN_PREAPPROVAL_SCAFFOLD",
        "source_contract": {
            "checkpoint_role": "last",
            "checkpoint_epoch": 200,
            "depth_tap": "penultimate",
            "feature_dim": 512,
            "artifact_namespace": RESNET18_REPLICATION_STUDY_ID,
        },
        "score_contract": {
            "targets": ["raw_md", "raw_rmd", "raw_marginal", "l2_md"],
            "pair_metrics": ["gain", "loss", "pair_order_churn", "delta_auroc"],
            "dataset_groups": {
                "near": list(NEAR_SPLITS),
                "far": list(FAR_SPLITS),
            },
            "statistical_unit": "training_seed",
        },
        "protected_paths": {split: None for split in EXPECTED_PROTECTED_SPLITS},
        "expected_coverage": {
            "runs": 20,
            "paired_seed_records": 10,
            "cells": 2,
            "seeds_per_cell": 5,
        },
        "records": records,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return validate_pending_resnet18_evaluation_plan(payload)


def validate_pending_resnet18_evaluation_plan(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    observed_hash = payload.pop("plan_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise ValueError("replication evaluation plan hash mismatch")
    if payload.get("schema_version") != RESNET18_REPLICATION_EVALUATION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported replication evaluation plan schema")
    if payload.get("protected_data_access") is not False:
        raise ValueError("preapproval plan must declare protected_data_access=false")
    if payload.get("launch_authorization") != "PENDING_EXPLICIT_OWNER_APPROVAL":
        raise ValueError("preapproval plan must remain pending explicit approval")
    if payload.get("protected_executor") != "NOT_IMPLEMENTED_IN_PREAPPROVAL_SCAFFOLD":
        raise ValueError("preapproval plan must not expose a protected executor")
    paths = payload.get("protected_paths")
    if not isinstance(paths, Mapping) or set(paths) != set(EXPECTED_PROTECTED_SPLITS):
        raise ValueError("replication evaluation plan split catalog mismatch")
    if any(path is not None for path in paths.values()):
        raise ValueError("preapproval plan must not contain protected paths")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("replication evaluation plan must cover exactly 20 runs")
    if len({row.get("run_id") for row in records}) != 20:
        raise ValueError("replication evaluation plan run IDs must be unique")
    for row in records:
        if (
            row.get("checkpoint_role") != "last"
            or row.get("checkpoint_epoch") != 200
            or row.get("depth_tap") != "penultimate"
            or row.get("id_fit_splits") != ["id_train"]
            or tuple(row.get("evaluation_splits", ())) != EXPECTED_PROTECTED_SPLITS
        ):
            raise ValueError("replication evaluation plan is not endpoint-only")
    payload["plan_sha256"] = observed_hash
    return payload


def validate_endpoint_artifact_manifests(
    manifests: Sequence[Mapping[str, Any]], *, expected_run_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Validate only manifest metadata; no array or protected path is opened."""

    expected = set(str(run_id) for run_id in expected_run_ids)
    if len(expected) != 20 or len(manifests) != 20:
        raise ValueError("endpoint coverage requires exactly 20 unique runs")
    normalized = []
    for manifest in manifests:
        if (
            manifest.get("artifact_namespace") != RESNET18_REPLICATION_STUDY_ID
            or manifest.get("checkpoint_role") != "last"
            or manifest.get("checkpoint_epoch") != 200
            or manifest.get("depth_tap") != "penultimate"
            or manifest.get("dataset_split") != "id_train"
            or manifest.get("feature_shape", [None, None])[1] != 512
            or manifest.get("classifier_weight_shape") != [10, 512]
        ):
            raise ValueError("replication source artifact is not the frozen endpoint")
        normalized.append(copy.deepcopy(dict(manifest)))
    observed = [str(row.get("run_id")) for row in normalized]
    if len(set(observed)) != 20 or set(observed) != expected:
        raise ValueError("replication endpoint run coverage mismatch")
    return sorted(normalized, key=lambda row: str(row["run_id"]))


def _score_panel(
    train_features: Any, train_labels: Any, queries: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    train = np.asarray(train_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if train.ndim != 2 or len(train) != len(labels):
        raise ValueError("synthetic ID-train features and labels must align")
    if set(queries) != set(EXPECTED_PROTECTED_SPLITS):
        raise ValueError("synthetic query panel must use the exact seven split names")
    arrays: dict[str, np.ndarray] = {}
    for transform in ("raw", "l2"):
        transformed_train = task_f_feature_transform(train, transform)
        fit = fit_discriminant_geometry(transformed_train, labels)
        for split in EXPECTED_PROTECTED_SPLITS:
            values = np.asarray(queries[split], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != train.shape[1]:
                raise ValueError("synthetic query feature dimensions differ from ID train")
            transformed_values = task_f_feature_transform(values, transform)
            _, scored = score_discriminant_components(fit, transformed_values)
            for detector in DETECTORS:
                score = np.asarray(scored[detector], dtype=np.float64)
                arrays[f"{transform}__{split}__{detector}"] = score
            check = reconstruction_record(
                arrays[f"{transform}__{split}__md"],
                arrays[f"{transform}__{split}__rmd"],
                arrays[f"{transform}__{split}__marginal"],
                condition_number=(
                    fit.condition_number
                    if math.isfinite(fit.condition_number)
                    else None
                ),
            )
            if not check["pass"]:
                raise ValueError("synthetic MD=RMD+Marginal reconstruction failed")
    return arrays


def evaluate_synthetic_paired_endpoint_arrays(
    *,
    cell_id: str,
    training_seed: int,
    coupled_run_id: str,
    decoupled_run_id: str,
    coupled_train_features: Any,
    decoupled_train_features: Any,
    train_labels: Any,
    coupled_queries: Mapping[str, Any],
    decoupled_queries: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise production score/accounting math on synthetic arrays only."""

    if cell_id not in {LARGE_CELL, SMALL_CELL}:
        raise ValueError("synthetic fixture cell_id is not in the replication matrix")
    left = _score_panel(coupled_train_features, train_labels, coupled_queries)
    right = _score_panel(decoupled_train_features, train_labels, decoupled_queries)
    datasets: dict[str, Any] = {}
    for split in OOD_SPLITS:
        detector_rows = {}
        for target in EVALUATION_TARGETS:
            transform, detector = (
                ("l2", "md") if target == "l2_md" else ("raw", target)
            )
            comparison = compare_score_arrays(
                left=left,
                right=right,
                split=split,
                transform=transform,
                detector=detector,
            )
            left_outcome = pair_outcome_summary(
                left[f"{transform}__id_test__{detector}"],
                left[f"{transform}__{split}__{detector}"],
            )
            right_outcome = pair_outcome_summary(
                right[f"{transform}__id_test__{detector}"],
                right[f"{transform}__{split}__{detector}"],
            )
            detector_rows[target] = {
                "coupled_auroc": float(left_outcome["auroc_id_positive"]),
                "decoupled_auroc": float(right_outcome["auroc_id_positive"]),
                **comparison,
            }
        datasets[split] = detector_rows

    macro: dict[str, Any] = {}
    for group, splits in (("near", NEAR_SPLITS), ("far", FAR_SPLITS)):
        macro[group] = {}
        for target in EVALUATION_TARGETS:
            rows = [datasets[split][target] for split in splits]
            macro[group][target] = {
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in (
                    "coupled_auroc",
                    "decoupled_auroc",
                    "gain",
                    "loss",
                    "pair_order_churn",
                    "delta_auroc",
                    "balance_residual",
                )
            }
        attributions = [
            datasets[split]["md"]["component_attribution"] for split in splits
        ]
        macro[group]["md"]["phi_rmd"] = float(
            np.mean(
                [
                    row["component_auroc_attribution"]["rmd"]
                    for row in attributions
                ]
            )
        )
        macro[group]["md"]["phi_marginal"] = float(
            np.mean(
                [
                    row["component_auroc_attribution"]["marginal"]
                    for row in attributions
                ]
            )
        )
    return {
        "schema_version": RESNET18_REPLICATION_SEED_RECORD_SCHEMA_VERSION,
        "status": "SYNTHETIC_FIXTURE_ONLY",
        "research_evidence": False,
        "cell_id": cell_id,
        "training_seed": int(training_seed),
        "direction": "coupled_minus_decoupled",
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "source_run_ids": {
            "coupled": str(coupled_run_id),
            "decoupled": str(decoupled_run_id),
        },
        "datasets": datasets,
        "macro": macro,
    }


def adjudicate_resnet18_full_gate(
    *,
    expected_run_ids: Sequence[str],
    observed_run_ids: Sequence[str],
    seed_records: Sequence[Mapping[str, Any]],
    id_guardrail_by_cell: Mapping[str, str],
) -> dict[str, Any]:
    """Apply the prospective FULL/PARTIAL/FAIL rule after exact 20/20 coverage."""

    expected = set(str(value) for value in expected_run_ids)
    observed = [str(value) for value in observed_run_ids]
    coverage_pass = (
        len(expected) == 20
        and len(observed) == 20
        and len(set(observed)) == 20
        and set(observed) == expected
    )
    if not coverage_pass or len(seed_records) != 10:
        return {
            "schema_version": RESNET18_REPLICATION_GATE_SCHEMA_VERSION,
            "verdict": "BLOCKED",
            "coverage": {
                "expected_runs": len(expected),
                "observed_runs": len(set(observed)),
                "expected_seed_records": 10,
                "observed_seed_records": len(seed_records),
            },
            "full_gate_evaluated": False,
        }

    source_run_ids = [
        str(run_id)
        for row in seed_records
        for run_id in (
            row.get("source_run_ids", {}).get("coupled"),
            row.get("source_run_ids", {}).get("decoupled"),
        )
    ]
    source_coverage_pass = (
        len(source_run_ids) == 20
        and len(set(source_run_ids)) == 20
        and set(source_run_ids) == expected
    )
    if not source_coverage_pass:
        return {
            "schema_version": RESNET18_REPLICATION_GATE_SCHEMA_VERSION,
            "verdict": "BLOCKED",
            "coverage": {
                "expected_runs": 20,
                "observed_runs": 20,
                "source_runs": len(set(source_run_ids)),
                "expected_seed_records": 10,
                "observed_seed_records": 10,
            },
            "full_gate_evaluated": False,
        }
    if any(
        row.get("status") != "PASS" or row.get("research_evidence") is not True
        for row in seed_records
    ):
        raise ValueError(
            "FULL gate accepts only validated research-evidence seed records"
        )

    by_cell: dict[str, list[Mapping[str, Any]]] = {LARGE_CELL: [], SMALL_CELL: []}
    for row in seed_records:
        cell = str(row.get("cell_id"))
        if cell not in by_cell:
            raise ValueError("seed record contains an unknown replication cell")
        by_cell[cell].append(row)
    if any(len(rows) != 5 for rows in by_cell.values()):
        raise ValueError("each replication cell must contain exactly five seed records")
    if any(len({int(row["training_seed"]) for row in rows}) != 5 for rows in by_cell.values()):
        raise ValueError("replication cell contains duplicate seed records")

    def values(cell: str, group: str, detector: str, metric: str) -> np.ndarray:
        return np.asarray(
            [float(row["macro"][group][detector][metric]) for row in by_cell[cell]],
            dtype=np.float64,
        )

    checks: dict[str, bool] = {}
    for group in ("near", "far"):
        large_md = values(LARGE_CELL, group, "md", "delta_auroc")
        small_md = values(SMALL_CELL, group, "md", "delta_auroc")
        large_rmd = values(LARGE_CELL, group, "rmd", "delta_auroc")
        large_churn = values(LARGE_CELL, group, "md", "pair_order_churn")
        phi_marginal = values(LARGE_CELL, group, "md", "phi_marginal")
        checks[f"{group}_large_negative_mean"] = float(np.mean(large_md)) < 0.0
        checks[f"{group}_large_seed_direction_4_of_5"] = int(
            np.count_nonzero(large_md < 0.0)
        ) >= 4
        checks[f"{group}_large_churn_at_least_0_10"] = float(
            np.mean(large_churn)
        ) >= 0.10
        checks[f"{group}_small_gap_smaller"] = abs(float(np.mean(small_md))) < abs(
            float(np.mean(large_md))
        )
        checks[f"{group}_rmd_gap_smaller"] = abs(float(np.mean(large_rmd))) < abs(
            float(np.mean(large_md))
        )
        checks[f"{group}_marginal_same_sign"] = (
            float(np.mean(phi_marginal)) * float(np.mean(large_md)) > 0.0
        )
        checks[f"{group}_marginal_over_half"] = abs(
            float(np.mean(phi_marginal))
        ) > 0.5 * abs(float(np.mean(large_md)))
    checks["small_cell_id_guardrail_all_pass"] = (
        id_guardrail_by_cell.get(SMALL_CELL) == "PASS"
    )
    full = all(checks.values())
    core = all(
        checks[f"{group}_large_negative_mean"]
        and checks[f"{group}_large_seed_direction_4_of_5"]
        and checks[f"{group}_large_churn_at_least_0_10"]
        for group in ("near", "far")
    )
    verdict = "FULL" if full else "PARTIAL" if core else "FAIL"
    return {
        "schema_version": RESNET18_REPLICATION_GATE_SCHEMA_VERSION,
        "verdict": verdict,
        "coverage": {
            "expected_runs": 20,
            "observed_runs": 20,
            "expected_seed_records": 10,
            "observed_seed_records": 10,
        },
        "full_gate_evaluated": True,
        "checks": checks,
    }
