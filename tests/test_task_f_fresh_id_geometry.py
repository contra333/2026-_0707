import hashlib
import json
import math

import numpy as np
import pytest

from oge.analysis.discriminant_residual_preflight import fit_discriminant_geometry
from oge.analysis.task_f_fresh_id import (
    adjudicate_id_equivalence,
    aggregate_paired_records,
    analyze_alignment_arrays,
    analyze_bound_alignment,
    analyze_geometry_arrays,
    build_aggregation_contract,
    classify_alpha_interior,
    fit_affine_alignment,
    fit_gauge_alignment,
    paired_t_interval,
    render_table_templates,
    write_aggregation_artifacts,
)
import oge.analysis.task_f_fresh_id as fresh_id_module
from oge.evaluation.task_f_fresh import build_fresh_evaluation_plan
from oge.training import generate_research_run_matrix


def _geometry_fixture(seed=7, *, rank_deficient=False):
    rng = np.random.default_rng(seed)
    dimension = 4
    means = np.asarray(
        [
            [1.0, 0.2, -0.1, 0.3],
            [-0.4, 1.1, 0.2, -0.2],
            [-0.6, -0.8, 0.5, 0.1],
        ]
    )
    labels = np.repeat(np.arange(3), 24)
    validation_labels = np.repeat(np.arange(3), 10)
    train_noise = rng.normal(scale=0.35, size=(len(labels), dimension))
    validation_noise = rng.normal(scale=0.35, size=(len(validation_labels), dimension))
    if rank_deficient:
        train_noise[:, 2:] = train_noise[:, :2]
        validation_noise[:, 2:] = validation_noise[:, :2]
        means[:, 2:] = means[:, :2]
    train = means[labels] + train_noise
    validation = means[validation_labels] + validation_noise
    weight = np.asarray(
        [
            [1.1, -0.2, 0.3, 0.1],
            [-0.3, 0.9, 0.1, -0.2],
            [-0.6, -0.7, 0.4, 0.2],
        ]
    )
    bias = np.asarray([0.1, -0.1, 0.0])
    return {
        "train": train,
        "labels": labels,
        "validation": validation,
        "validation_labels": validation_labels,
        "weight": weight,
        "bias": bias,
        "train_logits": train @ weight.T + bias,
        "validation_logits": validation @ weight.T + bias,
    }


def test_aggregation_contract_is_derived_from_the_exact_1320_plan():
    run_plan = generate_research_run_matrix(
        anchor_seeds=(0, 1, 2, 3, 4),
        adam_factorial_seeds=(0, 1, 2),
        sgdm_seeds=(0, 1, 2),
    )
    contract = build_aggregation_contract(build_fresh_evaluation_plan(run_plan))
    assert contract["counts"] == {"cells": 5, "contexts": 186}
    assert contract["expected_seeds_by_cell"][
        "adam_lr1e-3_wd1e-4_anchor"
    ] == [0, 1, 2, 3, 4]
    assert contract["expected_seeds_by_cell"]["sgdm_lr0.1_wd5e-4"] == [0, 1, 2]
    assert any(
        row["checkpoint_role"] == "best_val"
        and row["dataset_split"] == "id_validation"
        for row in contract["contexts"]
    )


def test_raw_and_l2_geometry_reconstruct_components_and_record_nc():
    fixture = _geometry_fixture()
    summary, fits, arrays = analyze_geometry_arrays(
        train_features=fixture["train"],
        train_labels=fixture["labels"],
        validation_features=fixture["validation"],
        validation_labels=fixture["validation_labels"],
        depth_tap="penultimate",
        classifier_weight=fixture["weight"],
        classifier_bias=fixture["bias"],
        train_logits=fixture["train_logits"],
        validation_logits=fixture["validation_logits"],
        chunk_size=19,
    )
    assert set(summary["transforms"]) == {"raw", "l2"}
    assert set(fits) >= {"raw__subspace_basis", "l2__within_covariance"}
    for transform in ("raw", "l2"):
        components = arrays[f"{transform}__id_validation__components"]
        md = arrays[f"{transform}__id_validation__md"]
        assert np.allclose(md, np.sum(components, axis=1), rtol=1e-10, atol=1e-10)
        assert summary["transforms"][transform]["components"]["id_train"][
            "all_chunk_checks_pass"
        ]
        collapse = summary["transforms"][transform]["neural_collapse"]
        assert collapse["nc0_row_sum_raw"]["status"] == "success"
        assert collapse["nc4"]["agreement_with_bias"]["status"] == "success"
        assert summary["transforms"][transform]["id_utility"]["scope"].startswith(
            "descriptive_id_validation"
        )


def test_rank_deficient_fit_is_inapplicable_and_stage_classifier_metrics_are_na():
    fixture = _geometry_fixture(rank_deficient=True)
    summary, _, arrays = analyze_geometry_arrays(
        train_features=fixture["train"],
        train_labels=fixture["labels"],
        validation_features=fixture["validation"],
        validation_labels=fixture["validation_labels"],
        depth_tap="stage2",
        chunk_size=100,
    )
    assert summary["transforms"]["raw"]["status"] == "INAPPLICABLE"
    collapse = summary["transforms"]["raw"]["neural_collapse"]
    assert collapse["nc3_self_duality_raw"] == {
        "value": None,
        "status": "not_applicable",
        "reason_codes": ["classifier_reads_penultimate_only"],
    }
    assert collapse["nc4"]["status"] == "NOT_APPLICABLE"
    assert arrays["raw__id_train__q_perp"].shape == (len(fixture["train"]),)


def test_affine_alignment_generalizes_known_map_to_heldout_id():
    rng = np.random.default_rng(11)
    source = rng.normal(size=(120, 3))
    heldout_source = rng.normal(size=(40, 3))
    matrix = np.asarray([[1.2, 0.2, -0.1], [0.1, 0.8, 0.3], [-0.2, 0.1, 1.1]])
    bias = np.asarray([0.5, -0.4, 0.2])
    target = source @ matrix + bias
    heldout_target = heldout_source @ matrix + bias
    record, fitted = fit_affine_alignment(
        source_train=source,
        target_train=target,
        source_validation=heldout_source,
        target_validation=heldout_target,
        chunk_size=17,
    )
    assert record["status"] == "PASS"
    assert record["id_train"]["normalized_frobenius"] < 1e-10
    assert record["id_validation"]["normalized_frobenius"] < 1e-10
    assert np.allclose(fitted["matrix"], matrix, atol=1e-10)
    assert np.allclose(fitted["bias"], bias, atol=1e-10)


def test_whitened_gauge_recovers_rotation_and_principal_angles():
    fixture = _geometry_fixture()
    left = fixture["train"]
    left_validation = fixture["validation"]
    rotation, _ = np.linalg.qr(np.asarray(
        [[1.0, 0.3, 0.2, -0.1], [-0.2, 1.0, 0.1, 0.2], [0.1, -0.2, 1.0, 0.3], [0.2, 0.1, -0.3, 1.0]]
    ))
    right = left @ rotation
    right_validation = left_validation @ rotation
    left_fit = fit_discriminant_geometry(left, fixture["labels"])
    right_fit = fit_discriminant_geometry(right, fixture["labels"])
    record, state = fit_gauge_alignment(
        left_train=left,
        right_train=right,
        left_validation=left_validation,
        right_validation=right_validation,
        left_fit=left_fit,
        right_fit=right_fit,
        reference_role="zero",
        chunk_size=13,
    )
    assert record["status"] == "PASS"
    assert record["zero_decay_common_frame"]
    assert record["id_train_normalized_residual"] < 1e-9
    assert record["id_validation_normalized_residual"] < 1e-9
    assert max(record["principal_angles_degrees"], default=0.0) < 1e-5
    assert state["rotation"].shape == (4, 4)


def test_raw_l2_alignment_pipeline_records_zero_common_frame():
    fixture = _geometry_fixture()
    rotation, _ = np.linalg.qr(np.asarray(
        [[1.0, 0.2, -0.1, 0.0], [-0.2, 1.0, 0.1, 0.2], [0.1, 0.0, 1.0, -0.2], [0.0, -0.2, 0.2, 1.0]]
    ))
    record, states = analyze_alignment_arrays(
        left_train=fixture["train"],
        right_train=fixture["train"] @ rotation,
        left_validation=fixture["validation"],
        right_validation=fixture["validation"] @ rotation,
        labels=fixture["labels"],
        reference_role="zero",
        chunk_size=23,
    )
    assert set(record["transforms"]) == {"raw", "l2"}
    assert all(row["status"] == "PASS" for row in record["transforms"].values())
    assert all(
        row["gauge"]["zero_decay_common_frame"]
        for row in record["transforms"].values()
    )
    assert states["raw__matrix"].shape == (4, 4)
    assert states["l2__rotation"].shape == (4, 4)


def test_bound_alignment_publishes_checksummed_sibling_artifact(tmp_path, monkeypatch):
    fixture = _geometry_fixture()
    rotation, _ = np.linalg.qr(
        np.asarray(
            [
                [1.0, 0.2, -0.1, 0.0],
                [-0.2, 1.0, 0.1, 0.2],
                [0.1, 0.0, 1.0, -0.2],
                [0.0, -0.2, 0.2, 1.0],
            ]
        )
    )
    data = {
        "left_train": fixture["train"],
        "left_validation": fixture["validation"],
        "right_train": fixture["train"] @ rotation,
        "right_validation": fixture["validation"] @ rotation,
    }
    verified = {}
    for name, values in data.items():
        feature_root = tmp_path / name / "feature"
        bridge_root = tmp_path / name / "bridge"
        feature_root.mkdir(parents=True)
        bridge_root.mkdir()
        np.save(feature_root / "features.npy", values)
        if name.endswith("train"):
            np.save(bridge_root / "labels.npy", fixture["labels"])
        role = "alpha_1" if name.startswith("left") else "zero"
        run_id = "coupled-run" if name.startswith("left") else "zero-run"
        verified[name] = {
            "feature_root": feature_root,
            "bridge_root": bridge_root,
            "bridge": {
                "run_id": run_id,
                "family": "adam",
                "cell_id": (
                    "adam_lr1e-3_wd1e-3"
                    if name.startswith("left")
                    else "adam_lr1e-3_wd1e-4_anchor"
                ),
                "training_seed": 0,
                "sibling_group_id": "sibling-0",
                "sibling_role": role,
                "initialization_sha256": "1" * 64,
                "data_stream_sha256": "2" * 64,
                    "checkpoint_role": "last",
                    "checkpoint_epoch": 200,
                    "checkpoint_sha256": (
                        "3" * 64 if name.startswith("left") else "4" * 64
                    ),
                "depth_tap": "penultimate",
                "feature_output_identity_sha256": hashlib.sha256(name.encode()).hexdigest(),
            },
        }

    monkeypatch.setattr(
        fresh_id_module,
        "_verified_binding",
        lambda binding, expected_split: verified[binding["name"]],
    )
    output = analyze_bound_alignment(
        left_train_binding={"name": "left_train"},
        left_validation_binding={"name": "left_validation"},
        right_train_binding={"name": "right_train"},
        right_validation_binding={"name": "right_validation"},
        pair_direction="coupled_minus_zero",
        output_root=tmp_path / "alignments",
        chunk_size=17,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["pair_direction"] == "coupled_minus_zero"
    assert manifest["cell_id"] == "adam_lr1e-3_wd1e-3"
    assert manifest["transforms"]["raw"]["gauge"]["zero_decay_common_frame"]
    assert (output / "alignment_state.npz").is_file()
    assert analyze_bound_alignment(
        left_train_binding={"name": "left_train"},
        left_validation_binding={"name": "left_validation"},
        right_train_binding={"name": "right_train"},
        right_validation_binding={"name": "right_validation"},
        pair_direction="coupled_minus_zero",
        output_root=tmp_path / "alignments",
        chunk_size=17,
    ) == output


def _seed_record(
    *, seed, role, value, cell="adam_lr1e-3_wd1e-4_anchor", status="PASS"
):
    policies = {
        "zero": "zero_decay",
        "coupled": "adam_alpha_1" if cell.endswith("anchor") else "adam_coupled",
        "decoupled": "adamw_alpha_0" if cell.endswith("anchor") else "adamw_decoupled",
        "alpha_0_5": "adam_mixed_alpha_0_5",
    }
    sibling_roles = {
        "zero": "zero",
        "coupled": "alpha_1" if cell.endswith("anchor") else "adam_coupled_role",
        "decoupled": "alpha_0" if cell.endswith("anchor") else "adamw_decoupled_role",
        "alpha_0_5": "alpha_0_5",
    }
    return {
        "status": status,
        "run_id": f"{cell}-{seed}-{role}",
        "family": "adam",
        "cell_id": cell,
        "training_seed": seed,
        "sibling_group_id": f"sibling-{seed}",
        "sibling_role": sibling_roles[role],
        "branch_policy": policies[role],
        "initialization_sha256": f"{seed:064x}",
        "data_stream_sha256": f"{seed + 20:064x}",
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "dataset_split": "id_validation",
        "transform": "raw",
        "metrics": {
            "geometry": value,
            "accuracy": 0.80 + value / 1000,
            "nll": 0.50 + value / 100,
            "ece": 0.04 + value / 1000,
        },
    }


def _complete_records():
    records = []
    for seed in range(3):
        records.extend(
            [
                _seed_record(seed=seed, role="zero", value=0.0),
                _seed_record(seed=seed, role="decoupled", value=1.0),
                _seed_record(seed=seed, role="alpha_0_5", value=2.0),
                _seed_record(seed=seed, role="coupled", value=float(seed + 2)),
            ]
        )
    return records


def test_paired_t_ci_and_cell_seed_aggregation_are_descriptive():
    direct = paired_t_interval([1.0, 2.0, 3.0])
    assert direct["mean"] == pytest.approx(2.0)
    assert direct["sample_sd_ddof1"] == pytest.approx(1.0)
    expected_half_width = direct["critical_value"] / math.sqrt(3)
    assert direct["lower"] == pytest.approx(2.0 - expected_half_width)
    assert direct["upper"] == pytest.approx(2.0 + expected_half_width)
    assert not direct["formal_tost_claim"]

    payload = aggregate_paired_records(
        records=_complete_records(),
        expected_seeds_by_cell={"adam_lr1e-3_wd1e-4_anchor": [0, 1, 2]},
    )
    assert payload["status"] == "PASS"
    geometry = next(
        row
        for row in payload["paired_aggregate_records"]
        if row["direction"] == "coupled_minus_decoupled"
        and row["metric"] == "geometry"
    )
    assert [row["delta"] for row in geometry["seed_records"]] == [1.0, 2.0, 3.0]
    assert geometry["paired_summary"]["mean"] == pytest.approx(2.0)
    assert payload["id_equivalence"]["adam_lr1e-3_wd1e-4_anchor"]["status"] == (
        "PENDING_PROTECTED_ID_TEST"
    )


def test_missing_seed_and_whole_context_are_not_hidden():
    records = [
        row
        for row in _complete_records()
        if not (row["training_seed"] == 2 and row["sibling_role"] == "alpha_0")
    ]
    expected_context = {
        **records[0],
        "cell_id": "missing_cell",
    }
    payload = aggregate_paired_records(
        records=records,
        expected_seeds_by_cell={
            "adam_lr1e-3_wd1e-4_anchor": [0, 1, 2],
            "missing_cell": [0, 1, 2],
        },
        expected_contexts=[expected_context],
    )
    assert payload["status"] == "INCOMPLETE"
    assert payload["missing_contexts"]
    affected = [
        row
        for row in payload["paired_aggregate_records"]
        if row["direction"] == "coupled_minus_decoupled"
    ]
    assert any(row["status"] == "INCOMPLETE" for row in affected)
    assert any(
        seed["status"] == "MISSING"
        for row in affected
        for seed in row["seed_records"]
    )


def test_sibling_identity_mismatch_fails_without_dropping_seed():
    records = _complete_records()
    mismatched = next(
        row
        for row in records
        if row["training_seed"] == 1 and row["sibling_role"] == "alpha_0"
    )
    mismatched["initialization_sha256"] = "f" * 64
    payload = aggregate_paired_records(
        records=records,
        expected_seeds_by_cell={"adam_lr1e-3_wd1e-4_anchor": [0, 1, 2]},
    )
    assert payload["status"] == "FAILED"
    affected = next(
        row
        for row in payload["paired_aggregate_records"]
        if row["direction"] == "coupled_minus_decoupled"
        and row["metric"] == "geometry"
    )
    assert affected["status"] == "FAILED"
    assert affected["paired_summary"] is None
    assert affected["seed_records"][1]["reason"] == "sibling_identity_mismatch"


def test_id_guardrail_uses_mean_margin_joint_rule_but_not_validation():
    deltas = {
        "accuracy": [0.005, -0.003, 0.004],
        "nll": [0.02, 0.03, -0.01],
        "ece": [0.01, 0.0, -0.005],
    }
    pending = adjudicate_id_equivalence(
        deltas,
        evidence_scope="id_validation",
        protected_id_test_available=True,
    )
    assert pending["status"] == "PENDING_PROTECTED_ID_TEST"
    passed = adjudicate_id_equivalence(
        deltas,
        evidence_scope="protected_id_test",
        protected_id_test_available=True,
    )
    assert passed["status"] == "PASS"
    assert passed["comparable_id"]
    assert all(not row["ci_is_decision_gate"] for row in passed["metrics"].values())
    failed = adjudicate_id_equivalence(
        {**deltas, "nll": [0.09, 0.10, 0.11]},
        evidence_scope="protected_id_test",
        protected_id_test_available=True,
    )
    assert failed["status"] == "FAILED_GUARDRAIL"
    assert not failed["comparable_id"]
    assert failed["failure_action"]["keep_all_runs"]


def test_alpha_interior_and_degenerate_rules_and_tables_are_deterministic():
    assert classify_alpha_interior(
        alpha_0_mean=1.0,
        alpha_0_5_mean=2.0,
        alpha_1_mean=3.0,
    )["classification"] == "interior_compatible"
    assert classify_alpha_interior(
        alpha_0_mean=1.0,
        alpha_0_5_mean=1.2,
        alpha_1_mean=1.0 + 1e-13,
    )["classification"] == "undefined_degenerate_endpoints"
    assert classify_alpha_interior(
        alpha_0_mean=1.0,
        alpha_0_5_mean=4.0,
        alpha_1_mean=3.0,
    )["classification"] == "non_monotone_three_point_response"

    payload = aggregate_paired_records(
        records=_complete_records(),
        expected_seeds_by_cell={"adam_lr1e-3_wd1e-4_anchor": [0, 1, 2]},
    )
    first = render_table_templates(payload)
    second = render_table_templates(payload)
    assert first == second
    assert set(first) == {
        "id_utility_equivalence.md",
        "geometry_trajectory.md",
        "alpha_classification.md",
    }
    assert "PENDING_PROTECTED_ID_TEST" in first["id_utility_equivalence.md"]


def test_aggregation_artifacts_are_checksummed_and_no_overwrite(tmp_path):
    payload = aggregate_paired_records(
        records=_complete_records(),
        expected_seeds_by_cell={"adam_lr1e-3_wd1e-4_anchor": [0, 1, 2]},
    )
    destination = write_aggregation_artifacts(
        payload=payload,
        output_directory=tmp_path / "aggregate",
    )
    assert {path.name for path in destination.iterdir()} == {
        "paired_aggregation.json",
        "id_utility_equivalence.md",
        "geometry_trajectory.md",
        "alpha_classification.md",
        "checksums.sha256",
    }
    for line in (destination / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="overwrite"):
        write_aggregation_artifacts(payload=payload, output_directory=destination)
