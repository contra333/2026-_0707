import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
POLICY_DIR = REPOSITORY_ROOT / "configs/evaluation/fixed_readout_stage2"
CANDIDATE_REGISTRY_PATH = POLICY_DIR / "candidate_registry.json"
GATE_POLICY_PATH = POLICY_DIR / "gate_policy.json"
INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_bound_local_file(binding):
    path = REPOSITORY_ROOT / binding["path"]
    assert path.is_file(), binding["path"]
    assert binding["sha256"] == _sha256(path)


def test_stage2_authority_paths_and_checksums_are_frozen_and_valid():
    registry = _load(CANDIDATE_REGISTRY_PATH)
    gate = _load(GATE_POLICY_PATH)

    _assert_bound_local_file(registry["authority"])
    for binding in gate["authorities"].values():
        _assert_bound_local_file(binding)

    assert gate["authorities"]["candidate_registry"]["sha256"] == _sha256(
        CANDIDATE_REGISTRY_PATH
    )
    inventory = _load(INVENTORY_PATH)
    inventory_binding = gate["authorities"]["v1_2_checkpoint_inventory"]
    assert inventory_binding["inventory_hash"] == inventory["inventory_hash"]
    assert inventory_binding["dataset_policy_hash"] == inventory["dataset_policy_hash"]


def test_candidate_registry_freezes_only_two_selectable_radial_interventions():
    registry = _load(CANDIDATE_REGISTRY_PATH)
    boundary = registry["selection_boundary"]
    candidates = registry["candidates"]

    assert boundary["freeze_point"] == (
        "before_opening_any_stage2_raw_score_or_feature_array"
    )
    assert boundary["eligible_candidate_count"] == 2
    assert boundary["eligible_candidate_ids"] == [
        "radial_l2_mahalanobis",
        "radial_l2_knn",
    ]
    assert boundary["candidate_addition_after_freeze"].startswith("forbidden")
    assert boundary["diagnostic_can_enter_selection"] is False
    assert boundary["maximum_selected_candidates"] == 1

    assert [(row["candidate_id"], row["priority"]) for row in candidates] == [
        ("radial_l2_mahalanobis", 1),
        ("radial_l2_knn", 2),
    ]
    assert all(row["selection_status"] == "eligible" for row in candidates)

    mahalanobis, knn = candidates
    assert mahalanobis["raw_detector"]["paper_name"] == "Mahalanobis-Raw"
    assert mahalanobis["transformed_detector"]["paper_name"] == "Mahalanobis++"
    assert mahalanobis["targeted_transform"]["refit_after_transform"] is True
    assert set(mahalanobis["targeted_transform"]["refit_state"]) == {
        "class_means",
        "tied_within_class_covariance",
        "precision",
    }

    assert knn["raw_detector"]["paper_name"] == "kNN-Raw K=50"
    assert knn["transformed_detector"]["paper_name"] == "kNN-L2 K=50"
    assert knn["fixed_hyperparameters"] == {
        "k": 50,
        "distance": "squared_l2_float64",
        "search": "exact",
        "tie_break": "distance_then_sample_id",
    }
    assert "complete id_train bank" in knn["targeted_transform"]["fit_operation"]
    assert "every ID-test and OOD query" in knn["targeted_transform"]["query_operation"]


def test_candidates_freeze_exact_energy_null_invariants_and_noninvariant_witnesses():
    registry = _load(CANDIDATE_REGISTRY_PATH)
    energy_null = registry["shared_null_controls"]
    assert len(energy_null) == 1
    control = energy_null[0]
    assert control["control_id"] == "energy_t1_feature_only_exact"
    assert control["detector"] == "ood_score/energy_t1"
    assert control["operation"] == (
        "apply_the_candidate_feature_transform_post_hoc_without_changing_logits"
    )
    assert control["expected_score_relation"] == "bitwise_identical_score_array"
    assert control["expected_rank_relation"] == "identical"
    assert control["required_oracles"] == [
        "score_sha256_equal",
        "rank_disagreement_fraction_zero",
        "auroc_abs_drift_within_exact_tolerance",
    ]
    assert "independently_invoke" in control["execution"]

    oracle_policy = registry["oracle_execution_policy"]
    assert oracle_policy["production_scope"].startswith("all_30_bundles")
    assert oracle_policy["common_positive_scale"] == 2.0
    assert oracle_policy["selection_use"].startswith("oracles_are_hard")

    for candidate in registry["candidates"]:
        assert candidate["null_control_ids"] == ["energy_t1_feature_only_exact"]
        oracle_ids = {
            oracle["oracle_id"] for oracle in candidate["formula_invariant_oracles"]
        }
        assert any("signed_coordinate_permutation" in value for value in oracle_ids)
        assert any("common_positive_scale" in value for value in oracle_ids)
        assert candidate["non_invariant_witnesses"]
        assert all(
            "nonuniform_samplewise_positive_scaling" in witness["witness_id"]
            for witness in candidate["non_invariant_witnesses"]
        )


def test_diagnostic_channels_cannot_be_selected_without_targeted_transforms():
    registry = _load(CANDIDATE_REGISTRY_PATH)
    diagnostics = registry["diagnostic_only"]

    assert [row["diagnostic_id"] for row in diagnostics] == [
        "mahalanobis_spectral_tertiles",
        "ctm_angular_channel",
        "pure_residual_channel",
    ]
    assert all(row["selection_status"] == "diagnostic_only" for row in diagnostics)
    assert all(row["targeted_transform"] is None for row in diagnostics)
    assert all(
        "no" in row["reason_ineligible"].lower()
        and "targeted" in row["reason_ineligible"].lower()
        for row in diagnostics
    )


def test_gate_population_partition_and_clustered_analysis_unit_are_exact():
    gate = _load(GATE_POLICY_PATH)
    boundary = gate["evidence_boundary"]
    population = gate["population"]
    partition = gate["ood_partition"]
    units = gate["analysis_units"]

    assert boundary["procedural_bias_reduction"] is True
    assert boundary["blind_holdout"] is False
    assert boundary["interpretation"] == (
        "protected_discovery_not_confirmation_or_causality"
    )
    assert population["full_bundle_count_for_integrity_and_parity"] == 30
    assert population["unique_config_count"] == 10
    assert population["seed_set_for_integrity_and_parity"] == [0, 1, 2]
    assert population["seed_set_for_candidate_selection"] == [1, 2]
    assert population["seed_zero_role_selection_bias"] is True
    assert population["seed_zero_use"] == "post_selection_sensitivity_only"
    assert population["primary_independent_design_unit"] == "config_hash"

    assert partition["near"] == ["cifar100", "tin"]
    assert partition["far"] == ["mnist", "svhn", "texture", "places365"]
    assert partition["dataset_order"] == partition["near"] + partition["far"]
    assert partition["group_balanced_selection"] is True

    assert units["unique_config_pair_count"] == 45
    assert units["cross_config_pairs_are_independent_samples"] is False
    assert units["uncertainty_resampling_unit"] == "config_hash_cluster"
    assert units["seed_values_are_nested_within_config"] is True


def test_gate_cross_validation_support_margins_and_statuses_are_prespecified():
    gate = _load(GATE_POLICY_PATH)
    cv = gate["cross_validation"]
    support = gate["dataset_support"]
    margins = gate["practical_margins"]

    assert cv["lodo"]["fold_count"] == 6
    assert cv["lodo"]["training_dataset_count_per_fold"] == 5
    assert cv["lodo"]["minimum_same_candidate_and_heldout_pass_count"] == 5
    assert cv["lodo"]["heldout_dataset_is_not_used_for_fold_selection"] is True
    assert "five_training_datasets" in cv["lodo"]["fold_recomputation"]
    assert "full_10_config" in cv["lodo"]["heldout_absolute_attenuation"]

    assert cv["loco"]["fold_count"] == 10
    assert cv["loco"]["training_config_count_per_fold"] == 9
    assert cv["loco"]["hold_out_all_seeds_for_config"] is True
    assert cv["loco"]["minimum_same_candidate_and_heldout_pass_count"] == 8
    assert "nine_training_configs" in cv["loco"]["fold_recomputation"]
    assert "median_over_9_training_configs" in cv["loco"][
        "heldout_absolute_attenuation_per_dataset"
    ]
    assert cv["seed_stability"]["required_views"] == [[1], [2], [1, 2]]
    assert cv["seed_stability"][
        "single_seed_views_use_combined_seed1_and_seed2_frozen_margins"
    ] is True
    assert cv["seed_stability"][
        "seed1_seed2_and_combined_views_must_select_same_candidate"
    ] is True

    assert support["minimum_total_dataset_support"] == 4
    assert support["minimum_near_dataset_support"] == 1
    assert support["minimum_far_dataset_support"] == 2
    assert support["near_median_selective_attenuation_must_be_positive"] is True
    assert support["far_median_selective_attenuation_must_be_positive"] is True
    assert support["no_dataset_may_show_practical_reverse"] is True

    assert margins["raw_gap_margin_auroc"] == "max(0.01, seed_noise_q90)"
    assert margins["attenuation_margin_auroc"] == (
        "max(0.005, 0.25 * raw_gap, null_attenuation_q90)"
    )
    assert margins["minimum_relative_attenuation"] == 0.25
    assert margins["fpr95_secondary_absolute_margin"] == 0.02
    assert margins["seed_zero_must_not_enter_any_selection_margin"] is True
    assert gate["analysis_units"]["selection_seed_noise_excludes_seed_zero"] is True

    assert gate["selection_rule"]["deterministic_tie_break"] == [
        "ascending_priority",
        "ascending_candidate_id",
    ]
    assert gate["selection_rule"]["diagnostic_only_entries_are_excluded"] is True
    assert gate["selection_rule"]["null_controls_enter_ranking"] is False
    assert "max_null_drift_ratio" not in gate["selection_rule"][
        "dataset_selective_score"
    ]
    assert set(gate["decision_semantics"]) == {
        "PASS",
        "NO_GO",
        "INCONCLUSIVE",
        "FAILED",
        "decision_precedence",
        "scientific_launch_allowed_only_for",
        "candidate_replacement_to_rescue_result",
    }
    assert gate["decision_semantics"]["scientific_launch_allowed_only_for"] == "PASS"
    assert gate["decision_semantics"]["candidate_replacement_to_rescue_result"] is False
    assert gate["legacy_adam_adamw_contrasts"]["gate_role"].startswith(
        "reported_descriptive"
    )
    assert "legacy-direction" not in gate["decision_semantics"]["INCONCLUSIVE"]
    assert gate["decision_semantics"]["decision_precedence"][0].startswith(
        "FAILED"
    )


def test_pair_credit_and_formula_oracle_semantics_are_fully_prespecified():
    gate = _load(GATE_POLICY_PATH)
    overlap = gate["score_overlap"]
    oracle = gate["invariant_and_witness_policy"]

    assert overlap["state_definition"] == {
        "correct": "s_ID_greater_than_s_OOD",
        "tie": "s_ID_equal_to_s_OOD",
        "misordered": "s_ID_less_than_s_OOD",
    }
    assert overlap["ranking_credit"] == {
        "correct": 1.0,
        "tie": 0.5,
        "misordered": 0.0,
    }
    assert overlap["net_corrected_pair_mass"] == (
        "corrected_pair_mass_minus_harmed_pair_mass"
    )
    assert "transformed_auroc_minus_raw_auroc" in overlap[
        "required_identity_check"
    ]
    assert oracle["formula_reconstruction_scope"].startswith("all_30_bundles")
    assert oracle[
        "formula_component_outputs_are_diagnostic_not_selection_thresholds"
    ] is True


def test_legacy_adam_adamw_hashes_are_exact_inventory_roles_with_three_seeds():
    gate = _load(GATE_POLICY_PATH)
    inventory = _load(INVENTORY_PATH)
    authorized = [
        row
        for row in inventory["training_identities"]
        if row["disposition"] == "authorized_role"
    ]
    assert inventory["counts"]["authorized_training_identities"] == 30
    assert len(authorized) == 30
    assert len({row["config_hash"] for row in authorized}) == 10

    by_hash = {}
    for row in authorized:
        by_hash.setdefault(row["config_hash"], []).append(row)
    assert all({row["training_seed"] for row in rows} == {0, 1, 2} for rows in by_hash.values())

    contrasts = gate["legacy_adam_adamw_contrasts"]
    primary = contrasts["primary_discovery_roles"]
    sensitivity = contrasts["sensitivity_only"]
    assert [row["role_code"] for row in primary] == ["C3", "C4"]
    assert [row["role_code"] for row in sensitivity] == ["C1"]
    assert sensitivity[0]["adam_config_hash"] == primary[0]["adam_config_hash"]

    for contrast in [*primary, *sensitivity]:
        for family in ("adam", "adamw"):
            config_hash = contrast[f"{family}_config_hash"]
            rows = by_hash[config_hash]
            assert {row["optimizer_family"] for row in rows} == {family}
            assert all(f"role:{family}:{contrast['role_code']}" in row["labels"] for row in rows)


def test_gate_requires_complete_auditable_output_contract_before_stage3():
    gate = _load(GATE_POLICY_PATH)
    outputs = gate["required_output_artifacts"]
    assert list(outputs) == [
        "selection_manifest.json",
        "model_metrics.jsonl",
        "contrast_metrics.jsonl",
        "dataset_summaries.jsonl",
        "cv_folds.jsonl",
        "gate_decision.json",
        "preconfirmation_freeze.json",
    ]
    assert {
        "raw_gap",
        "transformed_gap",
        "absolute_attenuation",
        "relative_attenuation",
        "practical_gap_margin",
        "practical_attenuation_margin",
    }.issubset(outputs["dataset_summaries.jsonl"])
    assert {
        "status",
        "selected_candidate_id",
        "lodo_selection_count",
        "loco_selection_count",
        "seed1_selected_candidate",
        "seed2_selected_candidate",
    }.issubset(outputs["gate_decision.json"])
    assert {
        "gate_decision_sha256",
        "fresh_prefix_seed_list_sha256",
        "freeze_sha256",
        "post_freeze_change_policy",
    }.issubset(outputs["preconfirmation_freeze.json"])
    assert len(gate["preconfirmation_freeze_required"]) == 19
