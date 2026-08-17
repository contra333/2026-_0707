"""Prospective fresh-ID activation gate for the optional Task F RtMD slot.

This module contains no feature loader, protected split, OOD metric, or RtMD
detector.  It only adjudicates already-computed ID-validation tail summaries
under the rule frozen before fresh result values are inspected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from oge.studies.hashing import canonical_sha256

from .task_f_fresh_id import paired_t_interval


RTMD_GATE3_SCHEMA_VERSION = "task_f_rtmd_fresh_id_gate_v1"
RTMD_GATE3_SPEC = {
    "schema_version": RTMD_GATE3_SCHEMA_VERSION,
    "role": "optional_secondary_method_activation_not_main_study_gate",
    "evidence_scope": {
        "cell_id": "adam_lr1e-3_wd1e-4_anchor",
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "transform": "raw",
        "dataset_split": "id_validation",
        "roles": ["coupled", "decoupled"],
        "training_seeds": [0, 1, 2, 3, 4],
    },
    "tail_statistic": {
        "name": "log_qperp_q99_over_chi2_q99",
        "formula": "log(quantile_0.99(q_perp_id_validation)/chi2_ppf(0.99,k))",
        "k": "dim(S_perp) from the matching ID-train raw fit",
    },
    "tail_fit": {
        "source": "deterministic_class_stratified_two_fold_id_train",
        "finite_nu_domain": [2.05, 1000.0],
        "gaussian_candidate": "nu=infinity",
        "finite_selection": "2*(LL_t-LL_Gaussian)>log(N)",
        "minimum_finite_t_fits_per_role": 4,
        "all_five_paired_fits_must_be_numerically_applicable": True,
    },
    "paired_effect": {
        "direction": "coupled_minus_decoupled",
        "uncertainty": "two_sided_paired_90_percent_t_ci",
        "minimum_sign_consistency": 4,
        "tail_effect_absolute_floor": 0.10,
        "same_policy_reference": (
            "median absolute between-seed difference pooled within coupled "
            "and within decoupled"
        ),
        "same_policy_reference_floor": 1.0e-6,
    },
    "joint_pass_rule": [
        "exact_complete_five_seed_pairing",
        "all_paired_fits_numerically_applicable",
        "at_least_four_finite_t_selections_per_role",
        "paired_90_percent_ci_excludes_zero",
        "at_least_four_of_five_effects_share_the_mean_sign",
        "absolute_mean_effect_at_least_max_same_policy_reference_and_0.10",
    ],
    "failure_action": {
        "activate_rtmd": False,
        "close_optional_slot_for_this_anchor": True,
        "change_main_study": False,
        "replacement_estimator_allowed": False,
    },
    "protected_data_access": False,
}
RTMD_GATE3_SPEC_SHA256 = canonical_sha256(RTMD_GATE3_SPEC)


def _failure(status: str, reasons: Sequence[str], *, details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RTMD_GATE3_SCHEMA_VERSION,
        "status": status,
        "activated": False,
        "close_optional_slot_for_this_anchor": True,
        "specification_sha256": RTMD_GATE3_SPEC_SHA256,
        "reason_codes": list(reasons),
        "details": dict(details),
        "protected_data_access": False,
    }


def adjudicate_rtmd_gate3(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen five-seed, fresh-ID-only RtMD activation rule."""

    scope = RTMD_GATE3_SPEC["evidence_scope"]
    expected = {
        (role, seed)
        for role in scope["roles"]
        for seed in scope["training_seeds"]
    }
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    duplicates: list[list[Any]] = []
    malformed: list[list[Any]] = []
    for row in records:
        role = str(row.get("role"))
        try:
            seed = int(row.get("training_seed", -1))
            epoch = int(row.get("checkpoint_epoch", -1))
        except (TypeError, ValueError):
            malformed.append([role, str(row.get("training_seed"))])
            continue
        key = (role, seed)
        if key in indexed:
            duplicates.append([role, seed])
            continue
        indexed[key] = row
        if (
            row.get("cell_id") != scope["cell_id"]
            or row.get("checkpoint_role") != scope["checkpoint_role"]
            or epoch != scope["checkpoint_epoch"]
            or row.get("depth_tap") != scope["depth_tap"]
            or row.get("transform") != scope["transform"]
            or row.get("dataset_split") != scope["dataset_split"]
        ):
            malformed.append([role, seed])
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if duplicates or malformed or missing or unexpected:
        return _failure(
            "FAILED_INCOMPLETE",
            ["exact_complete_five_seed_pairing_failed"],
            details={
                "duplicates": duplicates,
                "malformed": malformed,
                "missing": [list(value) for value in missing],
                "unexpected": [list(value) for value in unexpected],
            },
        )

    inapplicable = [
        [role, seed]
        for role, seed in sorted(expected)
        if indexed[(role, seed)].get("numerically_applicable") is not True
    ]
    if inapplicable:
        return _failure(
            "FAILED_INAPPLICABLE",
            ["all_paired_fits_numerically_applicable_failed"],
            details={"inapplicable": inapplicable},
        )

    finite_by_role = {
        role: sum(
            indexed[(role, seed)].get("finite_t_selected") is True
            for seed in scope["training_seeds"]
        )
        for role in scope["roles"]
    }
    minimum_finite = int(
        RTMD_GATE3_SPEC["tail_fit"]["minimum_finite_t_fits_per_role"]
    )
    if any(count < minimum_finite for count in finite_by_role.values()):
        return _failure(
            "FAILED_TAIL_ESTIMABILITY",
            ["minimum_finite_t_fits_per_role_failed"],
            details={"finite_t_selected_by_role": finite_by_role},
        )

    values: dict[str, dict[int, float]] = defaultdict(dict)
    nonfinite = []
    for role, seed in sorted(expected):
        try:
            value = float(indexed[(role, seed)].get("tail_statistic", math.nan))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            nonfinite.append([role, seed])
        values[role][seed] = value
    if nonfinite:
        return _failure(
            "FAILED_INAPPLICABLE",
            ["tail_statistic_nonfinite"],
            details={"nonfinite": nonfinite},
        )

    deltas = [
        values["coupled"][seed] - values["decoupled"][seed]
        for seed in scope["training_seeds"]
    ]
    interval = paired_t_interval(deltas)
    mean = float(interval["mean"])
    sign_consistency = sum(value * mean > 0.0 for value in deltas) if mean else 0
    same_policy_differences = [
        abs(values[role][left] - values[role][right])
        for role in scope["roles"]
        for left, right in combinations(scope["training_seeds"], 2)
    ]
    reference = float(np.median(np.asarray(same_policy_differences, dtype=np.float64)))
    reference_floor = float(
        RTMD_GATE3_SPEC["paired_effect"]["same_policy_reference_floor"]
    )
    practical_floor = float(
        RTMD_GATE3_SPEC["paired_effect"]["tail_effect_absolute_floor"]
    )
    threshold = max(reference, reference_floor, practical_floor)
    ci_excludes_zero = float(interval["lower"]) > 0.0 or float(interval["upper"]) < 0.0
    enough_signs = sign_consistency >= int(
        RTMD_GATE3_SPEC["paired_effect"]["minimum_sign_consistency"]
    )
    large_enough = abs(mean) >= threshold
    checks = {
        "paired_90_percent_ci_excludes_zero": ci_excludes_zero,
        "minimum_sign_consistency": enough_signs,
        "effect_exceeds_same_policy_and_practical_floor": large_enough,
    }
    activated = all(checks.values())
    return {
        "schema_version": RTMD_GATE3_SCHEMA_VERSION,
        "status": "PASS" if activated else "FAILED_POLICY_EFFECT",
        "activated": activated,
        "close_optional_slot_for_this_anchor": not activated,
        "specification_sha256": RTMD_GATE3_SPEC_SHA256,
        "finite_t_selected_by_role": finite_by_role,
        "tail_statistic_by_role_and_seed": {
            role: {str(seed): values[role][seed] for seed in scope["training_seeds"]}
            for role in scope["roles"]
        },
        "paired_effects": deltas,
        "paired_90_percent_ci": interval,
        "same_policy_reference": reference,
        "decision_threshold": threshold,
        "sign_consistency_count": sign_consistency,
        "checks": checks,
        "reason_codes": [] if activated else [
            name for name, passed in checks.items() if not passed
        ],
        "protected_data_access": False,
    }


__all__ = [
    "RTMD_GATE3_SCHEMA_VERSION",
    "RTMD_GATE3_SPEC",
    "RTMD_GATE3_SPEC_SHA256",
    "adjudicate_rtmd_gate3",
]
