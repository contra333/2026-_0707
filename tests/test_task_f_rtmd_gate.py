import copy

from oge.analysis.task_f_rtmd_gate import (
    RTMD_GATE3_SPEC,
    RTMD_GATE3_SPEC_SHA256,
    adjudicate_rtmd_gate3,
)
from oge.studies.hashing import canonical_sha256


def _records(*, effect=0.5):
    rows = []
    decoupled = [0.00, 0.04, -0.03, 0.02, -0.01]
    for role in ("coupled", "decoupled"):
        for seed, base in enumerate(decoupled):
            rows.append(
                {
                    "cell_id": "adam_lr1e-3_wd1e-4_anchor",
                    "checkpoint_role": "last",
                    "checkpoint_epoch": 200,
                    "depth_tap": "penultimate",
                    "transform": "raw",
                    "dataset_split": "id_validation",
                    "role": role,
                    "training_seed": seed,
                    "numerically_applicable": True,
                    "finite_t_selected": seed != 4,
                    "tail_statistic": base + (effect if role == "coupled" else 0.0),
                }
            )
    return rows


def test_rtmd_gate3_specification_is_frozen_and_passes_large_repeatable_effect():
    assert RTMD_GATE3_SPEC_SHA256 == (
        "30e7f212c6e91b84885a7d06568820caa15c48fdcbe924af28818d07c428d270"
    )
    assert RTMD_GATE3_SPEC_SHA256 == canonical_sha256(RTMD_GATE3_SPEC)
    result = adjudicate_rtmd_gate3(_records())
    assert result["status"] == "PASS"
    assert result["activated"]
    assert result["sign_consistency_count"] == 5
    assert result["checks"] == {
        "paired_90_percent_ci_excludes_zero": True,
        "minimum_sign_consistency": True,
        "effect_exceeds_same_policy_and_practical_floor": True,
    }
    assert not result["protected_data_access"]


def test_rtmd_gate3_closes_on_missing_inapplicable_or_insufficient_finite_fit():
    missing = adjudicate_rtmd_gate3(_records()[:-1])
    assert missing["status"] == "FAILED_INCOMPLETE"
    assert not missing["activated"]

    rows = _records()
    rows[0]["numerically_applicable"] = False
    inapplicable = adjudicate_rtmd_gate3(rows)
    assert inapplicable["status"] == "FAILED_INAPPLICABLE"

    rows = _records()
    for row in rows:
        if row["role"] == "coupled" and row["training_seed"] >= 3:
            row["finite_t_selected"] = False
    fallback = adjudicate_rtmd_gate3(rows)
    assert fallback["status"] == "FAILED_TAIL_ESTIMABILITY"
    assert fallback["details"]["finite_t_selected_by_role"]["coupled"] == 3


def test_rtmd_gate3_requires_effect_beyond_same_policy_reference_and_uncertainty():
    weak = adjudicate_rtmd_gate3(_records(effect=0.05))
    assert weak["status"] == "FAILED_POLICY_EFFECT"
    assert not weak["checks"]["effect_exceeds_same_policy_and_practical_floor"]

    noisy_rows = _records()
    coupled_values = [0.6, -0.5, 0.7, -0.4, 0.6]
    for row in noisy_rows:
        if row["role"] == "coupled":
            row["tail_statistic"] = coupled_values[row["training_seed"]]
    noisy = adjudicate_rtmd_gate3(noisy_rows)
    assert noisy["status"] == "FAILED_POLICY_EFFECT"
    assert not noisy["activated"]

    duplicate_rows = copy.deepcopy(_records())
    duplicate_rows.append(copy.deepcopy(duplicate_rows[0]))
    duplicate = adjudicate_rtmd_gate3(duplicate_rows)
    assert duplicate["status"] == "FAILED_INCOMPLETE"

    malformed_rows = copy.deepcopy(_records())
    malformed_rows[0]["training_seed"] = None
    malformed = adjudicate_rtmd_gate3(malformed_rows)
    assert malformed["status"] == "FAILED_INCOMPLETE"

    nonfinite_rows = copy.deepcopy(_records())
    nonfinite_rows[0]["tail_statistic"] = "not-a-number"
    nonfinite = adjudicate_rtmd_gate3(nonfinite_rows)
    assert nonfinite["status"] == "FAILED_INAPPLICABLE"
