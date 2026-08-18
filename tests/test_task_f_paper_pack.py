import pytest

from oge.analysis.task_f_paper_pack import (
    CELLS,
    DATASETS,
    EXPECTED_SEEDS,
    endpoint_macro_seed_rows,
    mean_sd_t90,
    negative_gate_rows,
    summarize_macro_seed_rows,
    validate_source,
)


def _endpoint_row(*, dataset, seed=0, left=0.7, right=0.6, pair_count=1):
    return {
        "cell": CELLS[0],
        "contrast": "C-D",
        "dataset": dataset,
        "transform": "raw",
        "detector": "md",
        "seed": seed,
        "left": {"auroc": left},
        "right": {"auroc": right},
        "delta_auroc": left - right,
        "delta_fpr95": 0.0,
        "gain": 0.2,
        "loss": 0.1,
        "pair_order_churn": 0.3,
        "pair_count": pair_count,
        "identity": {"pass": True},
    }


def test_near_far_macro_is_formed_within_seed_before_inference():
    rows = [
        _endpoint_row(dataset="cifar100", left=0.2, right=0.1, pair_count=10**9),
        _endpoint_row(dataset="tin", left=0.8, right=0.4, pair_count=1),
        _endpoint_row(dataset="mnist", left=0.3, right=0.1),
        _endpoint_row(dataset="svhn", left=0.5, right=0.2),
        _endpoint_row(dataset="texture", left=0.7, right=0.3),
        _endpoint_row(dataset="places365", left=0.9, right=0.4),
    ]
    output = endpoint_macro_seed_rows({"endpoint_score_rows": rows})
    near = next(row for row in output if row["region"] == "Near")
    far = next(row for row in output if row["region"] == "Far")
    assert near["dataset_count"] == 2
    assert near["left_auroc"] == pytest.approx(0.5)
    assert near["right_auroc"] == pytest.approx(0.25)
    assert near["delta_auroc"] == pytest.approx(0.25)
    assert far["dataset_count"] == 4
    assert far["left_auroc"] == pytest.approx(0.6)
    assert far["right_auroc"] == pytest.approx(0.25)


def test_two_sided_90_percent_t_interval_matches_oracle():
    result = mean_sd_t90([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["sd"] == pytest.approx(2.5**0.5)
    assert result["t90_low"] == pytest.approx(1.4925566809376778)
    assert result["t90_high"] == pytest.approx(4.507443319062322)


def test_summary_keeps_seed_as_the_statistical_unit():
    rows = []
    for seed, delta in enumerate((0.1, 0.2, 0.6)):
        rows.append(
            {
                "scope": "endpoint",
                "cell": CELLS[0],
                "contrast": "C-D",
                "transform": "raw",
                "detector": "md",
                "axis": "endpoint",
                "axis_value": 200,
                "seed": seed,
                "region": "Near",
                "dataset_count": 2,
                "delta_auroc": delta,
            }
        )
    summary = summarize_macro_seed_rows(rows)
    row = next(value for value in summary if value["metric"] == "delta_auroc")
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(0.3)


def _full_source_fixture():
    endpoint = []
    localization = []
    for cell in CELLS:
        for seed in EXPECTED_SEEDS[cell]:
            for contrast in ("C-D", "C-Z", "D-Z"):
                for dataset in DATASETS:
                    for transform in ("raw", "l2"):
                        localization.append(
                            {
                                "cell": cell,
                                "contrast": contrast,
                                "dataset": dataset,
                                "transform": transform,
                                "seed": seed,
                                "delta_auroc": 0.1,
                                "rmd_marginal_replacement": {
                                    "phi_rmd": 0.04,
                                    "phi_marginal": 0.06,
                                },
                            }
                        )
                        for detector in ("md", "rmd", "marginal"):
                            endpoint.append(
                                {
                                    "cell": cell,
                                    "contrast": contrast,
                                    "dataset": dataset,
                                    "transform": transform,
                                    "detector": detector,
                                    "seed": seed,
                                    "gain": 0.2,
                                    "loss": 0.1,
                                    "delta_auroc": 0.1,
                                    "pair_order_churn": 0.3,
                                    "identity": {"pass": True},
                                }
                            )
    return {
        "validation": {
            "status": "PASS",
            "all_sibling_identity_checks_pass": True,
        },
        "endpoint_score_rows": endpoint,
        "score_localization_rows": localization,
        "formation_score_rows": [{} for _ in range(960)],
    }


def test_source_coverage_and_pair_localization_identities_are_enforced():
    source = _full_source_fixture()
    result = validate_source(source)
    assert result["status"] == "PASS"
    assert result["cell_seed_contexts"] == 14
    source["endpoint_score_rows"][0]["identity"]["pass"] = False
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source(source)


def test_source_pair_accounting_mismatch_is_rejected():
    source = _full_source_fixture()
    source["endpoint_score_rows"][0]["pair_order_churn"] = 0.31
    with pytest.raises(ValueError, match="pair-accounting identity"):
        validate_source(source)


def test_source_localization_reconstruction_mismatch_is_rejected():
    source = _full_source_fixture()
    source["score_localization_rows"][0]["rmd_marginal_replacement"]["phi_marginal"] = 0.05
    with pytest.raises(ValueError, match="score-localization identity"):
        validate_source(source)


def test_nested_classifier_kill_decision_is_flattened_for_csv():
    kill = {
        "decision": {"decision": "FAIL", "reason_codes": ["criterion_failed"]},
        "cell_summary": {
            cell: {"seed_count": len(EXPECTED_SEEDS[cell]), "rho_above_one_count": 0, "rho_median": 0.01}
            for cell in CELLS
        },
    }
    rows = negative_gate_rows({"spectral_allocation_rows": []}, kill)
    assert {row["decision"] for row in rows[1:]} == {"FAIL"}
