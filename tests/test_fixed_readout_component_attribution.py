import numpy as np
import pytest

from oge.analysis.fixed_readout_component_attribution import (
    mahalanobis_score_components,
    pair_outcome_summary,
    pair_transition_summary,
    paired_component_attribution,
    quadratic_size_stretch,
    reconstruction_record,
    symmetric_product_change,
    woodbury_low_rank_record,
)


def _brute_pair_auroc(ids, oods):
    margins = np.asarray(ids)[:, None] - np.asarray(oods)[None, :]
    return float(np.mean((margins > 0.0) + 0.5 * (margins == 0.0)))


def test_mahalanobis_components_reconstruct_scores_and_pair_margins():
    class_distances = np.asarray([[1.0, 4.0], [2.5, 2.0], [8.0, 3.0]])
    global_distances = np.asarray([3.0, 1.5, 5.0])
    scores = mahalanobis_score_components(class_distances, global_distances)

    np.testing.assert_allclose(scores["md"], scores["rmd"] + scores["marginal"])
    check = reconstruction_record(scores["md"], scores["rmd"], scores["marginal"])
    assert check["pass"]
    md_margin = scores["md"][0] - scores["md"][2]
    component_margin = (
        scores["rmd"][0]
        - scores["rmd"][2]
        + scores["marginal"][0]
        - scores["marginal"][2]
    )
    assert md_margin == pytest.approx(component_margin)


def test_pair_outcome_summary_counts_ties_as_half_not_error():
    ids = np.asarray([1.0, 2.0])
    oods = np.asarray([0.0, 1.0, 3.0])
    result = pair_outcome_summary(ids, oods)

    assert result == {
        "pair_count": 6,
        "correct": 3,
        "tie": 1,
        "incorrect": 2,
        "auroc_id_positive": 3.5 / 6.0,
    }
    assert result["auroc_id_positive"] == _brute_pair_auroc(ids, oods)


def test_fast_pair_transition_table_matches_brute_force_with_ties():
    id0 = np.asarray([1.0, 2.0, 2.0])
    ood0 = np.asarray([0.0, 1.0, 2.0, 3.0])
    id1 = np.asarray([2.0, 1.0, 2.0])
    ood1 = np.asarray([0.0, 2.0, 2.0, 1.0])
    result = pair_transition_summary(id0, ood0, id1, ood1)
    brute = np.zeros((3, 3), dtype=np.int64)
    for left in range(id0.size):
        for right in range(ood0.size):
            state0 = 2 if id0[left] > ood0[right] else int(id0[left] == ood0[right])
            state1 = 2 if id1[left] > ood1[right] else int(id1[left] == ood1[right])
            brute[state0, state1] += 1
    observed = np.asarray(
        [
            [result["transitions"][source][target] for target in ("incorrect", "tie", "correct")]
            for source in ("incorrect", "tie", "correct")
        ]
    )
    np.testing.assert_array_equal(observed, brute)


def test_paired_component_shapley_exactly_accounts_for_pair_transitions():
    branch_0 = {
        "id": {
            "rmd": np.asarray([0.5, 1.0]),
            "marginal": np.asarray([0.5, 1.0]),
            "md": np.asarray([1.0, 2.0]),
        },
        "ood": {
            "rmd": np.asarray([0.0, 1.0]),
            "marginal": np.asarray([0.0, 1.0]),
            "md": np.asarray([0.0, 2.0]),
        },
    }
    branch_1 = {
        "id": {
            "rmd": np.asarray([1.5, 1.0]),
            "marginal": np.asarray([0.5, 1.0]),
            "md": np.asarray([2.0, 2.0]),
        },
        "ood": {
            "rmd": np.asarray([0.0, 0.0]),
            "marginal": np.asarray([0.0, 1.0]),
            "md": np.asarray([0.0, 1.0]),
        },
    }

    result = paired_component_attribution(branch_0, branch_1)

    assert result["pass"]
    assert result["pair_count"] == 4
    assert result["branch_0_auroc"] == _brute_pair_auroc(
        branch_0["id"]["md"], branch_0["ood"]["md"]
    )
    assert result["branch_1_auroc"] == _brute_pair_auroc(
        branch_1["id"]["md"], branch_1["ood"]["md"]
    )
    attribution = result["component_auroc_attribution"]
    assert attribution["rmd"] + attribution["marginal"] == pytest.approx(
        result["auroc_delta"]
    )
    assert sum(
        count
        for source in result["pair_transitions"].values()
        for count in source.values()
    ) == 4


def test_component_attribution_rejects_non_reconstructing_scores():
    invalid = {
        "id": {"md": [2.0], "rmd": [0.5], "marginal": [0.5]},
        "ood": {"md": [0.0], "rmd": [0.0], "marginal": [0.0]},
    }
    with pytest.raises(ValueError, match="MD score reconstruction failed"):
        paired_component_attribution(invalid, invalid)


def test_size_stretch_and_symmetric_branch_change_reconstruct():
    precision = np.asarray([[2.0, 0.25], [0.25, 0.75]])
    deviations_0 = np.asarray([[1.0, 2.0], [2.0, -1.0]])
    deviations_1 = np.asarray([[1.5, 1.0], [1.0, -2.0]])
    term_0 = quadratic_size_stretch(deviations_0, precision)
    term_1 = quadratic_size_stretch(deviations_1, precision)

    np.testing.assert_allclose(
        term_0["quadratic"], term_0["size"] * term_0["stretch"]
    )
    change = symmetric_product_change(
        term_0["size"],
        term_0["stretch"],
        term_1["size"],
        term_1["stretch"],
    )
    assert change["pass"]
    np.testing.assert_allclose(
        change["total_change"],
        term_1["quadratic"] - term_0["quadratic"],
    )


def test_woodbury_rank_bound_is_only_asserted_for_positive_ridge():
    within = np.diag([1.0, 2.0, 3.0, 4.0])
    between_factor = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]
    )
    record = woodbury_low_rank_record(within, between_factor, ridge=1.0e-3)

    assert record["pass"]
    assert record["applicability"] == "ridge_full_rank_only"
    assert record["observed_correction_rank"] <= record["rank_bound"] == 2
    with pytest.raises(ValueError, match="positive finite ridge"):
        woodbury_low_rank_record(within, between_factor, ridge=0.0)
