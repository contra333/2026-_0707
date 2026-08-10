import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from oge.evaluation.detectors import exact_neighbor_distances


SCRIPT_ROOT = Path(__file__).parents[1] / "scripts"


def _load_script(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BACKEND = _load_script(
    "stage2_raw_knn_backend", "fixed_readout_stage2_raw_knn_backend.py"
)
BENCHMARK = _load_script(
    "stage2_raw_knn_benchmark", "benchmark_fixed_readout_stage2_raw_knn.py"
)
BACKEND_ID = BACKEND.BACKEND_ID
exact_raw_knn_topk = BACKEND.exact_raw_knn_topk
estimate_workload = BENCHMARK.estimate_workload
run_synthetic_benchmark = BENCHMARK.run_synthetic_benchmark


@pytest.mark.parametrize(
    ("seed", "bank_count", "query_count", "feature_dim", "k", "query_chunk", "bank_chunk"),
    [
        (7, 101, 23, 17, 11, 7, 29),
        (19, 73, 31, 8, 5, 9, 17),
        (41, 64, 12, 32, 50, 4, 50),
    ],
)
def test_random_exact_parity_with_frozen_neighbor_oracle(
    seed,
    bank_count,
    query_count,
    feature_dim,
    k,
    query_chunk,
    bank_chunk,
):
    rng = np.random.default_rng(seed)
    bank = rng.normal(size=(bank_count, feature_dim))
    queries = rng.normal(size=(query_count, feature_dim))
    id_order = rng.permutation(bank_count)
    sample_ids = np.asarray([f"sample-{value:04d}" for value in id_order])

    observed = exact_raw_knn_topk(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=k,
        query_chunk_size=query_chunk,
        bank_chunk_size=bank_chunk,
    )
    expected_distances, expected_ids = exact_neighbor_distances(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=k,
        query_chunk_size=query_chunk,
        bank_chunk_size=bank_chunk,
    )

    np.testing.assert_array_equal(
        observed["neighbor_squared_distances"], expected_distances
    )
    np.testing.assert_array_equal(observed["neighbor_sample_ids"], expected_ids)
    np.testing.assert_array_equal(
        sample_ids[observed["neighbor_bank_indices"]], expected_ids
    )
    np.testing.assert_array_equal(observed["score"], -expected_distances[:, -1])
    assert observed["backend"] == BACKEND_ID
    assert observed["k"] == k
    assert observed["query_chunk_size"] == query_chunk
    assert observed["bank_chunk_size"] == bank_chunk
    assert observed["score"].dtype == np.float64


def test_boundary_ties_are_lexicographic_and_chunk_invariant():
    bank = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    sample_ids = np.asarray(["z", "a", "m", "b", "c", "d", "aa"])
    queries = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    reference_distances, reference_ids = exact_neighbor_distances(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=3,
        query_chunk_size=1,
        bank_chunk_size=3,
    )

    assert reference_ids.tolist() == [["a", "b", "m"], ["aa", "c", "d"]]
    for query_chunk in (1, 2):
        for bank_chunk in (3, 5, 7):
            observed = exact_raw_knn_topk(
                bank,
                queries,
                fit_sample_ids=sample_ids,
                k=3,
                query_chunk_size=query_chunk,
                bank_chunk_size=bank_chunk,
            )
            np.testing.assert_array_equal(
                observed["neighbor_squared_distances"], reference_distances
            )
            np.testing.assert_array_equal(
                observed["neighbor_sample_ids"], reference_ids
            )
            np.testing.assert_array_equal(
                sample_ids[observed["neighbor_bank_indices"]], reference_ids
            )


def test_memory_budget_resolves_a_safe_bank_block_and_preserves_parity():
    rng = np.random.default_rng(123)
    bank = rng.normal(size=(37, 9))
    queries = rng.normal(size=(11, 9))
    sample_ids = np.asarray([f"id-{index:03d}" for index in range(len(bank))])
    query_chunk = 4
    k = 5
    conservative_bytes_per_bank_item = query_chunk * 24
    budget = conservative_bytes_per_bank_item * 7

    observed = exact_raw_knn_topk(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=k,
        query_chunk_size=query_chunk,
        memory_budget_bytes=budget,
    )
    expected_distances, expected_ids = exact_neighbor_distances(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=k,
        query_chunk_size=query_chunk,
        bank_chunk_size=7,
    )

    assert observed["bank_chunk_size"] == 7
    np.testing.assert_array_equal(
        observed["neighbor_squared_distances"], expected_distances
    )
    np.testing.assert_array_equal(observed["neighbor_sample_ids"], expected_ids)
    np.testing.assert_array_equal(
        sample_ids[observed["neighbor_bank_indices"]], expected_ids
    )
    with pytest.raises(ValueError, match="too small"):
        exact_raw_knn_topk(
            bank,
            queries,
            fit_sample_ids=sample_ids,
            k=k,
            query_chunk_size=query_chunk,
            memory_budget_bytes=conservative_bytes_per_bank_item * (k - 1),
        )


def test_explicit_bank_chunk_smaller_than_k_resolves_to_k_with_or_without_budget():
    rng = np.random.default_rng(321)
    bank = rng.normal(size=(23, 7))
    queries = rng.normal(size=(6, 7))
    sample_ids = np.asarray([f"id-{index:03d}" for index in range(len(bank))])
    k = 9
    query_chunk = 3
    minimum_budget = query_chunk * 24 * k
    for budget in (None, minimum_budget):
        observed = exact_raw_knn_topk(
            bank,
            queries,
            fit_sample_ids=sample_ids,
            k=k,
            query_chunk_size=query_chunk,
            bank_chunk_size=2,
            memory_budget_bytes=budget,
        )
        expected_distances, expected_ids = exact_neighbor_distances(
            bank,
            queries,
            fit_sample_ids=sample_ids,
            k=k,
            query_chunk_size=query_chunk,
            bank_chunk_size=k,
        )
        assert observed["bank_chunk_size"] == k
        np.testing.assert_array_equal(
            observed["neighbor_squared_distances"], expected_distances
        )
        np.testing.assert_array_equal(observed["neighbor_sample_ids"], expected_ids)


def test_neighbor_strings_can_be_suppressed_with_lossless_bank_indices():
    bank = np.eye(5, dtype=np.float64)
    queries = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]])
    sample_ids = np.asarray(["z", "a", "m", "b", "c"])

    observed = exact_raw_knn_topk(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=3,
        return_neighbor_sample_ids=False,
    )

    assert observed["neighbor_sample_ids"] is None
    reconstructed = sample_ids[observed["neighbor_bank_indices"]]
    assert reconstructed.tolist() == [["z", "a", "b"]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must"),
        ({"query_chunk_size": 0}, "query_chunk_size"),
        ({"bank_chunk_size": 0}, "bank_chunk_size"),
    ],
)
def test_invalid_search_parameters_fail_closed(kwargs, message):
    bank = np.eye(4, dtype=np.float64)
    queries = np.eye(2, 4, dtype=np.float64)
    arguments = {"k": 2, **kwargs}
    with pytest.raises(ValueError, match=message):
        exact_raw_knn_topk(
            bank,
            queries,
            fit_sample_ids=["a", "b", "c", "d"],
            **arguments,
        )


def test_invalid_feature_and_sample_identity_inputs_fail_closed():
    bank = np.eye(4, dtype=np.float64)
    queries = np.eye(2, 4, dtype=np.float64)
    with pytest.raises(ValueError, match="unique"):
        exact_raw_knn_topk(
            bank,
            queries,
            fit_sample_ids=["a", "a", "c", "d"],
            k=2,
        )
    corrupted = bank.copy()
    corrupted[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        exact_raw_knn_topk(
            corrupted,
            queries,
            fit_sample_ids=["a", "b", "c", "d"],
            k=2,
        )
    with pytest.raises(ValueError, match="widths differ"):
        exact_raw_knn_topk(
            bank,
            np.ones((2, 3)),
            fit_sample_ids=["a", "b", "c", "d"],
            k=2,
        )


def test_synthetic_wrntype_benchmark_is_explicitly_not_production_evidence():
    report = run_synthetic_benchmark(
        benchmark_fit_count=64,
        benchmark_query_count=8,
        feature_dim=640,
        k=5,
        query_chunk_size=4,
        bank_chunk_size=32,
        repeats=1,
        seed=99,
        target_fit_count=45_000,
        target_query_count_per_bundle=10_000,
        target_bundle_count=30,
    )

    assert report["synthetic_only"] is True
    assert report["benchmark"]["feature_dim"] == 640
    assert report["target_scenario"] == {
        "fit_count": 45_000,
        "query_count_per_bundle": 10_000,
        "bundle_count": 30,
        "variant_count": 7,
        "feature_dim": 640,
    }
    assert report["estimate"]["target_distance_pairs_per_variant_per_bundle"] == 450_000_000
    assert report["estimate"]["target_distance_pairs_total"] == 94_500_000_000
    assert "no Stage-2 production payload was read" in report["estimate"][
        "limitations"
    ]


def test_workload_estimate_has_auditable_linear_arithmetic():
    estimate = estimate_workload(
        measured_fit_count=100,
        measured_query_count=20,
        measured_seconds=2.0,
        feature_dim=640,
        target_fit_count=45_000,
        target_query_count_per_bundle=10_000,
        target_bundle_count=30,
    )
    assert estimate["measured_distance_pairs_per_second"] == 1000.0
    assert estimate["target_distance_pairs_per_variant_per_bundle"] == 450_000_000
    assert estimate["target_distance_pairs_per_bundle_all_variants"] == 3_150_000_000
    assert estimate["target_distance_pairs_total"] == 94_500_000_000
    assert estimate["estimated_seconds"] == 94_500_000.0
    assert estimate["target_cross_term_flops_approx"] == 120_960_000_000_000


def test_benchmark_cli_defaults_cover_real_query_population_and_all_seven_variants():
    args = BENCHMARK.parse_args([])
    assert args.target_query_count_per_bundle == 163_660
    assert args.target_bundle_count == 30
    assert args.target_variant_count == 7
    expected = 45_000 * 163_660 * 30 * 7
    estimate = estimate_workload(
        measured_fit_count=100,
        measured_query_count=10,
        measured_seconds=1.0,
        feature_dim=640,
        target_fit_count=args.target_fit_count,
        target_query_count_per_bundle=args.target_query_count_per_bundle,
        target_bundle_count=args.target_bundle_count,
        target_variant_count=args.target_variant_count,
    )
    assert estimate["target_distance_pairs_total_all_variants"] == expected
    assert expected == 1_546_587_000_000
