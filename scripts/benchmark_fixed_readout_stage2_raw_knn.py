#!/usr/bin/env python3
"""Synthetic-only benchmark and linear workload estimate for Stage-2 raw kNN."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from statistics import median
from typing import Any, Sequence

import numpy as np

try:
    from scripts.fixed_readout_stage2_raw_knn_backend import (
        BACKEND_ID,
        exact_raw_knn_topk,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    try:
        from fixed_readout_stage2_raw_knn_backend import (  # type: ignore[no-redef]
            BACKEND_ID,
            exact_raw_knn_topk,
        )
    except ModuleNotFoundError:  # Loaded by path from a test or audit harness.
        backend_path = Path(__file__).with_name(
            "fixed_readout_stage2_raw_knn_backend.py"
        )
        backend_spec = importlib.util.spec_from_file_location(
            "fixed_readout_stage2_raw_knn_backend", backend_path
        )
        if backend_spec is None or backend_spec.loader is None:
            raise ImportError(f"cannot load raw-kNN backend: {backend_path}")
        backend_module = importlib.util.module_from_spec(backend_spec)
        sys.modules[backend_spec.name] = backend_module
        backend_spec.loader.exec_module(backend_module)
        BACKEND_ID = backend_module.BACKEND_ID
        exact_raw_knn_topk = backend_module.exact_raw_knn_topk


def estimate_workload(
    *,
    measured_fit_count: int,
    measured_query_count: int,
    measured_seconds: float,
    feature_dim: int,
    target_fit_count: int,
    target_query_count_per_bundle: int,
    target_bundle_count: int,
    target_variant_count: int = 7,
) -> dict[str, Any]:
    positive_ints = {
        "measured_fit_count": measured_fit_count,
        "measured_query_count": measured_query_count,
        "feature_dim": feature_dim,
        "target_fit_count": target_fit_count,
        "target_query_count_per_bundle": target_query_count_per_bundle,
        "target_bundle_count": target_bundle_count,
        "target_variant_count": target_variant_count,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in positive_ints.values()
    ):
        raise ValueError("all workload counts must be positive integers")
    if not np.isfinite(measured_seconds) or measured_seconds <= 0.0:
        raise ValueError("measured_seconds must be positive and finite")

    measured_pairs = measured_fit_count * measured_query_count
    target_pairs_per_variant_per_bundle = (
        target_fit_count * target_query_count_per_bundle
    )
    target_pairs_per_bundle_all_variants = (
        target_pairs_per_variant_per_bundle * target_variant_count
    )
    target_pairs_per_variant_all_bundles = (
        target_pairs_per_variant_per_bundle * target_bundle_count
    )
    target_pairs_total = (
        target_pairs_per_bundle_all_variants * target_bundle_count
    )
    pairs_per_second = measured_pairs / measured_seconds
    estimated_seconds = target_pairs_total / pairs_per_second
    # The matrix product dominates and performs approximately one multiply and
    # one add per feature coordinate.  Top-k and memory traffic are deliberately
    # excluded, so this is an operation-count descriptor, not a runtime model.
    target_cross_term_flops = 2 * target_pairs_total * feature_dim
    return {
        "estimate_kind": "linear_pair_throughput_not_production_measurement",
        "measured_distance_pairs": measured_pairs,
        "measured_distance_pairs_per_second": pairs_per_second,
        "target_variant_count": target_variant_count,
        "target_distance_pairs_per_variant_per_bundle": (
            target_pairs_per_variant_per_bundle
        ),
        "target_distance_pairs_per_bundle_all_variants": (
            target_pairs_per_bundle_all_variants
        ),
        "target_distance_pairs_per_variant_all_bundles": (
            target_pairs_per_variant_all_bundles
        ),
        "target_distance_pairs_total": target_pairs_total,
        "target_distance_pairs_total_all_variants": target_pairs_total,
        "target_cross_term_flops_approx": target_cross_term_flops,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600.0,
        "estimated_seconds_per_variant_per_bundle": (
            target_pairs_per_variant_per_bundle / pairs_per_second
        ),
        "estimated_seconds_per_variant_all_bundles": (
            target_pairs_per_variant_all_bundles / pairs_per_second
        ),
        "estimated_seconds_per_bundle_all_variants": (
            target_pairs_per_bundle_all_variants / pairs_per_second
        ),
        "limitations": [
            "synthetic Gaussian features only",
            "linear extrapolation ignores cache and BLAS scaling changes",
            "throughput is specific to this host and numerical-thread runtime",
            "target counts describe the frozen planning workload, not artifact evidence",
            "no Stage-2 production payload was read",
        ],
    }


def run_synthetic_benchmark(
    *,
    benchmark_fit_count: int,
    benchmark_query_count: int,
    feature_dim: int,
    k: int,
    query_chunk_size: int,
    bank_chunk_size: int | None,
    repeats: int,
    seed: int,
    target_fit_count: int,
    target_query_count_per_bundle: int,
    target_bundle_count: int,
    target_variant_count: int = 7,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    fit = rng.standard_normal((benchmark_fit_count, feature_dim), dtype=np.float64)
    query = rng.standard_normal((benchmark_query_count, feature_dim), dtype=np.float64)
    sample_ids = np.asarray(
        [f"synthetic-fit-{index:08d}" for index in range(benchmark_fit_count)]
    )

    # Warm BLAS dispatch and validate the complete backend once before timing.
    warm = exact_raw_knn_topk(
        fit,
        query[: min(query_chunk_size, len(query))],
        fit_sample_ids=sample_ids,
        k=k,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
    )
    elapsed = []
    result = warm
    for _ in range(repeats):
        start = time.perf_counter()
        result = exact_raw_knn_topk(
            fit,
            query,
            fit_sample_ids=sample_ids,
            k=k,
            query_chunk_size=query_chunk_size,
            bank_chunk_size=bank_chunk_size,
        )
        elapsed.append(time.perf_counter() - start)
    measured_seconds = float(median(elapsed))
    estimate = estimate_workload(
        measured_fit_count=benchmark_fit_count,
        measured_query_count=benchmark_query_count,
        measured_seconds=measured_seconds,
        feature_dim=feature_dim,
        target_fit_count=target_fit_count,
        target_query_count_per_bundle=target_query_count_per_bundle,
        target_bundle_count=target_bundle_count,
        target_variant_count=target_variant_count,
    )
    return {
        "schema_version": "fixed_readout_stage2_raw_knn_synthetic_benchmark_v2",
        "synthetic_only": True,
        "backend": BACKEND_ID,
        "seed": seed,
        "runtime": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "benchmark": {
            "fit_count": benchmark_fit_count,
            "query_count": benchmark_query_count,
            "feature_dim": feature_dim,
            "k": k,
            "query_chunk_size": query_chunk_size,
            "bank_chunk_size": result["bank_chunk_size"],
            "repeats": repeats,
            "elapsed_seconds": elapsed,
            "median_elapsed_seconds": measured_seconds,
        },
        "target_scenario": {
            "fit_count": target_fit_count,
            "query_count_per_bundle": target_query_count_per_bundle,
            "bundle_count": target_bundle_count,
            "variant_count": target_variant_count,
            "feature_dim": feature_dim,
        },
        "estimate": estimate,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-fit-count", type=int, default=45_000)
    parser.add_argument("--benchmark-query-count", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=640)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--bank-chunk-size", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--target-fit-count", type=int, default=45_000)
    parser.add_argument(
        "--target-query-count-per-bundle", type=int, default=163_660
    )
    parser.add_argument("--target-bundle-count", type=int, default=30)
    parser.add_argument("--target-variant-count", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_synthetic_benchmark(
        benchmark_fit_count=args.benchmark_fit_count,
        benchmark_query_count=args.benchmark_query_count,
        feature_dim=args.feature_dim,
        k=args.k,
        query_chunk_size=args.query_chunk_size,
        bank_chunk_size=args.bank_chunk_size,
        repeats=args.repeats,
        seed=args.seed,
        target_fit_count=args.target_fit_count,
        target_query_count_per_bundle=args.target_query_count_per_bundle,
        target_bundle_count=args.target_bundle_count,
        target_variant_count=args.target_variant_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
