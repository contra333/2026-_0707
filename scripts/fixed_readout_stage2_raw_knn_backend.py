#!/usr/bin/env python3
"""Scalable exact-float64 raw-feature kNN for Stage 2.

The backend performs exhaustive squared-Euclidean search.  It uses BLAS-friendly
query/bank blocks for the distance calculation, ``argpartition`` to reduce each
row to its exact K-boundary, and an explicit lexicographic boundary-tie repair
using the frozen sample IDs.  Consequently chunk boundaries do not alter the
ordered ``(distance, sample_id)`` result.

This module intentionally contains no Stage-2 artifact loading or result
selection.  Callers remain responsible for checksum/identity validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


BACKEND_ID = "numpy_blas_exact_float64_lexicographic_v1"
DEFAULT_K = 50
_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_INT64_BYTES = np.dtype(np.int64).itemsize


def _finite_float64_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validated_sample_ids(value: Any, *, count: int) -> np.ndarray:
    sample_ids = np.asarray(value, dtype=str).reshape(-1)
    if len(sample_ids) != count:
        raise ValueError("fit_sample_ids must match the reference-bank count")
    if any(not item for item in sample_ids.tolist()):
        raise ValueError("fit_sample_ids must be non-empty")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("fit_sample_ids must be unique")
    return sample_ids


def _lexicographic_id_ranks(sample_ids: np.ndarray) -> np.ndarray:
    """Return unique integer ranks with the same ordering as NumPy strings."""
    order = np.argsort(sample_ids, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    return ranks


def resolve_bank_chunk_size(
    *,
    bank_count: int,
    k: int,
    query_chunk_size: int,
    bank_chunk_size: int | None,
    memory_budget_bytes: int | None,
) -> int:
    if bank_chunk_size is not None:
        if isinstance(bank_chunk_size, bool) or not isinstance(bank_chunk_size, int):
            raise ValueError("bank_chunk_size must be an integer")
        if bank_chunk_size <= 0:
            raise ValueError("bank_chunk_size must be positive")
        requested = min(bank_count, bank_chunk_size)
    else:
        requested = bank_count
    # Every block that reaches the merge must be able to establish a complete
    # K-neighbor frontier.  This also freezes create/resume semantics when a
    # caller explicitly asks for a bank chunk smaller than K.
    requested = min(bank_count, max(k, requested))

    if memory_budget_bytes is None:
        return requested
    if isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, int):
        raise ValueError("memory_budget_bytes must be an integer")
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")

    # The dominant resident block is the Q x B float64 distance matrix.  The
    # per-row merge additionally needs one float64 distance and one int64 index
    # per bank candidate.  Reserving both for every query is conservative and
    # keeps the stated budget meaningful across NumPy/BLAS implementations.
    bytes_per_bank_item = query_chunk_size * (
        2 * _FLOAT64_BYTES + _INT64_BYTES
    )
    budget_limited = memory_budget_bytes // bytes_per_bank_item
    if budget_limited < k:
        minimum = k * bytes_per_bank_item
        raise ValueError(
            "memory_budget_bytes is too small for one exact K-sized bank block; "
            f"need at least {minimum} bytes"
        )
    return min(requested, bank_count, int(budget_limited))


def _exact_boundary_topk(
    distances: np.ndarray,
    bank_indices: np.ndarray,
    *,
    id_ranks: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and order exact K smallest ``(distance, sample_id)`` pairs."""
    if distances.ndim != 1 or bank_indices.ndim != 1:
        raise ValueError("top-k inputs must be rank-1")
    if len(distances) != len(bank_indices) or len(distances) < k:
        raise ValueError("top-k inputs have incompatible lengths")

    partition = np.argpartition(distances, k - 1)[:k]
    boundary = float(np.max(distances[partition]))
    strict = np.flatnonzero(distances < boundary)
    boundary_ties = np.flatnonzero(distances == boundary)
    needed = k - len(strict)
    if needed <= 0 or len(boundary_ties) < needed:
        raise AssertionError("argpartition boundary accounting failed")

    if len(boundary_ties) == needed:
        chosen_ties = boundary_ties
    else:
        tie_ranks = id_ranks[bank_indices[boundary_ties]]
        tie_partition = np.argpartition(tie_ranks, needed - 1)[:needed]
        chosen_ties = boundary_ties[tie_partition]
    chosen = np.concatenate((strict, chosen_ties))
    if len(chosen) != k:
        raise AssertionError("exact boundary repair did not select K neighbors")

    chosen_bank_indices = bank_indices[chosen]
    order = np.lexsort(
        (id_ranks[chosen_bank_indices], distances[chosen])
    )
    ordered = chosen[order]
    return distances[ordered], bank_indices[ordered]


def exact_raw_knn_topk(
    fit_features: Any,
    query_features: Any,
    *,
    fit_sample_ids: Any,
    k: int = DEFAULT_K,
    query_chunk_size: int = 64,
    bank_chunk_size: int | None = None,
    memory_budget_bytes: int | None = None,
    return_neighbor_sample_ids: bool = True,
) -> dict[str, Any]:
    """Return exhaustive raw-feature top-K neighbors and ID-like scores.

    Distances are computed in float64 as
    ``||q||^2 + ||x||^2 - 2 q @ x`` and clipped at zero to remove negative
    roundoff.  There is intentionally no self-exclusion: Stage-2 queries are
    ID-test or OOD examples, while the reference bank is ID-train.
    """
    bank = _finite_float64_matrix(fit_features, name="fit_features")
    queries = _finite_float64_matrix(query_features, name="query_features")
    if bank.shape[1] != queries.shape[1]:
        raise ValueError("fit/query feature widths differ")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or k > len(bank):
        raise ValueError("k must be a positive integer no larger than the bank")
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size <= 0
    ):
        raise ValueError("query_chunk_size must be a positive integer")

    sample_ids = _validated_sample_ids(fit_sample_ids, count=len(bank))
    if not isinstance(return_neighbor_sample_ids, bool):
        raise ValueError("return_neighbor_sample_ids must be boolean")
    id_ranks = _lexicographic_id_ranks(sample_ids)
    resolved_bank_chunk_size = resolve_bank_chunk_size(
        bank_count=len(bank),
        k=k,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
    )

    output_distances = np.empty((len(queries), k), dtype=np.float64)
    output_indices = np.empty((len(queries), k), dtype=np.int64)
    # Match the frozen v1.2 oracle's evaluation order for strict numerical
    # parity while leaving the dominant cross term to BLAS.
    bank_norms = np.sum(bank * bank, axis=1)
    if not np.isfinite(bank_norms).all():
        raise ValueError("fit feature squared norms overflowed float64")

    for query_start in range(0, len(queries), query_chunk_size):
        query_stop = min(query_start + query_chunk_size, len(queries))
        query_block = queries[query_start:query_stop]
        query_norms = np.sum(query_block * query_block, axis=1)
        if not np.isfinite(query_norms).all():
            raise ValueError("query feature squared norms overflowed float64")

        best_distances: np.ndarray | None = None
        best_indices: np.ndarray | None = None
        for bank_start in range(0, len(bank), resolved_bank_chunk_size):
            bank_stop = min(bank_start + resolved_bank_chunk_size, len(bank))
            bank_block = bank[bank_start:bank_stop]
            # Preserve the frozen oracle's addition/subtraction order so the
            # backend can be tested at bit-level parity for a fixed chunking.
            distances = (
                query_norms[:, None]
                + bank_norms[None, bank_start:bank_stop]
                - 2.0 * (query_block @ bank_block.T)
            )
            np.maximum(distances, 0.0, out=distances)
            if not np.isfinite(distances).all():
                raise ValueError("squared-distance computation overflowed float64")
            block_indices = np.arange(bank_start, bank_stop, dtype=np.int64)

            next_distances = np.empty(
                (len(query_block), min(k, bank_stop)), dtype=np.float64
            )
            next_indices = np.empty(next_distances.shape, dtype=np.int64)
            for row_index in range(len(query_block)):
                if best_distances is None or best_indices is None:
                    merged_distances = distances[row_index]
                    merged_indices = block_indices
                else:
                    merged_distances = np.concatenate(
                        (best_distances[row_index], distances[row_index])
                    )
                    merged_indices = np.concatenate(
                        (best_indices[row_index], block_indices)
                    )
                row_k = min(k, len(merged_distances))
                if row_k < k:
                    order = np.lexsort(
                        (id_ranks[merged_indices], merged_distances)
                    )
                    selected_distances = merged_distances[order]
                    selected_indices = merged_indices[order]
                else:
                    selected_distances, selected_indices = _exact_boundary_topk(
                        merged_distances,
                        merged_indices,
                        id_ranks=id_ranks,
                        k=k,
                    )
                next_distances[row_index, :row_k] = selected_distances
                next_indices[row_index, :row_k] = selected_indices
            best_distances = next_distances
            best_indices = next_indices

        if best_distances is None or best_indices is None:
            raise AssertionError("exact raw-kNN did not visit the reference bank")
        if best_distances.shape != (len(query_block), k):
            raise AssertionError("exact raw-kNN did not produce K neighbors")
        output_distances[query_start:query_stop] = best_distances
        output_indices[query_start:query_stop] = best_indices

    output_ids = sample_ids[output_indices] if return_neighbor_sample_ids else None
    return {
        "score": -output_distances[:, -1],
        "neighbor_squared_distances": output_distances,
        "neighbor_sample_ids": output_ids,
        # Exact bank positions avoid materializing millions of fixed-width
        # strings in resumable caches.  The caller binds these indices to the
        # checksummed, ordered fit_sample_ids array and can reconstruct every
        # neighbor identity losslessly.
        "neighbor_bank_indices": output_indices,
        "backend": BACKEND_ID,
        "k": int(k),
        "query_chunk_size": int(query_chunk_size),
        "bank_chunk_size": int(resolved_bank_chunk_size),
    }


__all__ = ["BACKEND_ID", "exact_raw_knn_topk", "resolve_bank_chunk_size"]
