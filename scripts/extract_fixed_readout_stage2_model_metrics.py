#!/usr/bin/env python3
"""Extract the complete per-model evidence table for the frozen Stage-2 gate.

This worker is intentionally narrower than the scientific reducer.  It verifies
the exact selective-reuse population and the frozen v1.2 scalar baseline,
reconstructs the two prespecified raw -> radial-normalized detector pairs, and
writes deterministic evidence.  It never ranks candidates and always records
the scientific gate as ``NOT_RUN``.

Production output is a new directory containing ``selection_manifest.json``,
``model_metrics.jsonl``, ``diagnostic_metrics.jsonl``, and
``checksums.sha256``.  The directory is created by an atomic rename and is
never overwritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    # Support both direct CLI execution and importlib-based pytest loading.  The
    # repository deliberately does not make ``scripts`` a Python package.
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_fixed_readout_stage2_mechanism import (  # noqa: E402
    REPOSITORY_ROOT,
    ReuseContext,
    SelectiveBundle,
    SelectiveBundleAccess,
    _assert_non_git_root,
    _load_raw_array,
    _verify_exact_retrieval_tree,
    load_reuse_context,
    load_selective_bundle,
    validate_selective_reuse_manifest,
)
from fixed_readout_stage2_raw_knn_backend import (  # noqa: E402
    BACKEND_ID as RAW_KNN_BACKEND_ID,
    exact_raw_knn_topk,
    resolve_bank_chunk_size,
)
from oge.evaluation.classification import logit_ood_scores  # noqa: E402
from oge.evaluation.common import (  # noqa: E402
    EPSILON_NORM,
    norm_diagnostics,
    normalize_rows,
)
from oge.evaluation.detectors import (  # noqa: E402
    TiedGaussianFit,
    fit_tied_gaussians,
    fit_vim,
    prototype_scores,
    score_tied_gaussians,
    score_vim,
)
from oge.evaluation.extraction import ordered_sample_id_digest, sha256_file  # noqa: E402
from oge.evaluation.metrics import compute_ood_metrics  # noqa: E402


SELECTION_MANIFEST_SCHEMA = "fixed_readout_stage2_selection_manifest_v1"
MODEL_METRIC_SCHEMA = "fixed_readout_stage2_model_metric_v1"
DIAGNOSTIC_METRIC_SCHEMA = "fixed_readout_stage2_diagnostic_metric_v1"
ANALYSIS_CONTRACT_VERSION = "fixed_readout_training_rule_intervention_v2"
ANALYSIS_ROLE = "protected_discovery_not_confirmation_or_causality"
STAGE2_BUNDLE_COUNT = 30
STAGE2_OOD_DATASET_COUNT = 6
EXPECTED_DIAGNOSTIC_ROW_COUNT = 360
KNN_K = 50
EXACT_ATOL = 1.0e-12
EXACT_RTOL = 1.0e-10
KNN_APPROXIMATE_AUROC_TOLERANCE = 1.0e-3
KNN_CACHE_SCHEMA = "fixed_readout_stage2_knn_variant_cache_v1"
NUMERIC_RUNTIME_SCHEMA = "fixed_readout_stage2_numeric_runtime_v1"
CACHE_STAGING_DIRECTORY_SUFFIX = ".stage2-knn-staging"
FIXED_QUANTILE_PROBABILITIES = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
)
KNN_VARIANT_IDS = (
    "raw_identity",
    "l2_targeted",
    "signed_coordinate_permutation",
    "raw_common_positive_scale_2",
    "l2_common_positive_scale_2",
    "raw_nonuniform_samplewise_scale",
    "l2_after_nonuniform_samplewise_scale",
)
NUMERIC_RUNTIME_LOCK_PATHS = (
    "pyproject.toml",
    "configs/environments/wrn_v1_2_common.yaml",
)

METRIC_ARTIFACTS = {
    "auroc": "ood_metric/auroc_id_positive",
    "fpr95": "ood_metric/fpr95_id_tpr",
    "fpr95_threshold": "ood_metric/fpr95_threshold",
    "achieved_id_tpr": "ood_metric/fpr95_achieved_id_tpr",
}
ID_METRIC_ARTIFACTS = {
    "accuracy": "id/accuracy",
    "nll": "id/nll_raw",
    "ece": "id/ece_raw_m15",
}
PARITY_DETECTORS = (
    "detector/mahalanobis_raw",
    "detector/mahalanobis_pp",
    "detector/knn_l2_k50",
    "detector/ctm_prototype_cosine",
    "ood_score/energy_t1",
)
PARITY_BASELINE_LOGICAL_PATHS = (
    "aggregate.json",
    "artifact_manifest.json",
    "detector_rank_concordance.jsonl",
    "per_checkpoint_records.jsonl",
    "seed_aggregates.jsonl",
)
DIAGNOSTIC_IDS = ("pure_residual_channel", "ctm_angular_channel")
CANDIDATES = {
    "radial_l2_mahalanobis": {
        "raw_detector": "detector/mahalanobis_raw",
        "transformed_detector": "detector/mahalanobis_pp",
        "transform_id": "sample_l2_normalize_then_refit_tied_gaussian",
        "witness_id": "mahalanobis_nonuniform_samplewise_positive_scaling",
    },
    "radial_l2_knn": {
        "raw_detector": "detector/knn_raw_k50",
        "transformed_detector": "detector/knn_l2_k50",
        "transform_id": "sample_l2_normalize_bank_and_query",
        "witness_id": "knn_raw_nonuniform_samplewise_positive_scaling",
    },
}
TRANSITION_STATES = ("correct", "tie", "misordered")
TRANSITION_FIELDS = tuple(
    f"{before}_to_{after}"
    for before in TRANSITION_STATES
    for after in TRANSITION_STATES
)
CODE_SOURCE_PATHS = (
    "scripts/extract_fixed_readout_stage2_model_metrics.py",
    "scripts/analyze_fixed_readout_stage2_mechanism.py",
    "scripts/fixed_readout_stage2_raw_knn_backend.py",
    "src/oge/evaluation/classification.py",
    "src/oge/evaluation/common.py",
    "src/oge/evaluation/detectors.py",
    "src/oge/evaluation/extraction.py",
    "src/oge/evaluation/metrics.py",
    *NUMERIC_RUNTIME_LOCK_PATHS,
)


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _numeric_runtime_fingerprint() -> dict[str, Any]:
    """Return the exact numeric runtime identity used by every cache shard."""
    repository_locks = []
    for relative in NUMERIC_RUNTIME_LOCK_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"numeric runtime lock source is absent: {relative}")
        repository_locks.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    try:
        numpy_build = np.__config__.show(mode="dicts")
    except TypeError:  # pragma: no cover - compatibility for older NumPy.
        stream = io.StringIO()
        with redirect_stdout(stream):
            np.__config__.show()
        numpy_build = {"legacy_show_output": stream.getvalue()}
    runtime_libraries = sorted(
        (dict(item) for item in threadpool_info()),
        key=lambda item: (
            str(item.get("user_api")),
            str(item.get("internal_api")),
            str(item.get("prefix")),
            str(item.get("filepath")),
        ),
    )
    source_root = (REPOSITORY_ROOT / "src").resolve()
    oge_module_origins = {}
    for module_name in (
        "oge",
        "oge.evaluation.classification",
        "oge.evaluation.common",
        "oge.evaluation.detectors",
        "oge.evaluation.extraction",
        "oge.evaluation.metrics",
    ):
        module = importlib.import_module(module_name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise ValueError(f"imported OGE module has no file origin: {module_name}")
        resolved_origin = Path(raw_origin).resolve()
        if resolved_origin != source_root and source_root not in resolved_origin.parents:
            raise ValueError(
                f"imported OGE module is outside repository src: {module_name}"
            )
        oge_module_origins[module_name] = {
            "raw_path": raw_origin,
            "resolved_path": str(resolved_origin),
            "sha256": sha256_file(resolved_origin),
        }
    raw_executable = sys.executable
    payload: dict[str, Any] = {
        "schema_version": NUMERIC_RUNTIME_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_string": sys.version,
            "executable": raw_executable,
            "resolved_executable": str(Path(raw_executable).resolve()),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "blas": {
            "numpy_build_configuration": numpy_build,
            "runtime_libraries": runtime_libraries,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "BLIS_NUM_THREADS",
                "MKL_DYNAMIC",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "python_import_environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        },
        "oge_module_origins": oge_module_origins,
        "repository_runtime_locks": repository_locks,
        "repository_runtime_lock_sha256": _sha256_bytes(
            _canonical_bytes(repository_locks)
        ),
    }
    payload["canonical_lock_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _validate_runtime_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    supplied = payload.pop("canonical_lock_sha256", None)
    if payload.get("schema_version") != NUMERIC_RUNTIME_SCHEMA:
        raise ValueError("numeric runtime fingerprint schema mismatch")
    expected = _sha256_bytes(_canonical_bytes(payload))
    if supplied != expected:
        raise ValueError("numeric runtime fingerprint canonical lock mismatch")
    payload["canonical_lock_sha256"] = expected
    return payload


def _distribution_summary(values: Any, *, label: str) -> dict[str, Any]:
    array = _finite_vector(values, label=label)
    probabilities = np.asarray(
        [probability for _, probability in FIXED_QUANTILE_PROBABILITIES],
        dtype=np.float64,
    )
    quantile_values = np.quantile(array, probabilities, method="linear")
    return {
        "summary_schema": "fixed_linear_quantiles_v1",
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std_ddof0": float(np.std(array, ddof=0)),
        "quantiles": {
            key: float(result)
            for (key, _), result in zip(
                FIXED_QUANTILE_PROBABILITIES, quantile_values.tolist()
            )
        },
        "array_sha256": _array_sha256(array),
    }


def _normalization_record(
    values: Any, *, denominator_epsilon: float, label: str
) -> dict[str, Any]:
    norms, diagnostics = norm_diagnostics(values, name=label)
    exact_zero = int(diagnostics["exact_zero_count"])
    near_zero = int(diagnostics["near_zero_count"])
    if exact_zero:
        status = "failed"
        reasons = ["exact_zero_norm"]
        if near_zero:
            reasons.append("near_zero_norm")
    elif near_zero:
        status = "degenerate"
        reasons = ["near_zero_norm"]
    else:
        status = "success"
        reasons = []
    if status == "failed":
        raise ValueError(f"{label} contains an exact-zero norm")
    return {
        "status": status,
        "reason_codes": reasons,
        "epsilon_norm": float(diagnostics["epsilon_norm"]),
        "denominator_epsilon": float(denominator_epsilon),
        "exact_zero_count": exact_zero,
        "near_zero_count": near_zero,
        "sample_count": int(len(norms)),
    }


def _geometry_channel_record(
    split_features: Mapping[str, np.ndarray],
    *,
    denominator_epsilon: float,
    active_ood_split: str,
) -> dict[str, Any]:
    split_rows = {}
    for split_name, features in split_features.items():
        norms, _ = norm_diagnostics(features, name=f"{split_name} penultimate features")
        split_rows[split_name] = {
            "feature_norm": _distribution_summary(
                norms, label=f"{split_name} penultimate L2 norms"
            ),
            "normalization": _normalization_record(
                features,
                denominator_epsilon=denominator_epsilon,
                label=f"{split_name} penultimate features",
            ),
        }
    if active_ood_split not in split_rows:
        raise ValueError("active OOD split is absent from geometry channel")
    return {
        "channel_id": "penultimate_feature_l2_norm",
        "normalization_contract": "oge.evaluation.common.normalize_rows",
        "normalization_denominator_epsilon": float(denominator_epsilon),
        "active_ood_split": active_ood_split,
        "splits": split_rows,
    }


def _normalization_status_preservation(
    base_features: Mapping[str, np.ndarray],
    witness_features: Mapping[str, np.ndarray],
    *,
    denominator_epsilon: float,
) -> dict[str, Any]:
    if tuple(base_features) != tuple(witness_features):
        raise ValueError("base/witness normalization split order mismatch")
    preserved_fields = (
        "status",
        "reason_codes",
        "epsilon_norm",
        "denominator_epsilon",
        "exact_zero_count",
        "near_zero_count",
        "sample_count",
    )
    split_rows = {}
    failed = []
    for split_name in base_features:
        base = _normalization_record(
            base_features[split_name],
            denominator_epsilon=denominator_epsilon,
            label=f"base {split_name} normalization status",
        )
        witness = _normalization_record(
            witness_features[split_name],
            denominator_epsilon=denominator_epsilon,
            label=f"witness {split_name} normalization status",
        )
        preserved = all(base[field] == witness[field] for field in preserved_fields)
        split_rows[split_name] = {
            "preserved": bool(preserved),
            "base": base,
            "witness": witness,
        }
        if not preserved:
            failed.append(split_name)
    if failed:
        raise AssertionError(
            f"witness normalization status changed: {sorted(failed)}"
        )
    return {
        "status": "PASS",
        "reason_codes": [],
        "denominator_epsilon": float(denominator_epsilon),
        "preserved_fields": list(preserved_fields),
        "splits": split_rows,
    }


def _finite_vector(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a non-empty finite vector")
    return array


def _max_errors(reference: Any, reconstructed: Any) -> tuple[float, float]:
    left = _finite_vector(reference, label="reference score")
    right = _finite_vector(reconstructed, label="reconstructed score")
    if left.shape != right.shape:
        raise ValueError("score reconstruction shape mismatch")
    absolute = np.abs(left - right)
    denominator = np.maximum(np.abs(left), np.finfo(np.float64).tiny)
    return float(np.max(absolute)), float(np.max(absolute / denominator))


def _rank_signature_disagreement(left: Any, right: Any) -> float:
    """Fraction of samples whose exact weak-order signature changed.

    A signature consists of the number of strictly smaller values and the size
    of the equality class.  Unlike an ordinal argsort rank, it does not invent
    an order inside ties.
    """
    a = _finite_vector(left, label="left rank score")
    b = _finite_vector(right, label="right rank score")
    if a.shape != b.shape:
        raise ValueError("rank comparison shape mismatch")

    def signatures(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, inverse, counts = np.unique(
            values, return_inverse=True, return_counts=True
        )
        lower_by_group = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
        )
        return lower_by_group[inverse], counts[inverse]

    a_lower, a_ties = signatures(a)
    b_lower, b_ties = signatures(b)
    return float(np.mean((a_lower != b_lower) | (a_ties != b_ties)))


class _Fenwick:
    def __init__(self, size: int) -> None:
        self._tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, zero_based_index: int) -> None:
        index = zero_based_index + 1
        while index < len(self._tree):
            self._tree[index] += 1
            index += index & -index

    def prefix(self, stop_exclusive: int) -> int:
        total = 0
        index = stop_exclusive
        while index:
            total += int(self._tree[index])
            index -= index & -index
        return total


def exact_pair_order_transitions(
    raw_id_scores: Any,
    raw_ood_scores: Any,
    transformed_id_scores: Any,
    transformed_ood_scores: Any,
) -> dict[str, Any]:
    """Count the raw/transformed 3x3 pair states without an N x M matrix."""
    raw_id = _finite_vector(raw_id_scores, label="raw ID scores")
    raw_ood = _finite_vector(raw_ood_scores, label="raw OOD scores")
    transformed_id = _finite_vector(
        transformed_id_scores, label="transformed ID scores"
    )
    transformed_ood = _finite_vector(
        transformed_ood_scores, label="transformed OOD scores"
    )
    if len(raw_id) != len(transformed_id) or len(raw_ood) != len(transformed_ood):
        raise ValueError("raw/transformed pair populations differ")

    transformed_values = np.unique(transformed_ood)
    transformed_ranks = np.searchsorted(transformed_values, transformed_ood)
    raw_ood_order = np.argsort(raw_ood, kind="stable")
    raw_id_order = np.argsort(raw_id, kind="stable")
    all_transformed_sorted = np.sort(transformed_ood, kind="stable")
    counts = {
        before: {after: 0 for after in TRANSITION_STATES}
        for before in TRANSITION_STATES
    }
    fenwick = _Fenwick(len(transformed_values))
    ood_cursor = 0
    id_cursor = 0
    while id_cursor < len(raw_id_order):
        raw_value = raw_id[raw_id_order[id_cursor]]
        id_stop = id_cursor + 1
        while id_stop < len(raw_id_order) and raw_id[raw_id_order[id_stop]] == raw_value:
            id_stop += 1
        while (
            ood_cursor < len(raw_ood_order)
            and raw_ood[raw_ood_order[ood_cursor]] < raw_value
        ):
            ood_index = raw_ood_order[ood_cursor]
            fenwick.add(int(transformed_ranks[ood_index]))
            ood_cursor += 1
        equal_stop = ood_cursor
        while (
            equal_stop < len(raw_ood_order)
            and raw_ood[raw_ood_order[equal_stop]] == raw_value
        ):
            equal_stop += 1
        equal_transformed = np.sort(
            transformed_ood[raw_ood_order[ood_cursor:equal_stop]], kind="stable"
        )
        less_raw_count = ood_cursor

        for ordered_id_index in raw_id_order[id_cursor:id_stop]:
            transformed_value = transformed_id[ordered_id_index]
            rank_left = int(np.searchsorted(transformed_values, transformed_value, side="left"))
            rank_right = int(np.searchsorted(transformed_values, transformed_value, side="right"))
            less_less = fenwick.prefix(rank_left)
            less_equal = fenwick.prefix(rank_right) - less_less
            less_greater = less_raw_count - less_less - less_equal

            equal_less = int(
                np.searchsorted(equal_transformed, transformed_value, side="left")
            )
            equal_right = int(
                np.searchsorted(equal_transformed, transformed_value, side="right")
            )
            equal_equal = equal_right - equal_less
            equal_greater = len(equal_transformed) - equal_right

            all_less = int(
                np.searchsorted(all_transformed_sorted, transformed_value, side="left")
            )
            all_right = int(
                np.searchsorted(all_transformed_sorted, transformed_value, side="right")
            )
            all_equal = all_right - all_less
            all_greater = len(transformed_ood) - all_right
            greater_less = all_less - less_less - equal_less
            greater_equal = all_equal - less_equal - equal_equal
            greater_greater = all_greater - less_greater - equal_greater

            for before, row in (
                ("correct", (less_less, less_equal, less_greater)),
                ("tie", (equal_less, equal_equal, equal_greater)),
                ("misordered", (greater_less, greater_equal, greater_greater)),
            ):
                for after, count in zip(TRANSITION_STATES, row):
                    if count < 0:
                        raise AssertionError("negative pair transition count")
                    counts[before][after] += int(count)
        id_cursor = id_stop

    pair_count = int(len(raw_id) * len(raw_ood))
    if sum(sum(row.values()) for row in counts.values()) != pair_count:
        raise AssertionError("pair transition counts do not cover the population")
    mass = {
        f"{before}_to_{after}": counts[before][after] / pair_count
        for before in TRANSITION_STATES
        for after in TRANSITION_STATES
    }
    credit = {"correct": 1.0, "tie": 0.5, "misordered": 0.0}
    corrected = sum(
        mass[f"{before}_to_{after}"] * max(credit[after] - credit[before], 0.0)
        for before in TRANSITION_STATES
        for after in TRANSITION_STATES
    )
    harmed = sum(
        mass[f"{before}_to_{after}"] * max(credit[before] - credit[after], 0.0)
        for before in TRANSITION_STATES
        for after in TRANSITION_STATES
    )
    raw_auroc = sum(
        mass[f"{before}_to_{after}"] * credit[before]
        for before in TRANSITION_STATES
        for after in TRANSITION_STATES
    )
    transformed_auroc = sum(
        mass[f"{before}_to_{after}"] * credit[after]
        for before in TRANSITION_STATES
        for after in TRANSITION_STATES
    )
    net = corrected - harmed
    identity_error = abs(net - (transformed_auroc - raw_auroc))
    return {
        "pair_count": pair_count,
        "counts": {
            f"{before}_to_{after}": counts[before][after]
            for before in TRANSITION_STATES
            for after in TRANSITION_STATES
        },
        "mass": mass,
        "corrected_pair_mass": float(corrected),
        "harmed_pair_mass": float(harmed),
        "net_corrected_pair_mass": float(net),
        "raw_auroc_from_pairs": float(raw_auroc),
        "transformed_auroc_from_pairs": float(transformed_auroc),
        "pair_credit_identity_error": float(identity_error),
        "pair_credit_identity_pass": bool(identity_error <= EXACT_ATOL),
        "rank_disagreement_fraction": float(
            sum(value for key, value in mass.items() if key.split("_to_")[0] != key.split("_to_")[1])
        ),
    }


def _git_bytes(relative_path: str, *, revision: str = "HEAD") -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def validate_committed_sources(
    reuse_manifest_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    resolved_manifest = reuse_manifest_path.resolve()
    try:
        manifest_relative = resolved_manifest.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("production reuse manifest must be inside the repository") from exc
    source_paths = (
        *CODE_SOURCE_PATHS,
        manifest_relative,
        "configs/evaluation/fixed_readout_stage2/candidate_registry.json",
        "configs/evaluation/fixed_readout_stage2/gate_policy.json",
    )
    rows = []
    for relative in source_paths:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required analysis source is absent: {relative}")
        working = path.read_bytes()
        committed = _git_bytes(relative)
        if working != committed:
            raise ValueError(f"analysis source differs from HEAD: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(working),
                "size": len(working),
            }
        )
    return head, rows


def load_single_bundle_reuse_context(
    reuse_manifest_path: str | Path,
    retrieval_root: str | Path,
    bundle_id: str,
) -> ReuseContext:
    """Load one exact allowlisted bundle without requiring the other 29 trees."""
    manifest_path = Path(reuse_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("reuse manifest must contain a JSON mapping")
    validate_selective_reuse_manifest(manifest)
    all_bundles = {str(row["bundle_id"]): dict(row) for row in manifest["bundles"]}
    if bundle_id not in all_bundles:
        raise ValueError("precompute bundle ID is absent from the reuse manifest")
    requested_root = Path(retrieval_root)
    if requested_root.is_symlink():
        raise ValueError("single-bundle retrieval root must not be a symlink")
    root = _assert_non_git_root(requested_root, label="single-bundle retrieval root")
    if not root.is_dir():
        raise FileNotFoundError(f"single-bundle retrieval root is absent: {root}")
    selected = {bundle_id: all_bundles[bundle_id]}
    _verify_exact_retrieval_tree(root, selected)
    bundle_spec = selected[bundle_id]
    access = SelectiveBundleAccess(
        bundle_id=bundle_id,
        bundle_root=(root / bundle_id).resolve(),
        allowed_files={
            str(item["logical_path"]): dict(item)
            for item in bundle_spec["remote_integrity"]["files"]
        },
    )
    access.verify_all()
    artifact = access.read_json("artifact_manifest.json")
    for field, expected in bundle_spec["identity"].items():
        if artifact.get(field) != expected:
            raise ValueError(
                f"retrieved artifact identity mismatch for {bundle_id}:{field}"
            )
    return ReuseContext(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        retrieval_root=root,
        manifest=manifest,
        bundles=selected,
        accesses={bundle_id: access},
    )


@dataclass(frozen=True)
class KNNPrecomputeBundle:
    bundle_id: str
    bundle_spec: dict[str, Any]
    splits: dict[str, dict[str, np.ndarray]]
    query_keys: tuple[str, ...]
    source_allowlist_sha256: str


def load_knn_precompute_bundle(
    context: ReuseContext, bundle_id: str
) -> KNNPrecomputeBundle:
    """Load only features/sample IDs needed by single-bundle kNN precompute."""
    if bundle_id not in context.bundles:
        raise ValueError("precompute bundle ID is absent from the loaded context")
    spec = context.bundles[bundle_id]
    access = context.accesses[bundle_id]
    raw_manifest = access.read_json("manifest.json")
    checkpoint = raw_manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("precompute raw manifest checkpoint identity is absent")
    expected = {
        "sha256": bundle_id,
        "role": spec["identity"]["checkpoint_role"],
        "config_hash": spec["config_hash"],
        "training_seed": int(spec["training_seed"]),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"precompute raw manifest identity mismatch: {field}")
    query_keys = tuple(context.manifest["path_contract"]["query_keys"])
    splits = {}
    for split_name in ("id_train", *query_keys):
        features = _load_raw_array(access, raw_manifest, split_name, "features")
        sample_ids = _load_raw_array(access, raw_manifest, split_name, "sample_ids")
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError(f"precompute feature array is invalid: {split_name}")
        ids = np.asarray(sample_ids, dtype=str).reshape(-1)
        if len(ids) != len(features) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"precompute sample identity is invalid: {split_name}")
        expected_digest = raw_manifest["dataset"]["splits"][split_name].get(
            "ordered_sample_id_sha256"
        )
        if expected_digest is None or ordered_sample_id_digest(ids) != expected_digest:
            raise ValueError(f"precompute sample identity digest mismatch: {split_name}")
        splits[split_name] = {"features": features, "sample_ids": sample_ids}
    return KNNPrecomputeBundle(
        bundle_id=bundle_id,
        bundle_spec=spec,
        splits=splits,
        query_keys=query_keys,
        source_allowlist_sha256=spec["remote_integrity"][
            "allowed_payload_catalog_sha256"
        ],
    )


def _parity_baseline_manifest_records(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    baseline = manifest.get("parity_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("reuse manifest parity baseline is absent")
    if baseline.get("role") != "completed_metric_contract_v1.2_scalar_parity_baseline":
        raise ValueError("reuse manifest parity baseline role mismatch")
    manifest_records: dict[str, Mapping[str, Any]] = {}
    for item in baseline.get("files", []):
        if not isinstance(item, Mapping):
            raise ValueError("parity baseline file record must be a mapping")
        logical = str(item.get("logical_path", ""))
        declared_path = Path(str(item.get("local_path", "")))
        digest = item.get("sha256")
        size = item.get("size")
        if (
            logical not in PARITY_BASELINE_LOGICAL_PATHS
            or logical in manifest_records
            or not declared_path.is_absolute()
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError("parity baseline file identity is invalid")
        manifest_records[logical] = item
    expected = set(PARITY_BASELINE_LOGICAL_PATHS)
    if set(manifest_records) != expected:
        raise ValueError("parity baseline file set mismatch")
    return manifest_records


def _effective_parity_baseline_root(
    manifest: Mapping[str, Any], parity_baseline_root: str | Path | None
) -> Path:
    manifest_records = _parity_baseline_manifest_records(manifest)
    if parity_baseline_root is not None:
        requested = Path(parity_baseline_root)
        if not requested.is_absolute():
            raise ValueError("parity baseline root must be absolute")
        return _absolute_without_symlink_resolution(requested)
    parents = {
        Path(str(item["local_path"])).parent for item in manifest_records.values()
    }
    if len(parents) != 1:
        raise ValueError("frozen parity baseline files must share one parent root")
    return _absolute_without_symlink_resolution(next(iter(parents)))


def _verify_baseline_files(
    manifest: Mapping[str, Any],
    parity_baseline_root: str | Path | None = None,
) -> dict[str, Path]:
    manifest_records = _parity_baseline_manifest_records(manifest)
    expected = set(PARITY_BASELINE_LOGICAL_PATHS)

    if parity_baseline_root is None:
        records = {
            logical: Path(str(item["local_path"]))
            for logical, item in manifest_records.items()
        }
    else:
        root = _effective_parity_baseline_root(manifest, parity_baseline_root)
        _assert_real_directory_chain(root, label="parity baseline root chain")
        observed: set[str] = set()
        with os.scandir(root) as entries:
            for entry in entries:
                observed.add(entry.name)
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ValueError(
                        f"parity baseline root contains a non-regular entry: {entry.name}"
                    )
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing:
            raise FileNotFoundError(f"parity baseline file is absent: {missing[0]}")
        if unexpected:
            raise ValueError(
                f"parity baseline root contains an unexpected file: {unexpected[0]}"
            )
        records = {logical: root / logical for logical in manifest_records}

    for logical, path in records.items():
        if not os.path.lexists(path):
            raise FileNotFoundError(f"parity baseline file is absent: {logical}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"parity baseline file is not regular: {logical}")
        item = manifest_records[logical]
        if path.stat().st_size != int(item["size"]):
            raise ValueError(f"parity baseline size mismatch: {logical}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"parity baseline checksum mismatch: {logical}")
    return records


def load_frozen_baseline(
    context: ReuseContext,
    *,
    parity_baseline_root: str | Path | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
]:
    files = _verify_baseline_files(context.manifest, parity_baseline_root)
    bundle_by_id = context.bundles
    metric_rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    id_rows: dict[str, dict[str, float]] = defaultdict(dict)
    path = files["per_checkpoint_records.jsonl"]
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid baseline JSONL at line {line_number}") from exc
            checkpoint = row.get("checkpoint_sha256")
            if checkpoint not in bundle_by_id:
                continue
            bundle = bundle_by_id[str(checkpoint)]
            identity = bundle["identity"]
            expected_identity = {
                "checkpoint_role": identity["checkpoint_role"],
                "config_hash": bundle["config_hash"],
                "training_seed": int(bundle["training_seed"]),
                "evaluation_git_sha": identity["evaluation_git_sha"],
                "optimizer_family": bundle["optimizer_family"],
                "labels": bundle["labels"],
            }
            for field, expected in expected_identity.items():
                observed = row.get(field)
                if observed != expected:
                    raise ValueError(
                        f"baseline identity mismatch for {checkpoint}:{field}"
                    )
            artifact_key = row.get("artifact_key")
            if artifact_key in ID_METRIC_ARTIFACTS.values():
                name = next(
                    key for key, value in ID_METRIC_ARTIFACTS.items() if value == artifact_key
                )
                if name in id_rows[str(checkpoint)]:
                    raise ValueError("duplicate frozen ID metric baseline row")
                if row.get("status") != "success" or row.get("reason_codes") != []:
                    raise ValueError("frozen ID metric baseline is not successful")
                id_rows[str(checkpoint)][name] = float(row["value"])
                continue
            if artifact_key not in METRIC_ARTIFACTS.values():
                continue
            detector = row.get("detector")
            dataset = row.get("ood_dataset")
            if detector not in PARITY_DETECTORS or dataset not in context.manifest[
                "path_contract"
            ]["query_keys"]:
                continue
            if dataset == "id_test":
                continue
            key = (str(checkpoint), str(detector), str(dataset))
            name = next(
                key_name
                for key_name, value in METRIC_ARTIFACTS.items()
                if value == artifact_key
            )
            if name in metric_rows[key]:
                raise ValueError("duplicate frozen OOD metric baseline row")
            if row.get("status") != "success" or row.get("reason_codes") != []:
                raise ValueError("frozen OOD metric baseline is not successful")
            metric_rows[key][name] = dict(row)

    ood_datasets = tuple(
        value
        for value in context.manifest["path_contract"]["query_keys"]
        if value != "id_test"
    )
    expected_keys = {
        (checkpoint, detector, dataset)
        for checkpoint in bundle_by_id
        for detector in PARITY_DETECTORS
        for dataset in ood_datasets
    }
    if set(metric_rows) != expected_keys:
        missing = sorted(expected_keys.difference(metric_rows))
        extra = sorted(set(metric_rows).difference(expected_keys))
        raise ValueError(
            f"frozen scalar baseline population mismatch: missing={missing[:1]} extra={extra[:1]}"
        )
    if any(set(rows) != set(METRIC_ARTIFACTS) for rows in metric_rows.values()):
        raise ValueError("frozen scalar baseline metric set is incomplete")
    if set(id_rows) != set(bundle_by_id) or any(
        set(rows) != set(ID_METRIC_ARTIFACTS) for rows in id_rows.values()
    ):
        raise ValueError("frozen ID metric baseline population is incomplete")
    source_rows = [
        {
            "logical_path": logical,
            "local_path": str(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for logical, path in sorted(files.items())
    ]
    return dict(metric_rows), dict(id_rows), source_rows


def _metric_values(id_scores: Any, ood_scores: Any) -> dict[str, float]:
    result = compute_ood_metrics(id_scores, ood_scores)
    return {
        "auroc": float(result["auroc"]),
        "fpr95": float(result["fpr95_id_tpr"]),
        "fpr95_threshold": float(result["fpr95_threshold"]),
        "achieved_id_tpr": float(result["fpr95_achieved_id_tpr"]),
    }


def _scalar_parity(
    computed: Mapping[str, float], baseline_rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_values = {name: float(baseline_rows[name]["value"]) for name in METRIC_ARTIFACTS}
    failures = [
        name
        for name in METRIC_ARTIFACTS
        if not np.isclose(
            computed[name],
            baseline_values[name],
            atol=EXACT_ATOL,
            rtol=EXACT_RTOL,
        )
    ]
    if failures:
        raise AssertionError(f"frozen scalar parity failed: {sorted(failures)}")
    return {
        "status": "PASS",
        "reason_codes": [],
        "metrics": dict(computed),
        "baseline_metrics": baseline_values,
        "absolute_tolerance": EXACT_ATOL,
        "relative_tolerance": EXACT_RTOL,
    }


def _split_concatenated(
    values: np.ndarray, slices: Mapping[str, slice]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values[index]) for name, index in slices.items()}


def _concatenate_queries(
    bundle: SelectiveBundle | KNNPrecomputeBundle, field: str
) -> tuple[np.ndarray, dict[str, slice]]:
    arrays = []
    slices = {}
    cursor = 0
    for split_name in bundle.query_keys:
        value = np.asarray(bundle.splits[split_name][field])
        arrays.append(value)
        slices[split_name] = slice(cursor, cursor + len(value))
        cursor += len(value)
    return np.concatenate(arrays, axis=0), slices


def _allowed_source_record(
    bundle: SelectiveBundle, logical_path: str
) -> Mapping[str, Any]:
    access = bundle.access
    allowed = getattr(access, "allowed_files", None)
    if not isinstance(allowed, Mapping) or logical_path not in allowed:
        raise ValueError(f"checksum-bound source is absent from allowlist: {logical_path}")
    record = allowed[logical_path]
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("sha256"), str)
        or len(str(record["sha256"])) != 64
        or not isinstance(record.get("size"), int)
        or int(record["size"]) <= 0
    ):
        raise ValueError(f"malformed allowlisted source record: {logical_path}")
    return record


def _selective_input_source_reference(
    bundle: SelectiveBundle,
    split_name: str,
    array_name: str,
) -> dict[str, Any]:
    split_record = bundle.raw_manifest.get("dataset", {}).get("splits", {}).get(
        split_name
    )
    if not isinstance(split_record, Mapping) or not isinstance(
        split_record.get("relative_directory"), str
    ):
        raise ValueError(f"raw source manifest omits split: {split_name}")
    logical = f"{split_record['relative_directory']}/{array_name}.npy"
    file_record = _allowed_source_record(bundle, logical)
    array = np.asarray(bundle.splits[split_name][array_name])
    source_kinds = {
        "features": "selective_reuse_feature_array",
        "logits": "selective_reuse_logit_array",
        "class_labels": "selective_reuse_class_label_array",
        "sample_ids": "selective_reuse_sample_id_array",
    }
    if array_name not in source_kinds:
        raise ValueError(f"unsupported selective input source: {array_name}")
    return {
        "source_kind": source_kinds[array_name],
        "logical_path": logical,
        "file_sha256": str(file_record["sha256"]),
        "file_size": int(file_record["size"]),
        "array_sha256": _array_sha256(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
    }


def _classifier_source_reference(
    bundle: SelectiveBundle, array_name: str
) -> dict[str, Any] | None:
    if array_name == "weight":
        logical = "classifier_weight.npy"
        array = np.asarray(bundle.classifier_weight)
    elif array_name == "bias":
        if bundle.classifier_bias is None:
            return None
        logical = "classifier_bias.npy"
        array = np.asarray(bundle.classifier_bias)
    else:
        raise ValueError(f"unsupported classifier source: {array_name}")
    file_record = _allowed_source_record(bundle, logical)
    return {
        "source_kind": f"selective_reuse_classifier_{array_name}_array",
        "logical_path": logical,
        "file_sha256": str(file_record["sha256"]),
        "file_size": int(file_record["size"]),
        "analysis_array_sha256": _array_sha256(array),
        "analysis_shape": list(array.shape),
        "analysis_dtype": str(array.dtype),
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
    }


def _nested_result_reference(
    bundle: SelectiveBundle, split_name: str, path: Sequence[str]
) -> Mapping[str, Any]:
    current: Any = bundle.result.get("scores", {}).get(split_name)
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(
                f"stored diagnostic reference is absent: {split_name}:{'/'.join(path)}"
            )
        current = current[key]
    if not isinstance(current, Mapping) or not isinstance(
        current.get("array_path"), str
    ):
        raise ValueError("stored diagnostic result reference is malformed")
    return current


def _full_array_parity_record(
    stored_values: Any, observed_values: Any, *, label: str
) -> dict[str, Any]:
    stored = _finite_vector(stored_values, label=f"{label} stored array")
    observed = _finite_vector(observed_values, label=f"{label} observed array")
    if stored.shape != observed.shape:
        raise AssertionError(f"{label} full-array parity shape mismatch")
    maximum_absolute, maximum_relative = _max_errors(stored, observed)
    allclose = bool(np.allclose(stored, observed, atol=EXACT_ATOL, rtol=EXACT_RTOL))
    stored_sha = _array_sha256(stored)
    observed_sha = _array_sha256(observed)
    exact_sha = stored_sha == observed_sha
    if not allclose:
        raise AssertionError(f"{label} tolerance-bound full-array parity failed")
    return {
        "status": "PASS",
        "count": int(len(stored)),
        "max_abs_error": maximum_absolute,
        "max_rel_error": maximum_relative,
        "observed_array_sha256": observed_sha,
        "stored_array_sha256": stored_sha,
        "exact_array_sha256_equal": exact_sha,
        "absolute_tolerance": EXACT_ATOL,
        "relative_tolerance": EXACT_RTOL,
    }


def _stored_diagnostic_source_reference(
    bundle: SelectiveBundle,
    split_name: str,
    reference_path: Sequence[str],
    stored_values: Any,
    observed_values: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = _nested_result_reference(bundle, split_name, reference_path)
    logical = f"metrics/{reference['array_path']}"
    file_record = _allowed_source_record(bundle, logical)
    if str(reference.get("sha256")) != str(file_record["sha256"]):
        raise ValueError("stored diagnostic reference/allowlist checksum mismatch")
    stored = _finite_vector(stored_values, label="stored diagnostic array")
    observed = _finite_vector(observed_values, label="observed diagnostic array")
    parity = _full_array_parity_record(
        stored, observed, label=f"{split_name}:{'/'.join(reference_path)}"
    )
    source = {
        "source_kind": "selective_reuse_diagnostic_array",
        "logical_path": logical,
        "file_sha256": str(file_record["sha256"]),
        "file_size": int(file_record["size"]),
        "array_sha256": _array_sha256(observed),
        "stored_array_sha256": _array_sha256(stored),
        "shape": list(observed.shape),
        "dtype": str(observed.dtype),
        "source_declared_shape": list(reference.get("shape", [])),
        "source_declared_dtype": str(reference.get("dtype")),
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
        "parity_status": "PASS",
    }
    return source, parity


def _stored_score_source_reference(
    bundle: SelectiveBundle,
    split_name: str,
    detector: str,
    observed_scores: Any,
) -> dict[str, Any]:
    reference = (
        bundle.result.get("scores", {}).get(split_name, {}).get(detector)
    )
    if not isinstance(reference, Mapping) or not isinstance(
        reference.get("array_path"), str
    ):
        raise ValueError(f"stored score reference is absent: {split_name}:{detector}")
    logical = f"metrics/{reference['array_path']}"
    file_record = _allowed_source_record(bundle, logical)
    if str(reference.get("sha256")) != str(file_record["sha256"]):
        raise ValueError("stored score reference/allowlist checksum mismatch")
    stored = np.asarray(bundle.scores[split_name][detector], dtype=np.float64)
    observed = _finite_vector(observed_scores, label="observed detector scores")
    if stored.shape != observed.shape or not np.allclose(
        stored, observed, atol=EXACT_ATOL, rtol=EXACT_RTOL
    ):
        raise AssertionError("stored score/source-reference parity failed")
    return {
        "source_kind": "selective_reuse_score_array",
        "logical_path": logical,
        "file_sha256": str(file_record["sha256"]),
        "file_size": int(file_record["size"]),
        "array_sha256": _array_sha256(observed),
        "stored_array_sha256": _array_sha256(stored),
        "shape": list(observed.shape),
        "dtype": str(observed.dtype),
        "source_declared_shape": list(reference.get("shape", [])),
        "source_declared_dtype": str(reference.get("dtype")),
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
        "parity_status": "PASS",
    }


def _cache_score_source_reference(
    *,
    cache_root: Path,
    analysis_id: str,
    bundle: SelectiveBundle,
    variant_id: str,
    split_name: str,
    observed_scores: Any,
) -> dict[str, Any]:
    cache_path = cache_root / analysis_id / bundle.bundle_id / variant_id
    checksums = _read_cache_checksum_inventory(cache_path)
    logical = f"splits/{split_name}/score.npy"
    if logical not in checksums or "manifest.json" not in checksums:
        raise ValueError("kNN cache source-reference inventory is incomplete")
    score_path = cache_path / logical
    observed = _finite_vector(observed_scores, label="cached detector scores")
    return {
        "source_kind": "stage2_knn_cache_score_array",
        "logical_path": logical,
        "file_sha256": checksums[logical],
        "file_size": int(score_path.stat().st_size),
        "array_sha256": _array_sha256(observed),
        "shape": list(observed.shape),
        "dtype": str(observed.dtype),
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
        "analysis_id": analysis_id,
        "variant_id": variant_id,
        "cache_schema_version": KNN_CACHE_SCHEMA,
        "cache_manifest_sha256": checksums["manifest.json"],
    }


def _normalized(values: Any, *, epsilon: float, label: str) -> np.ndarray:
    normalized, status, _, _ = normalize_rows(
        values, denominator_epsilon=epsilon, name=label
    )
    if status == "failed":
        raise ValueError(f"{label} contains an exact-zero norm")
    return normalized


def _samplewise_scales(sample_ids: Any) -> np.ndarray:
    identifiers = np.asarray(sample_ids, dtype=str).reshape(-1)
    denominator = float((1 << 64) - 1)
    return np.asarray(
        [
            0.5
            + 1.5
            * int(hashlib.sha256(item.encode("utf-8")).hexdigest()[:16], 16)
            / denominator
            for item in identifiers
        ],
        dtype=np.float64,
    )


def _signed_coordinate_permutation(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    signs = np.where(np.arange(array.shape[1]) % 2 == 0, 1.0, -1.0)
    return array[:, ::-1] * signs[None, :]


def _score_gaussian_all(
    fit: TiedGaussianFit, queries: np.ndarray, slices: Mapping[str, slice]
) -> dict[str, np.ndarray]:
    return _split_concatenated(
        np.asarray(score_tied_gaussians(fit, queries)["mahalanobis"], dtype=np.float64),
        slices,
    )


def _score_knn_all(
    bank: np.ndarray,
    queries: np.ndarray,
    *,
    sample_ids: np.ndarray,
    slices: Mapping[str, slice],
    query_chunk_size: int,
    bank_chunk_size: int | None,
    memory_budget_bytes: int | None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
]:
    result = exact_raw_knn_topk(
        bank,
        queries,
        fit_sample_ids=sample_ids,
        k=KNN_K,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        return_neighbor_sample_ids=False,
    )
    scores = _split_concatenated(
        np.asarray(result["score"], dtype=np.float64), slices
    )
    metadata = {
        "backend": result["backend"],
        "k": int(result["k"]),
        "query_chunk_size": int(result["query_chunk_size"]),
        "bank_chunk_size": int(result["bank_chunk_size"]),
    }
    component_arrays = {
        split_name: {
            "neighbor_squared_distances": np.asarray(
                result["neighbor_squared_distances"][split_slice], dtype=np.float64
            ),
            "neighbor_bank_indices": np.asarray(
                result["neighbor_bank_indices"][split_slice]
            ),
        }
        for split_name, split_slice in slices.items()
    }
    del result
    return scores, metadata, component_arrays


def _neighbor_overlap_summary(
    raw_neighbor_indices: Any,
    normalized_neighbor_indices: Any,
    *,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    raw = np.asarray(raw_neighbor_indices)
    normalized = np.asarray(normalized_neighbor_indices)
    if raw.ndim != 2 or normalized.shape != raw.shape or raw.shape[1] != KNN_K:
        raise ValueError("raw/L2 top-50 neighbor identity shape mismatch")
    counts = np.empty(len(raw), dtype=np.int64)
    for start in range(0, len(raw), chunk_size):
        stop = min(start + chunk_size, len(raw))
        # Each row contains unique bank IDs.  Chunked broadcast gives the exact
        # set-intersection size without allocating a population-scale Q x K x K
        # tensor.
        counts[start:stop] = np.sum(
            np.any(
                raw[start:stop, :, None]
                == normalized[start:stop, None, :],
                axis=2,
            ),
            axis=1,
        )
    if np.any(counts < 0) or np.any(counts > KNN_K):
        raise AssertionError("raw/L2 neighbor overlap is outside [0,K]")
    return {
        "sample_count": int(len(counts)),
        "k": KNN_K,
        "total_shared_neighbor_count": int(np.sum(counts)),
        "mean_overlap_fraction": float(np.mean(counts / KNN_K)),
        "minimum_overlap_fraction": float(np.min(counts) / KNN_K),
        "maximum_overlap_fraction": float(np.max(counts) / KNN_K),
        "per_sample_shared_count_sha256": _array_sha256(counts),
        "identity_encoding": "exact_bank_indices_bound_to_train_sample_id_digest",
        "full_query_array_evaluated": True,
    }


def _scan_regular_cache_tree(root: Path) -> tuple[set[str], set[str]]:
    mode = root.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("cache root must be a real directory, not a symlink/special entry")
    files: set[str] = set()
    directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directory_names, *file_names]:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if name.endswith(".tmp"):
                raise ValueError(f"temporary cache entry is forbidden: {relative}")
            child_mode = candidate.lstat().st_mode
            if stat.S_ISLNK(child_mode):
                raise ValueError(f"symlink is forbidden in cache tree: {relative}")
            if name in directory_names:
                if not stat.S_ISDIR(child_mode):
                    raise ValueError(f"non-directory cache traversal entry: {relative}")
                directories.add(relative)
            else:
                if not stat.S_ISREG(child_mode):
                    raise ValueError(f"non-regular cache file: {relative}")
                files.add(relative)
    return files, directories


def _read_cache_checksum_inventory(root: Path) -> dict[str, str]:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ValueError("kNN variant cache checksum inventory is absent")
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("malformed kNN variant cache checksum row") from exc
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative == "checksums.sha256"
            or relative in rows
        ):
            raise ValueError("invalid kNN variant cache checksum identity")
        rows[relative] = digest
    actual_files, actual_directories = _scan_regular_cache_tree(root)
    if actual_files != set(rows) | {"checksums.sha256"}:
        raise ValueError("kNN variant cache contains missing or unexpected files")
    expected_directories = {
        parent.as_posix()
        for relative in rows
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_directories != expected_directories:
        raise ValueError("kNN variant cache contains missing or unexpected directories")
    return rows


def _cache_checksum_rows(root: Path) -> dict[str, str]:
    rows = _read_cache_checksum_inventory(root)
    for relative, digest in rows.items():
        path = root / relative
        if sha256_file(path) != digest:
            raise ValueError(f"kNN variant cache checksum mismatch: {relative}")
    return rows


def _write_cache_checksums(root: Path) -> None:
    files, _ = _scan_regular_cache_tree(root)
    if "checksums.sha256" in files:
        raise ValueError("cache checksum inventory already exists")
    rows = [
        f"{sha256_file(root / relative)}  {relative}"
        for relative in sorted(files)
    ]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _prepare_cache_root(path: Path) -> Path:
    requested = Path(path)
    if os.path.lexists(requested):
        mode = requested.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("Stage-2 kNN cache root must be a real directory")
    root = _assert_non_git_root(requested, label="Stage-2 kNN cache root")
    root.mkdir(parents=True, exist_ok=True)
    files, directories = _scan_regular_cache_tree(root)
    if any(len(Path(relative).parts) < 4 for relative in files):
        raise ValueError("unexpected file above a Stage-2 cache variant root")
    hexadecimal = re.compile(r"[0-9a-f]{64}")
    for relative in sorted(directories, key=lambda value: (value.count("/"), value)):
        parts = Path(relative).parts
        if len(parts) in (1, 2):
            if hexadecimal.fullmatch(parts[-1]) is None:
                raise ValueError(f"unexpected cache identity directory: {relative}")
        elif len(parts) == 3:
            if parts[-1] not in KNN_VARIANT_IDS:
                raise ValueError(f"unexpected kNN cache variant directory: {relative}")
            _read_cache_checksum_inventory(root / relative)
        elif len(parts) > 3:
            # Descendants are validated against the checksum-bound exact tree
            # when their variant root (depth three) is visited.
            continue
    return root


def _cache_staging_root_path(cache_root: str | Path) -> Path:
    root = _absolute_without_symlink_resolution(cache_root)
    return root.parent / f".{root.name}{CACHE_STAGING_DIRECTORY_SUFFIX}"


def _prepare_cache_staging_root(cache_root: Path) -> Path:
    root = _absolute_without_symlink_resolution(cache_root)
    staging = _cache_staging_root_path(root)
    if not os.path.lexists(staging):
        try:
            staging.mkdir(mode=0o700)
        except FileExistsError:
            # Another cache worker may have created the shared staging root.
            pass
    mode = staging.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("Stage-2 cache staging root must be a real directory")
    _assert_real_directory_chain(staging, label="Stage-2 cache staging root chain")
    if root.stat().st_dev != staging.stat().st_dev:
        raise ValueError("Stage-2 cache staging root must share the cache filesystem")
    return staging


def _knn_cache_identity(
    *,
    analysis_id: str,
    bundle: SelectiveBundle | KNNPrecomputeBundle,
    variant_id: str,
    include_components: bool,
    train_sample_ids: np.ndarray,
    query_slices: Mapping[str, slice],
    query_chunk_size: int,
    bank_chunk_size: int | None,
    memory_budget_bytes: int | None,
    numeric_runtime_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = _validate_runtime_fingerprint(numeric_runtime_fingerprint)
    return {
        "schema_version": KNN_CACHE_SCHEMA,
        "analysis_id": analysis_id,
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
        "variant_id": variant_id,
        "include_top50_components": bool(include_components),
        "backend": RAW_KNN_BACKEND_ID,
        "k": KNN_K,
        "query_keys": list(query_slices),
        "query_counts": {
            split: int(index.stop - index.start)
            for split, index in query_slices.items()
        },
        "train_sample_id_sha256": _array_sha256(train_sample_ids),
        "query_sample_id_sha256": {
            split: _array_sha256(bundle.splits[split]["sample_ids"])
            for split in query_slices
        },
        "backend_file_sha256": sha256_file(
            SCRIPT_DIRECTORY / "fixed_readout_stage2_raw_knn_backend.py"
        ),
        "requested_query_chunk_size": query_chunk_size,
        "requested_bank_chunk_size": bank_chunk_size,
        "requested_memory_budget_bytes": memory_budget_bytes,
        "numeric_runtime_fingerprint": runtime,
        "numeric_runtime_fingerprint_sha256": runtime[
            "canonical_lock_sha256"
        ],
        "neighbor_cache_encoding": (
            "int32_bank_indices_losslessly_mapped_through_bound_train_sample_ids"
            if len(train_sample_ids) <= np.iinfo(np.int32).max
            else "int64_bank_indices_losslessly_mapped_through_bound_train_sample_ids"
        ),
    }


def _load_knn_variant_cache(
    cache_path: Path,
    *,
    expected_identity: Mapping[str, Any],
    train_sample_ids: np.ndarray,
    query_slices: Mapping[str, slice],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
]:
    _cache_checksum_rows(cache_path)
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"kNN variant cache identity mismatch: {key}")
    backend_metadata = manifest.get("backend_metadata")
    if not isinstance(backend_metadata, dict) or backend_metadata.get(
        "backend"
    ) != RAW_KNN_BACKEND_ID:
        raise ValueError("kNN variant cache backend metadata mismatch")
    if backend_metadata.get("k") != KNN_K or backend_metadata.get(
        "query_chunk_size"
    ) != expected_identity["requested_query_chunk_size"]:
        raise ValueError("kNN variant cache chunk/K metadata mismatch")
    expected_bank = resolve_bank_chunk_size(
        bank_count=len(train_sample_ids),
        k=KNN_K,
        query_chunk_size=int(expected_identity["requested_query_chunk_size"]),
        bank_chunk_size=expected_identity["requested_bank_chunk_size"],
        memory_budget_bytes=expected_identity["requested_memory_budget_bytes"],
    )
    if backend_metadata.get("bank_chunk_size") != expected_bank:
        raise ValueError("kNN variant cache resolved bank chunk mismatch")
    scores: dict[str, np.ndarray] = {}
    components: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_slice in query_slices.items():
        count = int(split_slice.stop - split_slice.start)
        split_root = cache_path / "splits" / split_name
        score = np.load(split_root / "score.npy", allow_pickle=False)
        if (
            score.shape != (count,)
            or score.dtype != np.float64
            or not np.isfinite(score).all()
        ):
            raise ValueError("kNN variant cache score schema mismatch")
        scores[split_name] = score
        if expected_identity["include_top50_components"]:
            distances = np.load(
                split_root / "neighbor_squared_distances.npy", allow_pickle=False
            )
            indices = np.load(split_root / "neighbor_bank_indices.npy", allow_pickle=False)
            if distances.shape != (count, KNN_K) or distances.dtype != np.float64:
                raise ValueError("kNN variant cache distance component schema mismatch")
            if indices.shape != distances.shape or indices.dtype.kind not in "iu":
                raise ValueError("kNN variant cache bank-index schema mismatch")
            if np.any(indices < 0) or np.any(indices >= len(train_sample_ids)):
                raise ValueError("kNN variant cache bank index is out of range")
            if not np.isfinite(distances).all() or np.any(
                np.diff(distances, axis=1) < 0.0
            ):
                raise ValueError("kNN variant cache distances are invalid or unsorted")
            if any(len(set(row.tolist())) != KNN_K for row in indices):
                raise ValueError("kNN variant cache repeats a bank index within a row")
            if not np.array_equal(score, -distances[:, -1]):
                raise ValueError("kNN variant cache score/Kth-distance identity failed")
            components[split_name] = {
                "neighbor_squared_distances": distances,
                "neighbor_bank_indices": indices,
            }
    return scores, dict(backend_metadata), components


def _write_knn_variant_cache(
    cache_path: Path,
    *,
    cache_root: Path,
    identity: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    backend_metadata: Mapping[str, Any],
    components: Mapping[str, Mapping[str, np.ndarray]],
    train_sample_ids: np.ndarray,
) -> None:
    verified_cache_root = _prepare_cache_root(cache_root)
    try:
        relative = cache_path.relative_to(verified_cache_root)
    except ValueError as exc:
        raise ValueError("kNN variant destination is outside the cache root") from exc
    if len(relative.parts) != 3:
        raise ValueError("kNN variant destination depth is invalid")
    if os.path.lexists(cache_path):
        raise FileExistsError("kNN variant cache is no-overwrite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_cache_root(verified_cache_root)
    staging_root = _prepare_cache_staging_root(verified_cache_root)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"{identity['analysis_id']}.{identity['bundle_id']}."
            f"{identity['variant_id']}.",
            suffix=f".{uuid.uuid4().hex}.tmp",
            dir=staging_root,
        )
    )
    try:
        manifest = dict(identity)
        manifest["backend_metadata"] = dict(backend_metadata)
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
        index_dtype = np.int32 if len(train_sample_ids) <= np.iinfo(np.int32).max else np.int64
        for split_name, score in scores.items():
            split_root = temporary / "splits" / split_name
            split_root.mkdir(parents=True)
            np.save(split_root / "score.npy", np.asarray(score, dtype=np.float64), allow_pickle=False)
            if identity["include_top50_components"]:
                component = components[split_name]
                distances = np.asarray(
                    component["neighbor_squared_distances"], dtype=np.float64
                )
                indices = np.asarray(
                    component["neighbor_bank_indices"], dtype=index_dtype
                )
                np.save(
                    split_root / "neighbor_squared_distances.npy",
                    distances,
                    allow_pickle=False,
                )
                np.save(
                    split_root / "neighbor_bank_indices.npy", indices, allow_pickle=False
                )
        _write_cache_checksums(temporary)
        _cache_checksum_rows(temporary)
        _prepare_cache_root(verified_cache_root)
        _assert_real_directory_chain(
            cache_path.parent, label="kNN variant cache destination parent chain"
        )
        if temporary.stat().st_dev != cache_path.parent.stat().st_dev:
            raise ValueError("kNN cache staging/destination filesystems differ")
        if os.path.lexists(cache_path):
            raise FileExistsError("kNN variant cache is no-overwrite")
        os.rename(temporary, cache_path)
        _cache_checksum_rows(cache_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _score_knn_all_cached(
    bank: np.ndarray,
    queries: np.ndarray,
    *,
    sample_ids: np.ndarray,
    slices: Mapping[str, slice],
    query_chunk_size: int,
    bank_chunk_size: int | None,
    memory_budget_bytes: int | None,
    cache_root: Path | None,
    analysis_id: str,
    bundle: SelectiveBundle | KNNPrecomputeBundle,
    variant_id: str,
    include_components: bool,
    numeric_runtime_fingerprint: Mapping[str, Any] | None = None,
    require_existing: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
]:
    runtime = _validate_runtime_fingerprint(
        numeric_runtime_fingerprint or _numeric_runtime_fingerprint()
    )
    identity = _knn_cache_identity(
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id=variant_id,
        include_components=include_components,
        train_sample_ids=sample_ids,
        query_slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        numeric_runtime_fingerprint=runtime,
    )
    cache_path: Path | None = None
    verified_cache_root: Path | None = None
    if cache_root is not None:
        verified_cache_root = _prepare_cache_root(cache_root)
        cache_path = verified_cache_root / analysis_id / bundle.bundle_id / variant_id
        if os.path.lexists(cache_path):
            if cache_path.is_symlink() or not cache_path.is_dir():
                raise ValueError("kNN variant cache path is a symlink/special entry")
            return _load_knn_variant_cache(
                cache_path,
                expected_identity=identity,
                train_sample_ids=sample_ids,
                query_slices=slices,
            )
        if require_existing:
            raise FileNotFoundError(
                f"required Stage-2 kNN variant cache is absent: {bundle.bundle_id}:{variant_id}"
            )
    elif require_existing:
        raise ValueError("requiring an existing kNN cache needs a cache root")
    scores, metadata, components = _score_knn_all(
        bank,
        queries,
        sample_ids=sample_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
    )
    if cache_path is not None:
        if verified_cache_root is None:
            raise AssertionError("cache destination omitted its verified cache root")
        _write_knn_variant_cache(
            cache_path,
            cache_root=verified_cache_root,
            identity=identity,
            scores=scores,
            backend_metadata=metadata,
            components=components,
            train_sample_ids=sample_ids,
        )
    if not include_components:
        components = {}
    return scores, metadata, components


def _oracle_record(
    oracle_id: str,
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    expected_scale: float = 1.0,
    approximate_auroc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64) * float(expected_scale)
    absolute, relative = _max_errors(expected, observed)
    rank_drift = _rank_signature_disagreement(expected, observed)
    auroc_drift = 0.0
    if approximate_auroc is not None:
        ref_id, ref_ood, got_id, got_ood = approximate_auroc
        auroc_drift = abs(
            _metric_values(ref_id, ref_ood)["auroc"]
            - _metric_values(got_id, got_ood)["auroc"]
        )
        passed = auroc_drift <= KNN_APPROXIMATE_AUROC_TOLERANCE
    else:
        passed = bool(
            np.allclose(expected, observed, atol=EXACT_ATOL, rtol=EXACT_RTOL)
            and rank_drift == 0.0
        )
    if not passed:
        raise AssertionError(f"full-scope invariant oracle failed: {oracle_id}")
    return {
        "oracle_id": oracle_id,
        "status": "PASS",
        "reason_codes": [],
        "score_max_abs_error": absolute,
        "score_max_rel_error": relative,
        "rank_disagreement_fraction": rank_drift,
        "auroc_abs_drift": float(auroc_drift),
    }


def _targeted_witness_accepts(
    candidate_id: str,
    *,
    score_allclose: bool,
    rank_disagreement_fraction: float,
    auroc_abs_drift: float,
) -> bool:
    if candidate_id == "radial_l2_mahalanobis":
        return bool(
            score_allclose
            and rank_disagreement_fraction == 0.0
            and auroc_abs_drift <= EXACT_ATOL
        )
    if candidate_id == "radial_l2_knn":
        return bool(auroc_abs_drift <= KNN_APPROXIMATE_AUROC_TOLERANCE)
    raise ValueError(f"unknown targeted-witness candidate: {candidate_id}")


def _matrix_rank_condition(value: Any) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rank/condition diagnostic requires a square matrix")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        np.max(singular_values) * max(matrix.shape) * np.finfo(np.float64).eps
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(np.max(singular_values) / np.min(singular_values))
        if np.min(singular_values) > 0.0
        else None
    )
    return {
        "numerical_rank": rank,
        "condition_number": condition,
        "rank_tolerance": float(tolerance),
    }


def _mahalanobis_component_evidence(
    fit: TiedGaussianFit,
    queries: Any,
    *,
    expected_class_distances: Any | None = None,
    expected_nearest_class: Any | None = None,
    query_chunk_size: int = 2048,
) -> dict[str, Any]:
    values = np.asarray(queries, dtype=np.float64)
    scored = score_tied_gaussians(fit, values, query_chunk_size=query_chunk_size)
    class_distances = np.asarray(scored["class_distances"], dtype=np.float64)
    nearest = np.asarray(scored["nearest_class"], dtype=np.int64)
    score = np.asarray(scored["mahalanobis"], dtype=np.float64)
    geometry_values = (
        _normalized(
            values,
            epsilon=0.0,
            label="Mahalanobis normalized component queries",
        )
        if fit.normalized
        else values
    )
    precision = (fit.within_precision + fit.within_precision.T) / 2.0
    precision_eigenvalues, precision_eigenvectors = np.linalg.eigh(precision)
    covariance_eigenvalues = np.linalg.eigvalsh(
        (fit.within_covariance + fit.within_covariance.T) / 2.0
    )
    contribution_digest = hashlib.sha256()
    contribution_digest.update(b"float64\0")
    contribution_digest.update(
        json.dumps(
            [len(values), values.shape[1]], separators=(",", ":")
        ).encode("ascii")
    )
    contribution_digest.update(b"\0")
    bands = np.array_split(np.arange(values.shape[1]), 3)
    band_sums = np.zeros(3, dtype=np.float64)
    band_values = np.empty((len(values), 3), dtype=np.float64)
    reconstruction_max_abs_error = 0.0
    for start in range(0, len(values), query_chunk_size):
        stop = min(start + query_chunk_size, len(values))
        delta = (
            geometry_values[start:stop]
            - fit.class_means[nearest[start:stop]]
        )
        coordinates = delta @ precision_eigenvectors
        contributions = (
            coordinates * coordinates * precision_eigenvalues[None, :]
        )
        contribution_digest.update(
            np.ascontiguousarray(contributions, dtype=np.float64).tobytes(order="C")
        )
        selected_distance = class_distances[
            np.arange(start, stop), nearest[start:stop]
        ]
        reconstructed = np.sum(contributions, axis=1)
        reconstruction_max_abs_error = max(
            reconstruction_max_abs_error,
            float(np.max(np.abs(reconstructed - selected_distance))),
        )
        for index, band in enumerate(bands):
            current = np.sum(contributions[:, band], axis=1)
            band_values[start:stop, index] = current
            band_sums[index] += float(np.sum(current))
    if reconstruction_max_abs_error > EXACT_ATOL + EXACT_RTOL * float(
        np.max(np.abs(class_distances))
    ):
        raise AssertionError("Mahalanobis eigendirection reconstruction failed")
    selected = class_distances[np.arange(len(values)), nearest]
    second = np.partition(class_distances, 1, axis=1)[:, 1]
    nearest_margin = second - selected
    if np.any(nearest_margin < -EXACT_ATOL):
        raise AssertionError("Mahalanobis nearest/second distance margin is negative")
    if not np.allclose(score, -selected, atol=EXACT_ATOL, rtol=EXACT_RTOL):
        raise AssertionError("Mahalanobis score/class-distance identity failed")
    stored_component_max_abs_error = 0.0
    if expected_class_distances is not None:
        expected_distances = np.asarray(expected_class_distances, dtype=np.float64)
        if expected_distances.shape != class_distances.shape or not np.allclose(
            expected_distances,
            class_distances,
            atol=EXACT_ATOL,
            rtol=EXACT_RTOL,
        ):
            raise AssertionError("stored Mahalanobis class-distance parity failed")
        stored_component_max_abs_error = float(
            np.max(np.abs(expected_distances - class_distances))
        )
        if expected_nearest_class is None:
            expected_nearest_class = np.argmin(expected_distances, axis=1)
    if expected_nearest_class is not None and not np.array_equal(
        np.asarray(expected_nearest_class, dtype=np.int64), nearest
    ):
        raise AssertionError("stored Mahalanobis nearest-class parity failed")
    rank_condition = _matrix_rank_condition(fit.within_covariance)
    return {
        "class_distances": {
            "shape": list(class_distances.shape),
            "sha256": _array_sha256(class_distances),
        },
        "nearest_class": {
            "shape": list(nearest.shape),
            "sha256": _array_sha256(nearest),
        },
        "class_means": {
            "shape": list(fit.class_means.shape),
            "sha256": _array_sha256(fit.class_means),
        },
        "within_covariance": {
            "shape": list(fit.within_covariance.shape),
            "sha256": _array_sha256(fit.within_covariance),
        },
        "within_precision": {
            "shape": list(fit.within_precision.shape),
            "sha256": _array_sha256(fit.within_precision),
        },
        "covariance_eigenvalues": {
            "shape": list(covariance_eigenvalues.shape),
            "sha256": _array_sha256(covariance_eigenvalues),
            "minimum": float(np.min(covariance_eigenvalues)),
            "maximum": float(np.max(covariance_eigenvalues)),
        },
        "precision_eigenvalues": {
            "shape": list(precision_eigenvalues.shape),
            "sha256": _array_sha256(precision_eigenvalues),
            "minimum": float(np.min(precision_eigenvalues)),
            "maximum": float(np.max(precision_eigenvalues)),
        },
        "covariance_eigendirection_contributions": {
            "shape": [len(values), values.shape[1]],
            "sha256": contribution_digest.hexdigest(),
            "sha256_encoding": "dtype_shape_prefix_then_row_major_float64_chunks",
            "max_abs_distance_reconstruction_error": reconstruction_max_abs_error,
            "precision_eigenvalue_tertile_mean_contributions": {
                name: float(total / len(values))
                for name, total in zip(
                    ("low", "middle", "high"), band_sums.tolist()
                )
            },
            "precision_eigenvalue_tertile_distribution": {
                name: _distribution_summary(
                    band_values[:, index],
                    label=f"Mahalanobis {name} precision-band contributions",
                )
                for index, name in enumerate(("low", "middle", "high"))
            },
        },
        "selected_class_distance_distribution": _distribution_summary(
            selected, label="Mahalanobis selected class distance"
        ),
        "nearest_vs_second_distance_margin_distribution": _distribution_summary(
            nearest_margin, label="Mahalanobis nearest/second distance margin"
        ),
        "nearest_class_counts": {
            str(class_index): int(count)
            for class_index, count in enumerate(
                np.bincount(nearest, minlength=class_distances.shape[1]).tolist()
            )
        },
        "numerical_rank": rank_condition["numerical_rank"],
        "condition_number": rank_condition["condition_number"],
        "stored_component_parity_pass": True,
        "stored_component_max_abs_error": stored_component_max_abs_error,
        "full_query_array_evaluated": True,
    }


def _validate_registry_and_gate(
    context: ReuseContext,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], dict[str, str]]:
    bindings = context.manifest["analysis_policy"]
    registry_path = REPOSITORY_ROOT / bindings["candidate_registry"]["path"]
    gate_path = REPOSITORY_ROOT / bindings["gate_policy"]["path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "1.0" or gate.get("schema_version") != "1.0":
        raise ValueError("Stage-2 policy schema mismatch")
    eligible = {
        row["candidate_id"]: row
        for row in registry.get("candidates", [])
        if row.get("selection_status") == "eligible"
    }
    if set(eligible) != set(CANDIDATES):
        raise ValueError("eligible candidate population differs from worker contract")
    for candidate_id, layout in CANDIDATES.items():
        row = eligible[candidate_id]
        observed = {
            "raw_detector": row["raw_detector"]["artifact_key"],
            "transformed_detector": row["transformed_detector"]["artifact_key"],
            "transform_id": row["targeted_transform"]["transform_id"],
            "witness_id": row["non_invariant_witnesses"][0]["witness_id"],
        }
        if observed != layout:
            raise ValueError(f"candidate registry drift: {candidate_id}")
    ood_datasets = tuple(gate["ood_partition"]["dataset_order"])
    if len(ood_datasets) != STAGE2_OOD_DATASET_COUNT:
        raise ValueError("Stage-2 gate must contain exactly six OOD datasets")
    if list(ood_datasets) != [
        value
        for value in context.manifest["path_contract"]["query_keys"]
        if value != "id_test"
    ]:
        raise ValueError("gate/reuse OOD dataset order mismatch")
    groups = {
        dataset: "near" if dataset in gate["ood_partition"]["near"] else "far"
        for dataset in ood_datasets
    }
    required_fields = set(gate["required_output_artifacts"]["model_metrics.jsonl"])
    if not required_fields:
        raise ValueError("gate model-metric schema is absent")
    return registry, gate, ood_datasets, groups


def _formula_record(reference: np.ndarray, recomputed: np.ndarray) -> dict[str, Any]:
    absolute, relative = _max_errors(reference, recomputed)
    passed = bool(
        np.allclose(reference, recomputed, atol=EXACT_ATOL, rtol=EXACT_RTOL)
    )
    if not passed:
        raise AssertionError("full-scope detector formula reconstruction failed")
    return {
        "pass": True,
        "max_abs_error": absolute,
        "max_rel_error": relative,
    }


def precompute_bundle_knn_variants(
    bundle: SelectiveBundle | KNNPrecomputeBundle,
    *,
    analysis_id: str,
    cache_root: Path,
    query_chunk_size: int,
    bank_chunk_size: int | None,
    memory_budget_bytes: int | None,
    numeric_runtime_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute only the seven frozen kNN cache variants for one bundle."""
    runtime = _validate_runtime_fingerprint(
        numeric_runtime_fingerprint or _numeric_runtime_fingerprint()
    )
    cache_root = _prepare_cache_root(cache_root)
    train = bundle.splits["id_train"]
    train_features = np.asarray(train["features"], dtype=np.float64)
    train_ids = np.asarray(train["sample_ids"], dtype=str)
    query_features, slices = _concatenate_queries(bundle, "features")
    query_ids, id_slices = _concatenate_queries(bundle, "sample_ids")
    if slices != id_slices:
        raise ValueError("precompute query feature/sample-ID boundaries differ")
    scale = 2.0

    def variants() -> Iterable[tuple[str, np.ndarray, np.ndarray, bool]]:
        # Each branch is constructed only after the previous cache has been
        # atomically written and released.  This keeps one transformed bank and
        # query population resident rather than seven WRN-sized copies.
        yield "raw_identity", train_features, query_features, True
        yield (
            "l2_targeted",
            _normalized(
                train_features, epsilon=1.0e-10, label="precompute kNN-L2 bank"
            ),
            _normalized(
                query_features,
                epsilon=1.0e-10,
                label="precompute kNN-L2 queries",
            ),
            True,
        )
        yield (
            "signed_coordinate_permutation",
            _signed_coordinate_permutation(train_features),
            _signed_coordinate_permutation(query_features),
            False,
        )
        yield (
            "raw_common_positive_scale_2",
            train_features * scale,
            query_features * scale,
            False,
        )
        yield (
            "l2_common_positive_scale_2",
            _normalized(
                train_features * scale,
                epsilon=1.0e-10,
                label="precompute scaled kNN-L2 bank",
            ),
            _normalized(
                query_features * scale,
                epsilon=1.0e-10,
                label="precompute scaled kNN-L2 queries",
            ),
            False,
        )
        train_witness = train_features * _samplewise_scales(train_ids)[:, None]
        query_witness = query_features * _samplewise_scales(query_ids)[:, None]
        yield "raw_nonuniform_samplewise_scale", train_witness, query_witness, False
        yield (
            "l2_after_nonuniform_samplewise_scale",
            _normalized(
                train_witness,
                epsilon=1.0e-10,
                label="precompute witness kNN-L2 bank",
            ),
            _normalized(
                query_witness,
                epsilon=1.0e-10,
                label="precompute witness kNN-L2 queries",
            ),
            False,
        )

    records = []
    for variant_id, bank, queries, include_components in variants():
        scores, metadata, components = _score_knn_all_cached(
            bank,
            queries,
            sample_ids=train_ids,
            slices=slices,
            query_chunk_size=query_chunk_size,
            bank_chunk_size=bank_chunk_size,
            memory_budget_bytes=memory_budget_bytes,
            cache_root=cache_root,
            analysis_id=analysis_id,
            bundle=bundle,
            variant_id=variant_id,
            include_components=include_components,
            numeric_runtime_fingerprint=runtime,
            require_existing=False,
        )
        if set(scores) != set(bundle.query_keys):
            raise AssertionError("precomputed kNN cache query population mismatch")
        if include_components and set(components) != set(bundle.query_keys):
            raise AssertionError("precomputed kNN component population mismatch")
        records.append(
            {
                "variant_id": variant_id,
                "include_top50_components": include_components,
                "backend_metadata": metadata,
                "cache_path": str(
                    cache_root / analysis_id / bundle.bundle_id / variant_id
                ),
                "status": "PASS",
            }
        )
        del scores, components, bank, queries
    return {
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
        "variant_count": len(records),
        "variants": records,
        "numeric_runtime_fingerprint": runtime,
        "status": "PASS",
    }


def extract_bundle_rows(
    bundle: SelectiveBundle,
    *,
    baseline_metrics: Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]],
    id_metrics: Mapping[str, Mapping[str, float]],
    ood_datasets: Sequence[str],
    ood_groups: Mapping[str, str],
    analysis_id: str,
    cache_root: Path | None = None,
    require_existing_knn_cache: bool = False,
    query_chunk_size: int = 64,
    bank_chunk_size: int | None = None,
    memory_budget_bytes: int | None = None,
    numeric_runtime_fingerprint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    runtime = _validate_runtime_fingerprint(
        numeric_runtime_fingerprint or _numeric_runtime_fingerprint()
    )
    if cache_root is None:
        raise ValueError("Stage-2 model evidence requires a checksum-bound kNN cache")
    cache_root = _prepare_cache_root(cache_root)
    train = bundle.splits["id_train"]
    train_features = np.asarray(train["features"], dtype=np.float64)
    train_labels = np.asarray(train["class_labels"], dtype=np.int64)
    train_ids = np.asarray(train["sample_ids"], dtype=str)
    if len(train_features) < KNN_K:
        raise ValueError("Stage-2 raw-kNN requires at least 50 ID-train samples")
    query_features, slices = _concatenate_queries(bundle, "features")
    query_logits, logit_slices = _concatenate_queries(bundle, "logits")
    query_ids, id_slices = _concatenate_queries(bundle, "sample_ids")
    if slices != logit_slices or slices != id_slices:
        raise ValueError("query split boundaries differ across arrays")
    base_feature_splits = {
        "id_train": train_features,
        **{
            split_name: np.asarray(
                bundle.splits[split_name]["features"], dtype=np.float64
            )
            for split_name in bundle.query_keys
        },
    }

    class_count = bundle.classifier_weight.shape[0]
    raw_fit = fit_tied_gaussians(
        train_features, train_labels, num_classes=class_count
    )
    pp_fit = fit_tied_gaussians(
        train_features, train_labels, num_classes=class_count, normalize=True
    )
    maha_raw = _score_gaussian_all(raw_fit, query_features, slices)
    maha_pp = _score_gaussian_all(pp_fit, query_features, slices)

    if require_existing_knn_cache:
        # The verified variant cache is bound to analysis/code/source identity;
        # its loader never consumes these placeholder arrays.  Avoid another
        # WRN-sized normalized copy during the full local evidence reduction.
        normalized_bank = train_features
        normalized_queries = query_features
    else:
        normalized_bank = _normalized(
            train_features, epsilon=1.0e-10, label="kNN-L2 fit features"
        )
        normalized_queries = _normalized(
            query_features, epsilon=1.0e-10, label="kNN-L2 query features"
        )
    knn_raw, raw_knn_metadata, raw_knn_components = _score_knn_all_cached(
        train_features,
        query_features,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="raw_identity",
        include_components=True,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    knn_l2, l2_knn_metadata, l2_knn_components = _score_knn_all_cached(
        normalized_bank,
        normalized_queries,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="l2_targeted",
        include_components=True,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    if raw_knn_metadata["backend"] != RAW_KNN_BACKEND_ID or l2_knn_metadata[
        "backend"
    ] != RAW_KNN_BACKEND_ID:
        raise AssertionError("unexpected raw-kNN backend identity")

    knn_overlap: dict[str, dict[str, Any]] = {}
    knn_component_evidence: dict[str, dict[str, dict[str, Any]]] = {
        "detector/knn_raw_k50": {},
        "detector/knn_l2_k50": {},
    }
    for split_name in bundle.query_keys:
        raw_distances = raw_knn_components[split_name][
            "neighbor_squared_distances"
        ]
        raw_indices = np.asarray(
            raw_knn_components[split_name]["neighbor_bank_indices"], dtype=np.int64
        )
        l2_distances = l2_knn_components[split_name][
            "neighbor_squared_distances"
        ]
        l2_indices = np.asarray(
            l2_knn_components[split_name]["neighbor_bank_indices"], dtype=np.int64
        )
        for detector, distances, indices, scores in (
            ("detector/knn_raw_k50", raw_distances, raw_indices, knn_raw),
            ("detector/knn_l2_k50", l2_distances, l2_indices, knn_l2),
        ):
            if distances.shape != (len(bundle.splits[split_name]["features"]), KNN_K):
                raise AssertionError("kNN top-50 distance component shape mismatch")
            if indices.shape != distances.shape:
                raise AssertionError("kNN top-50 identity component shape mismatch")
            if not np.allclose(
                scores[split_name], -distances[:, -1], atol=0.0, rtol=0.0
            ):
                raise AssertionError("kNN score is not the negative Kth distance")
            if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(
                indices >= len(train_ids)
            ):
                raise AssertionError("kNN component contains an invalid bank index")
            if any(len(set(row.tolist())) != KNN_K for row in indices):
                raise AssertionError("kNN component repeats a bank index within a row")
            kth_neighbor_ids = train_ids[indices[:, -1]]
            identity_mapping_sha = _sha256_bytes(
                _canonical_bytes(
                    {
                        "train_sample_id_sha256": _array_sha256(train_ids),
                        "neighbor_bank_indices_sha256": _array_sha256(indices),
                    }
                )
            )
            knn_component_evidence[detector][split_name] = {
                "kth_squared_distance_sha256": _array_sha256(distances[:, -1]),
                "kth_neighbor_sample_id_sha256": _array_sha256(kth_neighbor_ids),
                "top50_squared_distances_sha256": _array_sha256(distances),
                "top50_neighbor_bank_indices_sha256": _array_sha256(indices),
                "top50_neighbor_identity_mapping_sha256": identity_mapping_sha,
                "sample_count": int(len(distances)),
                "k": KNN_K,
                "neighbor_identity_encoding": "exact_train_sample_id_strings",
                "train_sample_id_mapping_is_lossless": True,
                "full_query_array_evaluated": True,
                "first_squared_distance_distribution": _distribution_summary(
                    distances[:, 0], label=f"{detector} first-neighbor distance"
                ),
                "middle_rank": KNN_K // 2,
                "middle_rank_squared_distance_distribution": _distribution_summary(
                    distances[:, KNN_K // 2 - 1],
                    label=f"{detector} middle-rank neighbor distance",
                ),
                "kth_squared_distance_distribution": _distribution_summary(
                    distances[:, -1], label=f"{detector} Kth-neighbor distance"
                ),
                "mean_top50_squared_distance_distribution": _distribution_summary(
                    np.mean(distances, axis=1),
                    label=f"{detector} per-query mean top-50 distance",
                ),
            }
        if not np.allclose(
            l2_distances[:, -1],
            bundle.diagnostics[split_name]["knn_kth_squared_distance"],
            atol=EXACT_ATOL,
            rtol=EXACT_RTOL,
        ) or not np.array_equal(
            train_ids[l2_indices[:, -1]].astype(str),
            bundle.diagnostics[split_name]["knn_kth_neighbor_sample_id"].astype(str),
        ):
            raise AssertionError("kNN-L2 stored Kth-neighbor component parity failed")
        knn_overlap[split_name] = _neighbor_overlap_summary(
            raw_indices, l2_indices
        )

    formulas: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    mahalanobis_component_evidence: dict[
        str, dict[str, dict[str, Any]]
    ] = {
        "detector/mahalanobis_raw": {},
        "detector/mahalanobis_pp": {},
    }
    for split_name in bundle.query_keys:
        formulas["detector/mahalanobis_raw"][split_name] = _formula_record(
            bundle.scores[split_name]["detector/mahalanobis_raw"],
            maha_raw[split_name],
        )
        formulas["detector/mahalanobis_pp"][split_name] = _formula_record(
            bundle.scores[split_name]["detector/mahalanobis_pp"],
            maha_pp[split_name],
        )
        formulas["detector/knn_l2_k50"][split_name] = _formula_record(
            bundle.scores[split_name]["detector/knn_l2_k50"],
            knn_l2[split_name],
        )
        raw_knn_formula = _formula_record(
            knn_raw[split_name],
            -raw_knn_components[split_name]["neighbor_squared_distances"][:, -1],
        )
        raw_knn_formula["basis"] = (
            "independent_score_equals_negative_exact_float64_kth_squared_distance"
        )
        formulas["detector/knn_raw_k50"][split_name] = raw_knn_formula
        mahalanobis_component_evidence["detector/mahalanobis_raw"][
            split_name
        ] = _mahalanobis_component_evidence(
            raw_fit,
            bundle.splits[split_name]["features"],
            expected_class_distances=bundle.diagnostics[split_name][
                "raw_class_distances"
            ],
            expected_nearest_class=bundle.diagnostics[split_name][
                "raw_nearest_class"
            ],
        )
        mahalanobis_component_evidence["detector/mahalanobis_pp"][
            split_name
        ] = _mahalanobis_component_evidence(
            pp_fit,
            bundle.splits[split_name]["features"],
            expected_class_distances=bundle.diagnostics[split_name][
                "normalized_class_distances"
            ],
        )

    # Full-population exact and approximate invariant oracles.
    permuted_bank = _signed_coordinate_permutation(train_features)
    permuted_queries = _signed_coordinate_permutation(query_features)
    permuted_fit = fit_tied_gaussians(
        permuted_bank, train_labels, num_classes=class_count
    )
    maha_permuted = _score_gaussian_all(permuted_fit, permuted_queries, slices)
    knn_permuted, _, _ = _score_knn_all_cached(
        permuted_bank,
        permuted_queries,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="signed_coordinate_permutation",
        include_components=False,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    del permuted_bank, permuted_queries
    scale = 2.0
    scaled_fit = fit_tied_gaussians(
        train_features * scale, train_labels, num_classes=class_count
    )
    maha_scaled = _score_gaussian_all(
        scaled_fit, query_features * scale, slices
    )
    knn_scaled, _, _ = _score_knn_all_cached(
        train_features if require_existing_knn_cache else train_features * scale,
        query_features if require_existing_knn_cache else query_features * scale,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="raw_common_positive_scale_2",
        include_components=False,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    scaled_l2_bank = (
        train_features
        if require_existing_knn_cache
        else _normalized(
            train_features * scale,
            epsilon=1.0e-10,
            label="scaled kNN-L2 fit features",
        )
    )
    scaled_l2_queries = (
        query_features
        if require_existing_knn_cache
        else _normalized(
            query_features * scale,
            epsilon=1.0e-10,
            label="scaled kNN-L2 query features",
        )
    )
    knn_l2_scaled, _, _ = _score_knn_all_cached(
        scaled_l2_bank,
        scaled_l2_queries,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="l2_common_positive_scale_2",
        include_components=False,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    del scaled_l2_bank, scaled_l2_queries

    train_witness = train_features * _samplewise_scales(train_ids)[:, None]
    query_witness = query_features * _samplewise_scales(query_ids)[:, None]
    witness_raw_fit = fit_tied_gaussians(
        train_witness, train_labels, num_classes=class_count
    )
    witness_pp_fit = fit_tied_gaussians(
        train_witness, train_labels, num_classes=class_count, normalize=True
    )
    maha_witness_raw = _score_gaussian_all(
        witness_raw_fit, query_witness, slices
    )
    maha_witness_targeted = _score_gaussian_all(
        witness_pp_fit, query_witness, slices
    )
    knn_witness_raw, _, _ = _score_knn_all_cached(
        train_witness,
        query_witness,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="raw_nonuniform_samplewise_scale",
        include_components=False,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    witness_l2_bank = (
        train_witness
        if require_existing_knn_cache
        else _normalized(
            train_witness, epsilon=1.0e-10, label="witness kNN-L2 fit features"
        )
    )
    witness_l2_queries = (
        query_witness
        if require_existing_knn_cache
        else _normalized(
            query_witness, epsilon=1.0e-10, label="witness kNN-L2 query features"
        )
    )
    knn_witness_targeted, _, _ = _score_knn_all_cached(
        witness_l2_bank,
        witness_l2_queries,
        sample_ids=train_ids,
        slices=slices,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        cache_root=cache_root,
        analysis_id=analysis_id,
        bundle=bundle,
        variant_id="l2_after_nonuniform_samplewise_scale",
        include_components=False,
        numeric_runtime_fingerprint=runtime,
        require_existing=require_existing_knn_cache,
    )
    witness_query_splits = _split_concatenated(query_witness, slices)
    witness_feature_splits = {
        "id_train": train_witness,
        **witness_query_splits,
    }
    candidate_normalization_epsilon = {
        "radial_l2_mahalanobis": 0.0,
        "radial_l2_knn": 1.0e-10,
    }
    if not ood_datasets:
        raise ValueError("Stage-2 extraction requires at least one OOD dataset")
    geometry_channel_profiles = {
        candidate_id: _geometry_channel_record(
            base_feature_splits,
            denominator_epsilon=epsilon,
            active_ood_split=str(ood_datasets[0]),
        )
        for candidate_id, epsilon in candidate_normalization_epsilon.items()
    }
    witness_normalization_preservation = {
        candidate_id: _normalization_status_preservation(
            base_feature_splits,
            witness_feature_splits,
            denominator_epsilon=epsilon,
        )
        for candidate_id, epsilon in candidate_normalization_epsilon.items()
    }
    del (
        witness_l2_bank,
        witness_l2_queries,
        witness_feature_splits,
        witness_query_splits,
        train_witness,
        query_witness,
        base_feature_splits,
    )

    # Independent Energy calls before and after execution of the feature-only
    # transform.  The transformed feature value is deliberately not used by
    # the second score call.
    energy_before_all = np.asarray(
        logit_ood_scores(query_logits)["energy_t1"], dtype=np.float64
    )
    feature_only_path = _normalized(
        query_features,
        epsilon=0.0,
        label="Energy feature-only null transformed features",
    )
    if feature_only_path.shape != query_features.shape:
        raise AssertionError("Energy feature-only transform shape changed")
    energy_after_all = np.asarray(
        logit_ood_scores(query_logits)["energy_t1"], dtype=np.float64
    )
    energy_before = _split_concatenated(energy_before_all, slices)
    energy_after = _split_concatenated(energy_after_all, slices)
    if _array_sha256(energy_before_all) != _array_sha256(energy_after_all):
        raise AssertionError("Energy feature-only null score SHA mismatch")
    energy_formula = {
        split: _formula_record(
            bundle.scores[split]["ood_score/energy_t1"], energy_before[split]
        )
        for split in bundle.query_keys
    }

    detector_scores = {
        "detector/mahalanobis_raw": maha_raw,
        "detector/mahalanobis_pp": maha_pp,
        "detector/knn_raw_k50": knn_raw,
        "detector/knn_l2_k50": knn_l2,
    }
    checkpoint = bundle.bundle_id
    model_id_metrics = dict(id_metrics[checkpoint])
    rows = []
    for dataset in ood_datasets:
        id_split = "id_test"
        energy_raw_metric = _metric_values(
            energy_before[id_split], energy_before[dataset]
        )
        energy_transformed_metric = _metric_values(
            energy_after[id_split], energy_after[dataset]
        )
        energy_parity = _scalar_parity(
            energy_raw_metric,
            baseline_metrics[(checkpoint, "ood_score/energy_t1", dataset)],
        )
        energy_rank_drift = _rank_signature_disagreement(
            np.concatenate((energy_before[id_split], energy_before[dataset])),
            np.concatenate((energy_after[id_split], energy_after[dataset])),
        )
        energy_auroc_drift = abs(
            energy_raw_metric["auroc"] - energy_transformed_metric["auroc"]
        )
        exact_null = {
            "control_id": "energy_t1_feature_only_exact",
            "status": "PASS",
            "reason_codes": [],
            "score_sha256_equal": True,
            "raw_score_sha256": _array_sha256(
                np.concatenate((energy_before[id_split], energy_before[dataset]))
            ),
            "transformed_score_sha256": _array_sha256(
                np.concatenate((energy_after[id_split], energy_after[dataset]))
            ),
            "rank_disagreement_fraction": energy_rank_drift,
            "raw_auroc": energy_raw_metric["auroc"],
            "transformed_auroc": energy_transformed_metric["auroc"],
            "auroc_abs_drift": energy_auroc_drift,
            "formula_reconstruction_pass": bool(
                energy_formula[id_split]["pass"] and energy_formula[dataset]["pass"]
            ),
            "baseline_parity": energy_parity,
        }
        if energy_rank_drift != 0.0 or energy_auroc_drift > EXACT_ATOL:
            raise AssertionError("Energy exact feature-only null oracle failed")

        for candidate_id, layout in CANDIDATES.items():
            raw_key = layout["raw_detector"]
            transformed_key = layout["transformed_detector"]
            raw_id = detector_scores[raw_key][id_split]
            raw_ood = detector_scores[raw_key][dataset]
            transformed_id = detector_scores[transformed_key][id_split]
            transformed_ood = detector_scores[transformed_key][dataset]
            raw_metric = _metric_values(raw_id, raw_ood)
            transformed_metric = _metric_values(transformed_id, transformed_ood)
            transition = exact_pair_order_transitions(
                raw_id, raw_ood, transformed_id, transformed_ood
            )
            if not np.isclose(
                transition["raw_auroc_from_pairs"],
                raw_metric["auroc"],
                atol=EXACT_ATOL,
                rtol=0.0,
            ) or not np.isclose(
                transition["transformed_auroc_from_pairs"],
                transformed_metric["auroc"],
                atol=EXACT_ATOL,
                rtol=0.0,
            ):
                raise AssertionError("pair transition AUROC reconstruction failed")

            if candidate_id == "radial_l2_mahalanobis":
                scale_oracle = _oracle_record(
                    "mahalanobis_common_positive_scale",
                    np.concatenate((maha_raw[id_split], maha_raw[dataset])),
                    np.concatenate((maha_scaled[id_split], maha_scaled[dataset])),
                )
                rank_before = _matrix_rank_condition(raw_fit.within_covariance)
                rank_after = _matrix_rank_condition(scaled_fit.within_covariance)
                scale_oracle.update(
                    {
                        "numerical_rank_before": rank_before["numerical_rank"],
                        "numerical_rank_after": rank_after["numerical_rank"],
                        "condition_number_before": rank_before["condition_number"],
                        "condition_number_after": rank_after["condition_number"],
                        "rank_tolerance_before": rank_before["rank_tolerance"],
                        "rank_tolerance_after": rank_after["rank_tolerance"],
                    }
                )
                if (
                    scale_oracle["numerical_rank_before"]
                    != scale_oracle["numerical_rank_after"]
                ):
                    raise AssertionError(
                        "Mahalanobis common-scale numerical-rank regime changed"
                    )
                invariants = [
                    _oracle_record(
                        "mahalanobis_signed_coordinate_permutation",
                        np.concatenate((maha_raw[id_split], maha_raw[dataset])),
                        np.concatenate(
                            (maha_permuted[id_split], maha_permuted[dataset])
                        ),
                    ),
                    scale_oracle,
                ]
                witness_raw_reference = np.concatenate(
                    (maha_raw[id_split], maha_raw[dataset])
                )
                witness_raw_observed = np.concatenate(
                    (maha_witness_raw[id_split], maha_witness_raw[dataset])
                )
                witness_target_reference = np.concatenate(
                    (maha_pp[id_split], maha_pp[dataset])
                )
                witness_target_observed = np.concatenate(
                    (
                        maha_witness_targeted[id_split],
                        maha_witness_targeted[dataset],
                    )
                )
                target_auroc_drift = abs(
                    _metric_values(maha_pp[id_split], maha_pp[dataset])["auroc"]
                    - _metric_values(
                        maha_witness_targeted[id_split],
                        maha_witness_targeted[dataset],
                    )["auroc"]
                )
                targeted_acceptance_rule = (
                    "allclose_and_exact_rank_and_exact_auroc"
                )
            else:
                invariants = [
                    _oracle_record(
                        "knn_signed_coordinate_permutation",
                        np.concatenate((knn_raw[id_split], knn_raw[dataset])),
                        np.concatenate(
                            (knn_permuted[id_split], knn_permuted[dataset])
                        ),
                    ),
                    _oracle_record(
                        "knn_raw_common_positive_scale",
                        np.concatenate((knn_raw[id_split], knn_raw[dataset])),
                        np.concatenate((knn_scaled[id_split], knn_scaled[dataset])),
                        expected_scale=scale * scale,
                    ),
                    _oracle_record(
                        "knn_l2_common_positive_scale_approximate",
                        np.concatenate((knn_l2[id_split], knn_l2[dataset])),
                        np.concatenate(
                            (knn_l2_scaled[id_split], knn_l2_scaled[dataset])
                        ),
                        approximate_auroc=(
                            knn_l2[id_split],
                            knn_l2[dataset],
                            knn_l2_scaled[id_split],
                            knn_l2_scaled[dataset],
                        ),
                    ),
                ]
                witness_raw_reference = np.concatenate(
                    (knn_raw[id_split], knn_raw[dataset])
                )
                witness_raw_observed = np.concatenate(
                    (knn_witness_raw[id_split], knn_witness_raw[dataset])
                )
                witness_target_reference = np.concatenate(
                    (knn_l2[id_split], knn_l2[dataset])
                )
                witness_target_observed = np.concatenate(
                    (
                        knn_witness_targeted[id_split],
                        knn_witness_targeted[dataset],
                    )
                )
                target_auroc_drift = abs(
                    _metric_values(knn_l2[id_split], knn_l2[dataset])["auroc"]
                    - _metric_values(
                        knn_witness_targeted[id_split],
                        knn_witness_targeted[dataset],
                    )["auroc"]
                )
                targeted_acceptance_rule = "approximate_auroc_only"
            raw_witness_abs, _ = _max_errors(
                witness_raw_reference, witness_raw_observed
            )
            raw_witness_rank = _rank_signature_disagreement(
                witness_raw_reference, witness_raw_observed
            )
            targeted_abs, _ = _max_errors(
                witness_target_reference, witness_target_observed
            )
            targeted_rank = _rank_signature_disagreement(
                witness_target_reference, witness_target_observed
            )
            targeted_allclose = bool(
                np.allclose(
                    witness_target_reference,
                    witness_target_observed,
                    atol=EXACT_ATOL,
                    rtol=EXACT_RTOL,
                )
            )
            target_pass = _targeted_witness_accepts(
                candidate_id,
                score_allclose=targeted_allclose,
                rank_disagreement_fraction=targeted_rank,
                auroc_abs_drift=target_auroc_drift,
            )
            witness_changed = raw_witness_abs > EXACT_ATOL or raw_witness_rank > 0.0
            if not target_pass:
                raise AssertionError(
                    f"targeted witness invariance failed: {candidate_id}:{dataset}"
                )
            witness = {
                "witness_id": layout["witness_id"],
                "status": "PASS",
                "reason_codes": [],
                "raw_score_max_abs_change": raw_witness_abs,
                "raw_rank_disagreement_fraction": raw_witness_rank,
                "raw_change_observed": bool(witness_changed),
                "targeted_score_max_abs_error": targeted_abs,
                "targeted_score_allclose": targeted_allclose,
                "targeted_rank_disagreement_fraction": targeted_rank,
                "targeted_auroc_abs_drift": target_auroc_drift,
                "targeted_acceptance_rule": targeted_acceptance_rule,
                "targeted_exact_atol": EXACT_ATOL,
                "targeted_exact_rtol": EXACT_RTOL,
                "targeted_approximate_auroc_tolerance": (
                    KNN_APPROXIMATE_AUROC_TOLERANCE
                    if candidate_id == "radial_l2_knn"
                    else None
                ),
                "normalization_status_preservation": (
                    witness_normalization_preservation[candidate_id]
                ),
            }

            for state, detector, transform_id, metric in (
                ("raw", raw_key, "raw_identity", raw_metric),
                (
                    "transformed",
                    transformed_key,
                    layout["transform_id"],
                    transformed_metric,
                ),
            ):
                formula_id = formulas[detector]
                formula_pass = bool(
                    formula_id[id_split]["pass"] and formula_id[dataset]["pass"]
                )
                formula_abs = max(
                    formula_id[id_split]["max_abs_error"],
                    formula_id[dataset]["max_abs_error"],
                )
                formula_rel = max(
                    formula_id[id_split]["max_rel_error"],
                    formula_id[dataset]["max_rel_error"],
                )
                if detector == "detector/knn_raw_k50":
                    scalar_parity = {
                        "status": "NOT_APPLICABLE",
                        "reason_codes": ["V2_ONLY_RAW_KNN_HAS_NO_V1_2_BASELINE"],
                        "metrics": dict(metric),
                    }
                else:
                    scalar_parity = _scalar_parity(
                        metric, baseline_metrics[(checkpoint, detector, dataset)]
                    )
                transition_mass = dict(transition["mass"])
                current_id_scores = detector_scores[detector][id_split]
                current_ood_scores = detector_scores[detector][dataset]
                score_distribution = {
                    "summary_schema": "fixed_score_distribution_v1",
                    "score_direction": "higher_is_more_id_like",
                    "quantile_method": "numpy_linear",
                    "quantile_probabilities": {
                        key: probability
                        for key, probability in FIXED_QUANTILE_PROBABILITIES
                    },
                    "id_test": _distribution_summary(
                        current_id_scores, label=f"{detector} ID-test scores"
                    ),
                    "ood": _distribution_summary(
                        current_ood_scores, label=f"{detector} {dataset} scores"
                    ),
                }
                if detector == "detector/knn_raw_k50":
                    score_source_refs = {
                        "id_test": _cache_score_source_reference(
                            cache_root=cache_root,
                            analysis_id=analysis_id,
                            bundle=bundle,
                            variant_id="raw_identity",
                            split_name=id_split,
                            observed_scores=current_id_scores,
                        ),
                        "ood": _cache_score_source_reference(
                            cache_root=cache_root,
                            analysis_id=analysis_id,
                            bundle=bundle,
                            variant_id="raw_identity",
                            split_name=dataset,
                            observed_scores=current_ood_scores,
                        ),
                    }
                else:
                    score_source_refs = {
                        "id_test": _stored_score_source_reference(
                            bundle, id_split, detector, current_id_scores
                        ),
                        "ood": _stored_score_source_reference(
                            bundle, dataset, detector, current_ood_scores
                        ),
                    }
                source_references = {
                    "source_allowlist_sha256": bundle.source_allowlist_sha256,
                    "score_arrays": score_source_refs,
                    "feature_arrays": {
                        "id_train": _selective_input_source_reference(
                            bundle, "id_train", "features"
                        ),
                        "id_test": _selective_input_source_reference(
                            bundle, id_split, "features"
                        ),
                        "ood": _selective_input_source_reference(
                            bundle, dataset, "features"
                        ),
                    },
                    "null_logit_arrays": {
                        "id_test": _selective_input_source_reference(
                            bundle, id_split, "logits"
                        ),
                        "ood": _selective_input_source_reference(
                            bundle, dataset, "logits"
                        ),
                    },
                }
                for split_alias in ("id_test", "ood"):
                    if (
                        score_distribution[split_alias]["array_sha256"]
                        != score_source_refs[split_alias]["array_sha256"]
                    ):
                        raise AssertionError(
                            "score distribution/source-reference SHA mismatch"
                        )
                geometry_channel = {
                    **geometry_channel_profiles[candidate_id],
                    "active_ood_split": dataset,
                }
                row = {
                    "schema_version": MODEL_METRIC_SCHEMA,
                    "analysis_id": analysis_id,
                    "candidate_id": candidate_id,
                    "state": state,
                    "config_hash": bundle.bundle_spec["config_hash"],
                    "optimizer_family": bundle.bundle_spec["optimizer_family"],
                    "role_labels": list(bundle.bundle_spec["labels"]),
                    "training_seed": int(bundle.bundle_spec["training_seed"]),
                    "checkpoint_sha256": checkpoint,
                    "checkpoint_role": "last",
                    "ood_dataset": dataset,
                    "ood_group": ood_groups[dataset],
                    "detector": detector,
                    "transform_id": transform_id,
                    "paired_detector": (
                        transformed_key if state == "raw" else raw_key
                    ),
                    "paired_transform_id": layout["transform_id"],
                    "num_id": int(len(raw_id)),
                    "num_ood": int(len(raw_ood)),
                    "auroc": metric["auroc"],
                    "fpr95": metric["fpr95"],
                    "fpr95_threshold": metric["fpr95_threshold"],
                    "achieved_id_tpr": metric["achieved_id_tpr"],
                    "misordering_mass": float(
                        sum(
                            value
                            for key, value in transition_mass.items()
                            if (
                                key.startswith("misordered_to_")
                                if state == "raw"
                                else key.endswith("_to_misordered")
                            )
                        )
                    ),
                    "tie_mass": float(
                        sum(
                            value
                            for key, value in transition_mass.items()
                            if (key.startswith("tie_to_") if state == "raw" else key.endswith("_to_tie"))
                        )
                    ),
                    "pair_order_transition_mass": transition_mass,
                    "pair_order_transition_counts": dict(transition["counts"]),
                    "pair_count": transition["pair_count"],
                    "corrected_pair_mass": transition["corrected_pair_mass"],
                    "harmed_pair_mass": transition["harmed_pair_mass"],
                    "net_corrected_pair_mass": transition[
                        "net_corrected_pair_mass"
                    ],
                    "score_reconstruction_max_abs_error": formula_abs,
                    "score_reconstruction_max_rel_error": formula_rel,
                    "formula_reconstruction_pass": formula_pass,
                    "pair_credit_identity_pass": transition[
                        "pair_credit_identity_pass"
                    ],
                    "pair_credit_identity_error": transition[
                        "pair_credit_identity_error"
                    ],
                    "rank_disagreement_fraction": transition[
                        "rank_disagreement_fraction"
                    ],
                    "scalar_parity": scalar_parity,
                    "exact_null_oracle": exact_null,
                    "invariant_oracles": invariants,
                    "non_invariant_witness": witness,
                    "score_distribution": score_distribution,
                    "geometry_channel": geometry_channel,
                    "source_references": source_references,
                    "id_metrics": model_id_metrics,
                    "score_sha256": _array_sha256(
                        np.concatenate(
                            (
                                detector_scores[detector][id_split],
                                detector_scores[detector][dataset],
                            )
                        )
                    ),
                    "score_components": (
                        {
                            "id_test": knn_component_evidence[detector][id_split],
                            "ood": knn_component_evidence[detector][dataset],
                            "raw_to_l2_neighbor_overlap": {
                                "id_test": knn_overlap[id_split],
                                "ood": knn_overlap[dataset],
                            },
                            "exact_bank_mapping": {
                                "encoding": "exact_train_sample_id_strings",
                                "train_sample_id_sha256": _array_sha256(train_ids),
                                "lossless_from_neighbor_component": True,
                            },
                        }
                        if candidate_id == "radial_l2_knn"
                        else {
                            "id_test": mahalanobis_component_evidence[detector][
                                id_split
                            ],
                            "ood": mahalanobis_component_evidence[detector][dataset],
                        }
                    ),
                    "source_allowlist_sha256": bundle.source_allowlist_sha256,
                    "numeric_runtime_fingerprint_sha256": runtime[
                        "canonical_lock_sha256"
                    ],
                    "raw_knn_backend": raw_knn_metadata,
                    "transformed_knn_backend": l2_knn_metadata,
                    "status": "PASS",
                    "reason_codes": [],
                }
                rows.append(row)
    return rows


def _component_array_record(values: Any, *, label: str) -> dict[str, Any]:
    array = np.asarray(values)
    if not array.size or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be a non-empty numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "array_sha256": _array_sha256(array),
        "distribution": _distribution_summary(array, label=label),
    }


def _vim_author_dimensions(feature_dim: int) -> tuple[int, int]:
    width = int(feature_dim)
    if width < 2:
        raise ValueError("ViM author dimension requires feature_dim >= 2")
    author_dim = width // 2
    return author_dim, width - author_dim


def _matched_class_record(values: Any, *, class_count: int, label: str) -> dict[str, Any]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu" or not len(array):
        raise ValueError(f"{label} must be a non-empty integer vector")
    if np.any(array < 0) or np.any(array >= class_count):
        raise ValueError(f"{label} is outside the class range")
    return {
        "count": int(len(array)),
        "dtype": str(array.dtype),
        "array_sha256": _array_sha256(array),
        "class_counts": np.bincount(array, minlength=class_count).astype(int).tolist(),
    }


def _score_distribution_record(
    id_scores: Any, ood_scores: Any, *, detector: str, dataset: str
) -> dict[str, Any]:
    return {
        "summary_schema": "fixed_score_distribution_v1",
        "score_direction": "higher_is_more_id_like",
        "quantile_method": "numpy_linear",
        "quantile_probabilities": {
            key: probability for key, probability in FIXED_QUANTILE_PROBABILITIES
        },
        "id_test": _distribution_summary(
            id_scores, label=f"{detector} ID-test diagnostic scores"
        ),
        "ood": _distribution_summary(
            ood_scores, label=f"{detector} {dataset} diagnostic scores"
        ),
    }


def _formula_reconstruction_record(
    reference: Any, reconstructed: Any, *, label: str
) -> dict[str, Any]:
    left = _finite_vector(reference, label=f"{label} reference")
    right = _finite_vector(reconstructed, label=f"{label} reconstruction")
    maximum_absolute, maximum_relative = _max_errors(left, right)
    passed = bool(
        np.all(np.abs(left - right) <= EXACT_ATOL + EXACT_RTOL * np.abs(left))
    )
    if not passed:
        raise AssertionError(f"{label} formula reconstruction failed")
    return {
        "status": "PASS",
        "reason_codes": [],
        "max_abs_error": maximum_absolute,
        "max_rel_error": maximum_relative,
        "reference_array_sha256": _array_sha256(left),
        "reconstructed_array_sha256": _array_sha256(right),
        "absolute_tolerance": EXACT_ATOL,
        "relative_tolerance": EXACT_RTOL,
    }


def _derived_negative_component_source_reference(
    component_source: Mapping[str, Any], scores: Any
) -> dict[str, Any]:
    array = _finite_vector(scores, label="derived negative-component score")
    return {
        "source_kind": "derived_negative_of_selective_reuse_diagnostic_array",
        "derivation": "score_equals_negative_stored_vim_residual",
        "component_logical_path": component_source["logical_path"],
        "component_file_sha256": component_source["file_sha256"],
        "component_array_sha256": component_source["array_sha256"],
        "array_sha256": _array_sha256(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bundle_id": component_source["bundle_id"],
        "source_allowlist_sha256": component_source["source_allowlist_sha256"],
    }


def extract_bundle_diagnostic_rows(
    bundle: SelectiveBundle,
    *,
    baseline_metrics: Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]],
    ood_datasets: Sequence[str],
    ood_groups: Mapping[str, str],
    analysis_id: str,
    numeric_runtime_fingerprint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Materialize the two frozen diagnostic-only Stage-2 channels."""
    runtime = _validate_runtime_fingerprint(
        numeric_runtime_fingerprint or _numeric_runtime_fingerprint()
    )
    if tuple(ood_datasets) != bundle.ood_keys:
        raise ValueError("diagnostic OOD order differs from the selective bundle")
    train = bundle.splits["id_train"]
    train_features = np.asarray(train["features"])
    train_logits = np.asarray(train["logits"])
    train_labels = np.asarray(train["class_labels"])
    class_count = int(bundle.classifier_weight.shape[0])
    feature_dim = int(train_features.shape[1])

    vim_fit = fit_vim(
        train_features,
        train_logits,
        bundle.classifier_weight,
        bundle.classifier_bias,
        dim=None,
    )
    expected_author_dim, expected_residual_dim = _vim_author_dimensions(feature_dim)
    if (
        vim_fit.dim != expected_author_dim
        or vim_fit.residual_subspace.shape != (feature_dim, expected_residual_dim)
    ):
        raise AssertionError("Pure Residual author-DIM contract mismatch")

    train_prototype = prototype_scores(
        train_features,
        train_labels,
        train_features,
        num_classes=class_count,
    )
    train_ctm_status = {
        "status": str(train_prototype["status"]),
        "reason_codes": list(train_prototype["reason_codes"]),
        "normalization_diagnostics": dict(
            train_prototype["normalization_diagnostics"]
        ),
    }
    if train_ctm_status["status"] != "success":
        raise AssertionError("CTM ID-train normalization status is not successful")
    class_means = np.asarray(train_prototype["class_means"], dtype=np.float64)
    normalized_train = _normalized(
        train_features, epsilon=0.0, label="CTM ID-train angular concentration"
    )
    normalized_means = _normalized(
        class_means, epsilon=0.0, label="CTM class means"
    )
    own_class_cosine = np.sum(
        normalized_train * normalized_means[train_labels.astype(np.int64)], axis=1
    )
    class_counts = np.bincount(train_labels, minlength=class_count).astype(int)
    classwise_concentration = {
        "definition": (
            "cosine_of_each_raw_id_train_feature_to_its_own_raw_before_average_"
            "class_mean"
        ),
        "all_id_train": _distribution_summary(
            own_class_cosine, label="CTM all ID-train own-class cosine"
        ),
        "classes": [
            {
                "class_index": class_index,
                "sample_count": int(class_counts[class_index]),
                "own_class_cosine": _distribution_summary(
                    own_class_cosine[train_labels == class_index],
                    label=f"CTM ID-train class {class_index} own-class cosine",
                ),
            }
            for class_index in range(class_count)
        ],
    }
    pure_fit_components = {
        "feature_dim": feature_dim,
        "author_dim": int(vim_fit.dim),
        "residual_subspace_dim": int(vim_fit.residual_subspace.shape[1]),
        "author_dim_rule": "floor(feature_dim/2)",
        "origin": _component_array_record(vim_fit.origin, label="ViM origin"),
        "covariance_eigenvalues": _component_array_record(
            vim_fit.eigenvalues, label="ViM covariance eigenvalues"
        ),
        "residual_subspace": _component_array_record(
            vim_fit.residual_subspace, label="ViM residual subspace"
        ),
        "alpha": float(vim_fit.alpha),
        "residual_mean": float(vim_fit.residual_mean),
    }
    ctm_fit_components = {
        "class_means": _component_array_record(
            class_means, label="CTM raw-before-average class means"
        ),
        "class_counts": class_counts.tolist(),
        "normalization_status": train_ctm_status,
        "classwise_id_train_angular_concentration": classwise_concentration,
    }

    vim_by_split: dict[str, Mapping[str, np.ndarray]] = {}
    ctm_by_split: dict[str, Mapping[str, Any]] = {}
    pure_scores: dict[str, np.ndarray] = {}
    full_vim_sources: dict[str, Mapping[str, Any]] = {}
    residual_sources: dict[str, Mapping[str, Any]] = {}
    ctm_sources: dict[str, Mapping[str, Any]] = {}
    vim_score_parity: dict[str, Mapping[str, Any]] = {}
    residual_parity: dict[str, Mapping[str, Any]] = {}
    ctm_score_parity: dict[str, Mapping[str, Any]] = {}
    pure_formula: dict[str, Mapping[str, Any]] = {}
    vim_formula: dict[str, Mapping[str, Any]] = {}
    ctm_formula: dict[str, Mapping[str, Any]] = {}
    pure_components: dict[str, Mapping[str, Any]] = {}
    ctm_components: dict[str, Mapping[str, Any]] = {}

    for split_name in bundle.query_keys:
        query = bundle.splits[split_name]
        vim = score_vim(vim_fit, query["features"], query["logits"])
        ctm = prototype_scores(
            train_features,
            train_labels,
            query["features"],
            num_classes=class_count,
        )
        ctm_status = {
            "status": str(ctm["status"]),
            "reason_codes": list(ctm["reason_codes"]),
            "normalization_diagnostics": dict(ctm["normalization_diagnostics"]),
        }
        if ctm_status["status"] != "success":
            raise AssertionError(f"CTM {split_name} normalization status is not successful")
        vim_by_split[split_name] = vim
        ctm_by_split[split_name] = ctm
        residual = np.asarray(vim["residual"], dtype=np.float64)
        pure = -residual
        pure_scores[split_name] = pure

        full_vim_sources[split_name] = _stored_score_source_reference(
            bundle, split_name, "detector/vim_author_dim", vim["score"]
        )
        vim_score_parity[split_name] = _full_array_parity_record(
            bundle.scores[split_name]["detector/vim_author_dim"],
            vim["score"],
            label=f"{split_name}:stored full ViM score",
        )
        residual_source, residual_check = _stored_diagnostic_source_reference(
            bundle,
            split_name,
            ("diagnostics", "vim", "residual"),
            bundle.diagnostics[split_name]["vim_residual"],
            residual,
        )
        residual_sources[split_name] = residual_source
        residual_parity[split_name] = residual_check
        ctm_sources[split_name] = _stored_score_source_reference(
            bundle, split_name, "detector/ctm_prototype_cosine", ctm["ctm_score"]
        )
        ctm_score_parity[split_name] = _full_array_parity_record(
            bundle.scores[split_name]["detector/ctm_prototype_cosine"],
            ctm["ctm_score"],
            label=f"{split_name}:stored CTM score",
        )

        pure_formula[split_name] = _formula_reconstruction_record(
            pure, -np.asarray(bundle.diagnostics[split_name]["vim_residual"]),
            label=f"{split_name}:Pure Residual",
        )
        vim_formula[split_name] = _formula_reconstruction_record(
            vim["score"],
            np.asarray(vim["energy"]) - np.asarray(vim["alpha_residual"]),
            label=f"{split_name}:full ViM component sum",
        )
        ctm_matrix = np.asarray(ctm["prototype_cosine_matrix"], dtype=np.float64)
        ctm_reconstructed = np.max(ctm_matrix, axis=1)
        ctm_formula[split_name] = _formula_reconstruction_record(
            ctm["ctm_score"], ctm_reconstructed, label=f"{split_name}:CTM"
        )
        if not np.array_equal(
            np.asarray(ctm["ctm_class"]), np.argmax(ctm_matrix, axis=1)
        ):
            raise AssertionError("CTM matched-class reconstruction failed")
        pure_components[split_name] = {
            "residual": _component_array_record(
                residual, label=f"{split_name} ViM residual"
            ),
            "pure_residual_score": _component_array_record(
                pure, label=f"{split_name} Pure Residual score"
            ),
            "vim_energy": _component_array_record(
                vim["energy"], label=f"{split_name} ViM energy"
            ),
            "vim_alpha_residual": _component_array_record(
                vim["alpha_residual"], label=f"{split_name} ViM alpha residual"
            ),
            "vim_full_score": _component_array_record(
                vim["score"], label=f"{split_name} full ViM score"
            ),
            "pure_residual_formula": pure_formula[split_name],
            "full_vim_formula": vim_formula[split_name],
        }
        ctm_components[split_name] = {
            "matched_cosine": _component_array_record(
                ctm["ctm_score"], label=f"{split_name} CTM matched cosine"
            ),
            "matched_class": _matched_class_record(
                ctm["ctm_class"],
                class_count=class_count,
                label=f"{split_name} CTM matched class",
            ),
            "prototype_cosine_matrix": _component_array_record(
                ctm_matrix, label=f"{split_name} CTM prototype cosine matrix"
            ),
            "normalization_status": ctm_status,
            "ctm_formula": ctm_formula[split_name],
        }

    checkpoint = bundle.bundle_id
    identity = {
        "analysis_id": analysis_id,
        "config_hash": bundle.bundle_spec["config_hash"],
        "optimizer_family": bundle.bundle_spec["optimizer_family"],
        "role_labels": list(bundle.bundle_spec["labels"]),
        "training_seed": int(bundle.bundle_spec["training_seed"]),
        "checkpoint_sha256": checkpoint,
        "checkpoint_role": "last",
    }
    shared_features = {
        "id_train": _selective_input_source_reference(bundle, "id_train", "features"),
        "id_test": _selective_input_source_reference(bundle, "id_test", "features"),
    }
    shared_logits = {
        "id_train": _selective_input_source_reference(bundle, "id_train", "logits"),
        "id_test": _selective_input_source_reference(bundle, "id_test", "logits"),
    }
    classifier_sources = {
        "weight": _classifier_source_reference(bundle, "weight"),
        "bias": _classifier_source_reference(bundle, "bias"),
    }
    rows: list[dict[str, Any]] = []
    for dataset in ood_datasets:
        id_split = "id_test"
        pure_id = pure_scores[id_split]
        pure_ood = pure_scores[dataset]
        pure_metric = _metric_values(pure_id, pure_ood)
        pure_distribution = _score_distribution_record(
            pure_id,
            pure_ood,
            detector="detector/vim_residual_author_dim",
            dataset=dataset,
        )
        pure_source_refs = {
            "source_allowlist_sha256": bundle.source_allowlist_sha256,
            "feature_arrays": {
                **shared_features,
                "ood": _selective_input_source_reference(bundle, dataset, "features"),
            },
            "logit_arrays": {
                **shared_logits,
                "ood": _selective_input_source_reference(bundle, dataset, "logits"),
            },
            "classifier_arrays": classifier_sources,
            "stored_full_vim_score_arrays": {
                "id_test": full_vim_sources[id_split],
                "ood": full_vim_sources[dataset],
            },
            "stored_vim_residual_arrays": {
                "id_test": residual_sources[id_split],
                "ood": residual_sources[dataset],
            },
            "derived_score_arrays": {
                "id_test": _derived_negative_component_source_reference(
                    residual_sources[id_split], pure_id
                ),
                "ood": _derived_negative_component_source_reference(
                    residual_sources[dataset], pure_ood
                ),
            },
        }
        pure_formula_records = (
            pure_formula[id_split],
            pure_formula[dataset],
            vim_formula[id_split],
            vim_formula[dataset],
        )
        pure_row = {
            "schema_version": DIAGNOSTIC_METRIC_SCHEMA,
            **identity,
            "diagnostic_id": "pure_residual_channel",
            "selection_status": "diagnostic_only",
            "candidate_eligibility": False,
            "ranking_use": "forbidden",
            "ood_dataset": dataset,
            "ood_group": ood_groups[dataset],
            "detector": "detector/vim_residual_author_dim",
            "channel_contract": {
                "pure_residual_uses_energy": False,
                "pure_residual_uses_alpha": False,
                "score_orientation": "negative_residual_id_like",
                "author_dim_rule": "floor(feature_dim/2)",
            },
            "num_id": int(len(pure_id)),
            "num_ood": int(len(pure_ood)),
            **pure_metric,
            "score_sha256": _array_sha256(np.concatenate((pure_id, pure_ood))),
            "score_distribution": pure_distribution,
            "formula_reconstruction_max_abs_error": max(
                float(record["max_abs_error"]) for record in pure_formula_records
            ),
            "formula_reconstruction_max_rel_error": max(
                float(record["max_rel_error"]) for record in pure_formula_records
            ),
            "formula_reconstruction_pass": True,
            "stored_artifact_parity": {
                "status": "PASS",
                "reason_codes": [],
                "full_vim_score": {
                    "status": "PASS",
                    "reason_codes": [],
                    "id_test": vim_score_parity[id_split],
                    "ood": vim_score_parity[dataset],
                },
                "vim_residual": {
                    "status": "PASS",
                    "reason_codes": [],
                    "id_test": residual_parity[id_split],
                    "ood": residual_parity[dataset],
                },
            },
            "scalar_parity": {
                "status": "NOT_APPLICABLE",
                "reason_codes": [
                    "V2_ONLY_PURE_RESIDUAL_HAS_NO_V1_2_SCALAR_BASELINE"
                ],
                "metrics": dict(pure_metric),
            },
            "score_components": {
                "fit": pure_fit_components,
                "id_test": pure_components[id_split],
                "ood": pure_components[dataset],
            },
            "source_references": pure_source_refs,
            "source_allowlist_sha256": bundle.source_allowlist_sha256,
            "numeric_runtime_fingerprint_sha256": runtime[
                "canonical_lock_sha256"
            ],
            "status": "PASS",
            "reason_codes": [],
        }
        rows.append(pure_row)

        ctm_id = np.asarray(ctm_by_split[id_split]["ctm_score"], dtype=np.float64)
        ctm_ood = np.asarray(ctm_by_split[dataset]["ctm_score"], dtype=np.float64)
        ctm_metric = _metric_values(ctm_id, ctm_ood)
        ctm_distribution = _score_distribution_record(
            ctm_id,
            ctm_ood,
            detector="detector/ctm_prototype_cosine",
            dataset=dataset,
        )
        ctm_formula_records = (ctm_formula[id_split], ctm_formula[dataset])
        ctm_row = {
            "schema_version": DIAGNOSTIC_METRIC_SCHEMA,
            **identity,
            "diagnostic_id": "ctm_angular_channel",
            "selection_status": "diagnostic_only",
            "candidate_eligibility": False,
            "ranking_use": "forbidden",
            "ood_dataset": dataset,
            "ood_group": ood_groups[dataset],
            "detector": "detector/ctm_prototype_cosine",
            "num_id": int(len(ctm_id)),
            "num_ood": int(len(ctm_ood)),
            **ctm_metric,
            "score_sha256": _array_sha256(np.concatenate((ctm_id, ctm_ood))),
            "score_distribution": ctm_distribution,
            "formula_reconstruction_max_abs_error": max(
                float(record["max_abs_error"]) for record in ctm_formula_records
            ),
            "formula_reconstruction_max_rel_error": max(
                float(record["max_rel_error"]) for record in ctm_formula_records
            ),
            "formula_reconstruction_pass": True,
            "stored_artifact_parity": {
                "status": "PASS",
                "reason_codes": [],
                "ctm_score": {
                    "status": "PASS",
                    "reason_codes": [],
                    "id_test": ctm_score_parity[id_split],
                    "ood": ctm_score_parity[dataset],
                },
            },
            "scalar_parity": _scalar_parity(
                ctm_metric,
                baseline_metrics[
                    (checkpoint, "detector/ctm_prototype_cosine", dataset)
                ],
            ),
            "score_components": {
                "fit": ctm_fit_components,
                "id_test": ctm_components[id_split],
                "ood": ctm_components[dataset],
            },
            "source_references": {
                "source_allowlist_sha256": bundle.source_allowlist_sha256,
                "feature_arrays": {
                    **shared_features,
                    "ood": _selective_input_source_reference(
                        bundle, dataset, "features"
                    ),
                },
                "class_label_arrays": {
                    "id_train": _selective_input_source_reference(
                        bundle, "id_train", "class_labels"
                    )
                },
                "stored_score_arrays": {
                    "id_test": ctm_sources[id_split],
                    "ood": ctm_sources[dataset],
                },
            },
            "source_allowlist_sha256": bundle.source_allowlist_sha256,
            "numeric_runtime_fingerprint_sha256": runtime[
                "canonical_lock_sha256"
            ],
            "status": "PASS",
            "reason_codes": [],
        }
        rows.append(ctm_row)
    return rows


def _analysis_identity_payload(
    context: ReuseContext,
    *,
    analysis_git_sha: str,
    code_sources: Sequence[Mapping[str, Any]],
    query_chunk_size: int,
    bank_chunk_size: int | None,
    memory_budget_bytes: int | None,
    numeric_runtime_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = _validate_runtime_fingerprint(numeric_runtime_fingerprint)
    return {
        "contract_version": ANALYSIS_CONTRACT_VERSION,
        "reuse_manifest_sha256": context.manifest_sha256,
        "analysis_git_sha": analysis_git_sha,
        "code_sources": list(code_sources),
        "candidate_registry_sha256": context.manifest["analysis_policy"][
            "candidate_registry"
        ]["sha256"],
        "gate_policy_sha256": context.manifest["analysis_policy"]["gate_policy"][
            "sha256"
        ],
        "checkpoint_ids": sorted(
            str(row["bundle_id"]) for row in context.manifest["bundles"]
        ),
        "knn_backend": RAW_KNN_BACKEND_ID,
        "knn_k": KNN_K,
        "query_chunk_size": query_chunk_size,
        "bank_chunk_size": bank_chunk_size,
        "memory_budget_bytes": memory_budget_bytes,
        "numeric_runtime_fingerprint": runtime,
    }


def build_stage2_evidence(
    *,
    reuse_manifest_path: str | Path,
    retrieval_root: str | Path,
    analysis_git_sha: str | None = None,
    code_sources: Sequence[Mapping[str, Any]] | None = None,
    validate_git_sources: bool = True,
    cache_root: str | Path | None = None,
    parity_baseline_root: str | Path | None = None,
    bundle_ids: Sequence[str] | None = None,
    cache_only: bool = False,
    precompute_bundle_id: str | None = None,
    compute_missing_cache: bool = False,
    query_chunk_size: int = 64,
    bank_chunk_size: int | None = None,
    memory_budget_bytes: int | None = None,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    manifest_path = Path(reuse_manifest_path).resolve()
    precompute_mode = cache_only or precompute_bundle_id is not None
    if precompute_mode and parity_baseline_root is not None:
        raise ValueError("cache precompute forbids --parity-baseline-root")
    if not precompute_mode:
        manifest_for_baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_for_baseline, Mapping):
            raise ValueError("reuse manifest must contain a JSON mapping")
        # Verify relocation before load_reuse_context hashes any protected arrays.
        _verify_baseline_files(manifest_for_baseline, parity_baseline_root)
    if validate_git_sources:
        observed_git_sha, observed_sources = validate_committed_sources(manifest_path)
        if analysis_git_sha is not None and analysis_git_sha != observed_git_sha:
            raise ValueError("requested analysis Git SHA differs from HEAD")
        analysis_git_sha = observed_git_sha
        code_sources = observed_sources
    elif analysis_git_sha is None or code_sources is None:
        raise ValueError(
            "tests bypassing Git validation must supply analysis_git_sha and code_sources"
        )
    numeric_runtime_fingerprint = _numeric_runtime_fingerprint()
    if precompute_bundle_id is not None:
        if bundle_ids is not None:
            raise ValueError("precompute bundle ID and bundle_ids are mutually exclusive")
        cache_only = True
        bundle_ids = [precompute_bundle_id]
        context = load_single_bundle_reuse_context(
            manifest_path, retrieval_root, precompute_bundle_id
        )
    else:
        context = load_reuse_context(manifest_path, retrieval_root)
    _, gate, ood_datasets, ood_groups = _validate_registry_and_gate(context)
    identity_payload = _analysis_identity_payload(
        context,
        analysis_git_sha=str(analysis_git_sha),
        code_sources=list(code_sources),
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        memory_budget_bytes=memory_budget_bytes,
        numeric_runtime_fingerprint=numeric_runtime_fingerprint,
    )
    analysis_id = _sha256_bytes(_canonical_bytes(identity_payload))
    resolved_cache_root: Path | None = None
    if cache_root is not None:
        resolved_cache_root = _prepare_cache_root(Path(cache_root))
    if cache_only and resolved_cache_root is None:
        raise ValueError("cache-only mode requires --cache-root")
    if not cache_only and resolved_cache_root is None:
        raise ValueError(
            "full production extraction requires --cache-root; use explicit "
            "--compute-missing-cache to populate absent variants"
        )
    if bundle_ids is None:
        selected_bundle_ids = sorted(context.bundles)
    else:
        selected_bundle_ids = sorted(set(bundle_ids))
        if not selected_bundle_ids or any(
            bundle_id not in context.bundles for bundle_id in selected_bundle_ids
        ):
            raise ValueError("bundle shard contains an unknown or empty bundle ID set")
    if not cache_only and set(selected_bundle_ids) != set(context.bundles):
        raise ValueError("partial bundle selection is allowed only with cache-only mode")
    if cache_only:
        precomputed = []
        for bundle_id in selected_bundle_ids:
            bundle = load_knn_precompute_bundle(context, bundle_id)
            precomputed.append(
                precompute_bundle_knn_variants(
                    bundle,
                    analysis_id=analysis_id,
                    cache_root=resolved_cache_root,
                    query_chunk_size=query_chunk_size,
                    bank_chunk_size=bank_chunk_size,
                    memory_budget_bytes=memory_budget_bytes,
                    numeric_runtime_fingerprint=numeric_runtime_fingerprint,
                )
            )
            del bundle
        return (
            {
                "schema_version": "fixed_readout_stage2_knn_cache_precompute_v1",
                "analysis_id": analysis_id,
                "contract_version": ANALYSIS_CONTRACT_VERSION,
                "analysis_git_sha": str(analysis_git_sha),
                "cache_root": str(resolved_cache_root),
                "bundle_ids": selected_bundle_ids,
                "bundle_count": len(selected_bundle_ids),
                "variant_count_per_bundle": len(KNN_VARIANT_IDS),
                "numeric_runtime_fingerprint": numeric_runtime_fingerprint,
                "bundles": precomputed,
                "status": "PASS",
                "scientific_gate": "NOT_RUN",
            },
            [],
            [],
        )
    if compute_missing_cache:
        # Explicit opt-in still uses the low-peak sequential precompute path;
        # row extraction subsequently becomes cache-only and deterministic.
        for bundle_id in selected_bundle_ids:
            precompute_bundle = load_knn_precompute_bundle(context, bundle_id)
            precompute_bundle_knn_variants(
                precompute_bundle,
                analysis_id=analysis_id,
                cache_root=resolved_cache_root,
                query_chunk_size=query_chunk_size,
                bank_chunk_size=bank_chunk_size,
                memory_budget_bytes=memory_budget_bytes,
                numeric_runtime_fingerprint=numeric_runtime_fingerprint,
            )
            del precompute_bundle
    baseline_metrics, id_metrics, baseline_sources = load_frozen_baseline(
        context, parity_baseline_root=parity_baseline_root
    )
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for bundle_id in selected_bundle_ids:
        bundle = load_selective_bundle(context, bundle_id)
        rows.extend(
            extract_bundle_rows(
                bundle,
                baseline_metrics=baseline_metrics,
                id_metrics=id_metrics,
                ood_datasets=ood_datasets,
                ood_groups=ood_groups,
                analysis_id=analysis_id,
                cache_root=resolved_cache_root,
                require_existing_knn_cache=True,
                query_chunk_size=query_chunk_size,
                bank_chunk_size=bank_chunk_size,
                memory_budget_bytes=memory_budget_bytes,
                numeric_runtime_fingerprint=numeric_runtime_fingerprint,
            )
        )
        diagnostic_rows.extend(
            extract_bundle_diagnostic_rows(
                bundle,
                baseline_metrics=baseline_metrics,
                ood_datasets=ood_datasets,
                ood_groups=ood_groups,
                analysis_id=analysis_id,
                numeric_runtime_fingerprint=numeric_runtime_fingerprint,
            )
        )
        del bundle
    expected_row_count = (
        STAGE2_BUNDLE_COUNT * len(ood_datasets) * len(CANDIDATES) * 2
    )
    if len(rows) != expected_row_count:
        raise AssertionError("Stage-2 model-metric row population is incomplete")
    rows.sort(
        key=lambda row: (
            row["candidate_id"],
            row["config_hash"],
            row["training_seed"],
            row["ood_dataset"],
            row["state"],
        )
    )
    expected_diagnostic_row_count = (
        STAGE2_BUNDLE_COUNT * len(ood_datasets) * len(DIAGNOSTIC_IDS)
    )
    if expected_diagnostic_row_count != EXPECTED_DIAGNOSTIC_ROW_COUNT:
        raise AssertionError("frozen Stage-2 diagnostic population must equal 360 rows")
    if len(diagnostic_rows) != expected_diagnostic_row_count:
        raise AssertionError("Stage-2 diagnostic row population is incomplete")
    diagnostic_rows.sort(
        key=lambda row: (
            row["diagnostic_id"],
            row["config_hash"],
            row["training_seed"],
            row["ood_dataset"],
        )
    )
    for row in diagnostic_rows:
        if row.get("diagnostic_id") not in DIAGNOSTIC_IDS:
            raise AssertionError("unexpected Stage-2 diagnostic ID")
        if (
            row.get("selection_status") != "diagnostic_only"
            or row.get("candidate_eligibility") is not False
            or row.get("ranking_use") != "forbidden"
        ):
            raise AssertionError("diagnostic row is not excluded from selection")
    if set(DIAGNOSTIC_IDS) & set(CANDIDATES):
        raise AssertionError("diagnostic IDs must be disjoint from candidate IDs")
    required = set(gate["required_output_artifacts"]["model_metrics.jsonl"])
    for row in rows:
        if not required.issubset(row):
            raise AssertionError(
                f"model-metric row omits frozen fields: {sorted(required - set(row))}"
            )
        if set(row["pair_order_transition_mass"]) != set(TRANSITION_FIELDS):
            raise AssertionError("model-metric transition schema mismatch")
    checkpoint_bundle_ids = sorted(
        context.bundles,
        key=lambda bundle_id: (
            context.bundles[bundle_id]["config_hash"],
            int(context.bundles[bundle_id]["training_seed"]),
            bundle_id,
        ),
    )
    checkpoints = [
        {
            "checkpoint_sha256": bundle_id,
            "config_hash": context.bundles[bundle_id]["config_hash"],
            "training_seed": int(context.bundles[bundle_id]["training_seed"]),
            "checkpoint_role": context.bundles[bundle_id]["identity"][
                "checkpoint_role"
            ],
            "optimizer_family": context.bundles[bundle_id]["optimizer_family"],
            "role_labels": list(context.bundles[bundle_id]["labels"]),
            "source_allowlist_sha256": context.bundles[bundle_id]["remote_integrity"][
                "allowed_payload_catalog_sha256"
            ],
        }
        for bundle_id in checkpoint_bundle_ids
    ]
    transformed_rows = [row for row in rows if row["state"] == "transformed"]
    witness_population = {}
    for candidate_id in CANDIDATES:
        candidate_rows = [
            row for row in transformed_rows if row["candidate_id"] == candidate_id
        ]
        passed_query_count = sum(
            row["non_invariant_witness"]["status"] == "PASS"
            and (
                row["non_invariant_witness"]["raw_score_max_abs_change"]
                > EXACT_ATOL
                or row["non_invariant_witness"]["raw_rank_disagreement_fraction"]
                > 0.0
            )
            for row in candidate_rows
        )
        if passed_query_count < 1:
            raise AssertionError(
                f"population non-invariant witness was not observed: {candidate_id}"
            )
        witness_population[candidate_id] = {
            "status": "PASS",
            "required_minimum_full_query_array_count": 1,
            "observed_full_query_array_count": int(passed_query_count),
            "evaluated_full_query_array_count": len(candidate_rows),
        }
    oracle_population = {
        "execution_scope": "all_30_bundles_full_id_test_and_all_six_ood_query_arrays",
        "formula_reconstruction": {
            "status": "PASS",
            "row_count": len(rows),
        },
        "exact_energy_feature_only_null": {
            "status": "PASS",
            "evaluated_full_query_array_count": len(transformed_rows)
            // len(CANDIDATES),
        },
        "invariant_oracles": {
            "status": "PASS",
            "oracle_record_count": sum(
                len(row["invariant_oracles"]) for row in transformed_rows
            ),
        },
        "non_invariant_witness": witness_population,
    }
    identity = next(iter(context.bundles.values()))["identity"]
    selection_manifest = {
        "schema_version": SELECTION_MANIFEST_SCHEMA,
        "analysis_id": analysis_id,
        "contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "analysis_git_sha": str(analysis_git_sha),
        "checkpoint_role": "last",
        "inventory_hash": identity["inventory_hash"],
        "dataset_policy_hash": identity["dataset_policy_hash"],
        "evaluation_git_sha": identity["evaluation_git_sha"],
        "checkpoint_count": STAGE2_BUNDLE_COUNT,
        "unique_config_count": 10,
        "seed_set": [0, 1, 2],
        "candidate_selection_seed_set": [1, 2],
        "seed_zero_role_selection_bias": True,
        "ood_datasets": list(ood_datasets),
        "candidate_registry_sha256": context.manifest["analysis_policy"][
            "candidate_registry"
        ]["sha256"],
        "gate_policy_sha256": context.manifest["analysis_policy"]["gate_policy"][
            "sha256"
        ],
        "source_reuse_manifest_sha256": context.manifest_sha256,
        "source_files": baseline_sources,
        "code_files": list(code_sources),
        "numeric_runtime_fingerprint": numeric_runtime_fingerprint,
        "candidate_ids": list(CANDIDATES),
        "diagnostic_ids": list(DIAGNOSTIC_IDS),
        "diagnostic_population": {
            "row_count": len(diagnostic_rows),
            "diagnostic_count": len(DIAGNOSTIC_IDS),
            "rows_per_diagnostic": STAGE2_BUNDLE_COUNT * len(ood_datasets),
            "selection_status": "diagnostic_only",
            "candidate_eligibility": False,
            "ranking_use": "forbidden",
            "status": "PASS",
            "reason_codes": [],
        },
        "knn_cache_contract": {
            "schema_version": KNN_CACHE_SCHEMA,
            "variant_count_per_bundle": len(KNN_VARIANT_IDS),
            "resume_policy": "verified_atomic_no_overwrite_per_bundle_per_variant",
            "neighbor_component_encoding": (
                "int32_or_int64_bank_indices_losslessly_mapped_through_the_"
                "manifest_bound_train_sample_id_array"
            ),
            "full_build_policy": (
                "all_30_cache_identities_required_unless_compute_missing_cache_"
                "was_explicitly_requested"
            ),
        },
        "checkpoints": checkpoints,
        "oracle_population": oracle_population,
        "files": [],
        "destination_policy": "atomic_new_directory_no_overwrite",
        "scientific_gate": {
            "status": "NOT_RUN",
            "reason_codes": ["STAGE2_GATE_REDUCER_NOT_RUN"],
        },
    }
    return selection_manifest, rows, diagnostic_rows


def write_evidence_directory_no_overwrite(
    selection_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    diagnostic_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    destination = _preflight_evidence_destination(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )
    try:
        model_bytes = _jsonl_bytes(rows)
        model_record = {
            "logical_path": "model_metrics.jsonl",
            "sha256": _sha256_bytes(model_bytes),
            "size": len(model_bytes),
            "row_count": len(rows),
            "schema_version": MODEL_METRIC_SCHEMA,
        }
        diagnostic_bytes = _jsonl_bytes(diagnostic_rows)
        diagnostic_record = {
            "logical_path": "diagnostic_metrics.jsonl",
            "sha256": _sha256_bytes(diagnostic_bytes),
            "size": len(diagnostic_bytes),
            "row_count": len(diagnostic_rows),
            "schema_version": DIAGNOSTIC_METRIC_SCHEMA,
        }
        manifest = dict(selection_manifest)
        population = manifest.get("diagnostic_population")
        if isinstance(population, Mapping) and int(population["row_count"]) != len(
            diagnostic_rows
        ):
            raise AssertionError("diagnostic manifest/file row counts differ")
        manifest["files"] = [model_record, diagnostic_record]
        manifest_bytes = _canonical_bytes(manifest)
        (temporary / "model_metrics.jsonl").write_bytes(model_bytes)
        (temporary / "diagnostic_metrics.jsonl").write_bytes(diagnostic_bytes)
        (temporary / "selection_manifest.json").write_bytes(manifest_bytes)
        file_records = [
            {
                "logical_path": "model_metrics.jsonl",
                "sha256": model_record["sha256"],
                "size": len(model_bytes),
            },
            {
                "logical_path": "diagnostic_metrics.jsonl",
                "sha256": diagnostic_record["sha256"],
                "size": len(diagnostic_bytes),
            },
            {
                "logical_path": "selection_manifest.json",
                "sha256": _sha256_bytes(manifest_bytes),
                "size": len(manifest_bytes),
            },
        ]
        checksum_text = "".join(
            f"{record['sha256']}  {record['logical_path']}\n"
            for record in sorted(file_records, key=lambda row: row["logical_path"])
        )
        (temporary / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        # Recheck the requested path immediately before the atomic publish.  In
        # particular, never resolve a newly introduced dangling symlink and
        # publish into its target.
        if _preflight_evidence_destination(destination) != destination:
            raise AssertionError("evidence destination identity changed before publish")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "output_directory": str(destination),
        "selection_manifest_sha256": sha256_file(
            destination / "selection_manifest.json"
        ),
        "model_metrics_sha256": sha256_file(destination / "model_metrics.jsonl"),
        "model_metric_row_count": len(rows),
        "diagnostic_metrics_sha256": sha256_file(
            destination / "diagnostic_metrics.jsonl"
        ),
        "diagnostic_metric_row_count": len(diagnostic_rows),
        "scientific_gate": "NOT_RUN",
    }


def _absolute_without_symlink_resolution(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_real_directory_chain(path: Path, *, label: str) -> None:
    for component in reversed((path, *path.parents)):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} must already exist: {component}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} must not contain a symlink: {component}")
        if not stat.S_ISDIR(mode):
            raise ValueError(f"{label} must contain only directories: {component}")


def _preflight_evidence_destination(
    output_directory: str | Path,
    *,
    disjoint_roots: Sequence[tuple[str, str | Path]] = (),
) -> Path:
    """Validate one new evidence path without following its leaf or parents."""
    destination = _absolute_without_symlink_resolution(output_directory)
    if os.path.lexists(destination):
        raise FileExistsError("Stage-2 evidence output directory is no-overwrite")
    _assert_real_directory_chain(destination.parent, label="evidence output parent chain")

    repository = REPOSITORY_ROOT.resolve(strict=True)
    if _paths_overlap(destination, repository):
        raise ValueError("Stage-2 evidence output must be disjoint from the Git worktree")
    for root_label, root_value in disjoint_roots:
        root = Path(root_value).resolve(strict=False)
        if _paths_overlap(destination, root):
            raise ValueError(
                f"Stage-2 evidence output must be disjoint from {root_label}"
            )
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="required for the full 30-bundle evidence build",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="non-Git verified per-bundle/per-variant cache root",
    )
    parser.add_argument(
        "--parity-baseline-root",
        type=Path,
        help=(
            "full-build override containing exactly the five frozen v1.2 parity "
            "baseline files; forbidden for single-bundle cache precompute"
        ),
    )
    parser.add_argument(
        "--precompute-bundle-id",
        help=(
            "compute one manifest-selected bundle from a retrieval root containing "
            "exactly that bundle; emit no scientific evidence"
        ),
    )
    parser.add_argument(
        "--compute-missing-cache",
        action="store_true",
        help=(
            "full-build opt-in to compute absent cache variants; otherwise every "
            "one of the 30 x 7 verified caches is required"
        ),
    )
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--bank-chunk-size", type=int)
    parser.add_argument("--memory-budget-bytes", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.precompute_bundle_id is None and args.output_directory is None:
        raise ValueError("full build requires --output-directory")
    if args.precompute_bundle_id is not None and args.output_directory is not None:
        raise ValueError("single-bundle precompute does not write an evidence directory")
    if args.precompute_bundle_id is not None and args.parity_baseline_root is not None:
        raise ValueError("single-bundle precompute forbids --parity-baseline-root")
    evidence_destination: Path | None = None
    if args.output_directory is not None:
        # This must precede build_stage2_evidence: a bad destination must not
        # trigger any protected-array read or mutate the retrieval/cache trees.
        disjoint_roots: list[tuple[str, str | Path]] = [
            ("retrieval root", args.retrieval_root),
            ("cache root", args.cache_root),
            ("cache staging root", _cache_staging_root_path(args.cache_root)),
        ]
        if args.parity_baseline_root is not None:
            disjoint_roots.append(
                ("parity baseline root", args.parity_baseline_root)
            )
        evidence_destination = _preflight_evidence_destination(
            args.output_directory,
            disjoint_roots=disjoint_roots,
        )
        if args.precompute_bundle_id is None:
            reuse_document = json.loads(
                args.reuse_manifest.read_text(encoding="utf-8")
            )
            if not isinstance(reuse_document, Mapping):
                raise ValueError("reuse manifest must contain a JSON mapping")
            effective_baseline_root = _effective_parity_baseline_root(
                reuse_document, args.parity_baseline_root
            )
            evidence_destination = _preflight_evidence_destination(
                args.output_directory,
                disjoint_roots=(
                    *disjoint_roots,
                    ("effective parity baseline root", effective_baseline_root),
                ),
            )
    selection_manifest, rows, diagnostic_rows = build_stage2_evidence(
        reuse_manifest_path=args.reuse_manifest,
        retrieval_root=args.retrieval_root,
        cache_root=args.cache_root,
        parity_baseline_root=args.parity_baseline_root,
        precompute_bundle_id=args.precompute_bundle_id,
        compute_missing_cache=args.compute_missing_cache,
        query_chunk_size=args.query_chunk_size,
        bank_chunk_size=args.bank_chunk_size,
        memory_budget_bytes=args.memory_budget_bytes,
    )
    if args.precompute_bundle_id is not None:
        print(json.dumps(selection_manifest, sort_keys=True))
        return 0
    if evidence_destination is None:
        raise AssertionError("full build omitted its preflighted evidence destination")
    result = write_evidence_directory_no_overwrite(
        selection_manifest,
        rows,
        evidence_destination,
        diagnostic_rows=diagnostic_rows,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
