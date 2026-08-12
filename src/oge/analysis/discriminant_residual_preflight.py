"""CPU-only historical discriminant--residual and residual-tail preflight.

This module implements only the ID-cache measurements frozen by Card 13
Sections 7.4, 7.5, and 13.  It does not load checkpoints, score OOD samples,
or expose a complete RtMD detector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2, kstest, pearsonr, spearmanr

from oge.studies.artifacts import atomic_write_json, sha256_file


SCHEMA_VERSION = "discriminant_residual_preflight_v1"
CONTRACT_SOURCE = "Card 13 Sections 7.4, 7.5, and 13"
TRANSFORMS = ("raw", "l2")
NU_LOWER = 2.05
NU_UPPER = 1000.0
CONDITION_CEILING = 1.0e8
COMMON_RIDGE_SCALE = 1.0e-6
MATRIX_TOLERANCE = 1.0e-10
EXPECTED_REUSE_MANIFEST_SHA256 = (
    "f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc"
)

ID_REQUIRED_PATHS = (
    "artifact_manifest.json",
    "manifest.json",
    "classifier_weight.npy",
    "cifar10/train/class_labels.npy",
    "cifar10/train/features.npy",
    "cifar10/train/sample_ids.npy",
    "cifar10/test/class_labels.npy",
    "cifar10/test/features.npy",
    "cifar10/test/sample_ids.npy",
    "metrics/arrays/scores/id_test/detector__mahalanobis_raw.npy",
    "metrics/arrays/scores/id_test/detector__relative_mahalanobis_raw.npy",
    "metrics/arrays/scores/id_test/diagnostics/raw_gaussian/global_distances.npy",
    "metrics/arrays/scores/id_test/detector__mahalanobis_pp.npy",
    "metrics/arrays/scores/id_test/detector__relative_mahalanobis_pp.npy",
    "metrics/arrays/scores/id_test/diagnostics/normalized_gaussian/global_distances.npy",
)


class PreflightInputError(RuntimeError):
    """Raised when the allowlisted verified cache is absent or inconsistent."""


class HardCapExceeded(RuntimeError):
    """Raised before starting another unit of work after the wall-time cap."""


@dataclass(frozen=True)
class GeometryFit:
    mean: np.ndarray
    class_means: np.ndarray
    class_counts: np.ndarray
    within_covariance: np.ndarray
    between_covariance: np.ndarray
    total_covariance: np.ndarray
    within_precision: np.ndarray
    global_precision: np.ndarray
    within_sqrt: np.ndarray
    within_invsqrt: np.ndarray
    subspace_basis: np.ndarray
    transformed_class_means: np.ndarray
    parallel_global_precision: np.ndarray
    dim: int
    residual_dim: int
    condition_number: float
    tau_alg: float
    ridge: float
    applicable: bool
    numerical: dict[str, Any]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return (values + values.T) / 2.0


def _matrix_relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(
        1.0,
        float(np.linalg.norm(left, ord="fro")),
        float(np.linalg.norm(right, ord="fro")),
    )
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


def _contract_scaled_max(
    residual: np.ndarray, direct: np.ndarray, *components: np.ndarray
) -> float:
    """Apply Card 13's ``max(1, |direct|, sum_j |component_j|)`` scale."""

    component_sum = np.zeros_like(np.asarray(residual, dtype=np.float64))
    for component in components:
        component_sum += np.abs(np.asarray(component, dtype=np.float64))
    scale = np.maximum.reduce(
        (
            np.ones_like(component_sum),
            np.abs(np.asarray(direct, dtype=np.float64)),
            component_sum,
        )
    )
    return float(np.max(np.abs(residual) / scale))


def _pointwise_parity_max(residual: np.ndarray, *terms: np.ndarray) -> float:
    """Return a diagnostic direct-array parity residual, not a frozen gate."""

    scale = np.ones_like(np.asarray(residual, dtype=np.float64))
    for term in terms:
        scale = np.maximum(scale, np.abs(np.asarray(term, dtype=np.float64)))
    return float(np.max(np.abs(residual) / scale))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "infinity" if number > 0.0 else "-infinity"
        return number
    return value


def _json_line(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(_json_safe(record), sort_keys=True, allow_nan=False)
            )
            handle.write("\n")


def _parse_checksum_catalog(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, logical_path = line.split(maxsplit=1)
        logical_path = logical_path.lstrip("* ")
        output[logical_path] = digest
    return output


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightInputError(f"{path} must contain a JSON object")
    return value


def _transform_features(features: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise PreflightInputError("features must be a finite two-dimensional array")
    if transform == "raw":
        return values
    if transform != "l2":
        raise ValueError(f"unknown transform: {transform}")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise PreflightInputError("L2 transform encountered a zero or non-finite norm")
    return values / norms[:, None]


def deterministic_class_stratified_twofold(
    sample_ids: Any, labels: Any
) -> np.ndarray:
    """Assign stable IDs to two nearly equal folds within every class."""

    identities = np.asarray(sample_ids)
    targets = np.asarray(labels, dtype=np.int64)
    if identities.ndim != 1 or targets.ndim != 1 or identities.shape != targets.shape:
        raise ValueError("sample_ids and labels must be aligned one-dimensional arrays")
    if len(np.unique(identities)) != len(identities):
        raise ValueError("stable sample identities must be unique")
    folds = np.empty(len(targets), dtype=np.int8)
    for class_index in np.unique(targets):
        selected = np.flatnonzero(targets == class_index)
        keyed = sorted(
            selected,
            key=lambda index: hashlib.sha256(
                f"{SCHEMA_VERSION}:twofold:{identities[index]}".encode("utf-8")
            ).digest(),
        )
        folds[np.asarray(keyed, dtype=np.int64)] = np.arange(len(keyed)) % 2
    return folds


def gaussian_residual_nll(q: Any, k: int) -> np.ndarray:
    radii = np.asarray(q, dtype=np.float64)
    if k < 0 or np.any(radii < 0.0) or not np.isfinite(radii).all():
        raise ValueError("Gaussian residual likelihood requires finite q >= 0 and k >= 0")
    return 0.5 * (k * math.log(2.0 * math.pi) + radii)


def residual_t_nll(q: Any, k: int, nu: float) -> np.ndarray:
    """Full covariance-normalized multivariate-t negative log likelihood."""

    radii = np.asarray(q, dtype=np.float64)
    if k < 0 or np.any(radii < 0.0) or not np.isfinite(radii).all():
        raise ValueError("Residual-t likelihood requires finite q >= 0 and k >= 0")
    if not np.isfinite(nu) or not (2.0 < nu < math.inf):
        raise ValueError("finite nu must be greater than two")
    return (
        -gammaln((nu + k) / 2.0)
        + gammaln(nu / 2.0)
        + (k / 2.0) * math.log((nu - 2.0) * math.pi)
        + ((nu + k) / 2.0) * np.log1p(radii / (nu - 2.0))
    )


def rtmd_residual_penalty(q: Any, k: int, nu: float | None) -> np.ndarray:
    """Return only the frozen RtMD residual likelihood penalty ``g``."""

    radii = np.asarray(q, dtype=np.float64)
    if k < 0 or np.any(radii < 0.0) or not np.isfinite(radii).all():
        raise ValueError("RtMD penalty requires finite q >= 0 and k >= 0")
    if nu is None or math.isinf(float(nu)):
        return radii.copy()
    finite_nu = float(nu)
    if not (2.0 < finite_nu < math.inf):
        raise ValueError("finite nu must be greater than two")
    return (finite_nu + k) * np.log1p(radii / (finite_nu - 2.0))


def fit_residual_nu(q: Any, k: int) -> dict[str, Any]:
    """Fit the deterministic finite candidate and apply the frozen BIC fallback."""

    radii = np.asarray(q, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("q must be a non-empty one-dimensional array")
    if np.any(radii < 0.0) or not np.isfinite(radii).all() or k < 0:
        raise ValueError("q and k are outside the residual likelihood domain")
    gaussian_ll = -float(np.sum(gaussian_residual_nll(radii, k)))
    theta_lower = math.log(NU_LOWER - 2.0)
    theta_upper = math.log(NU_UPPER - 2.0)

    def objective(theta: float) -> float:
        nu = 2.0 + math.exp(float(theta))
        value = float(np.sum(residual_t_nll(radii, k, nu)))
        return value if np.isfinite(value) else math.inf

    result = minimize_scalar(
        objective,
        bounds=(theta_lower, theta_upper),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 500},
    )
    candidate_nu = 2.0 + math.exp(float(result.x))
    candidate_ll = -float(result.fun)
    lower_solution = float(result.x) <= theta_lower + 1.0e-8 * max(
        1.0, theta_upper - theta_lower
    )
    optimization_valid = bool(
        result.success
        and np.isfinite(candidate_nu)
        and np.isfinite(candidate_ll)
        and not lower_solution
    )
    likelihood_ratio = (
        2.0 * (candidate_ll - gaussian_ll) if optimization_valid else None
    )
    bic_threshold = math.log(len(radii))
    finite_selected = bool(
        optimization_valid
        and likelihood_ratio is not None
        and likelihood_ratio > bic_threshold
    )
    if not optimization_valid:
        fallback_reason = "optimization_failure_or_lower_bound"
    elif not finite_selected:
        fallback_reason = "bic_prefers_gaussian"
    else:
        fallback_reason = None
    return {
        "sample_count": int(len(radii)),
        "residual_dim": int(k),
        "finite_candidate_nu": float(candidate_nu),
        "finite_candidate_log_likelihood": float(candidate_ll),
        "gaussian_log_likelihood": float(gaussian_ll),
        "likelihood_ratio": likelihood_ratio,
        "bic_threshold_log_n": float(bic_threshold),
        "optimization_success": bool(result.success),
        "optimization_iterations": int(result.nfev),
        "lower_bound_solution": bool(lower_solution),
        "selected_model": "finite_t" if finite_selected else "gaussian_infinity",
        "selected_nu": float(candidate_nu) if finite_selected else "infinity",
        "fallback_reason": fallback_reason,
    }


def _eigen_record(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    maximum = float(np.max(values))
    tau_spec = maximum * matrix.shape[0] * np.finfo(np.float64).eps
    if np.any(values < -tau_spec):
        raise ValueError("covariance has an eigenvalue below -tau_spec")
    clipped = int(np.count_nonzero((values < 0.0) & (values >= -tau_spec)))
    values = np.maximum(values, 0.0)
    rank = int(np.count_nonzero(values > tau_spec))
    condition = (
        float(maximum / values[0])
        if rank == matrix.shape[0] and values[0] > 0.0
        else math.inf
    )
    return values, vectors, {
        "lambda_max": maximum,
        "tau_spec": float(tau_spec),
        "rank": rank,
        "dimension": int(matrix.shape[0]),
        "condition_number": condition if np.isfinite(condition) else "infinity",
        "clipped_negative_count": clipped,
    }


def fit_discriminant_geometry(
    features: Any, labels: Any, *, ridge: float = 0.0
) -> GeometryFit:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets):
        raise ValueError("features and labels must have aligned sample axes")
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("features must be finite and non-empty")
    classes = np.unique(targets)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("labels must be contiguous and zero based")
    counts = np.bincount(targets, minlength=len(classes)).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("every class must be populated")
    means = np.stack([np.mean(values[targets == c], axis=0) for c in classes])
    probabilities = counts.astype(np.float64) / len(values)
    mean = np.sum(probabilities[:, None] * means, axis=0)
    centered_means = means - mean
    residuals = values - means[targets]
    within_unridged = _symmetrize(residuals.T @ residuals / len(values))
    between = _symmetrize(
        (centered_means * probabilities[:, None]).T @ centered_means
    )
    centered = values - mean
    total = _symmetrize(centered.T @ centered / len(values))
    identity = np.eye(values.shape[1], dtype=np.float64)
    within = _symmetrize(within_unridged + float(ridge) * identity)
    global_covariance = _symmetrize(total + float(ridge) * identity)

    within_values, within_vectors, within_record = _eigen_record(within)
    global_values, global_vectors, global_record = _eigen_record(global_covariance)
    full_rank = (
        within_record["rank"] == values.shape[1]
        and global_record["rank"] == values.shape[1]
    )
    finite_conditions = [
        item
        for item in (
            within_record["condition_number"],
            global_record["condition_number"],
        )
        if isinstance(item, float)
    ]
    condition = max(finite_conditions) if len(finite_conditions) == 2 else math.inf
    applicable = bool(full_rank and condition <= CONDITION_CEILING)
    tau_alg = float(
        max(
            1.0e-10,
            10.0 * (condition if np.isfinite(condition) else CONDITION_CEILING)
            * np.finfo(np.float64).eps,
        )
    )

    positive_within = within_values > 0.0
    reciprocal = np.zeros_like(within_values)
    reciprocal_sqrt = np.zeros_like(within_values)
    sqrt_values = np.sqrt(within_values)
    reciprocal[positive_within] = 1.0 / within_values[positive_within]
    reciprocal_sqrt[positive_within] = 1.0 / np.sqrt(
        within_values[positive_within]
    )
    within_precision = _symmetrize(
        (within_vectors * reciprocal) @ within_vectors.T
    )
    within_invsqrt = _symmetrize(
        (within_vectors * reciprocal_sqrt) @ within_vectors.T
    )
    within_sqrt = _symmetrize(
        (within_vectors * sqrt_values) @ within_vectors.T
    )
    positive_global = global_values > 0.0
    global_reciprocal = np.zeros_like(global_values)
    global_reciprocal[positive_global] = 1.0 / global_values[positive_global]
    global_precision = _symmetrize(
        (global_vectors * global_reciprocal) @ global_vectors.T
    )

    transformed_means = centered_means @ within_invsqrt
    e_matrix = transformed_means.T * np.sqrt(probabilities)[None, :]
    if e_matrix.size:
        basis, singular_values, _ = np.linalg.svd(e_matrix, full_matrices=False)
        sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    else:
        basis = np.empty((values.shape[1], 0), dtype=np.float64)
        singular_values = np.empty(0, dtype=np.float64)
        sigma_max = 0.0
    tau_s = sigma_max * max(values.shape[1], len(classes)) * np.finfo(
        np.float64
    ).eps
    dim = int(np.count_nonzero(singular_values > tau_s)) if sigma_max else 0
    subspace_basis = basis[:, :dim]
    rank_sensitive = bool(
        tau_s > 0.0
        and np.any((singular_values >= tau_s / 10.0) & (singular_values <= 10.0 * tau_s))
    )
    projector = subspace_basis @ subspace_basis.T
    projector_symmetry = _matrix_relative_residual(projector, projector.T)
    projector_idempotence = _matrix_relative_residual(projector @ projector, projector)
    z_global_precision = _symmetrize(
        within_sqrt @ global_precision @ within_sqrt
    )
    between_whitened = _symmetrize(e_matrix @ e_matrix.T)
    between_parallel = _symmetrize(
        subspace_basis.T @ between_whitened @ subspace_basis
    )
    parallel_global_precision = _symmetrize(
        np.linalg.inv(np.eye(dim, dtype=np.float64) + between_parallel)
    )

    covariance_identity = _matrix_relative_residual(total, within_unridged + between)
    backend_parity: dict[str, Any]
    backward_within: float | None
    backward_global: float | None
    if full_rank:
        within_direct = np.linalg.inv(within)
        global_direct = np.linalg.inv(global_covariance)
        backend_parity = {
            "within": _matrix_relative_residual(within_precision, within_direct),
            "global": _matrix_relative_residual(global_precision, global_direct),
        }
        backward_within = float(
            np.linalg.norm(within @ within_precision - identity, ord=2)
            / max(
                1.0,
                float(np.linalg.norm(within, ord=2) * np.linalg.norm(within_precision, ord=2)),
            )
        )
        backward_global = float(
            np.linalg.norm(global_covariance @ global_precision - identity, ord=2)
            / max(
                1.0,
                float(
                    np.linalg.norm(global_covariance, ord=2)
                    * np.linalg.norm(global_precision, ord=2)
                ),
            )
        )
    else:
        backend_parity = {"within": None, "global": None}
        backward_within = None
        backward_global = None

    # Construct the Woodbury correction through its at-most-(K-1) support.
    # Direct subtraction of two ill-conditioned full-rank precisions creates
    # spurious small eigenvalues and is retained only as a reconstruction check.
    correction_parallel = _symmetrize(
        np.eye(dim, dtype=np.float64) - parallel_global_precision
    )
    curvature = _symmetrize(
        within_invsqrt
        @ subspace_basis
        @ correction_parallel
        @ subspace_basis.T
        @ within_invsqrt
    )
    curvature_direct = _symmetrize(within_precision - global_precision)
    woodbury_reconstruction = _matrix_relative_residual(
        curvature, curvature_direct
    )
    curvature_values = np.linalg.eigvalsh(curvature)
    curvature_max = float(np.max(curvature_values))
    curvature_tau = max(0.0, curvature_max) * values.shape[1] * np.finfo(
        np.float64
    ).eps
    curvature_rank = int(np.count_nonzero(curvature_values > curvature_tau))
    psd_residual = max(0.0, -float(np.min(curvature_values))) / max(
        1.0, abs(curvature_max)
    )

    checks = {
        "covariance_identity": covariance_identity <= MATRIX_TOLERANCE,
        "projector_symmetry": projector_symmetry <= MATRIX_TOLERANCE,
        "projector_idempotence": projector_idempotence <= MATRIX_TOLERANCE,
        "backend_parity": bool(
            full_rank
            and backend_parity["within"] is not None
            and backend_parity["within"] <= tau_alg
            and backend_parity["global"] is not None
            and backend_parity["global"] <= tau_alg
        ),
        "inverse_backward": bool(
            backward_within is not None
            and backward_within <= tau_alg
            and backward_global is not None
            and backward_global <= tau_alg
        ),
        "curvature_psd": bool(psd_residual <= tau_alg),
        "curvature_rank_bound": bool(curvature_rank <= len(classes) - 1),
        "woodbury_reconstruction": bool(woodbury_reconstruction <= tau_alg),
    }
    numerical = {
        "within_spectrum": within_record,
        "global_spectrum": global_record,
        "condition_ceiling": CONDITION_CEILING,
        "tau_alg": float(tau_alg),
        "covariance_identity_relative_frobenius": covariance_identity,
        "projector_symmetry_relative_frobenius": projector_symmetry,
        "projector_idempotence_relative_frobenius": projector_idempotence,
        "backend_inverse_parity_relative_frobenius": backend_parity,
        "inverse_backward_error": {
            "within": backward_within,
            "global": backward_global,
        },
        "subspace": {
            "tau_s": float(tau_s),
            "dim": dim,
            "residual_dim": int(values.shape[1] - dim),
            "rank_sensitive": rank_sensitive,
            "rank_bound": int(len(classes) - 1),
        },
        "curvature": {
            "observed_rank": curvature_rank,
            "rank_bound": int(len(classes) - 1),
            "psd_residual": psd_residual,
            "trace": float(np.trace(curvature)),
            "woodbury_reconstruction_relative_frobenius": woodbury_reconstruction,
        },
        "checks": checks,
    }
    return GeometryFit(
        mean=mean,
        class_means=means,
        class_counts=counts,
        within_covariance=within,
        between_covariance=between,
        total_covariance=total,
        within_precision=within_precision,
        global_precision=global_precision,
        within_sqrt=within_sqrt,
        within_invsqrt=within_invsqrt,
        subspace_basis=subspace_basis,
        transformed_class_means=transformed_means,
        parallel_global_precision=parallel_global_precision,
        dim=dim,
        residual_dim=int(values.shape[1] - dim),
        condition_number=condition,
        tau_alg=float(tau_alg),
        ridge=float(ridge),
        applicable=applicable,
        numerical=numerical,
    )


def residual_radius(fit: GeometryFit, queries: Any) -> np.ndarray:
    values = np.asarray(queries, dtype=np.float64)
    z = (values - fit.mean) @ fit.within_invsqrt
    parallel = z @ fit.subspace_basis
    q = np.sum(z * z, axis=1) - np.sum(parallel * parallel, axis=1)
    tolerance = fit.tau_alg * np.maximum(1.0, np.sum(z * z, axis=1))
    if np.any(q < -tolerance):
        raise ValueError("residual radius is negative beyond tau_alg")
    return np.maximum(q, 0.0)


def _class_minimum_distances(
    queries: np.ndarray, means: np.ndarray, precision: np.ndarray, *, chunk: int = 2048
) -> np.ndarray:
    output = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        delta = queries[start:stop, None, :] - means[None, :, :]
        distances = np.einsum(
            "ncd,de,nce->nc", delta, precision, delta, optimize=True
        )
        output[start:stop] = np.min(distances, axis=1)
    return output


def score_discriminant_components(
    fit: GeometryFit, queries: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values = np.asarray(queries, dtype=np.float64)
    centered = values - fit.mean
    z = centered @ fit.within_invsqrt
    parallel = z @ fit.subspace_basis
    q_perp = residual_radius(fit, values)
    eta_parallel = fit.transformed_class_means @ fit.subspace_basis
    q_parallel_classes = (
        np.sum(parallel * parallel, axis=1)[:, None]
        + np.sum(eta_parallel * eta_parallel, axis=1)[None, :]
        - 2.0 * parallel @ eta_parallel.T
    )
    q_parallel_class = np.min(q_parallel_classes, axis=1)
    q_parallel_global = np.einsum(
        "nr,rs,ns->n",
        parallel,
        fit.parallel_global_precision,
        parallel,
        optimize=True,
    )
    direct_class = _class_minimum_distances(values, fit.class_means, fit.within_precision)
    direct_global = np.einsum(
        "nd,de,ne->n", centered, fit.global_precision, centered, optimize=True
    )
    s_md = -direct_class
    s_marginal = -direct_global
    s_rmd_direct = direct_global - direct_class
    s_perp = -q_perp
    s_parallel_marginal = -q_parallel_global
    s_rmd = -q_parallel_class + q_parallel_global
    reconstructed_md = s_perp + s_parallel_marginal + s_rmd
    components = np.column_stack((s_perp, s_parallel_marginal, s_rmd))
    covariance = np.cov(components, rowvar=False, bias=True)
    direct_variance = float(np.var(s_md, ddof=0))
    covariance_variance = float(np.ones(3) @ covariance @ np.ones(3))
    covariance_scale = max(1.0, abs(direct_variance), abs(covariance_variance))
    covariance_residual = abs(direct_variance - covariance_variance) / covariance_scale
    score_residual = _contract_scaled_max(
        s_md - reconstructed_md,
        s_md,
        s_perp,
        s_parallel_marginal,
        s_rmd,
    )
    cancellation_residual = _contract_scaled_max(
        s_rmd_direct - s_rmd, s_rmd_direct, s_rmd
    )
    marginal_residual = _contract_scaled_max(
        s_marginal - (s_perp + s_parallel_marginal),
        s_marginal,
        s_perp,
        s_parallel_marginal,
    )
    if len(s_md) > 1:
        direct_margin = np.diff(s_md)
        component_margins = np.diff(components, axis=0)
        reconstructed_margin = np.sum(component_margins, axis=1)
        margin_residual = _contract_scaled_max(
            direct_margin - reconstructed_margin,
            direct_margin,
            *[component_margins[:, index] for index in range(component_margins.shape[1])],
        )
    else:
        margin_residual = 0.0
    record = {
        "sample_count": int(len(values)),
        "score_reconstruction_scaled_max": score_residual,
        "marginal_reconstruction_scaled_max": marginal_residual,
        "rmd_cancellation_scaled_max": cancellation_residual,
        "id_adjacent_pair_margin_reconstruction_scaled_max": margin_residual,
        "component_variances": np.diag(covariance).tolist(),
        "component_covariance_biased": covariance.tolist(),
        "direct_md_variance": direct_variance,
        "covariance_reconstructed_variance": covariance_variance,
        "covariance_variance_relative_residual": covariance_residual,
        "checks": {
            "score_reconstruction": score_residual <= fit.tau_alg,
            "marginal_reconstruction": marginal_residual <= fit.tau_alg,
            "rmd_cancellation": cancellation_residual <= fit.tau_alg,
            "pair_margin_reconstruction": margin_residual <= fit.tau_alg,
            "component_covariance": covariance_residual <= MATRIX_TOLERANCE,
        },
    }
    return record, {
        "md": s_md,
        "marginal": s_marginal,
        "rmd": s_rmd_direct,
        "components": components,
        "q_perp": q_perp,
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if np.var(x) == 0.0 or np.var(y) == 0.0:
        return {"spearman": None, "pearson": None}
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
    }


def classifier_alignment(fit: GeometryFit, classifier_weight: Any) -> dict[str, Any]:
    weights = np.asarray(classifier_weight, dtype=np.float64)
    centered = weights - np.mean(weights, axis=0, keepdims=True)
    rows = centered @ fit.within_sqrt
    _, singular_values, right = np.linalg.svd(rows, full_matrices=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    tau = sigma_max * max(rows.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tau)) if sigma_max else 0
    if rank == 0:
        return {"classifier_rank": 0, "angles_degrees": None, "overlap_fraction": None}
    row_basis = right[:rank].T
    cosines = np.linalg.svd(
        fit.subspace_basis.T @ row_basis, compute_uv=False
    )
    cosines = np.clip(cosines, 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    overlap = float(
        np.linalg.norm(fit.subspace_basis.T @ row_basis, ord="fro") ** 2 / rank
    )
    return {
        "classifier_rank": rank,
        "angles_degrees": angles.tolist(),
        "overlap_fraction": overlap,
    }


def _cached_score_paths(transform: str) -> dict[str, str]:
    if transform == "raw":
        return {
            "md": "metrics/arrays/scores/id_test/detector__mahalanobis_raw.npy",
            "rmd": "metrics/arrays/scores/id_test/detector__relative_mahalanobis_raw.npy",
            "global_distance": (
                "metrics/arrays/scores/id_test/diagnostics/raw_gaussian/global_distances.npy"
            ),
        }
    return {
        "md": "metrics/arrays/scores/id_test/detector__mahalanobis_pp.npy",
        "rmd": "metrics/arrays/scores/id_test/detector__relative_mahalanobis_pp.npy",
        "global_distance": (
            "metrics/arrays/scores/id_test/diagnostics/normalized_gaussian/global_distances.npy"
        ),
    }


def _cached_score_parity(
    bundle_root: Path,
    transform: str,
    arrays: Mapping[str, np.ndarray],
    tau_alg: float,
) -> dict[str, Any]:
    paths = _cached_score_paths(transform)
    cached_md = np.load(bundle_root / paths["md"], allow_pickle=False).astype(np.float64)
    cached_rmd = np.load(bundle_root / paths["rmd"], allow_pickle=False).astype(np.float64)
    cached_global = np.load(
        bundle_root / paths["global_distance"], allow_pickle=False
    ).astype(np.float64)
    residuals = {
        "md": _pointwise_parity_max(
            arrays["md"] - cached_md, arrays["md"], cached_md
        ),
        "rmd": _pointwise_parity_max(
            arrays["rmd"] - cached_rmd, arrays["rmd"], cached_rmd
        ),
        "global_distance": _pointwise_parity_max(
            -arrays["marginal"] - cached_global,
            -arrays["marginal"],
            cached_global,
        ),
    }
    return {
        "scaled_max_residuals": residuals,
        "pass": all(value <= tau_alg for value in residuals.values()),
    }


def _heldout_tail_record(q: np.ndarray, k: int, nu_fit: Mapping[str, Any]) -> dict[str, Any]:
    gaussian_ll = -float(np.sum(gaussian_residual_nll(q, k)))
    candidate_nu = float(nu_fit["finite_candidate_nu"])
    candidate_ll = -float(np.sum(residual_t_nll(q, k, candidate_nu)))
    selected_gain = (
        candidate_ll - gaussian_ll
        if nu_fit["selected_model"] == "finite_t"
        else 0.0
    )
    quantiles = (0.95, 0.99, 0.999)
    exceedance = {
        str(probability): float(np.mean(q > chi2.ppf(probability, k)))
        for probability in quantiles
    }
    ks = kstest(q, "chi2", args=(k,))
    return {
        "sample_count": int(len(q)),
        "gaussian_log_likelihood": gaussian_ll,
        "finite_candidate_log_likelihood": candidate_ll,
        "finite_candidate_gain_per_sample": float(
            (candidate_ll - gaussian_ll) / len(q)
        ),
        "selected_gain_per_sample": float(selected_gain / len(q)),
        "mean_q": float(np.mean(q)),
        "mean_to_k_ratio": None if k == 0 else float(np.mean(q) / k),
        "variance_q": float(np.var(q, ddof=0)),
        "variance_to_chi2_ratio": None if k == 0 else float(np.var(q, ddof=0) / (2.0 * k)),
        "chi2_upper_tail_exceedance": exceedance,
        "chi2_ks_statistic": float(ks.statistic),
        "chi2_ks_pvalue": float(ks.pvalue),
    }


def _time_check(started: float, hard_cap_seconds: float) -> None:
    if time.monotonic() - started >= hard_cap_seconds:
        raise HardCapExceeded("four-hour preflight hard cap reached")


def verify_input_cache(
    manifest: Mapping[str, Any], cache_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if manifest.get("schema_version") != "discriminant_residual_preflight_inputs_v1":
        raise PreflightInputError("unexpected input manifest schema")
    if manifest.get("source_reuse_manifest_sha256") != EXPECTED_REUSE_MANIFEST_SHA256:
        raise PreflightInputError("source reuse manifest SHA-256 does not match")
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != manifest.get("expected_bundle_count"):
        raise PreflightInputError("input manifest bundle count is inconsistent")
    source_records: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = str(bundle["bundle_id"])
        root = cache_root / bundle_id
        if not root.is_dir():
            raise PreflightInputError(f"required bundle is absent: {bundle_id}")
        catalog_path = root / "bundle_checksums.sha256"
        if not catalog_path.is_file():
            raise PreflightInputError(f"bundle checksum catalog is absent: {bundle_id}")
        if sha256_file(catalog_path) != bundle["bundle_checksums_sha256"]:
            raise PreflightInputError(f"bundle checksum catalog SHA mismatch: {bundle_id}")
        catalog = _parse_checksum_catalog(catalog_path)
        for logical_path in ID_REQUIRED_PATHS:
            path = root / logical_path
            expected = catalog.get(logical_path)
            if expected is None or not path.is_file():
                raise PreflightInputError(
                    f"required allowlisted cache file is absent: {bundle_id}/{logical_path}"
                )
            observed = sha256_file(path)
            if observed != expected:
                raise PreflightInputError(
                    f"cache file SHA mismatch: {bundle_id}/{logical_path}"
                )
            source_records.append(
                {
                    "bundle_id": bundle_id,
                    "logical_path": logical_path,
                    "sha256": observed,
                    "size": path.stat().st_size,
                    "scope": "id_only_allowlist",
                }
            )
        artifact_manifest = _load_json(root / "artifact_manifest.json")
        raw_manifest = _load_json(root / "manifest.json")
        identity = raw_manifest.get("checkpoint", {})
        if (
            artifact_manifest.get("checkpoint_sha256") != bundle_id
            or identity.get("sha256") != bundle_id
            or identity.get("role") != "last"
            or artifact_manifest.get("checkpoint_role") != "last"
            or artifact_manifest.get("config_hash") != bundle["config_hash"]
            or identity.get("config_hash") != bundle["config_hash"]
            or artifact_manifest.get("training_seed") != bundle["training_seed"]
        ):
            raise PreflightInputError(f"bundle identity mismatch: {bundle_id}")
        verified.append({**bundle, "root": root})
    return verified, source_records


def _bundle_transform(
    bundle: Mapping[str, Any], transform: str
) -> tuple[dict[str, Any], dict[str, Any], GeometryFit]:
    root = Path(bundle["root"])
    train_raw = np.load(root / "cifar10/train/features.npy", mmap_mode="r")
    test_raw = np.load(root / "cifar10/test/features.npy", mmap_mode="r")
    labels = np.load(root / "cifar10/train/class_labels.npy", allow_pickle=False)
    sample_ids = np.load(root / "cifar10/train/sample_ids.npy", allow_pickle=False)
    classifier_weight = np.load(root / "classifier_weight.npy", allow_pickle=False)
    train = _transform_features(train_raw, transform)
    test = _transform_features(test_raw, transform)
    fit = fit_discriminant_geometry(train, labels)
    train_record, train_arrays = score_discriminant_components(fit, train)
    test_record, test_arrays = score_discriminant_components(fit, test)
    train_energy_residual = abs(float(np.mean(train_arrays["q_perp"])) - fit.residual_dim) / max(
        1.0, abs(float(np.mean(train_arrays["q_perp"]))), float(fit.residual_dim)
    )
    cached_parity = _cached_score_parity(root, transform, test_arrays, fit.tau_alg)
    raw_norm_correlations = None
    if transform == "raw":
        train_norms = np.linalg.norm(np.asarray(train_raw, dtype=np.float64), axis=1)
        test_norms = np.linalg.norm(np.asarray(test_raw, dtype=np.float64), axis=1)
        raw_norm_correlations = {
            "id_train": {
                name: _correlation(
                    train_norms, train_arrays["components"][:, index]
                )
                for index, name in enumerate(
                    ("s_perp", "s_parallel_marginal", "s_rmd")
                )
            },
            "id_test": {
                name: _correlation(test_norms, test_arrays["components"][:, index])
                for index, name in enumerate(
                    ("s_perp", "s_parallel_marginal", "s_rmd")
                )
            },
        }
    base_checks = fit.numerical["checks"]
    score_checks = list(train_record["checks"].values()) + list(
        test_record["checks"].values()
    )
    required_numerical_pass = bool(
        fit.applicable
        and all(base_checks.values())
        and all(score_checks)
        and train_energy_residual <= fit.tau_alg
    )

    within_lambda_max = float(fit.numerical["within_spectrum"]["lambda_max"])
    common_ridge = COMMON_RIDGE_SCALE * within_lambda_max
    if within_lambda_max <= 0.0:
        common_ridge_record = {
            "name": "discriminant_residual_common_ridge_lmax_1e-6",
            "status": "FAILED",
            "reason": "nonpositive_within_lambda_max",
        }
    else:
        ridge_fit = fit_discriminant_geometry(train, labels, ridge=common_ridge)
        ridge_train_record, _ = score_discriminant_components(ridge_fit, train)
        ridge_test_record, _ = score_discriminant_components(ridge_fit, test)
        common_ridge_pass = bool(
            ridge_fit.applicable
            and all(ridge_fit.numerical["checks"].values())
            and all(ridge_train_record["checks"].values())
            and all(ridge_test_record["checks"].values())
        )
        common_ridge_record = {
            "name": "discriminant_residual_common_ridge_lmax_1e-6",
            "status": "PASS" if common_ridge_pass else "FAILED",
            "lambda": common_ridge,
            "numerical": ridge_fit.numerical,
            "id_train": ridge_train_record,
            "id_test": ridge_test_record,
        }

    folds = deterministic_class_stratified_twofold(sample_ids, labels)
    oof_q = np.empty(len(train), dtype=np.float64)
    fold_records = []
    fold_dimensions = []
    fold_failure_reasons = []
    expected_classes = np.arange(len(fit.class_counts), dtype=np.int64)
    for heldout_fold in (0, 1):
        fit_mask = folds != heldout_fold
        query_mask = folds == heldout_fold
        observed_classes = np.unique(labels[fit_mask])
        if not np.array_equal(observed_classes, expected_classes):
            fold_failure_reasons.append(
                f"fold_{heldout_fold}_fit_does_not_contain_every_class"
            )
            fold_records.append(
                {
                    "heldout_fold": heldout_fold,
                    "fit_count": int(np.count_nonzero(fit_mask)),
                    "query_count": int(np.count_nonzero(query_mask)),
                    "expected_class_count": int(len(expected_classes)),
                    "observed_class_count": int(len(observed_classes)),
                    "status": "INAPPLICABLE",
                    "reason": "fit_fold_does_not_contain_every_class",
                }
            )
            fold_dimensions.append(None)
            continue
        fold_fit = fit_discriminant_geometry(train[fit_mask], labels[fit_mask])
        fold_dimensions.append(fold_fit.residual_dim)
        fold_pass = bool(
            fold_fit.applicable and all(fold_fit.numerical["checks"].values())
        )
        if fold_pass:
            oof_q[query_mask] = residual_radius(fold_fit, train[query_mask])
        else:
            fold_failure_reasons.append(
                f"fold_{heldout_fold}_geometry_not_numerically_applicable"
            )
        fold_records.append(
            {
                "heldout_fold": heldout_fold,
                "fit_count": int(np.count_nonzero(fit_mask)),
                "query_count": int(np.count_nonzero(query_mask)),
                "expected_class_count": int(len(expected_classes)),
                "observed_class_count": int(len(observed_classes)),
                "residual_dim": fold_fit.residual_dim,
                "condition_number": fold_fit.condition_number,
                "tau_alg": fold_fit.tau_alg,
                "status": "PASS" if fold_pass else "INAPPLICABLE",
                "checks": fold_fit.numerical["checks"],
            }
        )
    if any(dimension != fit.residual_dim for dimension in fold_dimensions):
        fold_failure_reasons.append("two_fold_and_final_residual_dimensions_differ")
    if fold_failure_reasons:
        nu_fit = None
        heldout_tail = None
        tail_status = "INCONCLUSIVE_NUMERICAL"
    else:
        nu_fit = fit_residual_nu(oof_q, fit.residual_dim)
        heldout_tail = _heldout_tail_record(
            test_arrays["q_perp"], fit.residual_dim, nu_fit
        )
        tail_status = "PASS"
    tail_record = {
        "bundle_id": bundle["bundle_id"],
        "config_hash": bundle["config_hash"],
        "optimizer_family": bundle["optimizer_family"],
        "training_seed": bundle["training_seed"],
        "transform": transform,
        "status": tail_status,
        "failure_reasons": fold_failure_reasons,
        "partition": {
            "method": "sha256_stable_id_rank_alternation_within_class",
            "fold_counts": [int(np.count_nonzero(folds == value)) for value in (0, 1)],
            "folds": fold_records,
        },
        "nu_fit": nu_fit,
        "heldout_id_test": heldout_tail,
    }
    record = {
        "bundle_id": bundle["bundle_id"],
        "config_hash": bundle["config_hash"],
        "optimizer_family": bundle["optimizer_family"],
        "training_seed": bundle["training_seed"],
        "transform": transform,
        "status": "PASS" if required_numerical_pass else "INAPPLICABLE",
        "applicable": fit.applicable,
        "required_numerical_pass": required_numerical_pass,
        "primary_channel": "S-perp" if required_numerical_pass else "none",
        "dim_s": fit.dim,
        "dim_s_perp": fit.residual_dim,
        "rank_sensitive": fit.numerical["subspace"]["rank_sensitive"],
        "numerical": fit.numerical,
        "id_train": train_record,
        "id_test": test_record,
        "id_train_residual_energy": {
            "observed_mean": float(np.mean(train_arrays["q_perp"])),
            "target": fit.residual_dim,
            "relative_residual": train_energy_residual,
            "pass": train_energy_residual <= fit.tau_alg,
        },
        "cached_metric_v1_2_id_test_parity": cached_parity,
        "raw_norm_component_correlations": raw_norm_correlations,
        "classifier_alignment": classifier_alignment(fit, classifier_weight),
        "common_ridge_diagnostic": common_ridge_record,
    }
    return record, tail_record, fit


def _aligned_pair_record(
    pair: Mapping[str, Any],
    transform: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fits: Mapping[tuple[str, str], GeometryFit],
) -> dict[str, Any]:
    left_key = (left["bundle_id"], transform)
    right_key = (right["bundle_id"], transform)
    if left_key not in fits or right_key not in fits:
        return {
            "pair_id": pair["pair_id"],
            "left_bundle_id": left["bundle_id"],
            "right_bundle_id": right["bundle_id"],
            "transform": transform,
            "status": "INAPPLICABLE",
            "reason": "one_or_both_primary_geometries_were_not_constructed",
            "pass_threshold": None,
        }
    left_root = Path(left["root"])
    right_root = Path(right["root"])
    left_ids = np.load(left_root / "cifar10/train/sample_ids.npy", allow_pickle=False)
    right_ids = np.load(right_root / "cifar10/train/sample_ids.npy", allow_pickle=False)
    if not np.array_equal(left_ids, right_ids):
        raise PreflightInputError(f"pair sample identities differ: {pair['pair_id']}")
    left_values = _transform_features(
        np.load(left_root / "cifar10/train/features.npy", mmap_mode="r"), transform
    )
    right_values = _transform_features(
        np.load(right_root / "cifar10/train/features.npy", mmap_mode="r"), transform
    )
    left_fit = fits[left_key]
    right_fit = fits[right_key]
    if not left_fit.applicable or not right_fit.applicable:
        return {
            "pair_id": pair["pair_id"],
            "left_bundle_id": left["bundle_id"],
            "right_bundle_id": right["bundle_id"],
            "transform": transform,
            "status": "INAPPLICABLE",
            "reason": "one_or_both_primary_geometries_exceed_the_frozen_applicability_gate",
            "pass_threshold": None,
        }
    cross = np.zeros((left_values.shape[1], left_values.shape[1]), dtype=np.float64)
    left_norm = 0.0
    right_norm = 0.0
    for start in range(0, len(left_values), 2048):
        stop = min(start + 2048, len(left_values))
        z_left = (left_values[start:stop] - left_fit.mean) @ left_fit.within_invsqrt
        z_right = (right_values[start:stop] - right_fit.mean) @ right_fit.within_invsqrt
        cross += z_right.T @ z_left
        left_norm += float(np.sum(z_left * z_left))
        right_norm += float(np.sum(z_right * z_right))
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    residual_sum = 0.0
    for start in range(0, len(left_values), 2048):
        stop = min(start + 2048, len(left_values))
        z_left = (left_values[start:stop] - left_fit.mean) @ left_fit.within_invsqrt
        z_right = (right_values[start:stop] - right_fit.mean) @ right_fit.within_invsqrt
        difference = z_right @ rotation - z_left
        residual_sum += float(np.sum(difference * difference))
    aligned_right_basis = rotation.T @ right_fit.subspace_basis
    overlap = left_fit.subspace_basis.T @ aligned_right_basis
    cosines = np.clip(np.linalg.svd(overlap, compute_uv=False), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    chordal = math.sqrt(
        max(
            0.0,
            left_fit.dim
            + right_fit.dim
            - 2.0 * float(np.linalg.norm(overlap, ord="fro") ** 2),
        )
    )
    return {
        "pair_id": pair["pair_id"],
        "left_bundle_id": left["bundle_id"],
        "right_bundle_id": right["bundle_id"],
        "transform": transform,
        "status": "PASS",
        "alignment_scope": "id_train_only_orthogonal_procrustes",
        "normalized_alignment_residual": float(
            math.sqrt(residual_sum / max(1.0, left_norm, right_norm))
        ),
        "left_rank": left_fit.dim,
        "right_rank": right_fit.dim,
        "rank_gap": int(left_fit.dim - right_fit.dim),
        "principal_angles_degrees": angles.tolist(),
        "chordal_projector_distance": chordal,
        "pass_threshold": None,
    }


def _gate2_summary(tail_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_transform: dict[str, Any] = {}
    reason_codes = ["aggregate_gate2_decision_rule_not_predeclared"]

    def distribution(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "minimum": min(values) if values else None,
            "median": float(np.median(values)) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "maximum": max(values) if values else None,
        }

    for transform in TRANSFORMS:
        selected = [row for row in tail_records if row["transform"] == transform]
        complete = [row for row in selected if row["status"] == "PASS"]
        finite = [
            row
            for row in complete
            if row["nu_fit"]["selected_model"] == "finite_t"
        ]
        finite_nus = [float(row["nu_fit"]["selected_nu"]) for row in finite]
        heldout_gains = [
            float(row["heldout_id_test"]["selected_gain_per_sample"])
            for row in complete
        ]
        ks_statistics = [
            float(row["heldout_id_test"]["chi2_ks_statistic"])
            for row in complete
        ]
        variance_ratios = [
            float(row["heldout_id_test"]["variance_to_chi2_ratio"])
            for row in complete
            if row["heldout_id_test"]["variance_to_chi2_ratio"] is not None
        ]
        upper_99_exceedance = [
            float(row["heldout_id_test"]["chi2_upper_tail_exceedance"]["0.99"])
            for row in complete
        ]
        by_transform[transform] = {
            "fit_count": len(selected),
            "complete_fit_count": len(complete),
            "inconclusive_fit_count": len(selected) - len(complete),
            "finite_t_selected_count": len(finite),
            "gaussian_fallback_count": len(complete) - len(finite),
            "selected_finite_nu": distribution(finite_nus),
            "selected_heldout_log_likelihood_gain_per_sample": distribution(
                heldout_gains
            ),
            "heldout_chi2_ks_statistic": distribution(ks_statistics),
            "heldout_variance_to_chi2_ratio": distribution(variance_ratios),
            "heldout_chi2_upper_0_99_exceedance": distribution(
                upper_99_exceedance
            ),
        }
        if len(complete) != len(selected):
            reason_codes.append(f"{transform}_required_numerical_fits_incomplete")
    return {
        "status": "DESCRIPTIVE_ONLY",
        "gate_status": "INCONCLUSIVE",
        "reason_codes": reason_codes,
        "aggregate_decision_rule": "NOT_PREDECLARED_IN_CARD_13",
        "by_transform": by_transform,
        "claim_boundary": (
            "historical mixed-recipe ID-only plausibility; cannot activate RtMD, "
            "predict OOD performance, or support a causal claim"
        ),
    }


def _write_payload_checksums(output_dir: Path, names: Iterable[str]) -> Path:
    path = output_dir / "checksums.sha256"
    lines = [f"{sha256_file(output_dir / name)}  {name}" for name in sorted(names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_preflight(
    *,
    input_manifest_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    hard_cap_seconds: float = 4.0 * 60.0 * 60.0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the bounded preflight and publish only compact, ID-only records."""

    started = time.monotonic()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(input_manifest_path)
    try:
        manifest = _load_json(manifest_path)
        verified, source_records = verify_input_cache(manifest, Path(cache_root))
        _json_line(destination / "input_manifest.jsonl", source_records)
        if progress_callback is not None:
            progress_callback(
                f"verified {len(verified)} bundles and {len(source_records)} ID-only files"
            )
        bundle_records: list[dict[str, Any]] = []
        tail_records: list[dict[str, Any]] = []
        fits: dict[tuple[str, str], GeometryFit] = {}
        for bundle in verified:
            for transform in TRANSFORMS:
                _time_check(started, hard_cap_seconds)
                try:
                    record, tail_record, fit = _bundle_transform(bundle, transform)
                except ValueError as error:
                    record = {
                        "bundle_id": bundle["bundle_id"],
                        "config_hash": bundle["config_hash"],
                        "optimizer_family": bundle["optimizer_family"],
                        "training_seed": bundle["training_seed"],
                        "transform": transform,
                        "status": "INCONCLUSIVE_NUMERICAL",
                        "applicable": False,
                        "required_numerical_pass": False,
                        "primary_channel": "none",
                        "reason": str(error),
                    }
                    tail_record = {
                        "bundle_id": bundle["bundle_id"],
                        "config_hash": bundle["config_hash"],
                        "optimizer_family": bundle["optimizer_family"],
                        "training_seed": bundle["training_seed"],
                        "transform": transform,
                        "status": "INCONCLUSIVE_NUMERICAL",
                        "failure_reasons": [str(error)],
                        "nu_fit": None,
                        "heldout_id_test": None,
                    }
                    fit = None
                bundle_records.append(record)
                tail_records.append(tail_record)
                if fit is not None:
                    fits[(bundle["bundle_id"], transform)] = fit
                if progress_callback is not None:
                    progress_callback(
                        f"completed fit {len(bundle_records)}/{len(verified) * len(TRANSFORMS)} "
                        f"{bundle['bundle_id'][:12]} {transform}"
                    )
        bundle_by_id = {bundle["bundle_id"]: bundle for bundle in verified}
        pair_records = []
        for pair in manifest.get("pairs", []):
            for transform in TRANSFORMS:
                _time_check(started, hard_cap_seconds)
                pair_records.append(
                    _aligned_pair_record(
                        pair,
                        transform,
                        bundle_by_id[pair["left_bundle_id"]],
                        bundle_by_id[pair["right_bundle_id"]],
                        fits,
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        f"completed pair {len(pair_records)}/"
                        f"{len(manifest.get('pairs', [])) * len(TRANSFORMS)} "
                        f"{pair['pair_id']} {transform}"
                    )
        _json_line(destination / "bundle_records.jsonl", bundle_records)
        _json_line(destination / "tail_records.jsonl", tail_records)
        _json_line(destination / "pair_records.jsonl", pair_records)
        required_numerical_pass_count = sum(
            row["required_numerical_pass"] for row in bundle_records
        )
        gate2 = _gate2_summary(tail_records)
        primary_channel_counts = {
            "S-perp": sum(
                row["primary_channel"] == "S-perp" for row in bundle_records
            ),
            "none": sum(row["primary_channel"] == "none" for row in bundle_records),
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "contract_source": CONTRACT_SOURCE,
            "status": "PASS",
            "issue": 73,
            "analysis_role": "historical_mixed_recipe_id_only_discovery",
            "source_reuse_manifest_sha256": manifest["source_reuse_manifest_sha256"],
            "bundle_count": len(verified),
            "transform_count": len(TRANSFORMS),
            "fit_count": len(bundle_records),
            "required_numerical_pass_count": required_numerical_pass_count,
            "primary_channel_scope": "per_fit_only",
            "primary_channel_counts": primary_channel_counts,
            "gate_2": gate2,
            "gates": {
                "gate_1": "NARROW_PASS_PREEXISTING",
                "gate_2": gate2["gate_status"],
                "gate_3": "NOT_RUN",
                "gate_4": "NOT_RUN",
                "gate_5": "NOT_RUN",
            },
            "pair_comparison_count": len(pair_records),
            "zero_common_frame": {
                "status": "NOT_AVAILABLE",
                "reason": manifest.get("zero_reference_reason"),
                "fabricated_substitute_used": False,
            },
            "not_run": [
                "GPU",
                "checkpoint reevaluation or inference",
                "protected OOD access or evaluation",
                "fresh training",
                "full RtMD detector or OOD evaluation",
                "RtMD AUROC, FPR95, or churn evaluation",
                "historical Raw-MD OOD gap concentration",
            ],
            "hard_cap_seconds": float(hard_cap_seconds),
            "wall_seconds": float(time.monotonic() - started),
            "payload_files": [
                "bundle_records.jsonl",
                "input_manifest.jsonl",
                "pair_records.jsonl",
                "tail_records.jsonl",
            ],
        }
        atomic_write_json(destination / "summary.json", summary)
        payload_names = [*summary["payload_files"], "summary.json"]
        _write_payload_checksums(destination, payload_names)
        return summary
    except (PreflightInputError, HardCapExceeded, ValueError) as error:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "contract_source": CONTRACT_SOURCE,
            "status": "INCONCLUSIVE",
            "issue": 73,
            "reason": str(error),
            "gate_2": {"gate_status": "INCONCLUSIVE"},
            "gates": {
                "gate_1": "NARROW_PASS_PREEXISTING",
                "gate_2": "INCONCLUSIVE",
                "gate_3": "NOT_RUN",
                "gate_4": "NOT_RUN",
                "gate_5": "NOT_RUN",
            },
            "wall_seconds": float(time.monotonic() - started),
        }
        atomic_write_json(destination / "summary.json", summary)
        _write_payload_checksums(destination, ["summary.json"])
        return summary
