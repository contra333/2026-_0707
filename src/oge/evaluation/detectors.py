"""Feature detector fits and scores frozen by Metric Contract v1.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .common import EPSILON_NORM, MetricStatus, finite_float64, finite_labels, normalize_rows


def _class_statistics(
    features: Any,
    labels: Any,
    *,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    values = finite_float64(features, name="features", ndim=2)
    if num_classes is None:
        num_classes = int(np.max(np.asarray(labels))) + 1
    targets = finite_labels(labels, count=len(values), num_classes=num_classes)
    means = np.empty((num_classes, values.shape[1]), dtype=np.float64)
    counts = np.empty(num_classes, dtype=np.int64)
    for class_index in range(num_classes):
        selected = values[targets == class_index]
        if len(selected) == 0:
            raise ValueError(f"class {class_index} is empty")
        counts[class_index] = len(selected)
        means[class_index] = np.mean(selected, axis=0)
    return values, targets, means, num_classes


def _quadratic_distances(
    queries: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    delta = queries[:, None, :] - means[None, :, :]
    distances = np.einsum("ncp,pq,ncq->nc", delta, precision, delta, optimize=True)
    if not np.isfinite(distances).all():
        raise ValueError("quadratic distances must be finite")
    return distances


@dataclass(frozen=True)
class TiedGaussianFit:
    class_means: np.ndarray
    class_counts: np.ndarray
    within_covariance: np.ndarray
    within_precision: np.ndarray
    global_mean: np.ndarray
    global_covariance: np.ndarray
    global_precision: np.ndarray
    normalized: bool
    normalization_status: MetricStatus
    normalization_reasons: tuple[str, ...]
    normalization_diagnostics: dict[str, Any]


def fit_tied_gaussians(
    features: Any,
    labels: Any,
    *,
    num_classes: int | None = None,
    normalize: bool = False,
) -> TiedGaussianFit:
    values = finite_float64(features, name="features", ndim=2)
    status: MetricStatus = "success"
    reasons: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = {}
    if normalize:
        values, status, reasons, diagnostics = normalize_rows(values, name="fit_features")
        if status == "failed":
            raise ValueError("fit features contain exact-zero norms")
    values, targets, means, num_classes = _class_statistics(
        values, labels, num_classes=num_classes
    )
    counts = np.bincount(targets, minlength=num_classes).astype(np.int64)
    residuals = values - means[targets]
    within_estimator = EmpiricalCovariance(assume_centered=True).fit(residuals)
    global_mean = np.mean(values, axis=0)
    global_estimator = EmpiricalCovariance(assume_centered=True).fit(
        values - global_mean
    )
    arrays = (
        within_estimator.covariance_,
        within_estimator.precision_,
        global_estimator.covariance_,
        global_estimator.precision_,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Gaussian covariance fit produced non-finite arrays")
    return TiedGaussianFit(
        class_means=means,
        class_counts=counts,
        within_covariance=np.asarray(within_estimator.covariance_, dtype=np.float64),
        within_precision=np.asarray(within_estimator.precision_, dtype=np.float64),
        global_mean=global_mean,
        global_covariance=np.asarray(global_estimator.covariance_, dtype=np.float64),
        global_precision=np.asarray(global_estimator.precision_, dtype=np.float64),
        normalized=normalize,
        normalization_status=status,
        normalization_reasons=reasons,
        normalization_diagnostics=diagnostics,
    )


def score_tied_gaussians(fit: TiedGaussianFit, queries: Any) -> dict[str, Any]:
    values = finite_float64(queries, name="query_features", ndim=2)
    status = fit.normalization_status
    reasons = list(fit.normalization_reasons)
    diagnostics: dict[str, Any] = {"fit": fit.normalization_diagnostics}
    if fit.normalized:
        values, query_status, query_reasons, query_diagnostics = normalize_rows(
            values, name="query_features"
        )
        diagnostics["query"] = query_diagnostics
        reasons.extend(query_reasons)
        if query_status == "failed":
            raise ValueError("query features contain exact-zero norms")
        if query_status == "degenerate":
            status = "degenerate"
    class_distances = _quadratic_distances(values, fit.class_means, fit.within_precision)
    global_distances = _quadratic_distances(
        values, fit.global_mean[None, :], fit.global_precision
    )[:, 0]
    nearest = np.argmin(class_distances, axis=1)
    return {
        "mahalanobis": -class_distances[np.arange(len(values)), nearest],
        "marginal_mahalanobis": -global_distances,
        "relative_mahalanobis": np.max(
            global_distances[:, None] - class_distances, axis=1
        ),
        "class_distances": class_distances,
        "global_distances": global_distances,
        "nearest_class": nearest.astype(np.int64),
        "status": status,
        "reason_codes": sorted(set(reasons)),
        "normalization_diagnostics": diagnostics,
    }


def gda_jitter_candidates() -> tuple[float, ...]:
    return (
        0.0,
        float(np.finfo(np.float64).tiny),
        *(float(10.0**exponent) for exponent in range(-308, 0)),
    )


@dataclass(frozen=True)
class GDAClassDensityFit:
    class_means: np.ndarray
    class_covariances: np.ndarray
    class_counts: np.ndarray
    log_priors: np.ndarray
    cholesky: np.ndarray
    log_determinants: np.ndarray
    selected_jitter: float


def fit_gda_class_density(
    features: Any,
    labels: Any,
    *,
    num_classes: int | None = None,
) -> GDAClassDensityFit:
    values, targets, means, num_classes = _class_statistics(
        features, labels, num_classes=num_classes
    )
    counts = np.bincount(targets, minlength=num_classes).astype(np.int64)
    if np.any(counts < 2):
        raise ValueError("GDA-ClassDensity requires at least two samples per class")
    covariances = np.stack(
        [
            np.cov(values[targets == class_index], rowvar=False, ddof=1)
            for class_index in range(num_classes)
        ]
    ).astype(np.float64)
    if covariances.ndim == 1:
        covariances = covariances[:, None, None]
    identity = np.eye(values.shape[1], dtype=np.float64)
    selected: float | None = None
    cholesky: np.ndarray | None = None
    for candidate in gda_jitter_candidates():
        try:
            candidate_cholesky = np.linalg.cholesky(
                covariances + candidate * identity[None, :, :]
            )
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(candidate_cholesky).all():
            selected = candidate
            cholesky = candidate_cholesky
            break
    if selected is None or cholesky is None:
        raise ValueError("GDA-ClassDensity exhausted the shared jitter ladder")
    log_determinants = 2.0 * np.sum(
        np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
    )
    return GDAClassDensityFit(
        class_means=means,
        class_covariances=covariances,
        class_counts=counts,
        log_priors=np.log(counts / np.sum(counts)),
        cholesky=cholesky,
        log_determinants=log_determinants,
        selected_jitter=selected,
    )


def score_gda_class_density(fit: GDAClassDensityFit, queries: Any) -> dict[str, Any]:
    values = finite_float64(queries, name="query_features", ndim=2)
    if values.shape[1] != fit.class_means.shape[1]:
        raise ValueError("GDA query feature width does not match fit")
    class_log_density = np.empty((len(values), len(fit.class_means)), dtype=np.float64)
    constant = fit.class_means.shape[1] * np.log(2.0 * np.pi)
    for class_index in range(len(fit.class_means)):
        delta = (values - fit.class_means[class_index]).T
        solved = np.linalg.solve(fit.cholesky[class_index], delta)
        quadratic = np.sum(solved * solved, axis=0)
        class_log_density[:, class_index] = -0.5 * (
            constant + fit.log_determinants[class_index] + quadratic
        )
    mixture_terms = class_log_density + fit.log_priors[None, :]
    maximum = np.max(mixture_terms, axis=1, keepdims=True)
    score = np.squeeze(maximum, axis=1) + np.log(
        np.sum(np.exp(mixture_terms - maximum), axis=1)
    )
    if not np.isfinite(score).all():
        raise ValueError("GDA-ClassDensity produced non-finite scores")
    return {
        "score": score,
        "class_log_density": class_log_density,
        "mixture_terms": mixture_terms,
    }


def _stable_k_smallest(
    distances: np.ndarray,
    sample_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    if k > len(distances):
        raise ValueError("k exceeds reference-bank size")
    candidate = np.argpartition(distances, k - 1)[:k]
    boundary = float(np.max(distances[candidate]))
    strict = np.flatnonzero(distances < boundary)
    tied = np.flatnonzero(distances == boundary)
    tied_order = tied[np.argsort(sample_ids[tied], kind="stable")]
    needed = k - len(strict)
    chosen = np.concatenate([strict, tied_order[:needed]])
    return chosen[np.lexsort((sample_ids[chosen], distances[chosen]))]


def exact_knn_l2_scores(
    fit_features: Any,
    query_features: Any,
    *,
    fit_sample_ids: Any,
    k: int = 50,
) -> dict[str, Any]:
    bank, fit_status, fit_reasons, fit_diagnostics = normalize_rows(
        fit_features,
        denominator_epsilon=1.0e-10,
        name="fit_features",
    )
    queries, query_status, query_reasons, query_diagnostics = normalize_rows(
        query_features,
        denominator_epsilon=1.0e-10,
        name="query_features",
    )
    if fit_status == "failed" or query_status == "failed":
        raise ValueError("kNN-L2 received exact-zero feature norms")
    sample_ids = np.asarray(fit_sample_ids, dtype=str).reshape(-1)
    if len(sample_ids) != len(bank) or len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("fit_sample_ids must be unique and match the bank")
    distances_out = np.empty(len(queries), dtype=np.float64)
    neighbor_ids = np.empty(len(queries), dtype=sample_ids.dtype)
    for query_index, query in enumerate(queries):
        distances = np.sum((bank - query) ** 2, axis=1)
        ordered = _stable_k_smallest(distances, sample_ids, k)
        neighbor = ordered[-1]
        distances_out[query_index] = distances[neighbor]
        neighbor_ids[query_index] = sample_ids[neighbor]
    status: MetricStatus = (
        "degenerate" if "degenerate" in (fit_status, query_status) else "success"
    )
    return {
        "score": -distances_out,
        "kth_squared_distance": distances_out,
        "kth_neighbor_sample_id": neighbor_ids,
        "k": k,
        "exact_search": True,
        "status": status,
        "reason_codes": sorted(set(fit_reasons + query_reasons)),
        "normalization_diagnostics": {
            "fit": fit_diagnostics,
            "query": query_diagnostics,
            "denominator_epsilon": 1.0e-10,
        },
    }


def prototype_scores(
    fit_features: Any,
    fit_labels: Any,
    query_features: Any,
    *,
    classifier_weight: Any | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    _, _, means, _ = _class_statistics(
        fit_features, fit_labels, num_classes=num_classes
    )
    queries = finite_float64(query_features, name="query_features", ndim=2)
    euclidean = np.linalg.norm(queries[:, None, :] - means[None, :, :], axis=2)
    nearest = np.argmin(euclidean, axis=1)
    normalized_queries, query_status, query_reasons, query_diagnostics = normalize_rows(
        queries, name="query_features"
    )
    normalized_means, mean_status, mean_reasons, mean_diagnostics = normalize_rows(
        means, name="class_means"
    )
    if "failed" in (query_status, mean_status):
        raise ValueError("prototype cosine received exact-zero norms")
    cosine = normalized_queries @ normalized_means.T
    matched = np.argmax(cosine, axis=1)
    result: dict[str, Any] = {
        "ncp_score": -euclidean[np.arange(len(queries)), nearest],
        "ncp_distance": euclidean[np.arange(len(queries)), nearest],
        "ncp_class": nearest.astype(np.int64),
        "ctm_score": cosine[np.arange(len(queries)), matched],
        "ctm_class": matched.astype(np.int64),
        "prototype_cosine_matrix": cosine,
        "class_means": means,
        "status": "degenerate"
        if "degenerate" in (query_status, mean_status)
        else "success",
        "reason_codes": sorted(set(query_reasons + mean_reasons)),
        "normalization_diagnostics": {
            "query": query_diagnostics,
            "class_mean": mean_diagnostics,
        },
    }
    if classifier_weight is not None:
        weights, weight_status, weight_reasons, weight_diagnostics = normalize_rows(
            classifier_weight, name="classifier_weight"
        )
        if weight_status == "failed":
            raise ValueError("classifier contains an exact-zero weight row")
        weight_cosine = normalized_queries @ weights.T
        result.update(
            {
                "weight_cosine_score": np.max(weight_cosine, axis=1),
                "weight_cosine_class": np.argmax(weight_cosine, axis=1).astype(np.int64),
                "weight_cosine_matrix": weight_cosine,
                "weight_normalization_diagnostics": weight_diagnostics,
            }
        )
        if weight_status == "degenerate":
            result["status"] = "degenerate"
            result["reason_codes"] = sorted(
                set(result["reason_codes"] + list(weight_reasons))
            )
    return result


@dataclass(frozen=True)
class ViMFit:
    origin: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    residual_subspace: np.ndarray
    dim: int
    alpha: float
    residual_mean: float


def fit_vim(
    train_features: Any,
    train_logits: Any,
    classifier_weight: Any,
    classifier_bias: Any | None,
    *,
    dim: int | None = None,
) -> ViMFit:
    features = finite_float64(train_features, name="train_features", ndim=2)
    logits = finite_float64(train_logits, name="train_logits", ndim=2)
    weight = finite_float64(classifier_weight, name="classifier_weight", ndim=2)
    if len(features) != len(logits) or features.shape[1] != weight.shape[1]:
        raise ValueError("ViM feature/logit/classifier shapes do not match")
    if classifier_bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float64)
    else:
        bias = finite_float64(classifier_bias, name="classifier_bias", ndim=1)
    if bias.shape != (weight.shape[0],):
        raise ValueError("ViM classifier bias shape does not match weight")
    feature_dim = features.shape[1]
    dim = feature_dim // 2 if dim is None else int(dim)
    if not (0 < dim < feature_dim):
        raise ValueError("ViM DIM must be between zero and feature width")
    origin = -np.linalg.pinv(weight) @ bias
    shifted = features - origin
    covariance = shifted.T @ shifted / len(shifted)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    residual_subspace = eigenvectors[:, : feature_dim - dim]
    residual = np.linalg.norm(shifted @ residual_subspace, axis=1)
    residual_mean = float(np.mean(residual))
    tolerance = float(np.max(eigenvalues) * feature_dim * np.finfo(np.float64).eps)
    if residual_mean <= tolerance:
        raise ValueError("ViM mean ID residual is numerically zero")
    alpha = float(np.mean(np.max(logits, axis=1)) / residual_mean)
    if not np.isfinite(alpha):
        raise ValueError("ViM alpha must be finite")
    return ViMFit(
        origin=origin,
        covariance=covariance,
        eigenvalues=eigenvalues[::-1],
        residual_subspace=residual_subspace,
        dim=dim,
        alpha=alpha,
        residual_mean=residual_mean,
    )


def score_vim(fit: ViMFit, features: Any, logits: Any) -> dict[str, Any]:
    values = finite_float64(features, name="features", ndim=2)
    logit_values = finite_float64(logits, name="logits", ndim=2)
    if len(values) != len(logit_values):
        raise ValueError("ViM feature and logit counts do not match")
    residual = np.linalg.norm((values - fit.origin) @ fit.residual_subspace, axis=1)
    maximum = np.max(logit_values, axis=1, keepdims=True)
    energy = np.squeeze(maximum, axis=1) + np.log(
        np.sum(np.exp(logit_values - maximum), axis=1)
    )
    alpha_residual = fit.alpha * residual
    score = energy - alpha_residual
    if not np.isfinite(score).all():
        raise ValueError("ViM produced non-finite scores")
    return {
        "score": score,
        "energy": energy,
        "residual": residual,
        "alpha_residual": alpha_residual,
    }


@dataclass(frozen=True)
class NECOFit:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    scaler_variance: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance: np.ndarray
    dim: int


def fit_neco(train_features: Any, *, dim: int = 100) -> NECOFit:
    features = finite_float64(train_features, name="train_features", ndim=2)
    if features.shape[1] < dim or len(features) < dim:
        raise ValueError(f"NECO requires both N and p >= {dim}")
    scaler = StandardScaler().fit(features)
    if np.any(scaler.scale_ <= 0.0) or not np.isfinite(scaler.scale_).all():
        raise ValueError("NECO StandardScaler contains zero or non-finite scale")
    standardized = scaler.transform(features)
    pca = PCA(n_components=dim, svd_solver="full").fit(standardized)
    arrays = (pca.mean_, pca.components_, pca.explained_variance_)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("NECO PCA produced non-finite arrays")
    return NECOFit(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        scaler_variance=np.asarray(scaler.var_, dtype=np.float64),
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        pca_explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        dim=dim,
    )


def score_neco(fit: NECOFit, features: Any) -> dict[str, Any]:
    values = finite_float64(features, name="features", ndim=2)
    if np.any(fit.scaler_scale <= 0.0) or not np.isfinite(fit.scaler_scale).all():
        raise ValueError("NECO fitted scaler contains zero or non-finite scale")
    standardized = (values - fit.scaler_mean) / fit.scaler_scale
    denominator = np.linalg.norm(standardized, axis=1)
    exact_zero = denominator == 0.0
    near_zero = (denominator > 0.0) & (denominator <= EPSILON_NORM)
    if np.any(exact_zero):
        raise ValueError("NECO query has an exact-zero standardized norm")
    coordinates = (standardized - fit.pca_mean) @ fit.pca_components.T
    numerator = np.linalg.norm(coordinates, axis=1)
    score = numerator / denominator
    if not np.isfinite(score).all():
        raise ValueError("NECO produced non-finite scores")
    return {
        "score": score,
        "numerator": numerator,
        "denominator": denominator,
        "status": "degenerate" if np.any(near_zero) else "success",
        "reason_codes": ["near_zero_norm"] if np.any(near_zero) else [],
        "near_zero_count": int(np.count_nonzero(near_zero)),
        "max_logit_used": False,
    }
