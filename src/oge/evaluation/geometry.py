"""Representation-geometry measurements for Metric Contract v1.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .common import (
    EPSILON_NORM,
    MetricValue,
    finite_float64,
    finite_labels,
    linear_quantiles,
    normalize_rows,
)


@dataclass(frozen=True)
class GeometryStatistics:
    features: np.ndarray
    labels: np.ndarray
    class_counts: np.ndarray
    class_means: np.ndarray
    global_class_mean: np.ndarray
    centered_class_means: np.ndarray
    within_covariance: np.ndarray
    between_covariance: np.ndarray
    total_covariance: np.ndarray


def fit_geometry_statistics(
    features: Any,
    labels: Any,
    *,
    num_classes: int | None = None,
) -> GeometryStatistics:
    values = finite_float64(features, name="features", ndim=2)
    if num_classes is None:
        num_classes = int(np.max(np.asarray(labels))) + 1
    targets = finite_labels(labels, count=len(values), num_classes=num_classes)
    counts = np.bincount(targets, minlength=num_classes).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("geometry requires every class to be non-empty")
    means = np.stack([np.mean(values[targets == c], axis=0) for c in range(num_classes)])
    global_class_mean = np.mean(means, axis=0)
    centered_means = means - global_class_mean
    residual = values - means[targets]
    within = residual.T @ residual / len(values)
    between = centered_means.T @ centered_means / num_classes
    sample_centered = values - np.mean(values, axis=0)
    total = sample_centered.T @ sample_centered / len(values)
    return GeometryStatistics(
        features=values,
        labels=targets,
        class_counts=counts,
        class_means=means,
        global_class_mean=global_class_mean,
        centered_class_means=centered_means,
        within_covariance=(within + within.T) / 2.0,
        between_covariance=(between + between.T) / 2.0,
        total_covariance=(total + total.T) / 2.0,
    )


def _distribution(values: np.ndarray) -> dict[str, Any]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return {
        "count": int(len(values)),
        "mean": mean,
        "population_std": std,
        "cv": None if mean == 0.0 else float(std / mean),
        "minimum": float(np.min(values)),
        "quantiles": linear_quantiles(values),
        "maximum": float(np.max(values)),
    }


def feature_norm_distribution(statistics: GeometryStatistics) -> dict[str, Any]:
    norms = np.linalg.norm(statistics.features, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("feature norms must be finite")
    exact_zero = int(np.count_nonzero(norms == 0.0))
    near_zero = int(np.count_nonzero((norms > 0.0) & (norms <= EPSILON_NORM)))
    status = "failed" if exact_zero else "degenerate" if near_zero else "success"
    reasons = []
    if exact_zero:
        reasons.append("exact_zero_norm")
    if near_zero:
        reasons.append("near_zero_norm")
    return {
        "metric": MetricValue(None, status=status, reason_codes=tuple(reasons)).to_dict(),
        "norms": norms,
        "global": _distribution(norms),
        "classwise": {
            str(class_index): _distribution(norms[statistics.labels == class_index])
            for class_index in range(len(statistics.class_counts))
        },
    }


def cdnv(statistics: GeometryStatistics) -> dict[str, Any]:
    class_variances = np.asarray(
        [
            np.mean(
                np.sum(
                    (
                        statistics.features[statistics.labels == class_index]
                        - statistics.class_means[class_index]
                    )
                    ** 2,
                    axis=1,
                )
            )
            for class_index in range(len(statistics.class_counts))
        ],
        dtype=np.float64,
    )
    class_count = len(class_variances)
    pair_matrix = np.zeros((class_count, class_count), dtype=np.float64)
    denominators = np.zeros_like(pair_matrix)
    values: list[float] = []
    for left in range(class_count):
        for right in range(left + 1, class_count):
            denominator = float(
                np.sum((statistics.class_means[left] - statistics.class_means[right]) ** 2)
            )
            denominators[left, right] = denominators[right, left] = denominator
            if denominator <= EPSILON_NORM:
                return {
                    "metric": MetricValue(
                        None,
                        status="failed",
                        reason_codes=("class_pair_distance_too_small",),
                    ).to_dict(),
                    "class_variances": class_variances,
                    "pair_denominators": denominators,
                }
            value = (class_variances[left] + class_variances[right]) / (2.0 * denominator)
            pair_matrix[left, right] = pair_matrix[right, left] = value
            values.append(float(value))
    return {
        "metric": MetricValue(float(np.mean(values))).to_dict(),
        "class_variances": class_variances,
        "pair_matrix": pair_matrix,
        "pair_denominators": denominators,
    }


def class_pair_geometry(statistics: GeometryStatistics) -> dict[str, Any]:
    means = statistics.class_means
    centered = statistics.centered_class_means
    distance = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
    centered_norms = np.linalg.norm(centered, axis=1)
    raw_norms = np.linalg.norm(means, axis=1)
    if np.any(centered_norms <= EPSILON_NORM):
        raise ValueError("centered class-mean norm is zero or near-zero")
    centered_cosine = (centered @ centered.T) / np.outer(centered_norms, centered_norms)
    raw_cosine = np.full_like(centered_cosine, np.nan)
    valid_raw = raw_norms > EPSILON_NORM
    if np.all(valid_raw):
        raw_cosine = (means @ means.T) / np.outer(raw_norms, raw_norms)
    off_diagonal = ~np.eye(len(means), dtype=bool)
    return {
        "distance": distance,
        "centered_cosine": centered_cosine,
        "raw_origin_cosine_appendix": raw_cosine,
        "distance_summary": _distribution(distance[off_diagonal]),
        "centered_cosine_summary": _distribution(centered_cosine[off_diagonal]),
    }


def _simplex_target(class_count: int) -> np.ndarray:
    return (
        np.eye(class_count, dtype=np.float64)
        - np.ones((class_count, class_count), dtype=np.float64) / class_count
    ) / np.sqrt(class_count - 1)


def _equinorm_equiangular(rows: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= EPSILON_NORM) or float(np.mean(norms)) <= EPSILON_NORM:
        raise ValueError("NC2 rows contain a zero or near-zero norm")
    equinorm = float(np.std(norms, ddof=1) / np.mean(norms))
    cosine = (rows @ rows.T) / (np.outer(norms, norms) + 1.0e-9)
    mask = ~np.eye(len(rows), dtype=bool)
    equiangular = float(np.mean(np.abs(cosine[mask] + 1.0 / (len(rows) - 1))))
    return equinorm, equiangular, norms, cosine


def neural_collapse_metrics(
    statistics: GeometryStatistics,
    classifier_weight: Any,
    classifier_bias: Any | None = None,
    *,
    query_features: Any | None = None,
    query_logits: Any | None = None,
    query_labels: Any | None = None,
) -> dict[str, Any]:
    weight = finite_float64(classifier_weight, name="classifier_weight", ndim=2)
    class_count, feature_dim = weight.shape
    if statistics.class_means.shape != (class_count, feature_dim):
        raise ValueError("classifier and class-mean shapes do not match")
    nc0_raw = float(np.linalg.norm(np.sum(weight, axis=0)))

    between = statistics.between_covariance
    eigenvalues, eigenvectors = np.linalg.eigh(between)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    maximum = float(np.max(eigenvalues))
    if maximum <= 0.0:
        raise ValueError("NC1 between covariance has zero retained rank")
    cutoff = 1.0e-15 * maximum
    retained = eigenvalues > cutoff
    if not np.any(retained):
        raise ValueError("NC1 between covariance has zero retained rank")
    pinv = (eigenvectors[:, retained] / eigenvalues[retained]) @ eigenvectors[:, retained].T
    trace_components = np.einsum("ij,ji->i", statistics.within_covariance, pinv)
    nc1 = float(np.trace(statistics.within_covariance @ pinv) / class_count)
    between_trace = float(np.trace(between))
    nc1_trace_quotient = (
        None if between_trace <= EPSILON_NORM else float(np.trace(statistics.within_covariance) / between_trace)
    )
    singular = np.linalg.svd(statistics.centered_class_means, compute_uv=False)
    nc1_svd = float(np.sum(singular[class_count - 1 :] ** 2) / np.sum(singular**2)) if np.sum(singular**2) else None

    centered = statistics.centered_class_means
    nc2n, nc2a, class_norms, class_cosine = _equinorm_equiangular(centered)
    gram = centered @ centered.T
    gram_norm = float(np.linalg.norm(gram, ord="fro"))
    if gram_norm <= EPSILON_NORM:
        raise ValueError("NC2 centered-mean Gram norm is zero")
    target = _simplex_target(class_count)
    nc2_etf = float(np.linalg.norm(gram / gram_norm - target, ord="fro"))

    nc2wn, nc2wa, weight_norms, weight_cosine = _equinorm_equiangular(weight)
    weight_gram = weight @ weight.T
    weight_gram_norm = float(np.linalg.norm(weight_gram, ord="fro"))
    nc2w = float(np.linalg.norm(weight_gram / weight_gram_norm - target, ord="fro"))

    mean_matrix_t = centered
    weight_norm = float(np.linalg.norm(weight, ord="fro"))
    mean_norm = float(np.linalg.norm(mean_matrix_t, ord="fro"))
    if min(weight_norm, mean_norm) <= EPSILON_NORM:
        raise ValueError("NC3 Frobenius norm is zero or near-zero")
    nc3 = float(np.linalg.norm(weight / weight_norm - mean_matrix_t / mean_norm, ord="fro"))
    result: dict[str, Any] = {
        "nc0_row_sum_raw": MetricValue(nc0_raw).to_dict(),
        "nc0_eq12_per_dim": MetricValue(nc0_raw / feature_dim).to_dict(),
        "nc0_theory_squared": MetricValue(nc0_raw**2 / class_count).to_dict(),
        "nc1_pinv": MetricValue(nc1).to_dict(),
        "nc1_svd_diagnostic": MetricValue(nc1_svd).to_dict(),
        "nc1_trace_quotient_diagnostic": MetricValue(nc1_trace_quotient).to_dict(),
        "nc1_diagnostics": {
            "between_eigenvalues": eigenvalues,
            "cutoff": cutoff,
            "retained_rank": int(np.count_nonzero(retained)),
            "trace_components": trace_components,
            "within_denominator": len(statistics.features),
            "between_denominator": class_count,
        },
        "nc2_equinorm": MetricValue(nc2n).to_dict(),
        "nc2_equiangular": MetricValue(nc2a).to_dict(),
        "nc2_etf_raw": MetricValue(nc2_etf).to_dict(),
        "nc2_etf_eq5_scaled": MetricValue(nc2_etf / class_count**2).to_dict(),
        "nc2_diagnostics": {
            "class_norms": class_norms,
            "centered_means": centered,
            "cosine": class_cosine,
            "gram": gram,
        },
        "nc2w_etf_raw": MetricValue(nc2w).to_dict(),
        "nc2w_equinorm": MetricValue(nc2wn).to_dict(),
        "nc2w_equiangular": MetricValue(nc2wa).to_dict(),
        "nc2w_diagnostics": {
            "weight_norms": weight_norms,
            "cosine": weight_cosine,
            "gram": weight_gram,
        },
        "nc3_self_duality_raw": MetricValue(nc3).to_dict(),
        "nc3_eq10_scaled": MetricValue(nc3 / (class_count * feature_dim)).to_dict(),
    }
    if query_features is not None or query_logits is not None:
        if query_features is None or query_logits is None:
            raise ValueError("NC4 requires both query features and query logits")
        features = finite_float64(query_features, name="query_features", ndim=2)
        logits = finite_float64(query_logits, name="query_logits", ndim=2)
        if len(features) != len(logits):
            raise ValueError("NC4 query feature/logit counts differ")
        ncc_distances = np.linalg.norm(
            features[:, None, :] - statistics.class_means[None, :, :], axis=2
        )
        ncc_prediction = np.argmin(ncc_distances, axis=1)
        affine_prediction = np.argmax(logits, axis=1)
        without_bias_logits = features @ weight.T
        without_bias_prediction = np.argmax(without_bias_logits, axis=1)
        nc4_payload: dict[str, Any] = {
            "agreement_with_bias": MetricValue(
                float(np.mean(affine_prediction == ncc_prediction))
            ).to_dict(),
            "agreement_without_bias": MetricValue(
                float(np.mean(without_bias_prediction == ncc_prediction))
            ).to_dict(),
            "classifier_prediction": affine_prediction.astype(np.int64),
            "classifier_prediction_without_bias": without_bias_prediction.astype(np.int64),
            "ncc_prediction": ncc_prediction.astype(np.int64),
        }
        if query_labels is not None:
            labels = finite_labels(query_labels, count=len(features), num_classes=class_count)
            nc4_payload["classifier_accuracy"] = float(np.mean(affine_prediction == labels))
            nc4_payload["ncc_accuracy"] = float(np.mean(ncc_prediction == labels))
        result["nc4"] = nc4_payload
    return result


def rankme(features: Any) -> dict[str, Any]:
    values = finite_float64(features, name="features", ndim=2)
    singular = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    total = float(np.sum(singular))
    if total <= 0.0:
        return {
            "metric": MetricValue(None, status="failed", reason_codes=("zero_singular_sum",)).to_dict(),
            "singular_values": singular,
        }
    probabilities = singular / total + 1.0e-7
    value = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "metric": MetricValue(value).to_dict(),
        "singular_values": singular,
        "probabilities_plus_epsilon": probabilities,
        "epsilon": 1.0e-7,
        "centered": False,
    }


def covariance_spectrum(covariance: Any) -> dict[str, Any]:
    matrix = finite_float64(covariance, name="covariance", ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    matrix = (matrix + matrix.T) / 2.0
    raw = np.linalg.eigvalsh(matrix)[::-1]
    maximum = float(raw[0])
    tolerance = maximum * len(raw) * np.finfo(np.float64).eps
    if np.any(raw < -tolerance):
        return {
            "status": "failed",
            "reason_codes": ["materially_negative_eigenvalue"],
            "raw_eigenvalues": raw,
            "tolerance": tolerance,
        }
    clipped = raw.copy()
    negative_clip = (clipped < 0.0) & (clipped >= -tolerance)
    clipped[negative_clip] = 0.0
    total = float(np.sum(clipped))
    if total <= 0.0:
        return {
            "status": "failed",
            "reason_codes": ["all_zero_spectrum"],
            "raw_eigenvalues": raw,
            "eigenvalues": clipped,
            "negative_clip_count": int(np.count_nonzero(negative_clip)),
            "tolerance": tolerance,
        }
    probabilities = clipped / total
    positive = probabilities > 0.0
    entropy_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    trace_top = float(total / clipped[0])
    participation = float(total**2 / np.sum(clipped**2))
    retained = clipped > tolerance
    numerical_rank = int(np.count_nonzero(retained))
    if numerical_rank == 0:
        raise ValueError("covariance has no eigenvalue above tolerance")
    condition = float(clipped[0] / clipped[retained][-1])
    result: dict[str, Any] = {
        "status": "success",
        "reason_codes": [],
        "raw_eigenvalues": raw,
        "eigenvalues": clipped,
        "negative_clip_count": int(np.count_nonzero(negative_clip)),
        "trace": total,
        "tolerance": tolerance,
        "entropy_rank": MetricValue(entropy_rank).to_dict(),
        "trace_top_rank": MetricValue(trace_top).to_dict(),
        "participation_ratio": MetricValue(participation).to_dict(),
        "numerical_rank": MetricValue(numerical_rank).to_dict(),
        "condition_number_retained": MetricValue(condition).to_dict(),
        "log10_condition": float(np.log10(condition)),
    }
    retained_values = clipped[retained]
    if len(retained_values) < 2:
        result["log_slope"] = MetricValue(
            None, status="failed", reason_codes=("retained_rank_lt_2",)
        ).to_dict()
        result["top20_mean_log_decay"] = MetricValue(
            None, status="failed", reason_codes=("retained_rank_lt_2",)
        ).to_dict()
    else:
        rank = np.arange(1, len(retained_values) + 1, dtype=np.float64)
        log_values = np.log(retained_values)
        design = np.column_stack([np.ones_like(rank), rank])
        coefficients, *_ = np.linalg.lstsq(design, log_values, rcond=None)
        predicted = design @ coefficients
        residuals = log_values - predicted
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((log_values - np.mean(log_values)) ** 2))
        r_squared = None if sst == 0.0 else float(1.0 - sse / sst)
        count = min(20, len(retained_values) - 1)
        decay = float(np.mean(log_values[:count] - log_values[1 : count + 1]))
        result.update(
            {
                "log_slope": MetricValue(float(coefficients[1])).to_dict(),
                "log_slope_abs": abs(float(coefficients[1])),
                "log_slope_intercept": float(coefficients[0]),
                "log_slope_r_squared": MetricValue(
                    r_squared,
                    status="degenerate" if r_squared is None else "success",
                    reason_codes=("constant_log_spectrum",) if r_squared is None else (),
                ).to_dict(),
                "log_slope_residuals": residuals,
                "top20_mean_log_decay": MetricValue(decay).to_dict(),
            }
        )
    return result


def all_covariance_spectra(statistics: GeometryStatistics) -> dict[str, Any]:
    return {
        "sw": covariance_spectrum(statistics.within_covariance),
        "total": covariance_spectrum(statistics.total_covariance),
        "between": covariance_spectrum(statistics.between_covariance),
    }


def _exact_self_neighbor_distances(
    features: np.ndarray,
    *,
    k: int,
    sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if k >= len(features):
        raise ValueError("neighbor k must be smaller than sample count")
    distances = np.empty((len(features), k), dtype=np.float64)
    neighbor_ids = np.empty((len(features), k), dtype=sample_ids.dtype)
    for index, query in enumerate(features):
        all_distances = np.linalg.norm(features - query, axis=1)
        all_distances[index] = np.inf
        candidate = np.argpartition(all_distances, k - 1)[:k]
        boundary = float(np.max(all_distances[candidate]))
        strict = np.flatnonzero(all_distances < boundary)
        tied = np.flatnonzero(all_distances == boundary)
        tied = tied[np.argsort(sample_ids[tied], kind="stable")]
        chosen = np.concatenate([strict, tied[: k - len(strict)]])
        chosen = chosen[np.lexsort((sample_ids[chosen], all_distances[chosen]))]
        distances[index] = all_distances[chosen]
        neighbor_ids[index] = sample_ids[chosen]
    return distances, neighbor_ids


def local_intrinsic_dimensionality(
    features: Any,
    *,
    sample_ids: Any,
    k_values: Iterable[int] = (10, 25, 50, 100),
) -> dict[str, Any]:
    values = finite_float64(features, name="features", ndim=2)
    ids = np.asarray(sample_ids, dtype=str).reshape(-1)
    if len(ids) != len(values) or len(set(ids.tolist())) != len(ids):
        raise ValueError("sample IDs must be unique and match features")
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    distances, neighbor_ids = _exact_self_neighbor_distances(
        values, k=max(k_values), sample_ids=ids
    )
    output: dict[str, Any] = {"neighbor_ids": neighbor_ids}
    for k in k_values:
        selected = distances[:, :k]
        kth = selected[:, -1]
        invalid_distance = np.any((selected <= EPSILON_NORM) | ~np.isfinite(selected), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_log = np.mean(np.log(selected / kth[:, None]), axis=1)
            lid = -1.0 / mean_log
        invalid_denominator = np.abs(mean_log) <= 1.0e-15
        invalid = invalid_distance | invalid_denominator | ~np.isfinite(lid)
        valid = ~invalid
        per_sample = lid.copy()
        per_sample[invalid] = np.nan
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count == len(values):
            metric = MetricValue(None, status="failed", reason_codes=("all_lid_samples_invalid",))
            summary = None
        else:
            metric = MetricValue(
                float(np.mean(per_sample[valid])),
                status="degenerate" if invalid_count else "success",
                reason_codes=("invalid_lid_samples",) if invalid_count else (),
            )
            summary = {
                **_distribution(per_sample[valid]),
                "median": float(np.median(per_sample[valid])),
            }
        output[str(k)] = {
            "metric": metric.to_dict(),
            "per_sample": per_sample,
            "valid_mask": valid,
            "invalid_count": invalid_count,
            "invalid_distance_count": int(np.count_nonzero(invalid_distance)),
            "invalid_denominator_count": int(np.count_nonzero(invalid_denominator)),
            "summary": summary,
        }
    return output


def twonn(
    features: Any,
    *,
    sample_ids: Any,
    mu_fraction: float = 0.9,
) -> dict[str, Any]:
    values = finite_float64(features, name="features", ndim=2)
    ids = np.asarray(sample_ids, dtype=str).reshape(-1)
    distances, neighbor_ids = _exact_self_neighbor_distances(values, k=2, sample_ids=ids)
    if np.any(distances[:, 0] <= EPSILON_NORM):
        return {
            "metric": MetricValue(
                None, status="degenerate", reason_codes=("duplicate_or_zero_first_neighbor",)
            ).to_dict(),
            "neighbor_distances": distances,
            "neighbor_ids": neighbor_ids,
        }
    mu = distances[:, 1] / distances[:, 0]
    log_mu = np.sort(np.log(mu))
    n_eff = int(np.floor(mu_fraction * len(values)))
    if n_eff < 1:
        raise ValueError("TwoNN effective sample count is zero")
    x = log_mu[:n_eff]
    indices = np.arange(1, n_eff + 1, dtype=np.float64)
    y = -np.log(1.0 - indices / len(values))
    denominator = float(x @ x)
    if denominator <= 1.0e-15:
        return {
            "metric": MetricValue(
                None, status="failed", reason_codes=("twonn_zero_ols_denominator",)
            ).to_dict(),
            "mu": mu,
            "x": x,
            "y": y,
        }
    slope = float((x @ y) / denominator)
    return {
        "metric": MetricValue(slope).to_dict(),
        "mu": mu,
        "sorted_log_mu": log_mu,
        "x": x,
        "y": y,
        "n_eff": n_eff,
        "mu_fraction": mu_fraction,
        "mean_first_distance": float(np.mean(distances[:, 0])),
        "mean_second_distance": float(np.mean(distances[:, 1])),
        "neighbor_ids": neighbor_ids,
    }


def hypersphere_alignment_uniformity(
    statistics: GeometryStatistics,
    *,
    pair_count: int = 100_000,
    seeds: Iterable[int] = (0, 1, 2),
) -> dict[str, Any]:
    normalized, status, reasons, diagnostics = normalize_rows(
        statistics.features, name="features"
    )
    if status == "failed":
        raise ValueError("hypersphere metrics received exact-zero norms")
    alignments = np.empty(len(statistics.class_counts), dtype=np.float64)
    for class_index, count in enumerate(statistics.class_counts):
        if count < 2:
            raise ValueError("class alignment requires at least two samples per class")
        class_values = normalized[statistics.labels == class_index]
        mean = np.mean(class_values, axis=0)
        alignments[class_index] = 2.0 * count / (count - 1) * (1.0 - float(mean @ mean))
    repeat_values: list[float] = []
    sample_count = len(normalized)
    if sample_count < 2:
        raise ValueError("uniformity requires at least two samples")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        collected = 0
        while collected < pair_count:
            draw = max(1024, int((pair_count - collected) * 1.1))
            left = rng.integers(0, sample_count, size=draw)
            right = rng.integers(0, sample_count, size=draw)
            valid = left != right
            left_parts.append(left[valid])
            right_parts.append(right[valid])
            collected += int(np.count_nonzero(valid))
        left = np.concatenate(left_parts)[:pair_count]
        right = np.concatenate(right_parts)[:pair_count]
        squared = np.sum((normalized[left] - normalized[right]) ** 2, axis=1)
        potentials = -2.0 * squared
        maximum = float(np.max(potentials))
        repeat_values.append(maximum + float(np.log(np.mean(np.exp(potentials - maximum)))))
    repeats = np.asarray(repeat_values, dtype=np.float64)
    return {
        "class_alignment": MetricValue(
            float(np.mean(alignments)), status=status, reason_codes=reasons
        ).to_dict(),
        "class_alignments": alignments,
        "uniformity": MetricValue(
            float(np.mean(repeats)), status=status, reason_codes=reasons
        ).to_dict(),
        "uniformity_repeats": repeats,
        "uniformity_sample_sd": float(np.std(repeats, ddof=1)),
        "pair_count": pair_count,
        "seeds": list(seeds),
        "normalization_diagnostics": diagnostics,
    }
