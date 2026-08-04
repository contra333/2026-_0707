"""Shared numerical contracts for Metric Contract v1.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

DEFINITION_VERSION = "metric_contract_v1.2"
EPSILON_NORM = 1.0e-12
REFERENCE_ATOL = 1.0e-12
REFERENCE_RTOL = 1.0e-10

MetricStatus = Literal["success", "degenerate", "failed"]


@dataclass(frozen=True)
class MetricValue:
    """JSON-safe scalar value with explicit numerical status."""

    value: float | int | None
    status: MetricStatus = "success"
    reason_codes: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def finite_float64(values: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def finite_labels(labels: Any, *, count: int, num_classes: int) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if array.size != count:
        raise ValueError(f"labels must contain {count} entries")
    if np.any(array < 0) or np.any(array >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes - 1}]")
    return array


def norm_diagnostics(values: Any, *, name: str = "features") -> tuple[np.ndarray, dict[str, Any]]:
    array = finite_float64(values, name=name, ndim=2)
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError(f"{name} norms must be finite")
    exact_zero = int(np.count_nonzero(norms == 0.0))
    near_zero = int(np.count_nonzero((norms > 0.0) & (norms <= EPSILON_NORM)))
    return norms, {
        "epsilon_norm": EPSILON_NORM,
        "exact_zero_count": exact_zero,
        "near_zero_count": near_zero,
    }


def normalize_rows(
    values: Any,
    *,
    denominator_epsilon: float = 0.0,
    name: str = "features",
) -> tuple[np.ndarray, MetricStatus, tuple[str, ...], dict[str, Any]]:
    array = finite_float64(values, name=name, ndim=2)
    norms, diagnostics = norm_diagnostics(array, name=name)
    reasons: list[str] = []
    status: MetricStatus = "success"
    if diagnostics["exact_zero_count"]:
        reasons.append("exact_zero_norm")
        status = "failed"
    if diagnostics["near_zero_count"]:
        reasons.append("near_zero_norm")
        if status != "failed":
            status = "degenerate"
    if status == "failed":
        return np.full_like(array, np.nan), status, tuple(reasons), diagnostics
    return (
        array / (norms[:, None] + float(denominator_epsilon)),
        status,
        tuple(reasons),
        diagnostics,
    )


def merge_status(*statuses: MetricStatus) -> MetricStatus:
    if "failed" in statuses:
        return "failed"
    if "degenerate" in statuses:
        return "degenerate"
    return "success"


def linear_quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    result = np.quantile(values, probabilities, method="linear")
    return {str(probability): float(value) for probability, value in zip(probabilities, result)}
