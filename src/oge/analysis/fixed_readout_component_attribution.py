"""Exact accounting tools for fixed Mahalanobis readout changes.

The module treats known Mahalanobis decompositions as measurement tools.  It
does not introduce a new detector.  All public functions use the project
convention that larger scores are more ID-like and count an exact score tie as
one half in AUROC accounting.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


SCHEMA_VERSION = "fixed_readout_component_attribution_v3"
PAIR_STATES = ("incorrect", "tie", "correct")


def _finite_vector(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return array


def scale_aware_tolerance(
    *values: Any,
    condition_number: float | None = None,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-10,
    roundoff_multiplier: float = 32.0,
) -> float:
    """Return a prespecifiable floating-point tolerance from scale and kappa.

    The condition-number term is only included when the caller has an
    independently computed finite condition number.  It is not inferred from
    an observed reconstruction residual.
    """

    if atol < 0.0 or rtol < 0.0 or roundoff_multiplier < 0.0:
        raise ValueError("tolerance parameters must be non-negative")
    scale = 1.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size and np.isfinite(array).all():
            scale = max(scale, float(np.max(np.abs(array))))
    kappa = 1.0 if condition_number is None else float(condition_number)
    if not np.isfinite(kappa) or kappa < 1.0:
        raise ValueError("condition_number must be finite and at least one")
    roundoff = roundoff_multiplier * np.finfo(np.float64).eps * kappa * scale
    return max(float(atol) + float(rtol) * scale, roundoff)


def reconstruction_record(
    total: Any,
    first: Any,
    second: Any,
    *,
    condition_number: float | None = None,
) -> dict[str, float | bool]:
    """Check the additive identity ``total = first + second``."""

    total_array = _finite_vector(total, label="total")
    first_array = _finite_vector(first, label="first component")
    second_array = _finite_vector(second, label="second component")
    if total_array.shape != first_array.shape or total_array.shape != second_array.shape:
        raise ValueError("score components must have identical shapes")
    residual = total_array - first_array - second_array
    tolerance = scale_aware_tolerance(
        total_array,
        first_array,
        second_array,
        condition_number=condition_number,
    )
    maximum = float(np.max(np.abs(residual)))
    return {
        "pass": maximum <= tolerance,
        "max_abs_residual": maximum,
        "tolerance": tolerance,
    }


def mahalanobis_score_components(
    class_distances: Any,
    global_distances: Any,
) -> dict[str, np.ndarray]:
    """Construct ID-like MD, Marginal, and RMD scores from distances."""

    classes = np.asarray(class_distances, dtype=np.float64)
    global_values = _finite_vector(global_distances, label="global distances")
    if classes.ndim != 2 or classes.shape[0] != global_values.shape[0]:
        raise ValueError("class distances must have shape [sample, class]")
    if classes.shape[1] == 0 or not np.isfinite(classes).all():
        raise ValueError("class distances must be finite and contain a class axis")
    nearest = np.argmin(classes, axis=1)
    nearest_distance = classes[np.arange(classes.shape[0]), nearest]
    md = -nearest_distance
    marginal = -global_values
    rmd = global_values - nearest_distance
    return {
        "md": md,
        "marginal": marginal,
        "rmd": rmd,
        "nearest_class": nearest.astype(np.int64, copy=False),
    }


def pair_outcome_summary(id_scores: Any, ood_scores: Any) -> dict[str, int | float]:
    """Count correct, tied, and incorrect ID--OOD pairs without a pair matrix."""

    ids = _finite_vector(id_scores, label="ID scores")
    oods = np.sort(_finite_vector(ood_scores, label="OOD scores"))
    lower = np.searchsorted(oods, ids, side="left")
    upper = np.searchsorted(oods, ids, side="right")
    correct = int(np.sum(lower, dtype=np.int64))
    ties = int(np.sum(upper - lower, dtype=np.int64))
    pair_count = int(ids.size * oods.size)
    incorrect = pair_count - correct - ties
    auroc = (correct + 0.5 * ties) / pair_count
    return {
        "pair_count": pair_count,
        "correct": correct,
        "tie": ties,
        "incorrect": incorrect,
        "auroc_id_positive": float(auroc),
    }


def _component_split(
    branch: Mapping[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split not in branch or not isinstance(branch[split], Mapping):
        raise ValueError(f"branch is missing the {split!r} score mapping")
    values = branch[split]
    md = _finite_vector(values.get("md"), label=f"{split} MD")
    rmd = _finite_vector(values.get("rmd"), label=f"{split} RMD")
    marginal = _finite_vector(
        values.get("marginal"), label=f"{split} Marginal"
    )
    if md.shape != rmd.shape or md.shape != marginal.shape:
        raise ValueError(f"{split} component score shapes differ")
    check = reconstruction_record(md, rmd, marginal)
    if not check["pass"]:
        raise ValueError(f"{split} MD score reconstruction failed: {check}")
    return md, rmd, marginal


def _pair_state(margin: np.ndarray) -> np.ndarray:
    # State codes map directly to AUROC pair utility: 0, 0.5, and 1.
    return (margin > 0.0).astype(np.int8) * 2 + (margin == 0.0).astype(np.int8)


class _FenwickCounts:
    def __init__(self, size: int) -> None:
        self.values = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int) -> None:
        cursor = index + 1
        while cursor < self.values.size:
            self.values[cursor] += 1
            cursor += cursor & -cursor

    def prefix(self, stop: int) -> int:
        total = 0
        cursor = stop
        while cursor > 0:
            total += int(self.values[cursor])
            cursor -= cursor & -cursor
        return total


def _dominance_count(
    id_x: np.ndarray,
    id_y: np.ndarray,
    ood_x: np.ndarray,
    ood_y: np.ndarray,
    *,
    x_inclusive: bool,
    y_inclusive: bool,
) -> int:
    """Count OOD points dominated by ID points in two score coordinates."""

    y_coordinates = np.unique(ood_y)
    ood_order = np.argsort(ood_x, kind="mergesort")
    id_order = np.argsort(id_x, kind="mergesort")
    tree = _FenwickCounts(len(y_coordinates))
    cursor = 0
    total = 0
    for id_index in id_order:
        boundary = id_x[id_index]
        while cursor < ood_order.size:
            ood_index = ood_order[cursor]
            value = ood_x[ood_index]
            eligible = value <= boundary if x_inclusive else value < boundary
            if not eligible:
                break
            y_index = int(np.searchsorted(y_coordinates, ood_y[ood_index], side="left"))
            tree.add(y_index)
            cursor += 1
        side = "right" if y_inclusive else "left"
        y_stop = int(np.searchsorted(y_coordinates, id_y[id_index], side=side))
        total += tree.prefix(y_stop)
    return total


def pair_transition_summary(
    id_scores_0: Any,
    ood_scores_0: Any,
    id_scores_1: Any,
    ood_scores_1: Any,
) -> dict[str, Any]:
    """Count the exact 3x3 pair-state transition table in O(N log N)."""

    id0 = _finite_vector(id_scores_0, label="branch 0 ID scores")
    ood0 = _finite_vector(ood_scores_0, label="branch 0 OOD scores")
    id1 = _finite_vector(id_scores_1, label="branch 1 ID scores")
    ood1 = _finite_vector(ood_scores_1, label="branch 1 OOD scores")
    if id0.shape != id1.shape or ood0.shape != ood1.shape:
        raise ValueError("paired branches must contain the same ordered samples")

    f_ll = _dominance_count(
        id0, id1, ood0, ood1, x_inclusive=False, y_inclusive=False
    )
    f_lle = _dominance_count(
        id0, id1, ood0, ood1, x_inclusive=False, y_inclusive=True
    )
    f_lel = _dominance_count(
        id0, id1, ood0, ood1, x_inclusive=True, y_inclusive=False
    )
    f_lele = _dominance_count(
        id0, id1, ood0, ood1, x_inclusive=True, y_inclusive=True
    )
    branch0 = pair_outcome_summary(id0, ood0)
    branch1 = pair_outcome_summary(id1, ood1)
    x_less = int(branch0["correct"])
    x_equal = int(branch0["tie"])
    y_less = int(branch1["correct"])
    y_equal = int(branch1["tie"])
    pair_count = int(branch0["pair_count"])

    matrix = np.zeros((3, 3), dtype=np.int64)
    # Rows/columns are incorrect, tie, correct; derive the four lower-left
    # prefix rectangles and their one-dimensional margins.
    matrix[2, 2] = f_ll
    matrix[2, 1] = f_lle - f_ll
    matrix[2, 0] = x_less - f_lle
    matrix[1, 2] = f_lel - f_ll
    matrix[1, 1] = f_lele - f_lle - f_lel + f_ll
    matrix[1, 0] = x_equal - matrix[1, 2] - matrix[1, 1]
    matrix[0, 2] = y_less - matrix[2, 2] - matrix[1, 2]
    matrix[0, 1] = y_equal - matrix[2, 1] - matrix[1, 1]
    matrix[0, 0] = pair_count - int(np.sum(matrix))
    if np.any(matrix < 0) or int(np.sum(matrix)) != pair_count:
        raise AssertionError("pair transition accounting produced invalid counts")
    return {
        "pair_count": pair_count,
        "transitions": {
            PAIR_STATES[source]: {
                PAIR_STATES[target]: int(matrix[source, target])
                for target in range(3)
            }
            for source in range(3)
        },
    }


def paired_component_attribution(
    branch_0: Mapping[str, Any],
    branch_1: Mapping[str, Any],
) -> dict[str, Any]:
    """Exactly attribute branch AUROC change to RMD and Marginal changes.

    The two-component Shapley accounting is evaluated for every ID--OOD pair.
    It is order-independent and its two contributions sum exactly (up to
    roundoff) to the change in tie-aware AUROC.
    """

    id0, id0_rmd, id0_marginal = _component_split(branch_0, "id")
    ood0, ood0_rmd, ood0_marginal = _component_split(branch_0, "ood")
    id1, id1_rmd, id1_marginal = _component_split(branch_1, "id")
    ood1, ood1_rmd, ood1_marginal = _component_split(branch_1, "ood")
    if id0.shape != id1.shape or ood0.shape != ood1.shape:
        raise ValueError("paired branches must contain the same ordered samples")

    pair_count = int(id0.size * ood0.size)
    delta_id_rmd = id1_rmd - id0_rmd
    delta_ood_rmd = ood1_rmd - ood0_rmd
    delta_id_marginal = id1_marginal - id0_marginal
    delta_ood_marginal = ood1_marginal - ood0_marginal
    hybrid_rmd_id = id0 + delta_id_rmd
    hybrid_rmd_ood = ood0 + delta_ood_rmd
    hybrid_marginal_id = id0 + delta_id_marginal
    hybrid_marginal_ood = ood0 + delta_ood_marginal

    auroc0 = float(pair_outcome_summary(id0, ood0)["auroc_id_positive"])
    auroc1 = float(pair_outcome_summary(id1, ood1)["auroc_id_positive"])
    auroc_rmd_first = float(
        pair_outcome_summary(hybrid_rmd_id, hybrid_rmd_ood)["auroc_id_positive"]
    )
    auroc_marginal_first = float(
        pair_outcome_summary(hybrid_marginal_id, hybrid_marginal_ood)[
            "auroc_id_positive"
        ]
    )
    rmd_auroc = 0.5 * (
        auroc_rmd_first - auroc0 + auroc1 - auroc_marginal_first
    )
    marginal_auroc = 0.5 * (
        auroc_marginal_first - auroc0 + auroc1 - auroc_rmd_first
    )
    exact_delta = auroc1 - auroc0
    shapley_residual = exact_delta - rmd_auroc - marginal_auroc
    transitions = pair_transition_summary(id0, ood0, id1, ood1)
    score_reconstruction_0 = max(
        float(np.max(np.abs(id0 - id0_rmd - id0_marginal))),
        float(np.max(np.abs(ood0 - ood0_rmd - ood0_marginal))),
    )
    score_reconstruction_1 = max(
        float(np.max(np.abs(id1 - id1_rmd - id1_marginal))),
        float(np.max(np.abs(ood1 - ood1_rmd - ood1_marginal))),
    )
    maximum_pair_reconstruction = 2.0 * max(
        score_reconstruction_0, score_reconstruction_1
    )
    tolerance = scale_aware_tolerance(
        np.asarray([auroc0, auroc1, exact_delta, rmd_auroc, marginal_auroc])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_count": pair_count,
        "branch_0_auroc": float(auroc0),
        "branch_1_auroc": float(auroc1),
        "auroc_delta": float(auroc1 - auroc0),
        "pair_utility_delta": float(exact_delta),
        "component_auroc_attribution": {
            "rmd": float(rmd_auroc),
            "marginal": float(marginal_auroc),
            "reconstruction_residual": float(shapley_residual),
        },
        "pair_transitions": transitions["transitions"],
        "max_abs_pair_margin_reconstruction_residual": float(
            maximum_pair_reconstruction
        ),
        "tolerance": tolerance,
        "pass": (
            abs((auroc1 - auroc0) - exact_delta) <= tolerance
            and abs(shapley_residual) <= tolerance
            and maximum_pair_reconstruction <= tolerance
            and transitions["pair_count"] == pair_count
        ),
    }


def quadratic_size_stretch(
    deviations: Any,
    precision: Any,
) -> dict[str, np.ndarray | float]:
    """Decompose a quadratic distance into Euclidean size and stretch."""

    delta = np.asarray(deviations, dtype=np.float64)
    matrix = np.asarray(precision, dtype=np.float64)
    if delta.ndim != 2 or matrix.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError("deviations and precision have incompatible shapes")
    if not np.isfinite(delta).all() or not np.isfinite(matrix).all():
        raise ValueError("quadratic inputs contain NaN or infinity")
    symmetric = (matrix + matrix.T) / 2.0
    size = np.einsum("nd,nd->n", delta, delta)
    quadratic = np.einsum("ni,ij,nj->n", delta, symmetric, delta)
    stretch = np.zeros_like(quadratic)
    nonzero = size > 0.0
    stretch[nonzero] = quadratic[nonzero] / size[nonzero]
    tolerance = scale_aware_tolerance(quadratic)
    if np.any(~nonzero & (np.abs(quadratic) > tolerance)):
        raise AssertionError("zero-size deviations have non-zero quadratic distance")
    reconstructed = size * stretch
    return {
        "quadratic": quadratic,
        "size": size,
        "stretch": stretch,
        "max_abs_reconstruction_residual": float(
            np.max(np.abs(quadratic - reconstructed))
        ),
    }


def symmetric_product_change(
    size_0: Any,
    stretch_0: Any,
    size_1: Any,
    stretch_1: Any,
) -> dict[str, np.ndarray | float | bool]:
    """Exactly split ``q1-q0`` into symmetric size and stretch terms."""

    r0 = _finite_vector(size_0, label="size_0")
    w0 = _finite_vector(stretch_0, label="stretch_0")
    r1 = _finite_vector(size_1, label="size_1")
    w1 = _finite_vector(stretch_1, label="stretch_1")
    if not (r0.shape == w0.shape == r1.shape == w1.shape):
        raise ValueError("size/stretch arrays must have identical shapes")
    total = r1 * w1 - r0 * w0
    size_contribution = 0.5 * (w1 + w0) * (r1 - r0)
    stretch_contribution = 0.5 * (r1 + r0) * (w1 - w0)
    residual = total - size_contribution - stretch_contribution
    tolerance = scale_aware_tolerance(total, size_contribution, stretch_contribution)
    maximum = float(np.max(np.abs(residual)))
    return {
        "total_change": total,
        "size_contribution": size_contribution,
        "stretch_contribution": stretch_contribution,
        "max_abs_reconstruction_residual": maximum,
        "tolerance": tolerance,
        "pass": maximum <= tolerance,
    }


def woodbury_low_rank_record(
    within_covariance: Any,
    between_factor: Any,
    *,
    ridge: float,
) -> dict[str, Any]:
    """Verify the full-rank Woodbury correction under explicit ridge.

    ``between_factor @ between_factor.T`` represents the between-class
    covariance.  No pseudoinverse claim is made by this function.
    """

    within = np.asarray(within_covariance, dtype=np.float64)
    factor = np.asarray(between_factor, dtype=np.float64)
    if within.ndim != 2 or within.shape[0] != within.shape[1]:
        raise ValueError("within_covariance must be square")
    if factor.ndim != 2 or factor.shape[0] != within.shape[0]:
        raise ValueError("between_factor has incompatible shape")
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("Woodbury verification requires a positive finite ridge")
    if not np.isfinite(within).all() or not np.isfinite(factor).all():
        raise ValueError("Woodbury inputs contain NaN or infinity")
    dimension = within.shape[0]
    base = (within + within.T) / 2.0 + ridge * np.eye(dimension)
    total = base + factor @ factor.T
    # Cholesky is the explicit full-rank/SPD applicability gate.
    np.linalg.cholesky(base)
    np.linalg.cholesky(total)
    base_inverse = np.linalg.inv(base)
    middle = np.eye(factor.shape[1]) + factor.T @ base_inverse @ factor
    woodbury = base_inverse - base_inverse @ factor @ np.linalg.solve(
        middle, factor.T @ base_inverse
    )
    direct = np.linalg.inv(total)
    correction = base_inverse - direct
    tolerance = scale_aware_tolerance(
        direct,
        woodbury,
        condition_number=max(float(np.linalg.cond(base)), float(np.linalg.cond(total))),
    )
    residual = float(np.max(np.abs(direct - woodbury)))
    singular_values = np.linalg.svd(correction, compute_uv=False)
    rank_tolerance = tolerance * max(1.0, float(singular_values[0]))
    observed_rank = int(np.sum(singular_values > rank_tolerance))
    rank_bound = int(factor.shape[1])
    return {
        "applicability": "ridge_full_rank_only",
        "dimension": int(dimension),
        "between_factor_columns": rank_bound,
        "observed_correction_rank": observed_rank,
        "rank_bound": rank_bound,
        "max_abs_reconstruction_residual": residual,
        "tolerance": tolerance,
        "pass": residual <= tolerance and observed_rank <= rank_bound,
    }
