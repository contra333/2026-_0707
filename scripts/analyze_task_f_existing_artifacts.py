#!/usr/bin/env python3
"""Read-only Task F artifact extraction and deterministic score accounting.

This program never loads a checkpoint or fits a detector.  It reads completed
Task F score, feature, geometry, alignment, and update-telemetry artifacts and
writes compact derived summaries with source provenance.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "task_f_existing_artifact_analysis_v1"
DATASETS = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
TRANSFORMS = ("raw", "l2")
DETECTORS = ("md", "rmd", "marginal")
ENDPOINT_READOUTS = tuple((t, d) for t in TRANSFORMS for d in DETECTORS)
FORMATION_READOUTS = (("raw", "md"), ("raw", "rmd"), ("raw", "marginal"), ("l2", "md"))
PRIMARY_CELL = "adam_lr1e-3_wd1e-4_anchor"
ADAM_CELLS = (
    PRIMARY_CELL,
    "adam_lr1e-3_wd1e-3",
    "adam_lr3e-4_wd1e-4",
    "adam_lr3e-4_wd1e-3",
)
CELL_ZERO_SOURCE = {
    PRIMARY_CELL: PRIMARY_CELL,
    "adam_lr1e-3_wd1e-3": PRIMARY_CELL,
    "adam_lr3e-4_wd1e-4": "adam_lr3e-4_wd1e-4",
    "adam_lr3e-4_wd1e-3": "adam_lr3e-4_wd1e-4",
}
CONTRASTS = {"C-D": ("C", "D"), "D-Z": ("D", "Z"), "C-Z": ("C", "Z")}
WITHIN_LR_WD_PAIRS = {
    "lr1e-3": (PRIMARY_CELL, "adam_lr1e-3_wd1e-3"),
    "lr3e-4": ("adam_lr3e-4_wd1e-4", "adam_lr3e-4_wd1e-3"),
}
PAIR_STATES = ("incorrect", "tie", "correct")
T90 = {2: 6.313751515, 3: 2.91998558, 4: 2.353363435, 5: 2.131846786}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def mean_sd_t90(values: Iterable[float]) -> dict[str, Any]:
    data = [float(v) for v in values]
    result: dict[str, Any] = {"n": len(data), "mean": statistics.fmean(data) if data else None}
    if len(data) < 2:
        result.update({"sd": None, "t90_low": None, "t90_high": None})
        return result
    sd = statistics.stdev(data)
    critical = T90.get(len(data))
    half = critical * sd / math.sqrt(len(data)) if critical is not None else None
    result.update(
        {
            "sd": sd,
            "t90_low": result["mean"] - half if half is not None else None,
            "t90_high": result["mean"] + half if half is not None else None,
            "interval_definition": "two-sided paired 90% t interval",
        }
    )
    return result


def normalized_role(manifest: Mapping[str, Any]) -> str | None:
    role = str(manifest.get("sibling_role", ""))
    policy = str(manifest.get("branch_policy", ""))
    if role == "zero" or policy == "zero_decay":
        return "Z"
    if role == "alpha_1" or "_adam_coupled" in role or "_sgdm_coupled" in role:
        return "C"
    if role == "alpha_0" or "_adamw_decoupled" in role or "_sgdw_decoupled" in role:
        return "D"
    if role == "alpha_0_5":
        return "M"
    return None


def context_key(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        manifest.get("cell_id"),
        normalized_role(manifest),
        int(manifest.get("training_seed")),
        int(manifest.get("checkpoint_epoch")),
        manifest.get("checkpoint_role"),
        manifest.get("depth_tap"),
    )


def integer_epoch(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def catalog_manifests(root: Path, collection: str) -> list[tuple[dict[str, Any], Path]]:
    output = []
    # Read only the host-owned top-level collection.  The precision_medicine
    # protected root also contains central copies of curie/lise scores.
    for path in (root / collection).glob("*/manifest.json"):
        manifest = read_json(path)
        output.append((manifest, path))
    return output


class PairAccounting:
    def __init__(self, library: Path) -> None:
        self.library = library.resolve()
        self.lib = ctypes.CDLL(str(self.library))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        integer = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
        self.lib.task_f_transition_matrix.argtypes = [
            vector, vector, vector, vector, ctypes.c_int64, ctypes.c_int64, integer
        ]
        self.lib.task_f_transition_matrix.restype = ctypes.c_int
        self.lib.task_f_query_burden.argtypes = [
            vector, vector, ctypes.c_int64, vector, vector, ctypes.c_int64,
            ctypes.c_int, vector, vector, vector,
        ]
        self.lib.task_f_query_burden.restype = ctypes.c_int

    @staticmethod
    def _vector(values: Any) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("score vector must be finite and one-dimensional")
        return result

    def transition(self, id0: Any, ood0: Any, id1: Any, ood1: Any) -> dict[str, Any]:
        vectors = [self._vector(v) for v in (id0, ood0, id1, ood1)]
        if vectors[0].shape != vectors[2].shape or vectors[1].shape != vectors[3].shape:
            raise ValueError("paired score vectors have different shapes")
        matrix = np.zeros(9, dtype=np.int64)
        status = self.lib.task_f_transition_matrix(
            *vectors, vectors[0].size, vectors[1].size, matrix
        )
        if status:
            raise RuntimeError(f"transition helper failed with status {status}")
        matrix = matrix.reshape(3, 3)
        utility = np.asarray([0.0, 0.5, 1.0])
        delta = utility[None, :] - utility[:, None]
        pair_count = int(vectors[0].size * vectors[1].size)
        gain = float(np.sum(np.maximum(delta, 0.0) * matrix) / pair_count)
        loss = float(np.sum(np.maximum(-delta, 0.0) * matrix) / pair_count)
        return {
            "pair_count": pair_count,
            "gain": gain,
            "loss": loss,
            "pair_order_churn": gain + loss,
            "delta_auroc": gain - loss,
            "transitions": {
                PAIR_STATES[i]: {PAIR_STATES[j]: int(matrix[i, j]) for j in range(3)}
                for i in range(3)
            },
        }

    def burden(
        self, point0: Any, point1: Any, query0: Any, query1: Any, *, point_minus_query: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p0, p1, q0, q1 = [self._vector(v) for v in (point0, point1, query0, query1)]
        gain = np.zeros(q0.size, dtype=np.float64)
        loss = np.zeros(q0.size, dtype=np.float64)
        churn = np.zeros(q0.size, dtype=np.float64)
        status = self.lib.task_f_query_burden(
            p0, p1, p0.size, q0, q1, q0.size, int(point_minus_query), gain, loss, churn
        )
        if status:
            raise RuntimeError(f"burden helper failed with status {status}")
        return gain, loss, churn


def pair_auroc(ids: Any, oods: Any) -> float:
    id_values = np.asarray(ids, dtype=np.float64)
    ood_values = np.sort(np.asarray(oods, dtype=np.float64))
    lower = np.searchsorted(ood_values, id_values, side="left")
    upper = np.searchsorted(ood_values, id_values, side="right")
    return float((np.sum(lower) + 0.5 * np.sum(upper - lower)) / (id_values.size * ood_values.size))


def metric_from_manifest(manifest: Mapping[str, Any], transform: str, detector: str, split: str) -> dict[str, float]:
    value = manifest["ood_metrics"][transform][detector]["per_dataset"][split]
    return {"auroc": float(value["auroc"]), "fpr95": float(value["fpr95_id_tpr"])}


def score_key(transform: str, split: str, detector: str) -> str:
    return f"{transform}__{split}__{detector}"


def identity_check(manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(manifests)
    fields = ("initialization_sha256", "data_stream_sha256", "sibling_group_id")
    equal = {field: len({row.get(field) for row in rows}) == 1 for field in fields}
    sample_orders = [row.get("sample_order_sha256") for row in rows]
    return {
        **equal,
        "sample_order_sha256_equal": len({json.dumps(v, sort_keys=True) for v in sample_orders}) == 1,
        "pass": all(equal.values()) and len({json.dumps(v, sort_keys=True) for v in sample_orders}) == 1,
        "values": {field: sorted({str(row.get(field)) for row in rows}) for field in fields},
    }


def replacement_accounting(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], transform: str, split: str
) -> dict[str, Any]:
    def values(branch: Mapping[str, np.ndarray], split_name: str, detector: str) -> np.ndarray:
        return np.asarray(branch[score_key(transform, split_name, detector)], dtype=np.float64)

    right_id = values(right, "id_test", "md")
    right_ood = values(right, split, "md")
    left_id = values(left, "id_test", "md")
    left_ood = values(left, split, "md")
    a00 = pair_auroc(right_id, right_ood)
    a11 = pair_auroc(left_id, left_ood)
    a10 = pair_auroc(left_id, right_ood)
    a01 = pair_auroc(right_id, left_ood)
    phi_id = 0.5 * ((a10 - a00) + (a11 - a01))
    phi_ood = 0.5 * ((a01 - a00) + (a11 - a10))

    hybrid_rmd_id = right_id + values(left, "id_test", "rmd") - values(right, "id_test", "rmd")
    hybrid_rmd_ood = right_ood + values(left, split, "rmd") - values(right, split, "rmd")
    hybrid_marginal_id = right_id + values(left, "id_test", "marginal") - values(right, "id_test", "marginal")
    hybrid_marginal_ood = right_ood + values(left, split, "marginal") - values(right, split, "marginal")
    a_rmd = pair_auroc(hybrid_rmd_id, hybrid_rmd_ood)
    a_marginal = pair_auroc(hybrid_marginal_id, hybrid_marginal_ood)
    phi_rmd = 0.5 * ((a_rmd - a00) + (a11 - a_marginal))
    phi_marginal = 0.5 * ((a_marginal - a00) + (a11 - a_rmd))
    reconstruction = 0.0
    for branch in (left, right):
        for split_name in ("id_test", split):
            residual = (
                values(branch, split_name, "md")
                - values(branch, split_name, "rmd")
                - values(branch, split_name, "marginal")
            )
            reconstruction = max(reconstruction, float(np.max(np.abs(residual))))
    delta = a11 - a00
    return {
        "right_auroc": a00,
        "left_auroc": a11,
        "delta_auroc": delta,
        "id_ood_replacement": {
            "left_id_right_ood_auroc": a10,
            "right_id_left_ood_auroc": a01,
            "phi_id": phi_id,
            "phi_ood": phi_ood,
            "residual": delta - phi_id - phi_ood,
        },
        "rmd_marginal_replacement": {
            "rmd_first_auroc": a_rmd,
            "marginal_first_auroc": a_marginal,
            "phi_rmd": phi_rmd,
            "phi_marginal": phi_marginal,
            "marginal_share_abs_delta": abs(phi_marginal) / abs(delta) if delta else None,
            "residual": delta - phi_rmd - phi_marginal,
        },
        "md_equals_rmd_plus_marginal_max_abs_residual": reconstruction,
        "interpretation_boundary": "score accounting, not causal or unique mediation",
    }


def burden_summary(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    total = float(np.sum(vector))
    ordered = np.sort(vector)[::-1]
    concentration = {}
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(fraction * vector.size)))
        concentration[f"top_{int(fraction * 100)}pct"] = float(np.sum(ordered[:count]) / total) if total else 0.0
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "median": float(np.median(vector)),
        "quantiles": {str(q): float(np.quantile(vector, q)) for q in (0.9, 0.95, 0.99)},
        "maximum": float(np.max(vector)),
        "concentration": concentration,
    }


def analyze_score_pair(
    accounting: PairAccounting,
    left_entry: tuple[dict[str, Any], Path],
    right_entry: tuple[dict[str, Any], Path],
    *,
    cell: str,
    contrast: str,
    scope: str,
    readouts: Iterable[tuple[str, str]],
    burden_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    left_manifest, left_path = left_entry
    right_manifest, right_path = right_entry
    identity = identity_check((left_manifest, right_manifest))
    if not identity["pass"]:
        raise ValueError(f"sibling identity failed for {cell} {contrast}: {identity}")
    seed = int(left_manifest["training_seed"])
    score_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    burden_rows: list[dict[str, Any]] = []
    with np.load(left_path.with_name("scores.npz"), allow_pickle=False) as left_npz, np.load(
        right_path.with_name("scores.npz"), allow_pickle=False
    ) as right_npz:
        left = {name: np.asarray(left_npz[name]) for name in left_npz.files}
        right = {name: np.asarray(right_npz[name]) for name in right_npz.files}
        for transform, detector in readouts:
            for split in DATASETS:
                id_key = score_key(transform, "id_test", detector)
                ood_key = score_key(transform, split, detector)
                transition = accounting.transition(right[id_key], right[ood_key], left[id_key], left[ood_key])
                left_metric = metric_from_manifest(left_manifest, transform, detector, split)
                right_metric = metric_from_manifest(right_manifest, transform, detector, split)
                transition["delta_auroc_residual"] = (
                    left_metric["auroc"] - right_metric["auroc"] - transition["delta_auroc"]
                )
                score_rows.append(
                    {
                        "scope": scope,
                        "cell": cell,
                        "contrast": contrast,
                        "seed": seed,
                        "checkpoint_epoch": int(left_manifest["checkpoint_epoch"]),
                        "checkpoint_role": left_manifest["checkpoint_role"],
                        "depth_tap": left_manifest["depth_tap"],
                        "transform": transform,
                        "detector": detector,
                        "dataset": split,
                        "left_role": normalized_role(left_manifest),
                        "right_role": normalized_role(right_manifest),
                        "left": left_metric,
                        "right": right_metric,
                        "delta_fpr95": left_metric["fpr95"] - right_metric["fpr95"],
                        **transition,
                        "identity": identity,
                        "left_manifest": str(left_path),
                        "right_manifest": str(right_path),
                    }
                )
        for transform in TRANSFORMS:
            for split in DATASETS:
                localization_rows.append(
                    {
                        "scope": scope,
                        "cell": cell,
                        "contrast": contrast,
                        "seed": seed,
                        "checkpoint_epoch": int(left_manifest["checkpoint_epoch"]),
                        "checkpoint_role": left_manifest["checkpoint_role"],
                        "depth_tap": left_manifest["depth_tap"],
                        "transform": transform,
                        "dataset": split,
                        **replacement_accounting(left, right, transform, split),
                    }
                )

        if burden_arrays is not None and cell == PRIMARY_CELL and contrast == "C-D" and scope == "endpoint":
            for split in DATASETS:
                id_key = score_key("raw", "id_test", "md")
                ood_key = score_key("raw", split, "md")
                ood_gain, ood_loss, ood_churn = accounting.burden(
                    right[id_key], left[id_key], right[ood_key], left[ood_key], point_minus_query=True
                )
                id_gain, id_loss, id_churn = accounting.burden(
                    right[ood_key], left[ood_key], right[id_key], left[id_key], point_minus_query=False
                )
                ood_rate = ood_churn / right[id_key].size
                id_rate = id_churn / right[ood_key].size
                prefix = f"seed{seed}__{split}"
                burden_arrays[f"{prefix}__ood_churn_rate"] = ood_rate
                burden_arrays[f"{prefix}__id_churn_rate"] = id_rate
                burden_rows.append(
                    {
                        "cell": cell,
                        "contrast": contrast,
                        "seed": seed,
                        "dataset": split,
                        "ood_sample": {
                            "gain": burden_summary(ood_gain / right[id_key].size),
                            "loss": burden_summary(ood_loss / right[id_key].size),
                            "churn": burden_summary(ood_rate),
                        },
                        "id_sample": {
                            "gain": burden_summary(id_gain / right[ood_key].size),
                            "loss": burden_summary(id_loss / right[ood_key].size),
                            "churn": burden_summary(id_rate),
                        },
                        "statistical_unit": "seed; sample burdens are descriptive and are not independent replicates",
                    }
                )
    return score_rows, localization_rows, burden_rows


def spectrum_summary(record: Mapping[str, Any], *, tau: float | None = None) -> dict[str, Any]:
    values = np.asarray(record.get("raw_eigenvalues", record.get("eigenvalues", [])), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"status": "NOT_AVAILABLE"}
    positive = values[values > (float(tau) if tau is not None else 0.0)]
    trace = float(np.sum(np.maximum(values, 0.0)))
    probabilities = positive / np.sum(positive) if positive.size and np.sum(positive) > 0 else np.asarray([])
    entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if probabilities.size else 0.0
    output: dict[str, Any] = {
        "dimension": int(values.size),
        "numerical_rank": int(positive.size),
        "trace": trace,
        "retained_eigenvalue_median": float(np.median(positive)) if positive.size else None,
        "retained_condition_number": float(np.max(positive) / np.min(positive)) if positive.size else None,
        "effective_rank": entropy_rank,
    }
    for count in (1, 10, 50):
        output[f"top_{count}_trace_share"] = float(np.sum(values[:count]) / trace) if trace else None
    return output


def geometry_record(manifest: Mapping[str, Any], path: Path, transform: str, scope: str) -> dict[str, Any]:
    summary = manifest["summary"]["transforms"][transform]
    fit = summary["fit"]
    numerical = fit["numerical"]
    norm = summary["norm"]
    sw_tau = numerical["within_spectrum"]["tau_spec"]
    total_tau = numerical["global_spectrum"]["tau_spec"]
    class_means = [float(norm["classwise"][str(i)]["mean"]) for i in range(len(norm["classwise"]))]
    return {
        "scope": scope,
        "cell": manifest["cell_id"],
        "role": normalized_role(manifest),
        "seed": int(manifest["training_seed"]),
        "checkpoint_epoch": int(manifest["checkpoint_epoch"]),
        "checkpoint_role": manifest["checkpoint_role"],
        "depth_tap": manifest["depth_tap"],
        "transform": transform,
        "feature_norm": {
            "mean": float(norm["global"]["mean"]),
            "population_std": float(norm["global"]["population_std"]),
            "cv": float(norm["global"]["cv"]),
            "classwise_means": class_means,
            "classwise_mean_cv": float(np.std(class_means) / np.mean(class_means)),
        },
        "within_covariance": spectrum_summary(summary["covariance_spectra"]["sw"], tau=sw_tau),
        "global_covariance": spectrum_summary(summary["covariance_spectra"]["total"], tau=total_tau),
        "between_covariance": spectrum_summary(summary["covariance_spectra"]["between"]),
        "fit": {
            "applicable": bool(fit["applicable"]),
            "condition_number": fit["condition_number"],
            "dim_s": int(fit["dim_s"]),
            "dim_s_perp": int(fit["dim_s_perp"]),
            "within_rank": int(numerical["within_spectrum"]["rank"]),
            "global_rank": int(numerical["global_spectrum"]["rank"]),
            "within_condition_number": numerical["within_spectrum"]["condition_number"],
            "global_condition_number": numerical["global_spectrum"]["condition_number"],
        },
        "rankme": summary.get("rankme", {}).get("metric", {}).get("value"),
        "manifest": str(path),
    }


def alignment_record(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    output = {
        "cell": manifest["cell_id"],
        "seed": int(manifest["training_seed"]),
        "checkpoint_epoch": int(manifest["checkpoint_epoch"]),
        "checkpoint_role": manifest["checkpoint_role"],
        "depth_tap": manifest["depth_tap"],
        "left_role": normalized_role({"sibling_role": manifest["left_role"]}),
        "right_role": normalized_role({"sibling_role": manifest["right_role"]}),
        "pair_direction": manifest["pair_direction"],
        "manifest": str(path),
        "transforms": {},
    }
    for transform in TRANSFORMS:
        value = manifest["transforms"][transform]
        output["transforms"][transform] = {
            "status": value["status"],
            "affine_status": value["affine"]["status"],
            "affine_id_validation_normalized_frobenius": value["affine"]["id_validation"]["normalized_frobenius"],
            "affine_id_validation_rmse": value["affine"]["id_validation"]["rmse"],
            "gauge_status": value["gauge"]["status"],
            "principal_angles_degrees": value["gauge"]["principal_angles_degrees"],
            "chordal_projector_distance": value["gauge"]["chordal_projector_distance"],
            "gauge_id_validation_normalized_residual": value["gauge"]["id_validation_normalized_residual"],
            "interpretation_boundary": "diagnostic only when theorem/component applicability is not applicable",
        }
    return output


def summarize_distribution(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "sd": float(np.std(vector)),
        "quantiles": {str(q): float(np.quantile(vector, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
    }


def spectral_band_slices(values_descending: np.ndarray) -> list[tuple[str, int, int]]:
    total = float(np.sum(values_descending))
    cumulative = np.cumsum(values_descending)
    boundaries = []
    for fraction in (0.5, 0.9):
        end = int(np.searchsorted(cumulative, fraction * total, side="left") + 1)
        while end < values_descending.size and values_descending[end] == values_descending[end - 1]:
            end += 1
        boundaries.append(end)
    return [
        ("trace_0_50", 0, boundaries[0]),
        ("trace_50_90", boundaries[0], boundaries[1]),
        ("trace_90_100", boundaries[1], values_descending.size),
    ]


def spectral_split(
    features_path: Path,
    mean: np.ndarray,
    vectors: np.ndarray,
    precision_weights: np.ndarray,
    bands: list[tuple[str, int, int]],
    marginal_scores: np.ndarray,
    *,
    batch_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    features = np.load(features_path, mmap_mode="r")
    accum = {name: {"A": [], "P": [], "Q": []} for name, _, _ in bands}
    full_q_parts = []
    null_parts = []
    retained = precision_weights > 0.0
    for start in range(0, features.shape[0], batch_size):
        delta = np.asarray(features[start : start + batch_size], dtype=np.float64) - mean
        projection = delta @ vectors
        squared = projection * projection
        denominator = np.sum(delta * delta, axis=1)
        full_q = np.sum(squared * precision_weights, axis=1)
        full_q_parts.append(full_q)
        null_parts.append(np.sum(squared[:, ~retained], axis=1))
        for name, begin, end in bands:
            energy = np.sum(squared[:, begin:end], axis=1)
            weighted = np.sum(
                squared[:, begin:end] * precision_weights[begin:end], axis=1
            )
            accum[name]["A"].append(energy)
            accum[name]["P"].append(np.divide(energy, denominator, out=np.zeros_like(energy), where=denominator > 0))
            accum[name]["Q"].append(weighted)
    arrays = {
        f"{name}__{quantity}": np.concatenate(parts)
        for name, quantities in accum.items()
        for quantity, parts in quantities.items()
    }
    full_q = np.concatenate(full_q_parts)
    null_energy = np.concatenate(null_parts)
    band_q_sum = sum(arrays[f"{name}__Q"] for name, _, _ in bands)
    residual = -full_q - np.asarray(marginal_scores, dtype=np.float64)
    return (
        {
            "bands": {
                name: {
                    "index_range": [begin, end],
                    "A_unweighted_energy": summarize_distribution(arrays[f"{name}__A"]),
                    "P_allocation_share": summarize_distribution(arrays[f"{name}__P"]),
                    "Q_mahalanobis_contribution": summarize_distribution(arrays[f"{name}__Q"]),
                }
                for name, begin, end in bands
            },
            "full_q": summarize_distribution(full_q),
            "null_space_energy": summarize_distribution(null_energy),
            "max_abs_band_sum_residual": float(np.max(np.abs(full_q - band_q_sum))),
            "max_abs_stored_marginal_residual": float(np.max(np.abs(residual))),
        },
        arrays,
    )


def analyze_spectral_context(
    score_entry: tuple[dict[str, Any], Path],
    fit_entry: tuple[dict[str, Any], Path],
    protected_root: Path,
) -> dict[str, Any]:
    score_manifest, score_path = score_entry
    fit_manifest, fit_path = fit_entry
    raw_fit_applicable = bool(
        fit_manifest["summary"]["transforms"]["raw"]["fit"]["applicable"]
    )
    with np.load(fit_path.with_name("fit_state.npz"), allow_pickle=False) as fit:
        mean = np.asarray(fit["raw__mean"], dtype=np.float64)
        covariance = np.asarray(fit["raw__total_covariance"], dtype=np.float64)
        precision = np.asarray(fit["raw__global_precision"], dtype=np.float64)
    covariance = (covariance + covariance.T) / 2.0
    precision = (precision + precision.T) / 2.0
    raw_weights, vectors = np.linalg.eigh(precision)
    tolerance = max(float(np.max(np.abs(raw_weights))) * 1e-12, np.finfo(np.float64).eps)
    if float(np.min(raw_weights)) < -tolerance:
        raise ValueError(f"global precision is not PSD within tolerance: {fit_path}")
    precision_weights = np.where(raw_weights > tolerance, raw_weights, 0.0)
    covariance_variances = np.einsum("ij,ij->j", vectors, covariance @ vectors)
    order = np.argsort(covariance_variances)[::-1]
    covariance_variances = np.maximum(covariance_variances[order], 0.0)
    precision_weights = precision_weights[order]
    vectors = vectors[:, order]
    bands = spectral_band_slices(covariance_variances)
    commutator = covariance @ precision - precision @ covariance
    identities = score_manifest["identity"]["protected_feature_identities"]
    output = {
        "cell": score_manifest["cell_id"],
        "role": normalized_role(score_manifest),
        "seed": int(score_manifest["training_seed"]),
        "checkpoint_epoch": int(score_manifest["checkpoint_epoch"]),
        "depth_tap": score_manifest["depth_tap"],
        "stable_precision_rank_at_relative_1e-12": int(np.count_nonzero(precision_weights > 0.0)),
        "dimension": int(covariance_variances.size),
        "eigenvalue_trace": float(np.sum(covariance_variances)),
        "precision_eigenvalue_tolerance": tolerance,
        "covariance_precision_commutator_relative_frobenius": float(
            np.linalg.norm(commutator, ord="fro")
            / max(np.linalg.norm(covariance, ord="fro") * np.linalg.norm(precision, ord="fro"), np.finfo(np.float64).eps)
        ),
        "band_rule": "descending covariance eigenvalues; cumulative trace [0,50], (50,90], (90,100]; ties not split",
        "splits": {},
        "fit_manifest": str(fit_path),
        "score_manifest": str(score_path),
        "interpretation_boundary": "exact branch-internal Marginal decomposition using prior spectrum-allocation lens; not a new detector or causal mediation",
        "raw_fit_applicable": raw_fit_applicable,
    }
    split_arrays = {}
    with np.load(score_path.with_name("scores.npz"), allow_pickle=False) as scores:
        for split in ("id_test",) + DATASETS:
            feature_dir = protected_root / "features" / identities[split]
            record, arrays = spectral_split(
                feature_dir / "features.npy",
                mean,
                vectors,
                precision_weights,
                bands,
                np.asarray(scores[score_key("raw", split, "marginal")]),
            )
            output["splits"][split] = record
            split_arrays[split] = arrays
    id_arrays = split_arrays["id_test"]
    for split in DATASETS:
        pair_margin = {}
        for name, _, _ in bands:
            # Marginal scores are -Q, so ID-minus-OOD margin is E[Q_ood]-E[Q_id].
            pair_margin[name] = float(
                np.mean(split_arrays[split][f"{name}__Q"]) - np.mean(id_arrays[f"{name}__Q"])
            )
        output["splits"][split]["mean_pair_margin_band_contribution"] = pair_margin
        output["splits"][split]["mean_pair_margin_reconstruction"] = {
            "sum_bands": float(sum(pair_margin.values())),
            "full_marginal": float(
                -np.mean(id_arrays[f"{bands[0][0]}__Q"] + id_arrays[f"{bands[1][0]}__Q"] + id_arrays[f"{bands[2][0]}__Q"])
                + np.mean(
                    split_arrays[split][f"{bands[0][0]}__Q"]
                    + split_arrays[split][f"{bands[1][0]}__Q"]
                    + split_arrays[split][f"{bands[2][0]}__Q"]
                )
            ),
        }
    max_reconstruction_residual = max(
        split["max_abs_stored_marginal_residual"] for split in output["splits"].values()
    )
    if not raw_fit_applicable:
        output["status"] = "NOT_APPLICABLE"
        output["result_use"] = "NOT AVAILABLE for cross-branch spectral attribution"
    elif max_reconstruction_residual <= 1e-6:
        output["status"] = "PASS"
        output["result_use"] = "branch-internal diagnostic only"
    else:
        output["status"] = "FAILED_RECONSTRUCTION"
        output["result_use"] = "NOT AVAILABLE"
    output["max_abs_stored_marginal_reconstruction_residual"] = max_reconstruction_residual
    return output


def telemetry_records(training_root: Path) -> list[dict[str, Any]]:
    output = []
    for path in training_root.rglob("update_telemetry.jsonl"):
        metadata_path = path.with_name("run_metadata.json")
        if not metadata_path.is_file():
            continue
        metadata = read_json(metadata_path)
        provenance = metadata.get("paired_control_provenance", {})
        run_id = metadata.get("run_id", path.parent.name)
        if "task-f-adam-lr1e-03-wd0e00" in run_id:
            cell = "adam_lr1e-3_zero"
        elif "task-f-adam-lr3e-04-wd0e00" in run_id:
            cell = "adam_lr3e-4_zero"
        elif "task-f-adam-lr1e-03-wd1e-04" in run_id:
            cell = PRIMARY_CELL
        elif "task-f-adam-lr1e-03-wd1e-03" in run_id:
            cell = "adam_lr1e-3_wd1e-3"
        elif "task-f-adam-lr3e-04-wd1e-04" in run_id:
            cell = "adam_lr3e-4_wd1e-4"
        elif "task-f-adam-lr3e-04-wd1e-03" in run_id:
            cell = "adam_lr3e-4_wd1e-3"
        elif "task-f-sgdm" in run_id:
            cell = "sgdm_lr0.1_wd5e-4"
        else:
            continue
        if "-zero" in run_id:
            role = "Z"
        elif "adamw" in run_id or "sgdw" in run_id:
            role = "D"
        elif "alpha-0-5" in run_id:
            role = "M"
        elif "adam-alpha-1" in run_id or "adam-coupled" in run_id or "sgdm-coupled" in run_id:
            role = "C"
        else:
            raise ValueError(f"unrecognized Task F telemetry arm: {run_id}")
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("record_type") == "parameter_update":
                by_step[int(record["global_step"])].append(record)
        for step, rows in sorted(by_step.items()):
            update = np.asarray([r["optimizer_update_norm"] for r in rows], dtype=np.float64)
            radial = np.asarray([r["radial_update_norm"] for r in rows], dtype=np.float64)
            tangential = np.asarray([r["tangential_update_norm"] for r in rows], dtype=np.float64)
            parameter = np.asarray([r["parameter_norm"] for r in rows], dtype=np.float64)
            output.append(
                {
                    "cell": cell,
                    "role": role,
                    "seed": int(provenance["training_seed"]),
                    "global_step": step,
                    "parameter_count": len(rows),
                    "update_norm_l2_aggregate": float(np.linalg.norm(update)),
                    "radial_fraction_l2": float(np.linalg.norm(radial) / np.linalg.norm(update)),
                    "tangential_fraction_l2": float(np.linalg.norm(tangential) / np.linalg.norm(update)),
                    "parameter_norm_l2_aggregate": float(np.linalg.norm(parameter)),
                    "source": str(path),
                    "trajectory_boundary": "passive witness at actual branch states; not a frozen-state causal decomposition",
                }
            )
    return output


def index_entries(entries: Iterable[tuple[dict[str, Any], Path]]) -> dict[tuple[Any, ...], tuple[dict[str, Any], Path]]:
    output = {}
    for manifest, path in entries:
        key = context_key(manifest)
        if key in output:
            raise ValueError(f"duplicate artifact context on host: {key}")
        output[key] = (manifest, path)
    return output


def analyze_host(args: argparse.Namespace) -> None:
    protected_root = args.protected_root.resolve()
    fresh_root = args.fresh_root.resolve()
    training_root = args.training_root.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accounting = PairAccounting(args.pair_library)
    score_entries = catalog_manifests(protected_root, "scores")
    geometry_entries = catalog_manifests(fresh_root / "results", "geometry")
    alignment_entries = catalog_manifests(fresh_root / "results", "alignments")
    score_index = index_entries(score_entries)
    geometry_index = index_entries(geometry_entries)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "host",
        "host": args.host,
        "roots": {
            "protected": str(protected_root),
            "fresh": str(fresh_root),
            "training": str(training_root),
        },
        "coverage": {
            "score_contexts": len(score_entries),
            "geometry_fit_contexts": len(geometry_entries),
            "alignment_records": len(alignment_entries),
            "telemetry_files": len(list(training_root.rglob("update_telemetry.jsonl"))),
        },
        "endpoint_score_rows": [],
        "score_localization_rows": [],
        "pair_burden_rows": [],
        "geometry_rows": [],
        "alignment_rows": [],
        "formation_score_rows": [],
        "formation_localization_rows": [],
        "within_lr_wd_score_rows": [],
        "within_lr_wd_localization_rows": [],
        "telemetry_rows": [],
        "spectral_allocation_rows": [],
    }
    burden_arrays: dict[str, np.ndarray] = {}

    for cell in ADAM_CELLS:
        zero_cell = CELL_ZERO_SOURCE[cell]
        seeds = sorted(
            key[2] for key in score_index
            if key[0] == cell and key[1] == "C" and key[3:] == (200, "last", "penultimate")
        )
        for seed in seeds:
            roles = {
                "C": score_index.get((cell, "C", seed, 200, "last", "penultimate")),
                "D": score_index.get((cell, "D", seed, 200, "last", "penultimate")),
                "Z": score_index.get((zero_cell, "Z", seed, 200, "last", "penultimate")),
            }
            if any(value is None for value in roles.values()):
                continue
            three_way_identity = identity_check(value[0] for value in roles.values())
            if not three_way_identity["pass"]:
                raise ValueError(f"three-way identity failed for {cell} seed {seed}: {three_way_identity}")
            for contrast, (left_role, right_role) in CONTRASTS.items():
                rows, localization, burdens = analyze_score_pair(
                    accounting, roles[left_role], roles[right_role], cell=cell, contrast=contrast,
                    scope="endpoint", readouts=ENDPOINT_READOUTS,
                    burden_arrays=burden_arrays,
                )
                result["endpoint_score_rows"].extend(rows)
                result["score_localization_rows"].extend(localization)
                result["pair_burden_rows"].extend(burdens)

    # The two WD levels at a fixed LR share the same sibling group.  These are
    # controlled WD contrasts; no analogous causal LR contrast is available.
    for lr_label, (low_wd_cell, high_wd_cell) in WITHIN_LR_WD_PAIRS.items():
        seeds = sorted(
            key[2] for key in score_index
            if key[0] == high_wd_cell and key[1] == "C" and key[3:] == (200, "last", "penultimate")
        )
        for seed in seeds:
            for role in ("C", "D"):
                left = score_index.get((high_wd_cell, role, seed, 200, "last", "penultimate"))
                right = score_index.get((low_wd_cell, role, seed, 200, "last", "penultimate"))
                if not left or not right:
                    continue
                rows, localization, _ = analyze_score_pair(
                    accounting, left, right,
                    cell=f"adam_{lr_label}_within_lr_wd",
                    contrast=f"WD1e-3-WD1e-4|{role}",
                    scope="within_lr_wd",
                    readouts=ENDPOINT_READOUTS,
                )
                result["within_lr_wd_score_rows"].extend(rows)
                result["within_lr_wd_localization_rows"].extend(localization)

    # Primary C-D formation contexts: five epochs and four depths.
    formation_contexts = [
        (epoch, "last" if epoch == 200 else "snapshot", "penultimate")
        for epoch in (10, 60, 120, 160, 200)
    ] + [(200, "last", depth) for depth in ("stage1", "stage2", "stage3")]
    primary_seeds = sorted({key[2] for key in score_index if key[0] == PRIMARY_CELL and key[1] == "C"})
    for seed in primary_seeds:
        for epoch, checkpoint_role, depth in formation_contexts:
            left = score_index.get((PRIMARY_CELL, "C", seed, epoch, checkpoint_role, depth))
            right = score_index.get((PRIMARY_CELL, "D", seed, epoch, checkpoint_role, depth))
            if not left or not right:
                continue
            rows, localization, _ = analyze_score_pair(
                accounting, left, right, cell=PRIMARY_CELL, contrast="C-D",
                scope="formation", readouts=FORMATION_READOUTS,
            )
            result["formation_score_rows"].extend(rows)
            result["formation_localization_rows"].extend(localization)

    for manifest, path in geometry_entries:
        is_endpoint = (
            manifest["cell_id"] in ADAM_CELLS
            and int(manifest["checkpoint_epoch"]) == 200
            and manifest["checkpoint_role"] == "last"
            and manifest["depth_tap"] == "penultimate"
            and normalized_role(manifest) in ("C", "D", "Z")
        )
        is_formation = (
            manifest["cell_id"] == PRIMARY_CELL
            and normalized_role(manifest) in ("C", "D")
            and (
                (manifest["depth_tap"] == "penultimate" and int(manifest["checkpoint_epoch"]) in (10, 60, 120, 160, 200))
                or (int(manifest["checkpoint_epoch"]) == 200 and manifest["depth_tap"] in ("stage1", "stage2", "stage3"))
            )
        )
        if is_endpoint or is_formation:
            for transform in TRANSFORMS:
                result["geometry_rows"].append(
                    geometry_record(manifest, path, transform, "endpoint" if is_endpoint else "formation")
                )

    for manifest, path in alignment_entries:
        if (
            manifest["cell_id"] in ADAM_CELLS
            and integer_epoch(manifest.get("checkpoint_epoch")) == 200
            and manifest["checkpoint_role"] == "last"
            and manifest["depth_tap"] == "penultimate"
        ):
            result["alignment_rows"].append(alignment_record(manifest, path))

    result["telemetry_rows"] = telemetry_records(training_root)

    if args.include_spectral:
        for seed in primary_seeds:
            for role in ("D", "C"):
                key = (PRIMARY_CELL, role, seed, 200, "last", "penultimate")
                if key in score_index and key in geometry_index:
                    result["spectral_allocation_rows"].append(
                        analyze_spectral_context(score_index[key], geometry_index[key], protected_root)
                    )

    burden_path = output_path.with_name(f"{output_path.stem}_burdens.npz")
    np.savez_compressed(burden_path, **burden_arrays)
    result["burden_array_artifact"] = {
        "path": str(burden_path),
        "sha256": sha256_file(burden_path),
        "array_count": len(burden_arrays),
    }
    result["pair_library"] = {"path": str(args.pair_library.resolve()), "sha256": sha256_file(args.pair_library)}
    payload = canonical_json_bytes(json_safe(result))
    output_path.write_bytes(payload)
    print(json.dumps({"output": str(output_path), "sha256": hashlib.sha256(payload).hexdigest(), "coverage": result["coverage"]}, indent=2))


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def merge_host_outputs(args: argparse.Namespace) -> None:
    hosts = [read_json(path) for path in args.inputs]
    merged: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "merged",
        "source_host_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.inputs
        ],
        "coverage": {
            key: sum(int(host["coverage"][key]) for host in hosts)
            for key in ("score_contexts", "geometry_fit_contexts", "alignment_records", "telemetry_files")
        },
    }
    row_keys = (
        "endpoint_score_rows", "score_localization_rows", "pair_burden_rows",
        "geometry_rows", "alignment_rows", "formation_score_rows",
        "formation_localization_rows", "within_lr_wd_score_rows",
        "within_lr_wd_localization_rows", "telemetry_rows", "spectral_allocation_rows",
    )
    for key in row_keys:
        merged[key] = [row for host in hosts for row in host[key]]

    expected_coverage = {
        "score_contexts": 360,
        "geometry_fit_contexts": 660,
        "alignment_records": 657,
        "telemetry_files": 50,
    }
    if merged["coverage"] != expected_coverage:
        raise ValueError(f"Task F coverage mismatch: {merged['coverage']} != {expected_coverage}")
    score_rows_for_validation = (
        merged["endpoint_score_rows"]
        + merged["formation_score_rows"]
        + merged["within_lr_wd_score_rows"]
    )
    localization_rows_for_validation = (
        merged["score_localization_rows"]
        + merged["formation_localization_rows"]
        + merged["within_lr_wd_localization_rows"]
    )
    max_delta_residual = max(abs(row["delta_auroc_residual"]) for row in score_rows_for_validation)
    max_md_residual = max(
        row["md_equals_rmd_plus_marginal_max_abs_residual"]
        for row in localization_rows_for_validation
    )
    max_component_residual = max(
        abs(row["rmd_marginal_replacement"]["residual"])
        for row in localization_rows_for_validation
    )
    max_side_residual = max(
        abs(row["id_ood_replacement"]["residual"])
        for row in localization_rows_for_validation
    )
    if not all(row["identity"]["pass"] for row in score_rows_for_validation):
        raise ValueError("at least one sibling comparison failed identity verification")
    if max_delta_residual > 1e-12:
        raise ValueError(f"pair/manifest DeltaAUROC residual too large: {max_delta_residual}")
    if max_md_residual > 1e-6:
        raise ValueError(f"MD=RMD+Marginal residual too large: {max_md_residual}")
    if max_component_residual > 1e-12 or max_side_residual > 1e-12:
        raise ValueError(
            f"replacement accounting residual too large: {max_component_residual}, {max_side_residual}"
        )
    if len(merged["telemetry_rows"]) != expected_coverage["telemetry_files"] * 11:
        raise ValueError(f"telemetry row coverage mismatch: {len(merged['telemetry_rows'])}")
    merged["validation"] = {
        "status": "PASS",
        "expected_coverage": expected_coverage,
        "all_sibling_identity_checks_pass": True,
        "max_pair_vs_manifest_delta_auroc_residual": max_delta_residual,
        "max_md_equals_rmd_plus_marginal_abs_residual": max_md_residual,
        "max_rmd_marginal_replacement_residual": max_component_residual,
        "max_id_ood_replacement_residual": max_side_residual,
        "telemetry_rows": len(merged["telemetry_rows"]),
    }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in merged["endpoint_score_rows"]:
        grouped[(row["cell"], row["contrast"], row["transform"], row["detector"], row["dataset"])].append(row)
    merged["endpoint_aggregates"] = []
    for key, rows in sorted(grouped.items()):
        cell, contrast, transform, detector, dataset = key
        merged["endpoint_aggregates"].append(
            {
                "cell": cell,
                "contrast": contrast,
                "transform": transform,
                "detector": detector,
                "dataset": dataset,
                "left_auroc": mean_sd_t90(row["left"]["auroc"] for row in rows),
                "right_auroc": mean_sd_t90(row["right"]["auroc"] for row in rows),
                "delta_auroc": mean_sd_t90(row["delta_auroc"] for row in rows),
                "left_fpr95": mean_sd_t90(row["left"]["fpr95"] for row in rows),
                "right_fpr95": mean_sd_t90(row["right"]["fpr95"] for row in rows),
                "delta_fpr95": mean_sd_t90(row["delta_fpr95"] for row in rows),
                "gain": mean_sd_t90(row["gain"] for row in rows),
                "loss": mean_sd_t90(row["loss"] for row in rows),
                "pair_order_churn": mean_sd_t90(row["pair_order_churn"] for row in rows),
                "seeds": sorted(row["seed"] for row in rows),
            }
        )

    wd_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in merged["within_lr_wd_score_rows"]:
        wd_grouped[(row["cell"], row["contrast"], row["transform"], row["detector"], row["dataset"])].append(row)
    merged["within_lr_wd_aggregates"] = []
    for key, rows in sorted(wd_grouped.items()):
        merged["within_lr_wd_aggregates"].append(
            {
                "cell": key[0], "contrast": key[1], "transform": key[2],
                "detector": key[3], "dataset": key[4],
                "delta_auroc": mean_sd_t90(row["delta_auroc"] for row in rows),
                "delta_fpr95": mean_sd_t90(row["delta_fpr95"] for row in rows),
                "gain": mean_sd_t90(row["gain"] for row in rows),
                "loss": mean_sd_t90(row["loss"] for row in rows),
                "pair_order_churn": mean_sd_t90(row["pair_order_churn"] for row in rows),
                "seeds": sorted(row["seed"] for row in rows),
                "causal_scope": "within fixed LR only; same initialization/data stream verified",
            }
        )

    # Difference-in-differences for coupling x WD at each fixed LR.  This is a
    # seed-paired contrast of the two controlled C-D effects, not a pair-level
    # transition metric of its own.
    endpoint_lookup = {
        (row["cell"], row["seed"], row["transform"], row["detector"], row["dataset"]): row
        for row in merged["endpoint_score_rows"] if row["contrast"] == "C-D"
    }
    did_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for lr_label, (low_cell, high_cell) in WITHIN_LR_WD_PAIRS.items():
        candidates = [key for key in endpoint_lookup if key[0] == high_cell]
        for _, seed, transform, detector, dataset in candidates:
            high = endpoint_lookup[(high_cell, seed, transform, detector, dataset)]
            low = endpoint_lookup.get((low_cell, seed, transform, detector, dataset))
            if low is not None:
                did_groups[(lr_label, transform, detector, dataset)].append(
                    float(high["delta_auroc"] - low["delta_auroc"])
                )
    merged["coupling_by_wd_difference_in_differences"] = [
        {
            "lr_context": key[0], "transform": key[1], "detector": key[2],
            "dataset": key[3], "did_delta_auroc": mean_sd_t90(values),
            "definition": "C-D at WD=1e-3 minus C-D at WD=1e-4",
            "causal_scope": "within fixed LR sibling group; no cross-LR causal interpretation",
        }
        for key, values in sorted(did_groups.items())
    ]

    localization_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in merged["score_localization_rows"]:
        localization_groups[(row["cell"], row["contrast"], row["transform"], row["dataset"])].append(row)
    merged["score_localization_aggregates"] = []
    for key, rows in sorted(localization_groups.items()):
        component = "rmd_marginal_replacement"
        side = "id_ood_replacement"
        delta_summary = mean_sd_t90(row["delta_auroc"] for row in rows)
        phi_rmd_summary = mean_sd_t90(row[component]["phi_rmd"] for row in rows)
        phi_marginal_summary = mean_sd_t90(row[component]["phi_marginal"] for row in rows)
        mean_delta = float(delta_summary["mean"])
        ratio_of_means = (
            abs(float(phi_marginal_summary["mean"])) / abs(mean_delta)
            if mean_delta != 0.0 else None
        )
        merged["score_localization_aggregates"].append(
            {
                "cell": key[0], "contrast": key[1], "transform": key[2], "dataset": key[3],
                "delta_auroc": delta_summary,
                "phi_rmd": phi_rmd_summary,
                "phi_marginal": phi_marginal_summary,
                "marginal_share_ratio_of_means": ratio_of_means,
                "marginal_share_mean_seed_ratio": mean_sd_t90(
                    row[component]["marginal_share_abs_delta"] for row in rows
                    if row[component]["marginal_share_abs_delta"] is not None
                ),
                "marginal_share_definitions": {
                    "ratio_of_means": "abs(mean(phi_marginal)) / abs(mean(delta_auroc))",
                    "mean_seed_ratio": "mean_seed(abs(phi_marginal_seed) / abs(delta_auroc_seed))",
                },
                "phi_id": mean_sd_t90(row[side]["phi_id"] for row in rows),
                "phi_ood": mean_sd_t90(row[side]["phi_ood"] for row in rows),
                "max_md_reconstruction_residual": max(row["md_equals_rmd_plus_marginal_max_abs_residual"] for row in rows),
                "interpretation_boundary": "score accounting, not causal or unique mediation",
            }
        )

    # Cross-seed burden-rank consistency uses canonical protected sample order.
    burden_sources = []
    burden_data = {}
    for path, host in zip(args.inputs, hosts):
        npz_path = Path(host["burden_array_artifact"]["path"])
        if not npz_path.is_file():
            npz_path = path.with_name(npz_path.name)
        if sha256_file(npz_path) != host["burden_array_artifact"]["sha256"]:
            raise ValueError(f"burden array checksum failed: {npz_path}")
        burden_sources.append({"path": str(npz_path.resolve()), "sha256": sha256_file(npz_path)})
        with np.load(npz_path, allow_pickle=False) as arrays:
            for name in arrays.files:
                if name in burden_data:
                    raise ValueError(f"duplicate burden array {name}")
                burden_data[name] = np.asarray(arrays[name])
    merged["burden_array_sources"] = burden_sources
    merged["pair_burden_rank_consistency"] = []
    for split in DATASETS:
        names = sorted(name for name in burden_data if f"__{split}__ood_churn_rate" in name)
        correlations = []
        jaccards = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = burden_data[names[i]], burden_data[names[j]]
                correlations.append(float(np.corrcoef(rank_average(x), rank_average(y))[0, 1]))
                count = max(1, int(math.ceil(0.10 * x.size)))
                top_x = set(np.argpartition(x, -count)[-count:].tolist())
                top_y = set(np.argpartition(y, -count)[-count:].tolist())
                jaccards.append(len(top_x & top_y) / len(top_x | top_y))
        merged["pair_burden_rank_consistency"].append(
            {
                "dataset": split,
                "seed_count": len(names),
                "pairwise_spearman": mean_sd_t90(correlations),
                "top_10pct_jaccard": mean_sd_t90(jaccards),
                "scope": "primary C-D Raw MD OOD-sample flip burden",
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(json_safe(merged))
    output.write_bytes(payload)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(payload).hexdigest(), "coverage": merged["coverage"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    host = subparsers.add_parser("host")
    host.add_argument("--host", required=True)
    host.add_argument("--protected-root", type=Path, required=True)
    host.add_argument("--fresh-root", type=Path, required=True)
    host.add_argument("--training-root", type=Path, required=True)
    host.add_argument("--pair-library", type=Path, required=True)
    host.add_argument("--output", type=Path, required=True)
    host.add_argument("--include-spectral", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "host":
        analyze_host(arguments)
    else:
        merge_host_outputs(arguments)
