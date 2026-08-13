"""Task F fresh ID-only geometry and paired aggregation.

The implementation consumes only verified ``id_train`` and ``id_validation``
feature exports.  It contains no protected-data loader, OOD score, detector
fit, or RtMD path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from oge.analysis.discriminant_residual_preflight import (
    GeometryFit,
    classifier_alignment,
    fit_discriminant_geometry,
    score_discriminant_components,
)
from oge.evaluation.classification import (
    expected_calibration_error,
    negative_log_likelihood,
    top1_accuracy,
)
from oge.evaluation.geometry import (
    all_covariance_spectra,
    cdnv,
    class_pair_geometry,
    feature_norm_distribution,
    fit_geometry_statistics,
    neural_collapse_metrics,
    rankme,
)
from oge.evaluation.task_f_fresh import (
    ALLOWED_SPLITS,
    EXPECTED_SPECIFICATION_SHA256,
    verify_bridge_artifact,
)
from oge.feature_export import verify_task_f_artifact
from oge.studies.hashing import canonical_json_bytes, canonical_sha256


GEOMETRY_SCHEMA_VERSION = "task_f_fresh_id_geometry_v1"
ALIGNMENT_SCHEMA_VERSION = "task_f_fresh_id_alignment_v1"
PAIR_SCHEMA_VERSION = "task_f_fresh_id_paired_aggregation_v1"
TABLE_SCHEMA_VERSION = "task_f_fresh_id_table_templates_v1"
TRANSFORMS = ("raw", "l2")
PAIR_DIRECTIONS = (
    ("coupled_minus_decoupled", "coupled", "decoupled"),
    ("coupled_minus_zero", "coupled", "zero"),
    ("decoupled_minus_zero", "decoupled", "zero"),
)
ID_EQUIVALENCE_MARGINS = {"accuracy": 0.01, "nll": 0.08, "ece": 0.02}
PRIMARY_ANCHOR_CELL = "adam_lr1e-3_wd1e-4_anchor"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    value = np.asarray(array)
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _metric_value(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("metric", payload)
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric value must be finite")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else "nan"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError("features must be two-dimensional")
    result = np.asarray(source, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("features must be finite")
    if transform == "raw":
        return result
    if transform != "l2":
        raise ValueError(f"unsupported transform: {transform}")
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("L2 transform encountered a zero or non-finite norm")
    return result / norms[:, None]


def _representation_nc(statistics: Any) -> dict[str, Any]:
    """NC1/NC2 values that are meaningful without a classifier at this depth."""

    class_count = len(statistics.class_counts)
    between = statistics.between_covariance
    eigenvalues, eigenvectors = np.linalg.eigh(between)
    maximum = float(np.max(eigenvalues))
    if maximum <= 0.0:
        raise ValueError("NC1 between covariance has zero retained rank")
    cutoff = 1.0e-15 * maximum
    retained = eigenvalues > cutoff
    pinv = (
        eigenvectors[:, retained] / eigenvalues[retained]
    ) @ eigenvectors[:, retained].T
    nc1 = float(np.trace(statistics.within_covariance @ pinv) / class_count)

    centered = statistics.centered_class_means
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("NC2 centered class mean has a zero norm")
    equinorm = float(np.std(norms, ddof=1) / np.mean(norms))
    cosine = (centered @ centered.T) / (np.outer(norms, norms) + 1.0e-9)
    mask = ~np.eye(class_count, dtype=bool)
    equiangular = float(
        np.mean(np.abs(cosine[mask] + 1.0 / (class_count - 1)))
    )
    gram = centered @ centered.T
    gram_norm = float(np.linalg.norm(gram, ord="fro"))
    target = (
        np.eye(class_count) - np.ones((class_count, class_count)) / class_count
    ) / np.sqrt(class_count - 1)
    etf = float(np.linalg.norm(gram / gram_norm - target, ord="fro"))
    return {
        "nc1_pinv": {"value": nc1, "status": "success", "reason_codes": []},
        "nc2_equinorm": {
            "value": equinorm,
            "status": "success",
            "reason_codes": [],
        },
        "nc2_equiangular": {
            "value": equiangular,
            "status": "success",
            "reason_codes": [],
        },
        "nc2_etf_raw": {"value": etf, "status": "success", "reason_codes": []},
        "diagnostics": {
            "between_eigenvalues": eigenvalues,
            "between_cutoff": cutoff,
            "retained_rank": int(np.count_nonzero(retained)),
            "class_mean_norms": norms,
            "class_mean_cosine": cosine,
        },
    }


def _compact_spectra(statistics: Any) -> dict[str, Any]:
    spectra = all_covariance_spectra(statistics)
    return _jsonable(spectra)


def _score_in_chunks(
    fit: GeometryFit, values: np.ndarray, *, chunk_size: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    chunk_checks: list[dict[str, bool]] = []
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        record, arrays = score_discriminant_components(fit, values[start:stop])
        chunk_checks.append(dict(record["checks"]))
        for name, array in arrays.items():
            parts[name].append(np.asarray(array))
    combined = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    components = combined["components"]
    covariance = np.cov(components, rowvar=False, bias=True)
    direct_variance = float(np.var(combined["md"], ddof=0))
    covariance_variance = float(np.ones(3) @ covariance @ np.ones(3))
    scale = max(1.0, abs(direct_variance), abs(covariance_variance))
    distributions = {}
    for name in ("q_perp", "md", "marginal", "rmd"):
        values_for_name = np.asarray(combined[name], dtype=np.float64)
        distributions[name] = {
            "mean": float(np.mean(values_for_name)),
            "population_variance": float(np.var(values_for_name, ddof=0)),
            "population_std": float(np.std(values_for_name, ddof=0)),
            "minimum": float(np.min(values_for_name)),
            "quantiles": {
                str(probability): float(np.quantile(values_for_name, probability))
                for probability in (0.01, 0.05, 0.5, 0.95, 0.99)
            },
            "maximum": float(np.max(values_for_name)),
        }
    return {
        "sample_count": int(len(values)),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunk_checks),
        "all_chunk_checks_pass": all(all(checks.values()) for checks in chunk_checks),
        "component_variances": np.diag(covariance).tolist(),
        "component_covariance_biased": covariance.tolist(),
        "direct_md_variance": direct_variance,
        "covariance_reconstructed_variance": covariance_variance,
        "covariance_variance_relative_residual": abs(
            direct_variance - covariance_variance
        ) / scale,
        "distributions": distributions,
    }, combined


def _fit_state_arrays(fit: GeometryFit, prefix: str) -> dict[str, np.ndarray]:
    names = (
        "mean",
        "class_means",
        "class_counts",
        "within_covariance",
        "between_covariance",
        "total_covariance",
        "within_precision",
        "global_precision",
        "within_sqrt",
        "within_invsqrt",
        "subspace_basis",
        "transformed_class_means",
        "parallel_global_precision",
    )
    return {f"{prefix}__{name}": np.asarray(getattr(fit, name)) for name in names}


def analyze_geometry_arrays(
    *,
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    depth_tap: str,
    classifier_weight: Any | None = None,
    classifier_bias: Any | None = None,
    train_logits: Any | None = None,
    validation_logits: Any | None = None,
    chunk_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Analyze one checkpoint/depth using only ID train and validation arrays."""

    train_source = np.asanyarray(train_features)
    validation_source = np.asanyarray(validation_features)
    labels = np.asarray(train_labels, dtype=np.int64)
    heldout_labels = np.asarray(validation_labels, dtype=np.int64)
    if train_source.ndim != 2 or validation_source.ndim != 2:
        raise ValueError("Task F features must be matrices")
    if train_source.shape[1] != validation_source.shape[1]:
        raise ValueError("ID train and validation feature dimensions differ")
    if len(labels) != len(train_source) or len(heldout_labels) != len(validation_source):
        raise ValueError("feature and label sample axes differ")
    penultimate = depth_tap == "penultimate"
    supplied_classifier = all(
        value is not None
        for value in (
            classifier_weight,
            classifier_bias,
            train_logits,
            validation_logits,
        )
    )
    if penultimate != supplied_classifier:
        raise ValueError(
            "penultimate requires classifier arrays; stage taps must not supply them"
        )

    summary: dict[str, Any] = {
        "depth_tap": depth_tap,
        "sample_counts": {
            "id_train": int(len(train_source)),
            "id_validation": int(len(validation_source)),
        },
        "transforms": {},
    }
    fit_arrays: dict[str, np.ndarray] = {}
    sample_arrays: dict[str, np.ndarray] = {}
    for transform in TRANSFORMS:
        train = _transform(train_source, transform)
        validation = _transform(validation_source, transform)
        statistics = fit_geometry_statistics(train, labels)
        fit = fit_discriminant_geometry(train, labels)
        train_score, train_components = _score_in_chunks(
            fit, train, chunk_size=chunk_size
        )
        validation_score, validation_components = _score_in_chunks(
            fit, validation, chunk_size=chunk_size
        )
        norms = feature_norm_distribution(statistics)
        cdnv_record = cdnv(statistics)
        collapse: dict[str, Any]
        if penultimate:
            collapse = neural_collapse_metrics(
                statistics,
                classifier_weight,
                classifier_bias,
                query_features=validation,
                query_logits=validation_logits,
                query_labels=heldout_labels,
            )
            classifier_record: dict[str, Any] = {
                "status": "READY",
                "nc0_checkpoint_level": collapse["nc0_row_sum_raw"],
                "alignment": classifier_alignment(fit, classifier_weight),
            }
        else:
            collapse = _representation_nc(statistics)
            collapse.update(
                {
                    "nc0_row_sum_raw": {
                        "value": None,
                        "status": "not_applicable",
                        "reason_codes": ["checkpoint_level_classifier_record"],
                    },
                    "nc3_self_duality_raw": {
                        "value": None,
                        "status": "not_applicable",
                        "reason_codes": ["classifier_reads_penultimate_only"],
                    },
                    "nc4": {
                        "status": "NOT_APPLICABLE",
                        "reason": "classifier_reads_penultimate_only",
                    },
                }
            )
            classifier_record = {
                "status": "NOT_APPLICABLE",
                "reason": "classifier_reads_penultimate_only",
            }
        utility: dict[str, Any]
        if penultimate:
            utility = {
                "scope": "descriptive_id_validation_not_equivalence_endpoint",
                "accuracy": _metric_value(
                    top1_accuracy(validation_logits, heldout_labels)
                ),
                "nll": _metric_value(
                    negative_log_likelihood(validation_logits, heldout_labels)
                ),
                "ece": _metric_value(
                    expected_calibration_error(validation_logits, heldout_labels)
                ),
            }
        else:
            utility = {
                "status": "NOT_APPLICABLE",
                "reason": "classifier_reads_penultimate_only",
            }
        summary["transforms"][transform] = {
            "status": "PASS" if fit.applicable else "INAPPLICABLE",
            "primary_channel": "S-perp" if fit.applicable else "none",
            "fit": {
                "dim_s": fit.dim,
                "dim_s_perp": fit.residual_dim,
                "ridge": fit.ridge,
                "condition_number": (
                    fit.condition_number
                    if math.isfinite(fit.condition_number)
                    else "infinity"
                ),
                "tau_alg": fit.tau_alg,
                "applicable": fit.applicable,
                "numerical": _jsonable(fit.numerical),
            },
            "class_geometry": _jsonable(class_pair_geometry(statistics)),
            "cdnv": _jsonable(cdnv_record),
            "norm": {
                "metric": _jsonable(norms["metric"]),
                "global": _jsonable(norms["global"]),
                "classwise": _jsonable(norms["classwise"]),
            },
            "covariance_spectra": _compact_spectra(statistics),
            "rankme": _jsonable(rankme(train)),
            "neural_collapse": _jsonable(collapse),
            "classifier": _jsonable(classifier_record),
            "id_utility": utility,
            "components": {
                "id_train": train_score,
                "id_validation": validation_score,
            },
        }
        fit_arrays.update(_fit_state_arrays(fit, transform))
        for split, arrays in (
            ("id_train", train_components),
            ("id_validation", validation_components),
        ):
            for name, array in arrays.items():
                sample_arrays[f"{transform}__{split}__{name}"] = np.asarray(array)
    if penultimate:
        summary["checkpoint_classifier"] = {
            "status": "READY",
            "nc0_row_sum_raw": summary["transforms"]["raw"]["neural_collapse"][
                "nc0_row_sum_raw"
            ],
            "alignment_by_transform": {
                transform: summary["transforms"][transform]["classifier"]["alignment"]
                for transform in TRANSFORMS
            },
            "depth_applicability": "penultimate_only",
        }
    else:
        summary["checkpoint_classifier"] = {
            "status": "NOT_APPLICABLE",
            "reason": "use_the_matching_penultimate_checkpoint_record",
        }
    return summary, fit_arrays, sample_arrays


def _verified_binding(binding: Mapping[str, Any], expected_split: str) -> dict[str, Any]:
    if expected_split not in ALLOWED_SPLITS:
        raise ValueError("fresh geometry permits only id_train and id_validation")
    bridge_root = Path(binding["bridge_path"])
    feature_root = Path(binding["feature_artifact_path"])
    bridge = verify_bridge_artifact(bridge_root)["manifest"]
    feature = verify_task_f_artifact(feature_root)["manifest"]
    checks = {
        "split": bridge["dataset_split"] == feature["dataset_split"] == expected_split,
        "feature_identity": bridge["feature_output_identity_sha256"]
        == feature["output_identity_sha256"],
        "run_id": bridge["run_id"] == feature["run_id"],
        "checkpoint": bridge["checkpoint_sha256"] == feature["checkpoint_sha256"],
        "depth": bridge["depth_tap"] == feature["depth_tap"],
        "sample_order": bridge["ordered_sample_id_sha256"]
        == feature["ordered_sample_id_sha256"],
        "specification": bridge["task_f_specification_sha256"]
        == feature["specification_sha256"]
        == EXPECTED_SPECIFICATION_SHA256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"geometry binding identity checks failed: {failed}")
    return {"bridge_root": bridge_root, "feature_root": feature_root, "bridge": bridge}


def _verify_output_directory(path: Path, expected_identity: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("output_identity_sha256") != expected_identity:
        raise ValueError("geometry output identity mismatch")
    expected = {"manifest.json", "fit_state.npz", "sample_components.npz", "checksums.sha256"}
    if {item.name for item in path.iterdir()} != expected:
        raise ValueError("geometry output contains unexpected files")
    rows = {}
    for line in (path / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in rows or name not in expected - {"checksums.sha256"}:
            raise ValueError("geometry checksum catalog is invalid")
        if _sha256_file(path / name) != digest:
            raise ValueError("geometry output checksum mismatch")
        rows[name] = digest
    if set(rows) != expected - {"checksums.sha256"}:
        raise ValueError("geometry checksum catalog is incomplete")
    return manifest


def analyze_bound_geometry(
    *,
    train_binding: Mapping[str, Any],
    validation_binding: Mapping[str, Any],
    output_root: str | Path,
    chunk_size: int = 2048,
) -> Path:
    """Validate, analyze, and atomically publish one ID-only geometry bundle."""

    train = _verified_binding(train_binding, "id_train")
    validation = _verified_binding(validation_binding, "id_validation")
    identity_fields = (
        "run_id",
        "checkpoint_role",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "depth_tap",
        "initialization_sha256",
        "data_stream_sha256",
    )
    mismatches = [
        field
        for field in identity_fields
        if train["bridge"].get(field) != validation["bridge"].get(field)
    ]
    if mismatches:
        raise ValueError(f"ID train/validation bindings differ: {mismatches}")
    depth = str(train["bridge"]["depth_tap"])
    train_features = np.load(train["feature_root"] / "features.npy", mmap_mode="r")
    validation_features = np.load(
        validation["feature_root"] / "features.npy", mmap_mode="r"
    )
    train_labels = np.load(train["bridge_root"] / "labels.npy", mmap_mode="r")
    validation_labels = np.load(
        validation["bridge_root"] / "labels.npy", mmap_mode="r"
    )
    classifier = {}
    if depth == "penultimate":
        classifier = {
            "classifier_weight": np.load(
                train["bridge_root"] / "classifier_weight.npy", mmap_mode="r"
            ),
            "classifier_bias": np.load(
                train["bridge_root"] / "classifier_bias.npy", mmap_mode="r"
            ),
            "train_logits": np.load(train["bridge_root"] / "logits.npy", mmap_mode="r"),
            "validation_logits": np.load(
                validation["bridge_root"] / "logits.npy", mmap_mode="r"
            ),
        }
    summary, fit_arrays, sample_arrays = analyze_geometry_arrays(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        depth_tap=depth,
        chunk_size=chunk_size,
        **classifier,
    )
    identity_payload = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "run_id": train["bridge"]["run_id"],
        "family": train["bridge"]["family"],
        "cell_id": train["bridge"]["cell_id"],
        "training_seed": train["bridge"]["training_seed"],
        "branch_policy": train["bridge"]["branch_policy"],
        "sibling_group_id": train["bridge"]["sibling_group_id"],
        "sibling_role": train["bridge"]["sibling_role"],
        "initialization_sha256": train["bridge"]["initialization_sha256"],
        "data_stream_sha256": train["bridge"]["data_stream_sha256"],
        "checkpoint_role": train["bridge"]["checkpoint_role"],
        "checkpoint_epoch": train["bridge"]["checkpoint_epoch"],
        "checkpoint_sha256": train["bridge"]["checkpoint_sha256"],
        "depth_tap": depth,
        "train_feature_identity": train["bridge"]["feature_output_identity_sha256"],
        "validation_feature_identity": validation["bridge"][
            "feature_output_identity_sha256"
        ],
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
    }
    output_identity = canonical_sha256(identity_payload)
    destination = Path(output_root) / output_identity
    if destination.exists():
        _verify_output_directory(destination, output_identity)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        np.savez(temporary / "fit_state.npz", **fit_arrays)
        np.savez(temporary / "sample_components.npz", **sample_arrays)
        manifest = {
            **identity_payload,
            "output_identity_sha256": output_identity,
            "protected_data_access": False,
            "memory_policy": {
                "feature_load": "numpy_mmap_one_record_at_a_time",
                "component_scoring": f"chunks_of_{chunk_size}",
                "sample_pair_matrix": False,
            },
            "summary": summary,
            "fit_arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "array_sha256": _array_sha256(value),
                }
                for name, value in sorted(fit_arrays.items())
            },
            "sample_arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "array_sha256": _array_sha256(value),
                }
                for name, value in sorted(sample_arrays.items())
            },
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        names = ("fit_state.npz", "manifest.json", "sample_components.npz")
        (temporary / "checksums.sha256").write_text(
            "".join(f"{_sha256_file(temporary / name)}  {name}\n" for name in names),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_output_directory(destination, output_identity)
    return destination


def geometry_seed_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Flatten verified geometry manifests into deterministic aggregation rows."""

    output: list[dict[str, Any]] = []
    for path in sorted((Path(value) for value in paths), key=str):
        manifest = _verify_output_directory(path, path.name)
        shared = {
            key: manifest[key]
            for key in (
                "run_id",
                "family",
                "cell_id",
                "training_seed",
                "branch_policy",
                "sibling_group_id",
                "sibling_role",
                "initialization_sha256",
                "data_stream_sha256",
                "checkpoint_role",
                "checkpoint_epoch",
                "depth_tap",
            )
        }
        for transform in TRANSFORMS:
            summary = manifest["summary"]["transforms"][transform]
            spectrum = summary["covariance_spectra"]
            collapse = summary["neural_collapse"]
            train_components = summary["components"]["id_train"]
            metrics: dict[str, float] = {
                "dim_s": float(summary["fit"]["dim_s"]),
                "dim_s_perp": float(summary["fit"]["dim_s_perp"]),
                "feature_norm_mean": float(summary["norm"]["global"]["mean"]),
                "feature_norm_population_std": float(
                    summary["norm"]["global"]["population_std"]
                ),
                "q_perp_mean": float(
                    train_components["distributions"]["q_perp"]["mean"]
                ),
                "q_perp_population_variance": float(
                    train_components["distributions"]["q_perp"][
                        "population_variance"
                    ]
                ),
                "s_perp_score_variance": float(
                    train_components["component_variances"][0]
                ),
                "parallel_marginal_score_variance": float(
                    train_components["component_variances"][1]
                ),
                "rmd_score_variance": float(
                    train_components["component_variances"][2]
                ),
            }
            cdnv_value = summary["cdnv"]["metric"]["value"]
            if cdnv_value is not None:
                metrics["cdnv"] = float(cdnv_value)
            for covariance_name in ("sw", "between", "total"):
                record = spectrum[covariance_name]
                if record["status"] == "success":
                    metrics[f"{covariance_name}_trace"] = float(record["trace"])
                    metrics[f"{covariance_name}_entropy_rank"] = float(
                        record["entropy_rank"]["value"]
                    )
                    metrics[f"{covariance_name}_participation_ratio"] = float(
                        record["participation_ratio"]["value"]
                    )
            for name in (
                "nc0_row_sum_raw",
                "nc1_pinv",
                "nc2_equinorm",
                "nc2_equiangular",
                "nc2_etf_raw",
                "nc3_self_duality_raw",
            ):
                value = collapse.get(name)
                if isinstance(value, Mapping) and value.get("value") is not None:
                    metrics[name] = float(value["value"])
            output.append(
                {
                    **shared,
                    "status": "PASS",
                    "fit_status": summary["status"],
                    "dataset_split": "id_train",
                    "transform": transform,
                    "metrics": dict(sorted(metrics.items())),
                    "geometry_output_identity_sha256": manifest[
                        "output_identity_sha256"
                    ],
                }
            )
        raw_utility = manifest["summary"]["transforms"]["raw"]["id_utility"]
        if raw_utility.get("scope") == "descriptive_id_validation_not_equivalence_endpoint":
            output.append(
                {
                    **shared,
                    "status": "PASS",
                    "fit_status": manifest["summary"]["transforms"]["raw"]["status"],
                    "dataset_split": "id_validation",
                    "transform": "raw",
                    "metrics": {
                        name: float(raw_utility[name])
                        for name in ("accuracy", "nll", "ece")
                    },
                    "geometry_output_identity_sha256": manifest[
                        "output_identity_sha256"
                    ],
                }
            )
    identities = [
        (
            row["run_id"],
            row["checkpoint_role"],
            row["checkpoint_epoch"],
            row["depth_tap"],
            row["dataset_split"],
            row["transform"],
        )
        for row in output
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("geometry seed-record collection contains duplicates")
    return sorted(
        output,
        key=lambda row: (
            row["family"],
            row["cell_id"],
            int(row["training_seed"]),
            row["run_id"],
            row["checkpoint_role"],
            int(row["checkpoint_epoch"]),
            row["depth_tap"],
            row["dataset_split"],
            row["transform"],
        ),
    )


def _chunked_residual(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, float]:
    squared_error = 0.0
    target_energy = 0.0
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        residual = source[start:stop] @ matrix + bias - target[start:stop]
        squared_error += float(np.sum(residual * residual))
        target_energy += float(np.sum(target[start:stop] * target[start:stop]))
    return {
        "rmse": math.sqrt(squared_error / max(1, source.size)),
        "normalized_frobenius": math.sqrt(
            squared_error / max(1.0, target_energy)
        ),
    }


def fit_affine_alignment(
    *,
    source_train: Any,
    target_train: Any,
    source_validation: Any,
    target_validation: Any,
    chunk_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit an ID-train affine map and evaluate its held-out ID residual."""

    source = np.asanyarray(source_train)
    target = np.asanyarray(target_train)
    heldout_source = np.asanyarray(source_validation)
    heldout_target = np.asanyarray(target_validation)
    if source.shape != target.shape or heldout_source.shape != heldout_target.shape:
        raise ValueError("affine alignment requires shape-matched paired features")
    if source.ndim != 2 or source.shape[1] != heldout_source.shape[1]:
        raise ValueError("affine alignment feature dimensions differ")
    dimension = source.shape[1]
    gram = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    cross = np.zeros((dimension + 1, dimension), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        source_chunk = np.asarray(source[start:stop], dtype=np.float64)
        target_chunk = np.asarray(target[start:stop], dtype=np.float64)
        if not np.isfinite(source_chunk).all() or not np.isfinite(target_chunk).all():
            raise ValueError("affine alignment features must be finite")
        augmented = np.column_stack(
            (source_chunk, np.ones(stop - start, dtype=np.float64))
        )
        gram += augmented.T @ augmented
        cross += augmented.T @ target_chunk
    coefficients, _, rank, singular_values = np.linalg.lstsq(gram, cross, rcond=None)
    matrix = coefficients[:-1]
    bias = coefficients[-1]
    full_rank = int(rank) == dimension + 1
    condition = (
        float(singular_values[0] / singular_values[-1])
        if full_rank and singular_values[-1] > 0.0
        else math.inf
    )
    return {
        "status": "PASS" if full_rank else "INAPPLICABLE_RANK_DEFICIENT",
        "fit_scope": "id_train_only_chunked_normal_equations",
        "rank": int(rank),
        "required_rank": dimension + 1,
        "condition_number": condition if math.isfinite(condition) else "infinity",
        "id_train": _chunked_residual(
            source, target, matrix, bias, chunk_size=chunk_size
        ),
        "id_validation": _chunked_residual(
            heldout_source,
            heldout_target,
            matrix,
            bias,
            chunk_size=chunk_size,
        ),
    }, {"matrix": matrix, "bias": bias}


def fit_gauge_alignment(
    *,
    left_train: Any,
    right_train: Any,
    left_validation: Any,
    right_validation: Any,
    left_fit: GeometryFit,
    right_fit: GeometryFit,
    reference_role: str | None = None,
    chunk_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit the existing ID-whitened orthogonal gauge and principal angles."""

    left = np.asanyarray(left_train)
    right = np.asanyarray(right_train)
    left_validation = np.asanyarray(left_validation)
    right_validation = np.asanyarray(right_validation)
    if left.shape != right.shape or left_validation.shape != right_validation.shape:
        raise ValueError("gauge alignment requires shape-matched paired features")
    cross = np.zeros((left.shape[1], left.shape[1]), dtype=np.float64)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        z_left = (left[start:stop] - left_fit.mean) @ left_fit.within_invsqrt
        z_right = (right[start:stop] - right_fit.mean) @ right_fit.within_invsqrt
        cross += z_right.T @ z_left
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt

    def residual(values_left: np.ndarray, values_right: np.ndarray) -> float:
        error = 0.0
        scale = 0.0
        for start in range(0, len(values_left), chunk_size):
            stop = min(start + chunk_size, len(values_left))
            z_left = (values_left[start:stop] - left_fit.mean) @ left_fit.within_invsqrt
            z_right = (
                values_right[start:stop] - right_fit.mean
            ) @ right_fit.within_invsqrt
            difference = z_right @ rotation - z_left
            error += float(np.sum(difference * difference))
            scale += max(float(np.sum(z_left * z_left)), float(np.sum(z_right * z_right)))
        return math.sqrt(error / max(1.0, scale))

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
    status = "PASS" if left_fit.applicable and right_fit.applicable else "INAPPLICABLE"
    return {
        "status": status,
        "alignment_scope": "id_train_only_whitened_orthogonal_procrustes",
        "reference_role": reference_role,
        "zero_decay_common_frame": reference_role == "zero",
        "id_train_normalized_residual": residual(left, right),
        "id_validation_normalized_residual": residual(
            left_validation, right_validation
        ),
        "left_rank": left_fit.dim,
        "right_rank": right_fit.dim,
        "principal_angles_degrees": angles.tolist(),
        "chordal_projector_distance": chordal,
    }, {"rotation": rotation}


def analyze_alignment_arrays(
    *,
    left_train: Any,
    right_train: Any,
    left_validation: Any,
    right_validation: Any,
    labels: Any,
    reference_role: str | None = None,
    chunk_size: int = 2048,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run raw/L2 affine and whitened-gauge alignment for one sibling pair."""

    labels = np.asarray(labels, dtype=np.int64)
    record: dict[str, Any] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "protected_data_access": False,
        "fit_scope": "id_train_only",
        "heldout_scope": "id_validation",
        "reference_role": reference_role,
        "transforms": {},
    }
    states: dict[str, np.ndarray] = {}
    for transform in TRANSFORMS:
        left_fit_values = _transform(np.asanyarray(left_train), transform)
        right_fit_values = _transform(np.asanyarray(right_train), transform)
        left_heldout = _transform(np.asanyarray(left_validation), transform)
        right_heldout = _transform(np.asanyarray(right_validation), transform)
        if len(labels) != len(left_fit_values):
            raise ValueError("alignment labels differ from the ID-train sample axis")
        left_fit = fit_discriminant_geometry(left_fit_values, labels)
        right_fit = fit_discriminant_geometry(right_fit_values, labels)
        affine, affine_state = fit_affine_alignment(
            source_train=right_fit_values,
            target_train=left_fit_values,
            source_validation=right_heldout,
            target_validation=left_heldout,
            chunk_size=chunk_size,
        )
        gauge, gauge_state = fit_gauge_alignment(
            left_train=left_fit_values,
            right_train=right_fit_values,
            left_validation=left_heldout,
            right_validation=right_heldout,
            left_fit=left_fit,
            right_fit=right_fit,
            reference_role=reference_role,
            chunk_size=chunk_size,
        )
        record["transforms"][transform] = {
            "status": (
                "PASS"
                if affine["status"] == "PASS" and gauge["status"] == "PASS"
                else "INAPPLICABLE"
            ),
            "affine": affine,
            "gauge": gauge,
        }
        for name, value in {**affine_state, **gauge_state}.items():
            states[f"{transform}__{name}"] = value
    return record, states


def paired_t_interval(
    values: Sequence[float], *, confidence: float = 0.90
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired t interval requires at least two finite seed deltas")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
    half_width = critical * sample_sd / math.sqrt(len(array))
    return {
        "count": int(len(array)),
        "mean": mean,
        "sample_sd_ddof1": sample_sd,
        "confidence": confidence,
        "method": "two_sided_paired_t",
        "degrees_of_freedom": int(len(array) - 1),
        "critical_value": critical,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "formal_tost_claim": False,
    }


def adjudicate_id_equivalence(
    paired_deltas: Mapping[str, Sequence[float]], *, evidence_scope: str,
    protected_id_test_available: bool,
) -> dict[str, Any]:
    if not protected_id_test_available or evidence_scope != "protected_id_test":
        return {
            "status": "PENDING_PROTECTED_ID_TEST",
            "comparable_id": None,
            "id_validation_substitution": False,
            "observed_scope": evidence_scope,
            "margins": dict(ID_EQUIVALENCE_MARGINS),
        }
    metrics: dict[str, Any] = {}
    for name, margin in ID_EQUIVALENCE_MARGINS.items():
        if name not in paired_deltas:
            raise ValueError(f"ID equivalence lacks paired {name} deltas")
        interval = paired_t_interval(paired_deltas[name])
        absolute_mean = abs(float(interval["mean"]))
        metrics[name] = {
            "absolute_paired_seed_mean": absolute_mean,
            "margin": margin,
            "status": "PASS" if absolute_mean <= margin else "FAILED",
            "paired_90_percent_ci": interval,
            "ci_is_decision_gate": False,
        }
    comparable = all(value["status"] == "PASS" for value in metrics.values())
    return {
        "status": "PASS" if comparable else "FAILED_GUARDRAIL",
        "comparable_id": comparable,
        "joint_rule": "all_three_required_for_comparable_id",
        "formal_tost_claim": False,
        "metrics": metrics,
        "failure_action": {
            "keep_all_runs": True,
            "exclude_or_retrain_runs": False,
            "interpretation": "report_failed_guardrail_and_id_ood_pareto",
        },
    }


def classify_alpha_interior(
    *, alpha_0_mean: float, alpha_0_5_mean: float, alpha_1_mean: float
) -> dict[str, Any]:
    values = np.asarray([alpha_0_mean, alpha_0_5_mean, alpha_1_mean], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("alpha means must be finite")
    tolerance = 1.0e-12 * max(1.0, abs(alpha_0_mean), abs(alpha_1_mean))
    if abs(alpha_1_mean - alpha_0_mean) <= tolerance:
        classification = "undefined_degenerate_endpoints"
    elif min(alpha_0_mean, alpha_1_mean) <= alpha_0_5_mean <= max(
        alpha_0_mean, alpha_1_mean
    ):
        classification = "interior_compatible"
    else:
        classification = "non_monotone_three_point_response"
    return {
        "classification": classification,
        "alpha_0_mean": float(alpha_0_mean),
        "alpha_0_5_mean": float(alpha_0_5_mean),
        "alpha_1_mean": float(alpha_1_mean),
        "endpoint_tolerance": tolerance,
        "role": "reported_confirmation_classification_not_a_gate",
    }


def _role(record: Mapping[str, Any]) -> str | None:
    sibling_role = str(record["sibling_role"])
    branch = str(record["branch_policy"])
    if sibling_role == "zero" or branch == "zero_decay":
        return "zero"
    if sibling_role == "alpha_0_5" or branch == "adam_mixed_alpha_0_5":
        return "alpha_0_5"
    if sibling_role == "alpha_0" or branch == "adamw_alpha_0":
        return "decoupled"
    if sibling_role == "alpha_1" or branch == "adam_alpha_1":
        return "coupled"
    if sibling_role.endswith(("_adam_coupled", "_sgdm_coupled")):
        return "coupled"
    if sibling_role.endswith(("_adamw_decoupled", "_sgdw_decoupled")):
        return "decoupled"
    if branch in {"adam", "sgd", "adam_coupled", "sgdm_coupled"}:
        return "coupled"
    if branch in {"adamw", "sgdw", "adamw_decoupled", "sgdw_decoupled"}:
        return "decoupled"
    return None


def _context(record: Mapping[str, Any], *, include_cell: bool) -> tuple[Any, ...]:
    values: list[Any] = [record["family"]]
    if include_cell:
        values.append(record["cell_id"])
    checkpoint_epoch: int | str
    if record["checkpoint_role"] == "best_val":
        checkpoint_epoch = "selected_by_id_validation"
    else:
        checkpoint_epoch = int(record["checkpoint_epoch"])
    values.extend(
        (
            record["checkpoint_role"],
            checkpoint_epoch,
            record["depth_tap"],
            record["dataset_split"],
            record["transform"],
        )
    )
    return tuple(values)


def build_aggregation_contract(
    evaluation_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive cells, seeds, and contexts from the versioned 1,320-record plan."""

    if evaluation_plan.get("schema_version") != "task_f_fresh_id_evaluation_plan_v1":
        raise ValueError("unsupported Task F fresh evaluation plan")
    if evaluation_plan.get("task_f_specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("Task F specification identity changed")
    records = evaluation_plan.get("records")
    if not isinstance(records, list) or len(records) != 1320:
        raise ValueError("aggregation contract requires exactly 1,320 logical records")
    by_run: dict[str, Mapping[str, Any]] = {}
    for record in records:
        by_run.setdefault(str(record["run_id"]), record)
    seeds_by_cell_role: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in by_run.values():
        role = _role(record)
        if role in {"coupled", "decoupled"}:
            seeds_by_cell_role[str(record["cell_id"])][role].add(
                int(record["training_seed"])
            )
    expected_seeds: dict[str, list[int]] = {}
    for cell_id, roles in sorted(seeds_by_cell_role.items()):
        if set(roles) != {"coupled", "decoupled"} or roles["coupled"] != roles["decoupled"]:
            raise ValueError(f"cell {cell_id} lacks matched coupled/decoupled seeds")
        expected_seeds[cell_id] = sorted(roles["coupled"])

    contexts: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record["dataset_split"] != "id_train" or _role(record) not in {
            "coupled",
            "decoupled",
        }:
            continue
        for transform in TRANSFORMS:
            context = {
                **record,
                "dataset_split": "id_train",
                "transform": transform,
            }
            contexts[_context(context, include_cell=True)] = context
        if record["depth_tap"] == "penultimate":
            utility = {
                **record,
                "dataset_split": "id_validation",
                "transform": "raw",
            }
            contexts[_context(utility, include_cell=True)] = utility
    return {
        "expected_seeds_by_cell": expected_seeds,
        "contexts": [contexts[key] for key in sorted(contexts, key=str)],
        "counts": {
            "cells": len(expected_seeds),
            "contexts": len(contexts),
        },
    }


def aggregate_paired_records(
    *,
    records: Sequence[Mapping[str, Any]],
    expected_seeds_by_cell: Mapping[str, Sequence[int]],
    expected_contexts: Sequence[Mapping[str, Any]] | None = None,
    protected_id_test_available: bool = False,
) -> dict[str, Any]:
    """Aggregate matched sibling seeds without pooling cells or hiding failures."""

    ordered_records = sorted(
        records,
        key=lambda record: (
            str(_context(record, include_cell=True)),
            int(record["training_seed"]),
            str(_role(record)),
            str(record["run_id"]),
        ),
    )
    by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    duplicate_keys: list[list[Any]] = []
    for record in ordered_records:
        key = (
            _context(record, include_cell=True),
            int(record["training_seed"]),
            _role(record),
        )
        if key in by_key:
            duplicate_keys.append(_jsonable(list(key)))
        else:
            by_key[key] = record
    zero_index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in ordered_records:
        if _role(record) != "zero":
            continue
        key = (
            record["sibling_group_id"],
            int(record["training_seed"]),
            _context(record, include_cell=False),
        )
        if key in zero_index:
            duplicate_keys.append(_jsonable(list(key)))
        else:
            zero_index[key] = record

    observed_contexts = {
            _context(record, include_cell=True)
            for record in ordered_records
            if _role(record) in {"coupled", "decoupled"}
        }
    planned_contexts = {
        _context(record, include_cell=True) for record in (expected_contexts or ())
    }
    missing_contexts = sorted(planned_contexts - observed_contexts, key=str)
    contexts = sorted(
        observed_contexts | planned_contexts,
        key=str,
    )
    aggregates: list[dict[str, Any]] = []
    alpha_records: list[dict[str, Any]] = []
    failure_count = 0
    incomplete_count = 0
    for context in contexts:
        family, cell_id, checkpoint_role, checkpoint_epoch, depth, split, transform = context
        expected_seeds = tuple(int(seed) for seed in expected_seeds_by_cell[cell_id])
        context_rows = [
            record
            for record in ordered_records
            if _context(record, include_cell=True) == context
        ]
        metric_names = sorted(
            {
                str(name)
                for record in context_rows
                for name, value in record.get("metrics", {}).items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        )
        if not metric_names:
            incomplete_count += 1
            continue
        for direction, left_role, right_role in PAIR_DIRECTIONS:
            for metric in metric_names:
                seed_rows: list[dict[str, Any]] = []
                deltas: list[float] = []
                for seed in expected_seeds:
                    left = by_key.get((context, seed, left_role))
                    right = by_key.get((context, seed, right_role))
                    if right_role == "zero" and left is not None:
                        right = zero_index.get(
                            (
                                left["sibling_group_id"],
                                seed,
                                _context(left, include_cell=False),
                            )
                        )
                    if left is None or right is None:
                        seed_rows.append(
                            {
                                "training_seed": seed,
                                "status": "MISSING",
                                "left_present": left is not None,
                                "right_present": right is not None,
                            }
                        )
                        continue
                    if left.get("status") != "PASS" or right.get("status") != "PASS":
                        seed_rows.append(
                            {
                                "training_seed": seed,
                                "status": "FAILED",
                                "left_status": left.get("status"),
                                "right_status": right.get("status"),
                            }
                        )
                        continue
                    identity_fields = (
                        "sibling_group_id",
                        "initialization_sha256",
                        "data_stream_sha256",
                    )
                    if any(left.get(field) != right.get(field) for field in identity_fields):
                        seed_rows.append(
                            {
                                "training_seed": seed,
                                "status": "FAILED",
                                "reason": "sibling_identity_mismatch",
                            }
                        )
                        continue
                    if (
                        metric not in left.get("metrics", {})
                        or metric not in right.get("metrics", {})
                    ):
                        seed_rows.append(
                            {
                                "training_seed": seed,
                                "status": "MISSING",
                                "reason": "metric_missing",
                            }
                        )
                        continue
                    delta = float(left["metrics"][metric]) - float(right["metrics"][metric])
                    deltas.append(delta)
                    seed_rows.append(
                        {
                            "training_seed": seed,
                            "status": "PASS",
                            "left_run_id": left["run_id"],
                            "right_run_id": right["run_id"],
                            "left_value": float(left["metrics"][metric]),
                            "right_value": float(right["metrics"][metric]),
                            "delta": delta,
                        }
                    )
                if any(row["status"] == "FAILED" for row in seed_rows):
                    status = "FAILED"
                    summary = None
                    failure_count += 1
                elif any(row["status"] != "PASS" for row in seed_rows):
                    status = "INCOMPLETE"
                    summary = None
                    incomplete_count += 1
                else:
                    status = "PASS"
                    summary = paired_t_interval(deltas)
                aggregates.append(
                    {
                        "family": family,
                        "cell_id": cell_id,
                        "checkpoint_role": checkpoint_role,
                        "checkpoint_epoch": checkpoint_epoch,
                        "depth_tap": depth,
                        "dataset_split": split,
                        "transform": transform,
                        "direction": direction,
                        "metric": metric,
                        "status": status,
                        "expected_seeds": list(expected_seeds),
                        "seed_records": seed_rows,
                        "paired_summary": summary,
                    }
                )

        if cell_id == PRIMARY_ANCHOR_CELL:
            for metric in metric_names:
                role_means: dict[str, float] = {}
                seed_rows: list[dict[str, Any]] = []
                alpha_role_names = {
                    "alpha_0": "decoupled",
                    "alpha_0_5": "alpha_0_5",
                    "alpha_1": "coupled",
                }
                for alpha_name, role_name in alpha_role_names.items():
                    values = []
                    missing = []
                    for seed in expected_seeds:
                        row = by_key.get((context, seed, role_name))
                        if (
                            row is None
                            or row.get("status") != "PASS"
                            or metric not in row.get("metrics", {})
                        ):
                            missing.append(seed)
                        else:
                            values.append(float(row["metrics"][metric]))
                    seed_rows.append(
                        {
                            "alpha": alpha_name,
                            "values": values,
                            "missing_or_failed_seeds": missing,
                        }
                    )
                    if not missing:
                        role_means[alpha_name] = float(np.mean(values))
                if len(role_means) == 3:
                    classification = classify_alpha_interior(
                        alpha_0_mean=role_means["alpha_0"],
                        alpha_0_5_mean=role_means["alpha_0_5"],
                        alpha_1_mean=role_means["alpha_1"],
                    )
                    status = "PASS"
                else:
                    classification = None
                    status = "INCOMPLETE"
                    incomplete_count += 1
                alpha_records.append(
                    {
                        "cell_id": cell_id,
                        "checkpoint_role": checkpoint_role,
                        "checkpoint_epoch": checkpoint_epoch,
                        "depth_tap": depth,
                        "dataset_split": split,
                        "transform": transform,
                        "metric": metric,
                        "status": status,
                        "seed_records": seed_rows,
                        "classification": classification,
                    }
                )

    utility_rows = [
        row
        for row in aggregates
        if row["direction"] == "coupled_minus_decoupled"
        and row["checkpoint_role"] == "last"
        and row["checkpoint_epoch"] == 200
        and row["depth_tap"] == "penultimate"
        and row["metric"] in ID_EQUIVALENCE_MARGINS
    ]
    by_cell_utility: dict[str, dict[str, Sequence[float]]] = defaultdict(dict)
    utility_scope: dict[str, str] = {}
    for row in utility_rows:
        if row["status"] == "PASS" and row["transform"] == "raw":
            by_cell_utility[row["cell_id"]][row["metric"]] = [
                float(seed["delta"]) for seed in row["seed_records"]
            ]
            utility_scope[row["cell_id"]] = str(row["dataset_split"])
    equivalence = {
        cell: adjudicate_id_equivalence(
            values,
            evidence_scope=utility_scope[cell],
            protected_id_test_available=protected_id_test_available,
        )
        for cell, values in sorted(by_cell_utility.items())
        if not protected_id_test_available or set(values) == set(ID_EQUIVALENCE_MARGINS)
    }
    status = (
        "FAILED" if duplicate_keys or failure_count else
        "INCOMPLETE" if incomplete_count or missing_contexts else "PASS"
    )
    payload = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "status": status,
        "protected_data_access": False,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "aggregation_policy": {
            "unit": "training_seed",
            "pool_across_family_or_cell": False,
            "directions": [name for name, _, _ in PAIR_DIRECTIONS],
            "uncertainty": "two_sided_paired_90_percent_t_ci_descriptive",
            "ci_is_id_equivalence_gate": False,
        },
        "counts": {
            "seed_records": len(ordered_records),
            "paired_records": len(aggregates),
            "alpha_records": len(alpha_records),
            "failed_aggregates": failure_count,
            "incomplete_aggregates": incomplete_count,
        },
        "duplicate_keys": duplicate_keys,
        "missing_contexts": _jsonable(missing_contexts),
        "terminal_manifest": {
            "status": status,
            "observed_context_count": len(observed_contexts),
            "expected_context_count": (
                len(planned_contexts) if planned_contexts else None
            ),
            "missing_contexts": _jsonable(missing_contexts),
            "duplicate_key_count": len(duplicate_keys),
            "failed_aggregate_count": failure_count,
            "incomplete_aggregate_count": incomplete_count,
            "missing_or_failed_records_are_excluded": False,
        },
        "seed_level_records": _jsonable(ordered_records),
        "paired_aggregate_records": aggregates,
        "alpha_classification_records": alpha_records,
        "id_equivalence": equivalence,
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    return payload


def render_table_templates(payload: Mapping[str, Any]) -> dict[str, str]:
    if payload.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError("unsupported paired aggregation payload")

    def markdown(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
        return "\n".join(lines) + "\n"

    utility = []
    for cell, record in sorted(payload["id_equivalence"].items()):
        utility.append(
            (
                cell,
                record["status"],
                record.get("comparable_id"),
                "0.01 / 0.08 / 0.02",
                "descriptive paired 90% t-CI",
            )
        )
    trajectory = []
    for row in payload["paired_aggregate_records"]:
        summary = row["paired_summary"] or {}
        trajectory.append(
            (
                row["cell_id"],
                row["checkpoint_epoch"],
                row["depth_tap"],
                row["transform"],
                row["direction"],
                row["metric"],
                row["status"],
                summary.get("mean", "NA"),
                summary.get("lower", "NA"),
                summary.get("upper", "NA"),
            )
        )
    alpha = []
    for row in payload["alpha_classification_records"]:
        classification = row["classification"] or {}
        alpha.append(
            (
                row["metric"],
                row["checkpoint_epoch"],
                row["depth_tap"],
                row["transform"],
                row["status"],
                classification.get("classification", "NA"),
            )
        )
    return {
        "id_utility_equivalence.md": markdown(
            ("Cell", "Status", "Comparable ID", "Margins A/N/E", "Uncertainty"),
            utility,
        ),
        "geometry_trajectory.md": markdown(
            (
                "Cell",
                "Epoch",
                "Depth",
                "Transform",
                "Contrast",
                "Metric",
                "Status",
                "Mean delta",
                "CI low",
                "CI high",
            ),
            trajectory,
        ),
        "alpha_classification.md": markdown(
            ("Metric", "Epoch", "Depth", "Transform", "Status", "Classification"),
            alpha,
        ),
    }


def write_aggregation_artifacts(
    *, payload: Mapping[str, Any], output_directory: str | Path
) -> Path:
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError("refusing to overwrite paired aggregation artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        (temporary / "paired_aggregation.json").write_bytes(
            canonical_json_bytes(payload) + b"\n"
        )
        tables = render_table_templates(payload)
        for name, content in tables.items():
            (temporary / name).write_text(content, encoding="utf-8")
        names = ("paired_aggregation.json", *sorted(tables))
        (temporary / "checksums.sha256").write_text(
            "".join(f"{_sha256_file(temporary / name)}  {name}\n" for name in names),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
