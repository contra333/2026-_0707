"""ID-only audit of the Card-13 discriminant--residual preflight archive.

The diagnostic reads only an immutable preflight tar archive and an explicitly
manifested set of ID-train NC1 arrays.  It never walks a metric bundle, opens a
checkpoint, or evaluates a detector or OOD sample.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr

from oge.studies.artifacts import atomic_write_json, sha256_file


SCHEMA_VERSION = "discriminant_residual_diagnostic_v1"
INPUT_SCHEMA_VERSION = "discriminant_residual_diagnostic_inputs_v1"
EXPECTED_PREFLIGHT_ARCHIVE_SHA256 = (
    "4f58de57551c42ac4c75a6621b05c8dbfdc5f3bb8072de08d137c614c9611a88"
)
EXPECTED_PREFLIGHT_SCHEMA = "discriminant_residual_preflight_v1"
EXPECTED_FIT_COUNT = 60
EXPECTED_BUNDLE_COUNT = 30
TRANSFORMS = ("raw", "l2")
NC1_LOGICAL_PATH = (
    "metrics/arrays/geometry/train/neural_collapse/"
    "nc1_diagnostics/trace_components.npy"
)
PREFLIGHT_MEMBERS = (
    "bundle_records.jsonl",
    "checksums.sha256",
    "summary.json",
    "tail_records.jsonl",
)
FORBIDDEN_PATH_FRAGMENTS = (
    "metrics/result.json",
    "/ood/",
    "checkpoint",
    ".pt",
)


class DiagnosticInputError(RuntimeError):
    """Raised before analysis when a required input is absent or inconsistent."""


class DiagnosticHardCap(RuntimeError):
    """Raised before another diagnostic unit starts after the four-hour cap."""


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


def _json_lines(payload: bytes, *, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise DiagnosticInputError(
                f"{member}:{line_number} must contain a JSON object"
            )
        rows.append(value)
    return rows


def _write_json_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def _write_checksums(output_dir: Path, names: Iterable[str]) -> None:
    lines = [
        f"{sha256_file(output_dir / name)}  {name}"
        for name in sorted(names)
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticInputError("diagnostic manifest must contain a JSON object")
    if value.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise DiagnosticInputError("unexpected diagnostic manifest schema")
    archive = value.get("preflight_archive", {})
    if archive.get("sha256") != EXPECTED_PREFLIGHT_ARCHIVE_SHA256:
        raise DiagnosticInputError("preflight archive SHA is not the frozen v1 SHA")
    if archive.get("required_members") != list(PREFLIGHT_MEMBERS):
        raise DiagnosticInputError("preflight archive member allowlist is not frozen")
    hypothesis = value.get("nc1_hypothesis", {})
    if hypothesis.get("predicted_spearman_sign") != "negative":
        raise DiagnosticInputError("NC1 direction prediction was not frozen")
    if value.get("historical_pass_rate_may_select_scale") is not False:
        raise DiagnosticInputError("historical pass rate must not select a scale")
    nc1_contract = value.get("nc1_array_contract", {})
    if nc1_contract != {
        "shape": [640],
        "dtype": "float64",
        "class_count": 10,
        "aggregation": "sum(trace_components)/class_count",
    }:
        raise DiagnosticInputError("NC1 array contract is not the frozen GEO-5 export")
    if value.get("v1_2_comparison") != {
        "covariance_denominator": "biased_1_over_N_in_both",
        "centering": "class_residual_and_global_mean_in_both",
        "l2": "per_sample_normalization_then_refit_in_both",
        "backend": "sklearn_scipy_pinvh_vs_card13_numpy_eigh",
    }:
        raise DiagnosticInputError("v1.2 comparison conventions are not frozen")
    entries = value.get("nc1_entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_BUNDLE_COUNT:
        raise DiagnosticInputError("NC1 manifest must enumerate exactly 30 bundles")
    seen: set[str] = set()
    hf_base = str(value.get("hf_base_uri", "")).rstrip("/")
    for entry in entries:
        bundle_id = str(entry.get("bundle_id", ""))
        if len(bundle_id) != 64 or any(char not in "0123456789abcdef" for char in bundle_id):
            raise DiagnosticInputError("NC1 bundle ID is not a lowercase SHA-256")
        if bundle_id in seen:
            raise DiagnosticInputError(f"duplicate NC1 bundle: {bundle_id}")
        seen.add(bundle_id)
        expected_local = f"{bundle_id}/trace_components.npy"
        expected_uri = f"{hf_base}/{bundle_id}/{NC1_LOGICAL_PATH}"
        if entry.get("local_path") != expected_local:
            raise DiagnosticInputError(f"unexpected NC1 local path: {bundle_id}")
        if entry.get("hf_uri") != expected_uri:
            raise DiagnosticInputError(f"unexpected NC1 URI: {bundle_id}")
        if len(str(entry.get("sha256", ""))) != 64:
            raise DiagnosticInputError(f"invalid NC1 checksum: {bundle_id}")
        for candidate in (expected_local, expected_uri):
            lowered = candidate.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_PATH_FRAGMENTS):
                raise DiagnosticInputError(
                    f"forbidden path in NC1 manifest: {candidate}"
                )
    return value


def _read_preflight_archive(
    archive_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not archive_path.is_file():
        raise DiagnosticInputError("required preflight archive is absent")
    observed = sha256_file(archive_path)
    if observed != expected_sha256:
        raise DiagnosticInputError("preflight archive SHA-256 mismatch")
    with tarfile.open(archive_path, mode="r:") as handle:
        members = {
            PurePosixPath(member.name).as_posix().removeprefix("./"): member
            for member in handle.getmembers()
            if member.isfile()
        }
        missing = sorted(set(PREFLIGHT_MEMBERS) - set(members))
        if missing:
            raise DiagnosticInputError(
                f"preflight archive lacks required members: {missing}"
            )
        payloads: dict[str, bytes] = {}
        for name in PREFLIGHT_MEMBERS:
            extracted = handle.extractfile(members[name])
            if extracted is None:
                raise DiagnosticInputError(f"cannot read preflight member: {name}")
            payloads[name] = extracted.read()
    checksums: dict[str, str] = {}
    for line in payloads["checksums.sha256"].decode("utf-8").splitlines():
        digest, logical_path = line.split(maxsplit=1)
        checksums[logical_path.lstrip("* ")] = digest
    for name in ("bundle_records.jsonl", "summary.json", "tail_records.jsonl"):
        observed_member = hashlib.sha256(payloads[name]).hexdigest()
        if checksums.get(name) != observed_member:
            raise DiagnosticInputError(f"preflight member checksum mismatch: {name}")
    summary = json.loads(payloads["summary.json"])
    if (
        summary.get("schema_version") != EXPECTED_PREFLIGHT_SCHEMA
        or summary.get("status") != "PASS"
        or summary.get("gate_2", {}).get("gate_status") != "INCONCLUSIVE"
    ):
        raise DiagnosticInputError("preflight summary is not the preserved v1 result")
    bundle_rows = _json_lines(
        payloads["bundle_records.jsonl"], member="bundle_records.jsonl"
    )
    tail_rows = _json_lines(
        payloads["tail_records.jsonl"], member="tail_records.jsonl"
    )
    source_records = [
        {
            "logical_input": "preflight_archive",
            "path": archive_path.name,
            "sha256": observed,
            "scope": "immutable_v1_archive_allowlisted_members_only",
        },
        *[
            {
                "logical_input": f"preflight_member:{name}",
                "path": name,
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                "scope": "archive_member_allowlist",
            }
            for name in ("bundle_records.jsonl", "summary.json", "tail_records.jsonl")
        ],
    ]
    return summary, bundle_rows, tail_rows, source_records


def _load_nc1(
    manifest: Mapping[str, Any], nc1_root: Path, *, started: float, hard_cap_seconds: float
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    expected_shape = tuple(manifest["nc1_array_contract"]["shape"])
    class_count = int(manifest["nc1_array_contract"]["class_count"])
    for entry in manifest["nc1_entries"]:
        if time.monotonic() - started >= hard_cap_seconds:
            raise DiagnosticHardCap("four-hour diagnostic hard cap reached")
        path = nc1_root / entry["local_path"]
        if not path.is_file():
            raise DiagnosticInputError(f"required NC1 input is absent: {entry['bundle_id']}")
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise DiagnosticInputError(f"NC1 checksum mismatch: {entry['bundle_id']}")
        components = np.load(path, allow_pickle=False)
        if components.shape != expected_shape or components.dtype != np.float64:
            raise DiagnosticInputError(f"NC1 shape/dtype mismatch: {entry['bundle_id']}")
        if not np.isfinite(components).all():
            raise DiagnosticInputError(f"NC1 contains non-finite values: {entry['bundle_id']}")
        nc1 = float(np.sum(components, dtype=np.float64) / class_count)
        values[entry["bundle_id"]] = nc1
        records.append(
            {
                "logical_input": "id_train_nc1_trace_components",
                "bundle_id": entry["bundle_id"],
                "path": entry["local_path"],
                "hf_uri": entry["hf_uri"],
                "sha256": observed,
                "shape": list(components.shape),
                "dtype": str(components.dtype),
                "nc1_pinv": nc1,
                "scope": "id_train_only",
            }
        )
    return values, records


def _condition(row: Mapping[str, Any]) -> float:
    spectra = (row["numerical"]["within_spectrum"], row["numerical"]["global_spectrum"])
    values: list[float] = []
    for spectrum in spectra:
        value = spectrum["condition_number"]
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            return math.inf
        values.append(float(value))
    return max(values)


def _applicability_reason(row: Mapping[str, Any]) -> str | None:
    if bool(row["applicable"]):
        return None
    for key in ("within_spectrum", "global_spectrum"):
        spectrum = row["numerical"][key]
        if int(spectrum["rank"]) < int(spectrum["dimension"]):
            return "rank_deficient"
    if _condition(row) > float(row["numerical"]["condition_ceiling"]):
        return "condition_ceiling"
    return "other_applicability_failure"


def _failed_required_checks(row: Mapping[str, Any]) -> list[str]:
    failures = [
        f"geometry.{name}"
        for name, passed in row["numerical"]["checks"].items()
        if not passed
    ]
    for split in ("id_train", "id_test"):
        failures.extend(
            f"{split}.{name}"
            for name, passed in row[split]["checks"].items()
            if not passed
        )
    if not row["id_train_residual_energy"]["pass"]:
        failures.append("id_train.residual_energy")
    return sorted(failures)


def _failed_cached_parity(row: Mapping[str, Any]) -> list[str]:
    tau = float(row["numerical"]["tau_alg"])
    return sorted(
        name
        for name, residual in row["cached_metric_v1_2_id_test_parity"][
            "scaled_max_residuals"
        ].items()
        if float(residual) > tau
    )


def _counter_rows(counter: Counter[tuple[str, ...]], names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {**dict(zip(names, key, strict=True)), "count": count}
        for key, count in sorted(counter.items())
    ]


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "median": float(np.median(values)) if values else None,
        "maximum": max(values) if values else None,
    }


def _safe_spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.var(x) == 0.0 or np.var(y) == 0.0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def cancellation_forward_error_fixture() -> dict[str, Any]:
    """Compare output- and operand-scaled cancellation under bounded errors."""

    getcontext().prec = 80
    exact_global = Decimal("640")
    exact_class = Decimal("632")
    exact_parallel_global = Decimal("640")
    exact_parallel_class = Decimal("632")
    exact_direct = exact_global - exact_class
    exact_decomposed = exact_parallel_global - exact_parallel_class
    algebraic_equivalence = exact_direct == exact_decomposed == Decimal("8")
    eps = np.finfo(np.float64).eps
    rows: list[dict[str, Any]] = []
    for condition in (1.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8):
        relative_error = 0.75 * condition * eps
        direct_global = 640.0 * (1.0 + relative_error)
        direct_class = 632.0 * (1.0 - relative_error)
        parallel_global = 640.0 * (1.0 - relative_error)
        parallel_class = 632.0 * (1.0 + relative_error)
        direct_rmd = direct_global - direct_class
        decomposed_rmd = parallel_global - parallel_class
        residual = abs(direct_rmd - decomposed_rmd)
        output_scale = max(1.0, abs(direct_rmd), abs(decomposed_rmd))
        operand_scale = max(
            1.0,
            abs(direct_global) + abs(direct_class),
            abs(parallel_global) + abs(parallel_class),
        )
        tau_alg = max(1.0e-10, 10.0 * condition * eps)
        output_scaled = residual / output_scale
        operand_scaled = residual / operand_scale
        rows.append(
            {
                "condition_number": condition,
                "tau_alg": tau_alg,
                "maximum_injected_relative_operand_error": relative_error,
                "relative_operand_error_bound": condition * eps,
                "forward_error_within_bound": relative_error <= condition * eps,
                "direct_rmd_float64": direct_rmd,
                "decomposed_rmd_float64": decomposed_rmd,
                "absolute_cancellation_residual": residual,
                "output_scaled_residual": output_scaled,
                "output_scale_pass": output_scaled <= tau_alg,
                "operand_scaled_residual": operand_scaled,
                "operand_scale_pass": operand_scaled <= tau_alg,
            }
        )
    criteria = {
        "high_precision_algebraic_equivalence": algebraic_equivalence,
        "all_injected_errors_within_condition_eps": all(
            row["forward_error_within_bound"] for row in rows
        ),
        "operand_scale_respects_tau_on_grid": all(
            row["operand_scale_pass"] for row in rows
        ),
        "output_scale_understates_error_on_grid": any(
            not row["output_scale_pass"] for row in rows
        ),
    }
    return {
        "exact_reference": {
            "direct_global": str(exact_global),
            "direct_class": str(exact_class),
            "parallel_global": str(exact_parallel_global),
            "parallel_class": str(exact_parallel_class),
            "rmd": str(exact_direct),
        },
        "model": "four exact operands with individually bounded float64 forward error",
        "historical_pass_rate_used": False,
        "rows": rows,
        "criteria": criteria,
        "supports_operand_scale_clarification": all(criteria.values()),
    }


def _analyze_rows(
    bundle_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
    nc1: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(bundle_rows) != EXPECTED_FIT_COUNT or len(tail_rows) != EXPECTED_FIT_COUNT:
        raise DiagnosticInputError("preflight records do not conserve 60 fits")
    keys = [(row["bundle_id"], row["transform"]) for row in bundle_rows]
    if len(set(keys)) != EXPECTED_FIT_COUNT:
        raise DiagnosticInputError("preflight bundle/transform keys are not unique")
    expected_keys = {(bundle_id, transform) for bundle_id in nc1 for transform in TRANSFORMS}
    if set(keys) != expected_keys:
        raise DiagnosticInputError("NC1 and preflight bundle populations differ")
    tail_keys = {(row["bundle_id"], row["transform"]) for row in tail_rows}
    if tail_keys != expected_keys:
        raise DiagnosticInputError("tail and preflight bundle populations differ")

    raw_rows = [row for row in bundle_rows if row["transform"] == "raw"]
    correlation_complete = True
    for row in raw_rows:
        correlations = row.get("raw_norm_component_correlations")
        if not isinstance(correlations, dict):
            correlation_complete = False
            break
        for split in ("id_train", "id_test"):
            if set(correlations.get(split, {})) != {
                "s_perp",
                "s_parallel_marginal",
                "s_rmd",
            }:
                correlation_complete = False
    if not correlation_complete:
        raise DiagnosticInputError("raw norm/component correlations lack train/test coverage")

    layer_a_reason = Counter()
    layer_a_transform = Counter()
    layer_a_optimizer = Counter()
    layer_a_config = Counter()
    layer_b_checks = Counter()
    layer_b_transform = Counter()
    layer_b_optimizer = Counter()
    layer_c_checks = Counter()
    layer_c_transform = Counter()
    layer_c_optimizer = Counter()
    diagnostic_rows: list[dict[str, Any]] = []
    nc1_by_reason: defaultdict[str, list[float]] = defaultdict(list)

    for row in sorted(bundle_rows, key=lambda item: (item["bundle_id"], item["transform"])):
        reason = _applicability_reason(row)
        failed_required = _failed_required_checks(row) if row["applicable"] else []
        failed_parity = _failed_cached_parity(row)
        if reason is not None:
            layer_a_reason[(reason,)] += 1
            layer_a_transform[(row["transform"], reason)] += 1
            layer_a_optimizer[(row["optimizer_family"], reason)] += 1
            layer_a_config[(row["config_hash"], row["optimizer_family"], reason)] += 1
        if row["applicable"] and not row["required_numerical_pass"]:
            layer_b_transform[(row["transform"],)] += 1
            layer_b_optimizer[(row["optimizer_family"],)] += 1
            for check in failed_required:
                layer_b_checks[(check,)] += 1
        if not row["cached_metric_v1_2_id_test_parity"]["pass"]:
            layer_c_transform[(row["transform"],)] += 1
            layer_c_optimizer[(row["optimizer_family"],)] += 1
            for check in failed_parity:
                layer_c_checks[(check,)] += 1
        group = reason or (
            "applicable_required_pass"
            if row["required_numerical_pass"]
            else "applicable_required_failure"
        )
        nc1_by_reason[group].append(nc1[row["bundle_id"]])
        diagnostic_rows.append(
            {
                "bundle_id": row["bundle_id"],
                "config_hash": row["config_hash"],
                "optimizer_family": row["optimizer_family"],
                "training_seed": row["training_seed"],
                "transform": row["transform"],
                "nc1_pinv": nc1[row["bundle_id"]],
                "condition_number": _condition(row),
                "layer_a_applicability_reason": reason,
                "layer_b_failed_required_checks": failed_required,
                "layer_c_failed_cached_parity_fields": failed_parity,
                "required_numerical_pass": row["required_numerical_pass"],
                "cached_parity_pass": row["cached_metric_v1_2_id_test_parity"]["pass"],
            }
        )

    associations: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        for optimizer in ("all", "adam", "adamw", "sgd"):
            selected = [
                row
                for row in diagnostic_rows
                if row["transform"] == transform
                and (optimizer == "all" or row["optimizer_family"] == optimizer)
                and isinstance(row["condition_number"], float)
                and math.isfinite(row["condition_number"])
                and row["condition_number"] > 0.0
            ]
            x = [float(row["nc1_pinv"]) for row in selected]
            y = [math.log10(float(row["condition_number"])) for row in selected]
            rho = _safe_spearman(x, y)
            associations.append(
                {
                    "transform": transform,
                    "optimizer_family": optimizer,
                    "fit_count": len(selected),
                    "excluded_nonfinite_condition_count": sum(
                        row["transform"] == transform
                        and (optimizer == "all" or row["optimizer_family"] == optimizer)
                        for row in diagnostic_rows
                    )
                    - len(selected),
                    "spearman_nc1_vs_log10_condition": rho,
                    "predicted_sign": "negative",
                    "direction_supported": None if rho is None else rho < 0.0,
                    "role": "descriptive_association_no_p_value_no_causal_claim",
                }
            )

    applicable_parity_failures = [
        row
        for row in diagnostic_rows
        if row["layer_a_applicability_reason"] is None
        and row["layer_c_failed_cached_parity_fields"]
    ]
    layer_b_failure_rows = [
        row for row in diagnostic_rows if row["layer_b_failed_required_checks"]
    ]
    evidence = {
        "count_conservation": {
            "fit_count": len(bundle_rows),
            "bundle_count": len(nc1),
            "transform_counts": dict(sorted(Counter(row["transform"] for row in bundle_rows).items())),
            "tail_fit_count": len(tail_rows),
            "pass": len(bundle_rows) == len(tail_rows) == EXPECTED_FIT_COUNT,
        },
        "layer_a_applicability": {
            "failure_fit_count": sum(layer_a_reason.values()),
            "reason_counts": _counter_rows(layer_a_reason, ("reason",)),
            "by_transform": _counter_rows(layer_a_transform, ("transform", "reason")),
            "by_optimizer_family": _counter_rows(layer_a_optimizer, ("optimizer_family", "reason")),
            "by_configuration": _counter_rows(
                layer_a_config, ("config_hash", "optimizer_family", "reason")
            ),
        },
        "layer_b_required_numerical": {
            "failure_fit_count": len(layer_b_failure_rows),
            "failed_check_counts": _counter_rows(layer_b_checks, ("check",)),
            "by_transform": _counter_rows(layer_b_transform, ("transform",)),
            "by_optimizer_family": _counter_rows(layer_b_optimizer, ("optimizer_family",)),
        },
        "layer_c_cached_parity_diagnostic": {
            "failure_fit_count": sum(layer_c_transform.values()),
            "failed_field_counts": _counter_rows(layer_c_checks, ("field",)),
            "by_transform": _counter_rows(layer_c_transform, ("transform",)),
            "by_optimizer_family": _counter_rows(layer_c_optimizer, ("optimizer_family",)),
            "applicable_failure_fit_count": len(applicable_parity_failures),
            "applicable_failures_only_rmd": all(
                row["layer_c_failed_cached_parity_fields"] == ["rmd"]
                for row in applicable_parity_failures
            ),
        },
        "nc1": {
            "hypothesis": "smaller NC1 predicts larger log10 condition number",
            "prediction_frozen_before_association": True,
            "associations": associations,
            "by_applicability_or_gate_reason": {
                key: _distribution(values) for key, values in sorted(nc1_by_reason.items())
            },
        },
        "raw_norm_component_correlation_regression": {
            "raw_fit_count": len(raw_rows),
            "id_train_complete_count": len(raw_rows),
            "id_test_complete_count": len(raw_rows),
            "pass": correlation_complete,
        },
        "tail_scope": {
            "status_counts": dict(sorted(Counter(row["status"] for row in tail_rows).items())),
            "gate_2": "INCONCLUSIVE_IMMUTABLE",
            "role": "conditional_discovery_on_numerically_applicable_subset",
        },
    }
    return evidence, diagnostic_rows


def run_diagnostic(
    *,
    input_manifest_path: str | Path,
    preflight_archive_path: str | Path,
    nc1_root: str | Path,
    output_dir: str | Path,
    hard_cap_seconds: float = 4.0 * 60.0 * 60.0,
) -> dict[str, Any]:
    """Run the deterministic ID-only archive diagnostic."""

    started = time.monotonic()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = Path(input_manifest_path)
        manifest = _load_manifest(manifest_path)
        preflight, bundle_rows, tail_rows, source_records = _read_preflight_archive(
            Path(preflight_archive_path), manifest["preflight_archive"]["sha256"]
        )
        nc1, nc1_records = _load_nc1(
            manifest,
            Path(nc1_root),
            started=started,
            hard_cap_seconds=hard_cap_seconds,
        )
        evidence, diagnostic_rows = _analyze_rows(bundle_rows, tail_rows, nc1)
        fixture = cancellation_forward_error_fixture()
        evidence["cancellation_forward_error_fixture"] = fixture
        layer_b_checks = {
            row["check"]
            for row in evidence["layer_b_required_numerical"]["failed_check_counts"]
        }
        cancellation_only = layer_b_checks == {
            "id_train.rmd_cancellation",
            "id_test.rmd_cancellation",
        }
        scale_supported = bool(fixture["supports_operand_scale_clarification"])
        applicable_parity_explained = bool(
            evidence["layer_c_cached_parity_diagnostic"][
                "applicable_failures_only_rmd"
            ]
        )
        inference = {
            "layer_a": (
                "Applicability failures are configuration- and optimizer-associated; "
                "the NC1 association is descriptive and does not establish causation."
            ),
            "layer_b": (
                "All applicable required-gate failures are confined to the two "
                "RMD-cancellation split checks."
                if cancellation_only
                else "Applicable required-gate failures are not cancellation-only."
            ),
            "layer_c": (
                "Applicable v1.2 parity failures are RMD-only and share the same "
                "cancellation-scale sensitivity; inapplicable fits remain outside the "
                "exact full-rank theorem claim."
                if applicable_parity_explained
                else "Applicable cached-parity differences contain unexplained fields."
            ),
            "nc1": (
                "The frozen-sign association is exploratory; rank-deficient fits have "
                "non-finite condition numbers and are excluded from Spearman summaries."
            ),
            "tail": (
                "Finite-tail evidence remains conditional on the numerically applicable "
                "historical subset and is not a Gate 2 PASS."
            ),
        }
        remediation_supported = bool(
            scale_supported and cancellation_only and applicable_parity_explained
        )
        decision = {
            "gate_2": {
                "status": "INCONCLUSIVE_IMMUTABLE",
                "retroactive_threshold_or_readjudication": False,
            },
            "rmd_cancellation": {
                "classification": (
                    "FROZEN_WORDING_AMBIGUITY_COMPLIANCE_CLARIFICATION"
                    if remediation_supported
                    else "NO_REMEDIATION_SUPPORTED"
                ),
                "candidate_scale": (
                    "max(1, abs(direct_global)+abs(direct_class), "
                    "abs(q_parallel_global)+abs(q_parallel_class))"
                ),
                "candidate_supported": scale_supported,
                "historical_pass_rate_used": False,
                "selection_basis": [
                    "high_precision_algebraic_equivalence",
                    "condition_aware_forward_error",
                    "synthetic_fixture_grid",
                ],
                "tau_alg_change": "NONE",
                "other_score_or_margin_scale_change": "NONE",
            },
            "metric_contract_v1_2_relationship": {
                "card_estimator_self_spec": "CONSISTENT_PENDING_V2_CONFIRMATION",
                "same_statistical_conventions": manifest["v1_2_comparison"],
                "applicable_parity_difference": (
                    "RMD_ONLY_CANCELLATION_SCALE_PENDING_V2_CONFIRMATION"
                    if applicable_parity_explained
                    else "UNEXPLAINED"
                ),
                "inapplicable_parity_difference": "OUTSIDE_FULL_RANK_THEOREM_SCOPE",
            },
            "common_ridge": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
            "estimator_status": (
                "CONDITIONAL_ONE_REMEDIATION_AND_V2_REQUIRED"
                if remediation_supported
                else "CLOSE_RTMD_OR_RETURN_TO_THEORY"
            ),
            "next_action": (
                "FREEZE_CARD13_V11_THEN_ONE_BOUNDED_V2"
                if remediation_supported
                else "NO_CODE_RESCUE"
            ),
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "issue": "https://github.com/contra333/2026-_0707/issues/75",
            "start_git_sha": "cc097cc318173cf5456a56e436ea4ccee262c7c2",
            "analysis_role": "historical_mixed_recipe_id_only_diagnostic",
            "claim_boundary": (
                "Evidence quality and numerical compliance diagnostic only; no causal, "
                "OOD, detector-performance, or Gate 2 activation claim."
            ),
            "input_manifest_sha256": sha256_file(Path(input_manifest_path)),
            "preflight_archive_sha256": EXPECTED_PREFLIGHT_ARCHIVE_SHA256,
            "preflight_gate_2": preflight["gate_2"],
            "evidence": evidence,
            "inference": inference,
            "decision": decision,
            "not_run": [
                "GPU",
                "checkpoint inference or reevaluation",
                "protected OOD access or evaluation",
                "fresh training",
                "full RtMD detector or OOD evaluation",
                "historical Gate 2 readjudication",
            ],
            "payload_files": ["diagnostic_records.jsonl", "input_records.jsonl"],
        }
        summary = _json_safe(summary)
        _write_json_lines(destination / "input_records.jsonl", [*source_records, *nc1_records])
        _write_json_lines(destination / "diagnostic_records.jsonl", diagnostic_rows)
        atomic_write_json(destination / "summary.json", summary)
        _write_checksums(destination, [*summary["payload_files"], "summary.json"])
        return summary
    except (DiagnosticInputError, DiagnosticHardCap, ValueError, OSError) as error:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "INCONCLUSIVE_INPUT",
            "issue": "https://github.com/contra333/2026-_0707/issues/75",
            "reason": str(error),
            "gate_2": "INCONCLUSIVE_IMMUTABLE",
            "not_run": [
                "diagnostic association and decision",
                "GPU",
                "checkpoint inference or reevaluation",
                "protected OOD access or evaluation",
                "fresh training",
            ],
        }
        summary = _json_safe(summary)
        atomic_write_json(destination / "summary.json", summary)
        _write_checksums(destination, ["summary.json"])
        return summary
