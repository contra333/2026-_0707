"""Execute the frozen Task F RtMD Gate 3 using verified fresh ID artifacts.

This module reads only ``id_train`` and ``id_validation`` feature artifacts.
It does not load checkpoints, traverse a dataset, score protected samples, or
implement the RtMD OOD detector.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from oge.analysis.discriminant_residual_preflight import (
    deterministic_class_stratified_twofold,
    fit_discriminant_geometry,
    fit_residual_nu,
    residual_radius,
)
from oge.analysis.task_f_fresh_id import verify_geometry_artifact
from oge.analysis.task_f_rtmd_gate import (
    RTMD_GATE3_SPEC,
    RTMD_GATE3_SPEC_SHA256,
    adjudicate_rtmd_gate3,
)
from oge.evaluation.task_f_fresh import (
    EXPECTED_SPECIFICATION_SHA256,
    verify_bridge_artifact,
)
from oge.evaluation.task_f_fresh_orchestration import SOURCE_TRAINING_SHA
from oge.feature_export import verify_task_f_artifact
from oge.studies.artifacts import sha256_file
from oge.studies.hashing import canonical_json_bytes, canonical_sha256


RECORD_SCHEMA_VERSION = "task_f_rtmd_gate3_record_v1"
HOST_SCHEMA_VERSION = "task_f_rtmd_gate3_host_records_v1"
TERMINAL_SCHEMA_VERSION = "task_f_rtmd_gate3_terminal_v1"
EXPECTED_HOST_COUNTS = {"curie": 4, "lise": 4, "precision_medicine": 2}
ROLE_BY_SIBLING = {"alpha_1": "coupled", "alpha_0": "decoupled"}


def _json_read(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _canonical_read(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = _json_read(source)
    if source.read_bytes() != canonical_json_bytes(payload) + b"\n":
        raise ValueError(f"{source} is not canonical JSON")
    return payload


def _write_directory(
    destination: str | Path,
    *,
    payloads: Mapping[str, Mapping[str, Any]],
) -> Path:
    output = Path(destination)
    expected_names = set(payloads) | {"checksums.sha256"}
    if output.exists():
        if {path.name for path in output.iterdir()} != expected_names:
            raise FileExistsError(f"refusing to overwrite different output: {output}")
        for name, payload in payloads.items():
            if _canonical_read(output / name) != payload:
                raise FileExistsError(
                    f"refusing to overwrite different output: {output / name}"
                )
        _verify_checksums(output, set(payloads))
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, payload in payloads.items():
            (temporary / name).write_bytes(canonical_json_bytes(payload) + b"\n")
        rows = [
            f"{sha256_file(temporary / name)}  {name}"
            for name in sorted(payloads)
        ]
        (temporary / "checksums.sha256").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_checksums(output, set(payloads))
    return output


def _verify_checksums(root: Path, names: set[str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in observed or name not in names or sha256_file(root / name) != digest:
            raise ValueError("RtMD Gate 3 checksum catalog is invalid")
        observed[name] = digest
    if set(observed) != names:
        raise ValueError("RtMD Gate 3 checksum catalog is incomplete")
    return observed


def _role(manifest: Mapping[str, Any]) -> str | None:
    return ROLE_BY_SIBLING.get(str(manifest.get("sibling_role")))


def _in_scope(manifest: Mapping[str, Any], *, split: str) -> bool:
    scope = RTMD_GATE3_SPEC["evidence_scope"]
    return bool(
        manifest.get("cell_id") == scope["cell_id"]
        and manifest.get("checkpoint_role") == scope["checkpoint_role"]
        and int(manifest.get("checkpoint_epoch", -1)) == scope["checkpoint_epoch"]
        and manifest.get("depth_tap") == scope["depth_tap"]
        and manifest.get("dataset_split") == split
        and _role(manifest) in scope["roles"]
        and int(manifest.get("training_seed", -1)) in scope["training_seeds"]
    )


def _verify_binding(binding: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    bridge_root = Path(binding["bridge_path"])
    feature_root = Path(binding["feature_artifact_path"])
    bridge = verify_bridge_artifact(bridge_root)["manifest"]
    feature = verify_task_f_artifact(feature_root)["manifest"]
    checks = {
        "split": bridge["dataset_split"] == feature["dataset_split"] == split,
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
        raise ValueError(f"RtMD Gate 3 binding identity checks failed: {failed}")
    if not _in_scope(bridge, split=split):
        raise ValueError("RtMD Gate 3 binding is outside the frozen scope")
    return {
        "bridge_root": bridge_root,
        "feature_root": feature_root,
        "bridge": bridge,
        "feature": feature,
    }


def _final_fit_applicable(manifest: Mapping[str, Any]) -> bool:
    raw = manifest["summary"]["transforms"]["raw"]
    checks = raw["fit"]["numerical"]["checks"]
    return bool(
        raw["status"] == "PASS"
        and raw["fit"]["applicable"] is True
        and all(bool(value) for value in checks.values())
    )


def fit_gate3_record(
    *,
    train_features: Any,
    train_labels: Any,
    train_sample_ids: Any,
    validation_q_perp: Any,
    residual_dim: int,
    final_fit_applicable: bool,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one frozen two-fold residual tail and build its Gate 3 record."""

    values = np.asanyarray(train_features)
    labels = np.asarray(train_labels, dtype=np.int64)
    sample_ids = np.asarray(train_sample_ids)
    heldout_q = np.asarray(validation_q_perp, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(labels) or len(labels) != len(sample_ids):
        raise ValueError("RtMD Gate 3 train arrays are not aligned")
    if heldout_q.ndim != 1 or heldout_q.size == 0:
        raise ValueError("RtMD Gate 3 held-out q_perp must be a non-empty vector")
    if residual_dim <= 0 or np.any(heldout_q < 0.0) or not np.isfinite(heldout_q).all():
        raise ValueError("RtMD Gate 3 held-out residual inputs are invalid")

    folds = deterministic_class_stratified_twofold(sample_ids, labels)
    oof_q = np.empty(len(values), dtype=np.float64)
    expected_classes = np.unique(labels)
    fold_records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    for heldout_fold in (0, 1):
        fit_mask = folds != heldout_fold
        query_mask = folds == heldout_fold
        observed_classes = np.unique(labels[fit_mask])
        if not np.array_equal(observed_classes, expected_classes):
            failure_reasons.append(
                f"fold_{heldout_fold}_fit_does_not_contain_every_class"
            )
            fold_records.append(
                {
                    "heldout_fold": heldout_fold,
                    "fit_count": int(np.count_nonzero(fit_mask)),
                    "query_count": int(np.count_nonzero(query_mask)),
                    "status": "INAPPLICABLE",
                    "reason": "fit_fold_does_not_contain_every_class",
                }
            )
            continue
        fold_fit = fit_discriminant_geometry(
            np.asarray(values[fit_mask], dtype=np.float64), labels[fit_mask]
        )
        fold_pass = bool(
            fold_fit.applicable
            and fold_fit.residual_dim == residual_dim
            and all(bool(value) for value in fold_fit.numerical["checks"].values())
        )
        if fold_pass:
            oof_q[query_mask] = residual_radius(
                fold_fit, np.asarray(values[query_mask], dtype=np.float64)
            )
        else:
            failure_reasons.append(
                f"fold_{heldout_fold}_geometry_not_numerically_applicable"
            )
        fold_records.append(
            {
                "heldout_fold": heldout_fold,
                "fit_count": int(np.count_nonzero(fit_mask)),
                "query_count": int(np.count_nonzero(query_mask)),
                "residual_dim": int(fold_fit.residual_dim),
                "condition_number": (
                    float(fold_fit.condition_number)
                    if math.isfinite(fold_fit.condition_number)
                    else "infinity"
                ),
                "status": "PASS" if fold_pass else "INAPPLICABLE",
                "checks": dict(fold_fit.numerical["checks"]),
            }
        )

    applicable = bool(final_fit_applicable and not failure_reasons)
    nu_fit = fit_residual_nu(oof_q, residual_dim) if applicable else None
    denominator = float(chi2.ppf(0.99, residual_dim))
    q99 = float(np.quantile(heldout_q, 0.99))
    tail_statistic = math.log(q99 / denominator) if applicable else None
    if applicable and (not math.isfinite(tail_statistic) or q99 <= 0.0):
        raise ValueError("RtMD Gate 3 tail statistic is outside its finite domain")
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        **dict(identity),
        "transform": "raw",
        "dataset_split": "id_validation",
        "numerically_applicable": applicable,
        "finite_t_selected": bool(
            nu_fit is not None and nu_fit["selected_model"] == "finite_t"
        ),
        "tail_statistic": tail_statistic,
        "tail_statistic_definition": RTMD_GATE3_SPEC["tail_statistic"]["name"],
        "residual_dim": int(residual_dim),
        "id_validation_q99": q99,
        "chi2_q99": denominator,
        "partition": {
            "method": "sha256_stable_id_rank_alternation_within_class",
            "fold_counts": [
                int(np.count_nonzero(folds == value)) for value in (0, 1)
            ],
            "folds": fold_records,
        },
        "failure_reasons": failure_reasons,
        "nu_fit": nu_fit,
        "protected_data_access": False,
        "rtmd_gate3_specification_sha256": RTMD_GATE3_SPEC_SHA256,
    }
    return record


def build_gate3_record_from_spec(
    *, worker_spec: Mapping[str, Any], geometry_root: str | Path
) -> dict[str, Any]:
    if worker_spec.get("stage") != "geometry":
        raise ValueError("RtMD Gate 3 worker spec must describe geometry")
    train = _verify_binding(worker_spec["train_binding"], split="id_train")
    validation = _verify_binding(
        worker_spec["validation_binding"], split="id_validation"
    )
    identity_fields = (
        "run_id",
        "cell_id",
        "training_seed",
        "sibling_group_id",
        "sibling_role",
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
        raise ValueError(f"RtMD Gate 3 train/validation bindings differ: {mismatches}")
    geometry = verify_geometry_artifact(geometry_root)["manifest"]
    bridge = train["bridge"]
    geometry_checks = {
        "run_id": geometry["run_id"] == bridge["run_id"],
        "checkpoint": geometry["checkpoint_sha256"] == bridge["checkpoint_sha256"],
        "train_feature": geometry["train_feature_identity"]
        == train["feature"]["output_identity_sha256"],
        "validation_feature": geometry["validation_feature_identity"]
        == validation["feature"]["output_identity_sha256"],
        "specification": geometry["task_f_specification_sha256"]
        == EXPECTED_SPECIFICATION_SHA256,
    }
    failed = sorted(name for name, passed in geometry_checks.items() if not passed)
    if failed:
        raise ValueError(f"RtMD Gate 3 geometry identity checks failed: {failed}")
    with np.load(
        Path(geometry_root) / "sample_components.npz", allow_pickle=False
    ) as arrays:
        validation_q_perp = np.asarray(
            arrays["raw__id_validation__q_perp"], dtype=np.float64
        )
    role = _role(bridge)
    if role is None:
        raise ValueError("RtMD Gate 3 sibling role is outside the frozen panel")
    identity = {
        "cell_id": bridge["cell_id"],
        "checkpoint_role": bridge["checkpoint_role"],
        "checkpoint_epoch": int(bridge["checkpoint_epoch"]),
        "depth_tap": bridge["depth_tap"],
        "role": role,
        "training_seed": int(bridge["training_seed"]),
        "run_id": bridge["run_id"],
        "sibling_group_id": bridge["sibling_group_id"],
        "initialization_sha256": bridge["initialization_sha256"],
        "data_stream_sha256": bridge["data_stream_sha256"],
        "checkpoint_sha256": bridge["checkpoint_sha256"],
        "geometry_output_identity_sha256": geometry["output_identity_sha256"],
        "train_feature_output_identity_sha256": train["feature"][
            "output_identity_sha256"
        ],
        "validation_feature_output_identity_sha256": validation["feature"][
            "output_identity_sha256"
        ],
    }
    train_features = np.load(
        train["feature_root"] / "features.npy", mmap_mode="r", allow_pickle=False
    )
    train_labels = np.load(
        train["bridge_root"] / "labels.npy", mmap_mode="r", allow_pickle=False
    )
    train_sample_ids = np.load(
        train["feature_root"] / "sample_ids.npy", mmap_mode="r", allow_pickle=False
    )
    raw_fit = geometry["summary"]["transforms"]["raw"]["fit"]
    return fit_gate3_record(
        train_features=train_features,
        train_labels=train_labels,
        train_sample_ids=train_sample_ids,
        validation_q_perp=validation_q_perp,
        residual_dim=int(raw_fit["dim_s_perp"]),
        final_fit_applicable=_final_fit_applicable(geometry),
        identity=identity,
    )


def run_gate3_host(
    *,
    host_id: str,
    worker_spec_root: str | Path,
    ledger_path: str | Path,
    expected_evaluation_git_sha: str,
    execution_git_sha: str,
    output_directory: str | Path,
) -> Path:
    if host_id not in EXPECTED_HOST_COUNTS:
        raise ValueError("unknown Task F Gate 3 host")
    ledger = _json_read(ledger_path)
    if ledger.get("status") != "PASS":
        raise ValueError("Task F ID-only ledger is not terminal PASS")
    identity = ledger.get("identity", {})
    if identity.get("evaluation_git_sha") != expected_evaluation_git_sha:
        raise ValueError("Task F evaluation Git SHA mismatch")
    if identity.get("source_training_sha") != SOURCE_TRAINING_SHA:
        raise ValueError("Task F source training SHA mismatch")
    if identity.get("task_f_specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("Task F specification identity mismatch")

    selected: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    for path in sorted(Path(worker_spec_root).glob("*.json")):
        spec = _json_read(path)
        if spec.get("stage") != "geometry":
            continue
        try:
            train_manifest = verify_bridge_artifact(
                spec["train_binding"]["bridge_path"]
            )["manifest"]
        except (KeyError, FileNotFoundError):
            continue
        if not _in_scope(train_manifest, split="id_train"):
            continue
        role = _role(train_manifest)
        key = (str(role), int(train_manifest["training_seed"]))
        if key in selected:
            raise ValueError(f"duplicate RtMD Gate 3 worker spec: {key}")
        selected[key] = (spec, path)
    expected_count = EXPECTED_HOST_COUNTS[host_id]
    if len(selected) != expected_count:
        raise ValueError(
            f"RtMD Gate 3 host selection differs: {len(selected)} != {expected_count}"
        )

    records: list[dict[str, Any]] = []
    for key, (spec, spec_path) in sorted(selected.items()):
        job_id = str(spec["job_id"])
        ledger_record = ledger["jobs"].get(job_id)
        if not isinstance(ledger_record, Mapping) or ledger_record.get("status") != "PASS":
            raise ValueError(f"RtMD Gate 3 source geometry is not PASS: {job_id}")
        geometry_root = Path(ledger_record["result"]["artifact_path"])
        record = build_gate3_record_from_spec(
            worker_spec=spec, geometry_root=geometry_root
        )
        record["host_id"] = host_id
        record["source_worker_spec_sha256"] = sha256_file(spec_path)
        records.append(record)
    payload = {
        "schema_version": HOST_SCHEMA_VERSION,
        "status": "PASS",
        "host_id": host_id,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": expected_evaluation_git_sha,
        "execution_git_sha": execution_git_sha,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "rtmd_gate3_specification_sha256": RTMD_GATE3_SPEC_SHA256,
        "record_count": len(records),
        "records": records,
        "protected_data_access": False,
    }
    return _write_directory(
        output_directory, payloads={"HOST_GATE3_RECORDS.json": payload}
    )


def verify_gate3_host_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if {item.name for item in root.iterdir()} != {
        "HOST_GATE3_RECORDS.json",
        "checksums.sha256",
    }:
        raise ValueError("RtMD Gate 3 host output inventory differs")
    _verify_checksums(root, {"HOST_GATE3_RECORDS.json"})
    payload = _canonical_read(root / "HOST_GATE3_RECORDS.json")
    if payload.get("schema_version") != HOST_SCHEMA_VERSION:
        raise ValueError("unsupported RtMD Gate 3 host schema")
    if payload.get("status") != "PASS" or payload.get("protected_data_access") is not False:
        raise ValueError("RtMD Gate 3 host output is not ID-only PASS")
    if payload.get("record_count") != len(payload.get("records", [])):
        raise ValueError("RtMD Gate 3 host record count mismatch")
    if payload.get("record_count") != EXPECTED_HOST_COUNTS.get(payload.get("host_id")):
        raise ValueError("RtMD Gate 3 host coverage mismatch")
    return payload


def collect_gate3(
    *, host_outputs: Sequence[str | Path], output_directory: str | Path
) -> Path:
    if len(host_outputs) != len(EXPECTED_HOST_COUNTS):
        raise ValueError("RtMD Gate 3 collection requires exactly three hosts")
    hosts: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    for path in host_outputs:
        payload = verify_gate3_host_output(path)
        host_id = str(payload["host_id"])
        if host_id in hosts:
            raise ValueError(f"duplicate RtMD Gate 3 host: {host_id}")
        identity = {
            key: payload[key]
            for key in (
                "source_training_sha",
                "evaluation_git_sha",
                "execution_git_sha",
                "task_f_specification_sha256",
                "rtmd_gate3_specification_sha256",
            )
        }
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError("RtMD Gate 3 host execution identities differ")
        hosts[host_id] = {
            "record_count": int(payload["record_count"]),
            "records_sha256": sha256_file(Path(path) / "HOST_GATE3_RECORDS.json"),
        }
        records.extend(payload["records"])
    if set(hosts) != set(EXPECTED_HOST_COUNTS) or len(records) != 10:
        raise ValueError("RtMD Gate 3 collection coverage differs")
    keys = [(str(row["role"]), int(row["training_seed"])) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("RtMD Gate 3 collection contains duplicate records")
    records = sorted(records, key=lambda row: (row["role"], row["training_seed"]))
    adjudication = adjudicate_rtmd_gate3(records)
    record_payload = {
        "schema_version": "task_f_rtmd_gate3_inputs_v1",
        **dict(shared_identity or {}),
        "record_count": len(records),
        "records": records,
        "protected_data_access": False,
    }
    terminal = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": "PASS",
        **dict(shared_identity or {}),
        "host_count": len(hosts),
        "hosts": dict(sorted(hosts.items())),
        "record_count": len(records),
        "gate3_verdict": adjudication,
        "rtmd_included_in_protected_plan": bool(adjudication["activated"]),
        "final_research_terminal": False,
        "protected_data_access": False,
    }
    terminal["gate3_inputs_sha256"] = canonical_sha256(record_payload)
    return _write_directory(
        output_directory,
        payloads={
            "gate3_inputs.json": record_payload,
            "GATE3_COMPLETE.json": terminal,
        },
    )


def verify_gate3_terminal(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    expected = {"gate3_inputs.json", "GATE3_COMPLETE.json", "checksums.sha256"}
    if {item.name for item in root.iterdir()} != expected:
        raise ValueError("RtMD Gate 3 terminal inventory differs")
    _verify_checksums(root, expected - {"checksums.sha256"})
    inputs = _canonical_read(root / "gate3_inputs.json")
    terminal = _canonical_read(root / "GATE3_COMPLETE.json")
    if terminal.get("schema_version") != TERMINAL_SCHEMA_VERSION:
        raise ValueError("unsupported RtMD Gate 3 terminal schema")
    if terminal.get("status") != "PASS" or terminal.get("protected_data_access") is not False:
        raise ValueError("RtMD Gate 3 terminal is not ID-only PASS")
    if terminal.get("gate3_inputs_sha256") != canonical_sha256(inputs):
        raise ValueError("RtMD Gate 3 input identity mismatch")
    expected_verdict = adjudicate_rtmd_gate3(inputs["records"])
    if terminal.get("gate3_verdict") != expected_verdict:
        raise ValueError("RtMD Gate 3 verdict mismatch")
    return terminal


__all__ = [
    "EXPECTED_HOST_COUNTS",
    "HOST_SCHEMA_VERSION",
    "RECORD_SCHEMA_VERSION",
    "TERMINAL_SCHEMA_VERSION",
    "build_gate3_record_from_spec",
    "collect_gate3",
    "fit_gate3_record",
    "run_gate3_host",
    "verify_gate3_host_output",
    "verify_gate3_terminal",
]
