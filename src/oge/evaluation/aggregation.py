"""Deterministic checkpoint/seed aggregation for Metric Contract v1.2."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oge.studies.supervisor import build_artifact_manifest

from .analysis import detector_rank_concordance
from .extraction import sha256_file
from .production import verify_production_bundle

IDENTITY_FIELDS = {
    "checkpoint_role",
    "reporting_role",
    "checkpoint_sha256",
    "completed_epoch",
    "config_hash",
    "training_seed",
    "optimizer_family",
    "labels",
    "evaluation_git_sha",
    "status",
    "reason_codes",
    "value",
    "runtime_diagnostics",
}

RANK_CONCORDANCE_DETECTORS = {
    "ood_score/msp",
    "ood_score/max_logit",
    "ood_score/energy_t1",
    "detector/mahalanobis_raw",
    "detector/relative_mahalanobis_raw",
    "detector/mahalanobis_pp",
    "detector/relative_mahalanobis_pp",
    "detector/knn_l2_k50",
    "detector/ctm_prototype_cosine",
    "detector/vim_author_dim",
    "detector/neco_author_std_dim100",
}


def _canonical_record_key(record: Mapping[str, Any]) -> str:
    dimensions = {
        key: value
        for key, value in record.items()
        if key not in IDENTITY_FIELDS
    }
    return json.dumps(dimensions, sort_keys=True, separators=(",", ":"))


def discover_bundles(bundle_roots: Sequence[str | Path]) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for raw_root in bundle_roots:
        root = Path(raw_root)
        candidates = [root] if (root / "JOB_COMPLETE.json").is_file() else root.glob("*")
        for candidate in candidates:
            if not candidate.is_dir() or not (candidate / "JOB_COMPLETE.json").is_file():
                continue
            sha = candidate.name
            if sha in discovered:
                raise ValueError(f"duplicate local bundle for checkpoint {sha}")
            discovered[sha] = candidate
    return discovered


def _verify_operational_root(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    sidecar_path = root / "artifact_manifest.json.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"operational shard lacks artifact manifest: {root}")
    sidecar = sidecar_path.read_text(encoding="utf-8").split()
    if not sidecar or sidecar[0] != sha256_file(manifest_path):
        raise ValueError(f"operational shard artifact-manifest checksum mismatch: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = root / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["size"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"operational shard file mismatch: {entry['path']}")
    ledger = json.loads((root / "host_ledger.json").read_text(encoding="utf-8"))
    if ledger.get("status") != "REMOTE_VERIFIED" or ledger.get("failed_job_ids"):
        raise ValueError("operational shard host ledger is not terminal-success")
    return ledger


def discover_operational_jobs(
    operational_roots: Sequence[str | Path],
) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for raw_root in operational_roots:
        root = Path(raw_root)
        candidates = (
            [root]
            if (root / "host_ledger.json").is_file()
            else [path for path in root.iterdir() if path.is_dir()]
        )
        for candidate in candidates:
            if not (candidate / "host_ledger.json").is_file():
                continue
            _verify_operational_root(candidate)
            for job_root in (candidate / "jobs").iterdir():
                if not job_root.is_dir():
                    continue
                if job_root.name in discovered:
                    raise ValueError(
                        f"duplicate operational shard for checkpoint {job_root.name}"
                    )
                discovered[job_root.name] = job_root
    return discovered


def aggregate_production_bundles(
    *,
    inventory: Mapping[str, Any],
    bundle_roots: Sequence[str | Path],
    output_dir: str | Path,
    operational_roots: Sequence[str | Path] = (),
) -> Path:
    jobs = list(inventory["checkpoint_jobs"])
    expected = {str(job["checkpoint_sha256"]): job for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError("inventory repeats checkpoint identities")
    if bundle_roots and operational_roots:
        raise ValueError("use either full bundle roots or operational shard roots")
    sources = (
        discover_operational_jobs(operational_roots)
        if operational_roots
        else discover_bundles(bundle_roots)
    )
    if set(sources) != set(expected):
        raise ValueError(
            "local bundle population differs from inventory: "
            f"missing={sorted(set(expected).difference(sources))}, "
            f"unexpected={sorted(set(sources).difference(expected))}"
        )

    checkpoint_records: list[dict[str, Any]] = []
    record_sets: dict[str, dict[str, dict[str, Any]]] = {}
    evaluation_shas: set[str] = set()
    checkpoint_index: list[dict[str, Any]] = []
    for checkpoint_sha, job in expected.items():
        source = sources[checkpoint_sha]
        if operational_roots:
            completion = json.loads(
                (source / "JOB_COMPLETE.json").read_text(encoding="utf-8")
            )
            upload = json.loads(
                (source / "upload_evidence.json").read_text(encoding="utf-8")
            )
            if (
                upload.get("status") != "REMOTE_VERIFIED"
                or upload.get("destination") != job["hf_output_uri"]
            ):
                raise ValueError(f"checkpoint {checkpoint_sha} lacks matching remote evidence")
            scalar_path = source / "scalar_records.jsonl"
            source_kind = "checksum_verified_operational_shard"
        else:
            verified = verify_production_bundle(source)
            completion = verified["completion"]
            scalar_path = source / "metrics/scalar_records.jsonl"
            source_kind = "full_local_bundle"
        checks = {
            "job_id": completion["job_id"] == job["job_id"],
            "checkpoint_role": completion["checkpoint_role"] == job["checkpoint_role"],
            "inventory_hash": completion["inventory_hash"] == inventory["inventory_hash"],
            "dataset_policy_hash": completion["dataset_policy_hash"] == inventory["dataset_policy_hash"],
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"bundle {checkpoint_sha} identity mismatch: {failed}")
        rows = [
            json.loads(row)
            for row in scalar_path.read_text(encoding="utf-8").splitlines()
            if row.strip()
        ]
        by_key: dict[str, dict[str, Any]] = {}
        for record in rows:
            if (
                record["checkpoint_sha256"] != checkpoint_sha
                or record["checkpoint_role"] != job["checkpoint_role"]
                or record["config_hash"] != job["config_hash"]
                or int(record["training_seed"]) != int(job["training_seed"])
            ):
                raise ValueError(f"bundle {checkpoint_sha} contains a foreign scalar record")
            key = _canonical_record_key(record)
            if key in by_key:
                raise ValueError(f"bundle {checkpoint_sha} repeats a scalar record identity")
            by_key[key] = record
            checkpoint_records.append(record)
            evaluation_shas.add(str(record["evaluation_git_sha"]))
        record_sets[checkpoint_sha] = by_key
        checkpoint_index.append(
            {
                "job_id": job["job_id"],
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_role": job["checkpoint_role"],
                "config_hash": job["config_hash"],
                "training_seed": int(job["training_seed"]),
                "scalar_record_count": len(rows),
                "source_reference": f"{source_kind}/{checkpoint_sha}",
                "source_kind": source_kind,
            }
        )
    if len(evaluation_shas) != 1:
        raise ValueError("bundles do not share one clean evaluation Git SHA")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in checkpoint_records:
        grouped[
            (
                str(record["config_hash"]),
                str(record["checkpoint_role"]),
                _canonical_record_key(record),
            )
        ].append(record)
    aggregates: list[dict[str, Any]] = []
    for (config_hash, checkpoint_role, record_key), rows in sorted(grouped.items()):
        seeds = [int(row["training_seed"]) for row in rows]
        if sorted(seeds) != [0, 1, 2] or len(set(seeds)) != 3:
            raise ValueError(
                f"aggregate {config_hash}/{checkpoint_role} lacks exact seeds 0,1,2"
            )
        template = json.loads(record_key)
        statuses = [str(row["status"]) for row in rows]
        values = [row["value"] for row in rows]
        if all(status == "success" and value is not None for status, value in zip(statuses, values)):
            numeric = np.asarray(values, dtype=np.float64)
            status = "success"
            mean = float(np.mean(numeric))
            sample_sd = float(np.std(numeric, ddof=1))
            reason_codes: list[str] = []
        else:
            status = "failed" if "failed" in statuses else "degenerate"
            mean = None
            sample_sd = None
            reason_codes = sorted(
                {
                    str(reason)
                    for row in rows
                    for reason in row.get("reason_codes", [])
                }
            )
            if not reason_codes:
                reason_codes = ["seed_status_propagation"]
        by_seed = {
            str(row["training_seed"]): {
                "checkpoint_sha256": row["checkpoint_sha256"],
                "status": row["status"],
                "reason_codes": row["reason_codes"],
                "value": row["value"],
            }
            for row in sorted(rows, key=lambda item: int(item["training_seed"]))
        }
        first = rows[0]
        aggregates.append(
            {
                "schema_version": "1.0",
                "definition_version": first["definition_version"],
                "inventory_hash": inventory["inventory_hash"],
                "dataset_policy_hash": inventory["dataset_policy_hash"],
                "evaluation_git_sha": next(iter(evaluation_shas)),
                "config_hash": config_hash,
                "checkpoint_role": checkpoint_role,
                "reporting_role": first["reporting_role"],
                "optimizer_family": first["optimizer_family"],
                "labels": first["labels"],
                **template,
                "seed_count": 3,
                "seed_values": by_seed,
                "aggregate_status": status,
                "aggregate_reason_codes": reason_codes,
                "mean": mean,
                "sample_sd_ddof1": sample_sd,
            }
        )

    concordance_inputs: dict[
        tuple[str, str, str], dict[str, dict[str, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in aggregates:
        if (
            row["artifact_key"]
            not in {
                "ood_metric/auroc/near_macro_mean",
                "ood_metric/auroc/far_macro_mean",
            }
            or row.get("detector") not in RANK_CONCORDANCE_DETECTORS
            or row["aggregate_status"] != "success"
        ):
            continue
        group = str(row["ood_group"])
        for label in row["labels"]:
            match = re.fullmatch(r"role:([^:]+):(C[1-4])", str(label))
            if match is None:
                continue
            optimizer, role_code = match.groups()
            concordance_inputs[(row["checkpoint_role"], group, role_code)][optimizer][
                str(row["detector"])
            ] = float(row["mean"])
    concordance_records: list[dict[str, Any]] = []
    for (checkpoint_role, group, role_code), optimizer_values in sorted(
        concordance_inputs.items()
    ):
        panel_sets = [set(values) for values in optimizer_values.values()]
        if len(optimizer_values) < 2 or any(
            panel != RANK_CONCORDANCE_DETECTORS for panel in panel_sets
        ):
            concordance_records.append(
                {
                    "artifact_key": "analysis/detector_rank_concordance_tau_b",
                    "formula_id": "OOD-6",
                    "checkpoint_role": checkpoint_role,
                    "ood_group": group,
                    "role_code": role_code,
                    "status": "failed",
                    "reason_codes": ["incomplete_optimizer_detector_panel"],
                    "optimizer_detector_counts": {
                        name: len(values) for name, values in optimizer_values.items()
                    },
                }
            )
            continue
        payload = detector_rank_concordance(optimizer_values)
        for optimizer_pair, pair in payload["pairs"].items():
            concordance_records.append(
                {
                    "artifact_key": "analysis/detector_rank_concordance_tau_b",
                    "formula_id": "OOD-6",
                    "checkpoint_role": checkpoint_role,
                    "ood_group": group,
                    "role_code": role_code,
                    "optimizer_pair": optimizer_pair,
                    "detectors": payload["detectors"],
                    "status": pair["metric"]["status"],
                    "reason_codes": pair["metric"]["reason_codes"],
                    "value": pair["metric"]["value"],
                    "pvalue": pair["pvalue"],
                    "left_ranks": pair["left_ranks"].tolist(),
                    "right_ranks": pair["right_ranks"].tolist(),
                }
            )

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"aggregate output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        with (temporary / "per_checkpoint_records.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record in sorted(
                checkpoint_records,
                key=lambda item: (
                    item["config_hash"],
                    item["checkpoint_role"],
                    int(item["training_seed"]),
                    item["artifact_key"],
                    item["query_split"],
                    json.dumps(
                        {
                            key: item.get(key)
                            for key in (
                                "detector",
                                "ood_dataset",
                                "ood_group",
                                "statistic",
                                "quantile_probability",
                                "class_index",
                            )
                        },
                        sort_keys=True,
                    ),
                ),
            ):
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        with (temporary / "seed_aggregates.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record in aggregates:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        with (temporary / "detector_rank_concordance.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record in concordance_records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        summary = {
            "schema_version": "1.0",
            "definition_version": checkpoint_records[0]["definition_version"],
            "status": "AGGREGATED",
            "inventory_hash": inventory["inventory_hash"],
            "dataset_policy_hash": inventory["dataset_policy_hash"],
            "evaluation_git_sha": next(iter(evaluation_shas)),
            "checkpoint_job_count": len(expected),
            "training_identity_count": len(
                {(job["config_hash"], int(job["training_seed"])) for job in jobs}
            ),
            "configuration_count": len({job["config_hash"] for job in jobs}),
            "checkpoint_roles": ["last", "best_val"],
            "seed_set": [0, 1, 2],
            "per_checkpoint_scalar_record_count": len(checkpoint_records),
            "seed_aggregate_record_count": len(aggregates),
            "detector_rank_concordance_record_count": len(concordance_records),
            "successful_aggregate_count": sum(
                item["aggregate_status"] == "success" for item in aggregates
            ),
            "non_success_aggregate_count": sum(
                item["aggregate_status"] != "success" for item in aggregates
            ),
            "checkpoint_index": checkpoint_index,
        }
        (temporary / "aggregate.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        build_artifact_manifest(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
    return destination
