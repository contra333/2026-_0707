import json
import shutil

import numpy as np
import pytest

from oge.evaluation.aggregation import aggregate_production_bundles
from oge.evaluation.extraction import sha256_file
from oge.studies.artifacts import atomic_write_json
from oge.studies.supervisor import build_artifact_manifest


def _write_bundle(root, job, *, evaluation_sha, value, status="success"):
    root.mkdir(parents=True)
    metrics = root / "metrics"
    metrics.mkdir()
    record = {
        "schema_version": "1.0",
        "definition_version": "metric_contract_v1.2",
        "formula_id": "ID-1",
        "artifact_key": "id/accuracy",
        "reporting_tier": "primary",
        "direction": "up",
        "checkpoint_role": job["checkpoint_role"],
        "reporting_role": job["reporting_role"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "completed_epoch": job["completed_epoch"],
        "config_hash": job["config_hash"],
        "training_seed": job["training_seed"],
        "optimizer_family": "sgd",
        "labels": ["role:sgd:C1"],
        "evaluation_git_sha": evaluation_sha,
        "fit_split": None,
        "query_split": "id_test",
        "transform": "raw_logits",
        "fit_dtype": "float64",
        "evaluation_dtype": "float64",
        "sample_count": 10000,
        "class_counts": [1000] * 10,
        "status": status,
        "reason_codes": [] if status == "success" else ["fixture_degenerate"],
        "value": value,
        "runtime_diagnostics": {},
        "source_commit": "project_metric_contract_v1.2",
        "tolerance": {"atol": 1e-12, "rtol": 1e-10},
    }
    (metrics / "scalar_records.jsonl").write_text(json.dumps(record) + "\n")
    completion = {
        "status": "LOCAL_COMPLETE",
        "job_id": job["job_id"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "checkpoint_role": job["checkpoint_role"],
        "inventory_hash": "i" * 64,
        "dataset_policy_hash": "p" * 64,
    }
    (root / "JOB_COMPLETE.json").write_text(json.dumps(completion))
    manifest = {
        "status": "LOCAL_COMPLETE",
        "checkpoint_sha256": job["checkpoint_sha256"],
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest))
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "bundle_checksums.sha256").write_text("\n".join(rows) + "\n")


def test_aggregation_keeps_roles_separate_and_uses_seed_sample_sd(tmp_path):
    jobs = []
    evaluation_sha = "e" * 40
    values = {"last": [0.7, 0.8, 0.9], "best_val": [0.6, 0.65, None]}
    for role in ("last", "best_val"):
        for seed in range(3):
            checkpoint_sha = f"{len(jobs) + 1:064x}"
            job = {
                "job_id": f"job-{role}-{seed}",
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_role": role,
                "reporting_role": (
                    "confirmatory_primary" if role == "last" else "deployment_control"
                ),
                "completed_epoch": 200 if role == "last" else 150,
                "config_hash": "c" * 64,
                "training_seed": seed,
                "hf_output_uri": f"hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2/{checkpoint_sha}",
            }
            jobs.append(job)
            status = "degenerate" if role == "best_val" and seed == 2 else "success"
            _write_bundle(
                tmp_path / "bundles" / checkpoint_sha,
                job,
                evaluation_sha=evaluation_sha,
                value=values[role][seed],
                status=status,
            )
    inventory = {
        "inventory_hash": "i" * 64,
        "dataset_policy_hash": "p" * 64,
        "checkpoint_jobs": jobs,
    }

    output = aggregate_production_bundles(
        inventory=inventory,
        bundle_roots=[tmp_path / "bundles"],
        output_dir=tmp_path / "aggregate",
    )

    aggregates = [
        json.loads(row)
        for row in (output / "seed_aggregates.jsonl").read_text().splitlines()
    ]
    by_role = {row["checkpoint_role"]: row for row in aggregates}
    assert by_role["last"]["aggregate_status"] == "success"
    assert by_role["last"]["mean"] == pytest.approx(0.8)
    assert by_role["last"]["sample_sd_ddof1"] == pytest.approx(
        np.std([0.7, 0.8, 0.9], ddof=1)
    )
    assert by_role["best_val"]["aggregate_status"] == "degenerate"
    assert by_role["best_val"]["mean"] is None
    assert by_role["best_val"]["sample_sd_ddof1"] is None
    summary = json.loads((output / "aggregate.json").read_text())
    assert summary["checkpoint_job_count"] == 6
    assert summary["training_identity_count"] == 3
    assert summary["configuration_count"] == 1

    operational = tmp_path / "operations" / "curie"
    operational.mkdir(parents=True)
    atomic_write_json(
        operational / "host_ledger.json",
        {"status": "REMOTE_VERIFIED", "failed_job_ids": []},
    )
    for job in jobs:
        checkpoint_sha = job["checkpoint_sha256"]
        source = tmp_path / "bundles" / checkpoint_sha
        job_root = operational / "jobs" / checkpoint_sha
        job_root.mkdir(parents=True)
        shutil.copy2(source / "JOB_COMPLETE.json", job_root / "JOB_COMPLETE.json")
        shutil.copy2(
            source / "metrics/scalar_records.jsonl",
            job_root / "scalar_records.jsonl",
        )
        (job_root / "upload_evidence.json").write_text(
            json.dumps(
                {
                    "status": "REMOTE_VERIFIED",
                    "destination": job["hf_output_uri"],
                }
            )
        )
    build_artifact_manifest(operational)
    operational_output = aggregate_production_bundles(
        inventory=inventory,
        bundle_roots=[],
        operational_roots=[tmp_path / "operations"],
        output_dir=tmp_path / "operational_aggregate",
    )
    operational_rows = (operational_output / "seed_aggregates.jsonl").read_text()
    assert operational_rows == (output / "seed_aggregates.jsonl").read_text()


def test_aggregation_rejects_missing_checkpoint_bundle(tmp_path):
    inventory = {
        "inventory_hash": "i" * 64,
        "dataset_policy_hash": "p" * 64,
        "checkpoint_jobs": [
            {
                "job_id": "missing",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_role": "last",
                "config_hash": "c" * 64,
                "training_seed": 0,
            }
        ],
    }
    with np.testing.assert_raises_regex(ValueError, "population differs"):
        aggregate_production_bundles(
            inventory=inventory,
            bundle_roots=[tmp_path],
            output_dir=tmp_path / "aggregate",
        )
