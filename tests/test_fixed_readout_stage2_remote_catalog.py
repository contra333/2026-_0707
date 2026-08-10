import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.build_fixed_readout_stage2_reuse_manifest import (
    DEFAULT_INVENTORY,
    EXPECTED_DATASET_POLICY_HASH,
    EXPECTED_EVALUATION_GIT_SHA,
    EXPECTED_INVENTORY_HASH,
    HF_ROOT,
    ROOT,
    SOURCE_PROTECTED_AUTHORIZATION,
    _expected_artifact_identity,
    _load_frozen_inventory,
    _required_logical_paths,
    _select_jobs,
    build_reuse_manifest,
    sha256_file,
    validate_reuse_manifest,
)
from scripts.collect_fixed_readout_stage2_remote_catalog import (
    DEFINITION_VERSION,
    build_remote_catalog,
    main,
    materialize_remote_catalog,
    validate_remote_catalog,
    verify_materialized_remote_catalog,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _synthetic_sha(checkpoint_sha: str, logical_path: str) -> str:
    return hashlib.sha256(
        f"{checkpoint_sha}\0{logical_path}".encode("utf-8")
    ).hexdigest()


def _synthetic_size(checkpoint_sha: str, logical_path: str) -> int:
    return 1000 + int(checkpoint_sha[:4], 16) + len(logical_path)


def _fixture_jobs():
    inventory, _ = _load_frozen_inventory(DEFAULT_INVENTORY, ROOT)
    return _select_jobs(inventory)


def _build_evidence_fixture(root: Path, *, wrong_identity: bool = False):
    jobs = _fixture_jobs()
    required_paths = _required_logical_paths()
    metadata_root = root / "metadata"
    bundle_container = (
        metadata_root / "evaluations" / DEFINITION_VERSION
    )
    plan_operations = []

    for job_index, job in enumerate(jobs):
        checkpoint_sha = job["checkpoint_sha256"]
        bundle = bundle_container / checkpoint_sha
        metrics = bundle / "metrics"
        metrics.mkdir(parents=True)

        raw_manifest = {
            "artifact_role": "raw_checkpoint_feature_cache",
            "checkpoint": {
                "config_hash": job["config_hash"],
                "role": "last",
                "sha256": checkpoint_sha,
                "training_seed": int(job["training_seed"]),
            },
            "definition_version": DEFINITION_VERSION,
            "evaluation": {
                "git_dirty": False,
                "oge_git_sha": EXPECTED_EVALUATION_GIT_SHA,
                "protected_split_authorization": SOURCE_PROTECTED_AUTHORIZATION,
                "smoke_only": False,
            },
        }
        result = {
            "authorization": SOURCE_PROTECTED_AUTHORIZATION,
            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
            "definition_version": DEFINITION_VERSION,
            "evaluation_git_sha": EXPECTED_EVALUATION_GIT_SHA,
            "inventory_hash": EXPECTED_INVENTORY_HASH,
            "job": copy.deepcopy(job),
            "protected_result": True,
            "smoke_only": False,
        }
        job_completion = {
            "authorization": SOURCE_PROTECTED_AUTHORIZATION,
            "checkpoint_role": "last",
            "checkpoint_sha256": checkpoint_sha,
            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
            "definition_version": DEFINITION_VERSION,
            "evaluation_git_sha": EXPECTED_EVALUATION_GIT_SHA,
            "inventory_hash": EXPECTED_INVENTORY_HASH,
            "job_id": job["job_id"],
            "metric_file_count": 498,
            "scalar_record_count": 1586,
            "status": "LOCAL_COMPLETE",
        }
        metric_completion = {
            "checkpoint_sha256": checkpoint_sha,
            "scalar_record_count": 1586,
            "status": "METRICS_COMPLETE",
        }
        _write_json(bundle / "manifest.json", raw_manifest)
        _write_json(metrics / "result.json", result)
        _write_json(bundle / "JOB_COMPLETE.json", job_completion)
        _write_json(metrics / "METRICS_COMPLETE.json", metric_completion)

        scalar_sha = _synthetic_sha(
            checkpoint_sha, "metrics/scalar_records.jsonl"
        )
        artifact = {
            **_expected_artifact_identity(job),
            "definition_version": DEFINITION_VERSION,
            "metric_result_sha256": sha256_file(metrics / "result.json"),
            "raw_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "scalar_records_sha256": scalar_sha,
        }
        if wrong_identity and job_index == 0:
            artifact["training_seed"] = 999
        _write_json(bundle / "artifact_manifest.json", artifact)

        checksum_rows = {}
        for logical_path in required_paths:
            if logical_path in {
                "REMOTE_COMPLETE.json",
                "bundle_checksums.sha256",
            }:
                continue
            local_path = bundle.joinpath(*logical_path.split("/"))
            checksum_rows[logical_path] = (
                sha256_file(local_path)
                if local_path.is_file()
                else _synthetic_sha(checkpoint_sha, logical_path)
            )
        checksum_rows["metrics/scalar_records.jsonl"] = scalar_sha
        (bundle / "bundle_checksums.sha256").write_text(
            "".join(
                f"{digest}  {logical_path}\n"
                for logical_path, digest in sorted(checksum_rows.items())
            ),
            encoding="utf-8",
        )
        _write_json(
            bundle / "REMOTE_COMPLETE.json",
            {
                "artifact_manifest_sha256": sha256_file(
                    bundle / "artifact_manifest.json"
                ),
                "destination": job["hf_output_uri"],
                "file_count": len(checksum_rows) + 1,
                "status": "REMOTE_VERIFIED",
                "total_size": 1_000_000,
            },
        )

        for logical_path in required_paths:
            local_path = bundle.joinpath(*logical_path.split("/"))
            size = (
                local_path.stat().st_size
                if local_path.is_file()
                else _synthetic_size(checkpoint_sha, logical_path)
            )
            plan_operations.append(
                {
                    "action": "download",
                    "path": f"{checkpoint_sha}/{logical_path}",
                    "reason": "new file",
                    "size": size,
                    "type": "operation",
                }
            )

    # The real control-plane plan contains other checkpoint bundles and
    # aggregate/control records.  These must be validated but not emitted.
    plan_operations.extend(
        [
            {
                "action": "download",
                "path": f"{'f' * 64}/unselected_payload.npy",
                "reason": "new file",
                "size": 321,
                "type": "operation",
            },
            {
                "action": "download",
                "path": "_aggregates/example/artifact_manifest.json",
                "reason": "new file",
                "size": 123,
                "type": "operation",
            },
        ]
    )
    plan = root / "remote.plan.jsonl"
    header = {
        "dest": str(root / "unused-destination"),
        "source": HF_ROOT,
        "summary": {
            "deletes": 0,
            "downloads": len(plan_operations),
            "skips": 0,
            "total_size": sum(row["size"] for row in plan_operations),
            "uploads": 0,
        },
        "type": "header",
    }
    plan.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [header, *plan_operations])
        + "\n",
        encoding="utf-8",
    )
    return {
        "bundle_container": bundle_container,
        "jobs": jobs,
        "metadata_root": metadata_root,
        "plan": plan,
        "required_paths": required_paths,
    }


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return _build_evidence_fixture(tmp_path_factory.mktemp("stage2-catalog"))


def _clone_evidence(evidence, tmp_path):
    metadata_root = tmp_path / "metadata"
    shutil.copytree(evidence["metadata_root"], metadata_root)
    plan = tmp_path / "remote.plan.jsonl"
    shutil.copy2(evidence["plan"], plan)
    return {
        **evidence,
        "bundle_container": metadata_root / "evaluations" / DEFINITION_VERSION,
        "metadata_root": metadata_root,
        "plan": plan,
    }


def _read_plan(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_plan(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_full_control_plane_extras_are_allowed_but_output_is_exact_selected_allowlist(
    evidence, tmp_path
):
    catalog = build_remote_catalog(
        metadata_root=evidence["metadata_root"],
        sync_plan_path=evidence["plan"],
    )

    assert len(catalog["bundles"]) == 30
    assert [row["checkpoint_sha256"] for row in catalog["bundles"]] == [
        job["checkpoint_sha256"] for job in evidence["jobs"]
    ]
    assert all(
        [item["logical_path"] for item in row["files"]]
        == evidence["required_paths"]
        for row in catalog["bundles"]
    )
    assert not any(
        row["checkpoint_sha256"] == "f" * 64 for row in catalog["bundles"]
    )
    evidence_binding = catalog["source_evidence"]
    assert evidence_binding["hf_sync_plan_sha256"] == sha256_file(evidence["plan"])
    assert evidence_binding["hf_sync_plan_size"] == evidence["plan"].stat().st_size
    assert evidence_binding["downloaded_metadata_file_count"] == 30 * 7

    output = tmp_path / "catalog.json"
    digest = materialize_remote_catalog(catalog, output)
    assert verify_materialized_remote_catalog(catalog, output) == digest
    reuse = build_reuse_manifest(checksum_catalog_path=output)
    validate_reuse_manifest(reuse)
    assert reuse["retrieval_ready"] is True

    tampered = copy.deepcopy(catalog)
    tampered["source_evidence"]["downloaded_metadata_catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="downloaded metadata digest mismatch"):
        validate_remote_catalog(tampered)


def test_downloaded_metadata_tamper_is_rejected(evidence, tmp_path):
    case = _clone_evidence(evidence, tmp_path)
    checkpoint_sha = case["jobs"][0]["checkpoint_sha256"]
    target = case["bundle_container"] / checkpoint_sha / "manifest.json"
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256 mismatch|metadata size"):
        build_remote_catalog(
            metadata_root=case["metadata_root"], sync_plan_path=case["plan"]
        )


def test_missing_downloaded_metadata_is_rejected(evidence, tmp_path):
    case = _clone_evidence(evidence, tmp_path)
    checkpoint_sha = case["jobs"][0]["checkpoint_sha256"]
    (case["bundle_container"] / checkpoint_sha / "JOB_COMPLETE.json").unlink()

    with pytest.raises(ValueError, match="metadata is missing"):
        build_remote_catalog(
            metadata_root=case["metadata_root"], sync_plan_path=case["plan"]
        )


def test_missing_authoritative_checksum_row_is_rejected(evidence, tmp_path):
    case = _clone_evidence(evidence, tmp_path)
    checkpoint_sha = case["jobs"][0]["checkpoint_sha256"]
    checksum_path = (
        case["bundle_container"] / checkpoint_sha / "bundle_checksums.sha256"
    )
    rows = checksum_path.read_text(encoding="utf-8").splitlines()
    omitted = next(
        logical_path
        for logical_path in case["required_paths"]
        if logical_path.endswith("/features.npy")
    )
    checksum_path.write_text(
        "\n".join(row for row in rows if not row.endswith(f"  {omitted}")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="misses required allowlisted"):
        build_remote_catalog(
            metadata_root=case["metadata_root"], sync_plan_path=case["plan"]
        )


def test_duplicate_plan_path_is_rejected(evidence, tmp_path):
    case = _clone_evidence(evidence, tmp_path)
    rows = _read_plan(case["plan"])
    rows.append(copy.deepcopy(rows[1]))
    _write_plan(case["plan"], rows)

    with pytest.raises(ValueError, match="duplicate remote path"):
        build_remote_catalog(
            metadata_root=case["metadata_root"], sync_plan_path=case["plan"]
        )


def test_wrong_plan_size_is_rejected(evidence, tmp_path):
    case = _clone_evidence(evidence, tmp_path)
    rows = _read_plan(case["plan"])
    checkpoint_sha = case["jobs"][0]["checkpoint_sha256"]
    target_path = f"{checkpoint_sha}/artifact_manifest.json"
    operation = next(row for row in rows[1:] if row["path"] == target_path)
    operation["size"] += 1
    rows[0]["summary"]["total_size"] += 1
    _write_plan(case["plan"], rows)

    with pytest.raises(ValueError, match="metadata size disagrees with plan"):
        build_remote_catalog(
            metadata_root=case["metadata_root"], sync_plan_path=case["plan"]
        )


def test_wrong_artifact_identity_is_rejected(tmp_path):
    case = _build_evidence_fixture(tmp_path, wrong_identity=True)

    with pytest.raises(ValueError, match="artifact manifest identity mismatch"):
        build_remote_catalog(
            metadata_root=case["metadata_root"], sync_plan_path=case["plan"]
        )


def test_no_overwrite_and_check_mode(evidence, tmp_path):
    output = tmp_path / "catalog.json"
    catalog = build_remote_catalog(
        metadata_root=evidence["metadata_root"],
        sync_plan_path=evidence["plan"],
    )
    materialize_remote_catalog(catalog, output)

    with pytest.raises(FileExistsError):
        materialize_remote_catalog(catalog, output)
    assert (
        main(
            [
                "--metadata-root",
                str(evidence["metadata_root"]),
                "--sync-plan",
                str(evidence["plan"]),
                "--output",
                str(output),
                "--check",
            ]
        )
        == 0
    )
    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match="not byte-identical"):
        main(
            [
                "--metadata-root",
                str(evidence["metadata_root"]),
                "--sync-plan",
                str(evidence["plan"]),
                "--output",
                str(output),
                "--check",
            ]
        )
