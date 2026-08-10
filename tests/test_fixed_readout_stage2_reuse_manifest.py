import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_fixed_readout_stage2_reuse_manifest import (
    AUTHORIZATION,
    CATALOG_SCHEMA_VERSION,
    EXPECTED_INVENTORY_FILE_SHA256,
    HF_ROOT,
    build_reuse_manifest,
    main,
    render_reuse_manifest,
    validate_reuse_manifest,
)


def _file_sha(checkpoint_sha, logical_path):
    return hashlib.sha256(
        f"{checkpoint_sha}\0{logical_path}".encode("utf-8")
    ).hexdigest()


def _catalog_payload(draft):
    required_paths = draft["path_contract"]["required_logical_paths"]
    bundles = []
    for bundle in draft["bundles"]:
        checkpoint_sha = bundle["bundle_id"]
        files = [
            {
                "logical_path": logical_path,
                "sha256": _file_sha(checkpoint_sha, logical_path),
                "size": 1000 + index + len(logical_path),
            }
            for index, logical_path in enumerate(required_paths)
        ]
        by_path = {row["logical_path"]: row for row in files}
        bundles.append(
            {
                "artifact_identity": copy.deepcopy(bundle["identity"]),
                "artifact_manifest_file_sha256": by_path[
                    "artifact_manifest.json"
                ]["sha256"],
                "bundle_checksums_file_sha256": by_path[
                    "bundle_checksums.sha256"
                ]["sha256"],
                "checkpoint_sha256": checkpoint_sha,
                "files": files,
                "hf_uri": bundle["hf_uri"],
            }
        )
    return {
        "bundles": bundles,
        "hf_root": HF_ROOT,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_inventory_sha256": EXPECTED_INVENTORY_FILE_SHA256,
    }


def _write_catalog(tmp_path, draft, mutation=None):
    payload = _catalog_payload(draft)
    if mutation == "missing_file":
        payload["bundles"][0]["files"].pop()
    elif mutation == "wrong_identity":
        payload["bundles"][0]["artifact_identity"]["training_seed"] = 999
    elif mutation == "bad_file_sha":
        payload["bundles"][0]["files"][0]["sha256"] = "not-a-sha"
    elif mutation == "extra_bundle":
        payload["bundles"].append(copy.deepcopy(payload["bundles"][0]))
        payload["bundles"][-1]["checkpoint_sha256"] = "f" * 64
    path = tmp_path / f"catalog-{mutation or 'valid'}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_frozen_inventory_selects_exactly_30_primary_last_bundles():
    manifest = build_reuse_manifest()

    assert manifest["schema_version"] == "fixed_readout_stage2_reuse_manifest_v1"
    assert manifest["selection"] == {
        "checkpoint_role": "last",
        "configuration_count": 10,
        "job_count": 30,
        "reporting_role": "confirmatory_primary",
        "seeds": [0, 1, 2],
    }
    assert len(manifest["bundles"]) == 30
    assert len({row["bundle_id"] for row in manifest["bundles"]}) == 30
    assert len({row["config_hash"] for row in manifest["bundles"]}) == 10
    assert {row["training_seed"] for row in manifest["bundles"]} == {0, 1, 2}
    assert Counter(row["optimizer_family"] for row in manifest["bundles"]) == {
        "adam": 9,
        "adamw": 9,
        "sgd": 12,
    }
    assert all(row["identity"]["checkpoint_role"] == "last" for row in manifest["bundles"])
    assert all(
        row["identity"]["reporting_role"] == "confirmatory_primary"
        for row in manifest["bundles"]
    )
    assert manifest["destination"] == {
        "create_automatically": False,
        "existing_file_policy": "verified_exact_resume_only",
        "kind": "absolute_non_git_directory",
        "mismatched_or_unexpected_existing_files": "reject",
        "must_be_outside_git_worktree": True,
        "root": (
            "/home/contra333/2026여름방학실험코드/"
            "fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1"
        ),
        "verified_existing_files": "skip_without_overwrite",
    }
    assert manifest["parity_baseline"]["required_metrics"] == ["auroc", "fpr95"]
    assert len(manifest["parity_baseline"]["files"]) == 5
    assert all(
        row["remote_uri"].startswith(
            "hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2/"
            "_aggregates/"
        )
        for row in manifest["parity_baseline"]["files"]
    )
    assert set(manifest["analysis_policy"]) == {
        "candidate_registry",
        "gate_policy",
    }
    for binding in manifest["analysis_policy"].values():
        path = Path(__file__).parents[1] / binding["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_missing_remote_catalog_is_structurally_visible_but_retrieval_validation_fails():
    manifest = build_reuse_manifest()

    assert manifest["remote_checksum_catalog"]["status"] == "missing"
    assert manifest["retrieval_ready"] is False
    assert all(row["remote_integrity"] is None for row in manifest["bundles"])
    validate_reuse_manifest(manifest, require_remote_catalog=False)
    with pytest.raises(ValueError, match="unsafe bulk fetch is forbidden"):
        validate_reuse_manifest(manifest, require_remote_catalog=True)


def test_complete_catalog_produces_deterministic_valid_exact_allowlist(tmp_path):
    draft = build_reuse_manifest()
    catalog_path = _write_catalog(tmp_path, draft)

    first = build_reuse_manifest(checksum_catalog_path=catalog_path)
    second = build_reuse_manifest(checksum_catalog_path=catalog_path)

    validate_reuse_manifest(first)
    assert render_reuse_manifest(first) == render_reuse_manifest(second)
    assert first["retrieval_ready"] is True
    assert first["authorization"] == AUTHORIZATION
    required_paths = first["path_contract"]["required_logical_paths"]
    assert required_paths == sorted(set(required_paths))
    assert "cifar10/train/logits.npy" in required_paths
    assert "places365/test/logits.npy" in required_paths
    for bundle in first["bundles"]:
        integrity = bundle["remote_integrity"]
        assert integrity["selected_file_count"] == len(required_paths)
        assert [row["logical_path"] for row in integrity["files"]] == required_paths
        assert integrity["catalog_file_count"] == len(required_paths)
        assert integrity["selected_total_size"] == sum(
            row["size"] for row in integrity["files"]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_file", "missing required allowlisted files"),
        ("wrong_identity", "artifact identity mismatch"),
        ("bad_file_sha", "must be a lowercase 64-character SHA-256"),
        ("extra_bundle", "exactly the 30 selected bundles"),
    ],
)
def test_catalog_validation_rejects_incomplete_or_mismatched_metadata(
    tmp_path, mutation, message
):
    draft = build_reuse_manifest()
    catalog_path = _write_catalog(tmp_path, draft, mutation)

    with pytest.raises(ValueError, match=message):
        build_reuse_manifest(checksum_catalog_path=catalog_path)


def test_manifest_validator_rejects_policy_or_payload_drift(tmp_path):
    draft = build_reuse_manifest()
    catalog_path = _write_catalog(tmp_path, draft)
    manifest = build_reuse_manifest(checksum_catalog_path=catalog_path)

    policy_drift = copy.deepcopy(manifest)
    policy_drift["authorization"]["remote_upload_allowed"] = True
    with pytest.raises(ValueError, match="authorization policy mismatch"):
        validate_reuse_manifest(policy_drift)

    parity_drift = copy.deepcopy(manifest)
    parity_drift["parity_baseline"]["files"][0]["size"] += 1
    with pytest.raises(ValueError, match="parity baseline mismatch"):
        validate_reuse_manifest(parity_drift)

    analysis_policy_drift = copy.deepcopy(manifest)
    analysis_policy_drift["analysis_policy"]["gate_policy"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="analysis policy mismatch"):
        validate_reuse_manifest(analysis_policy_drift)

    identity_drift = copy.deepcopy(manifest)
    identity_drift["bundles"][0]["optimizer_family"] = "adamw"
    with pytest.raises(ValueError, match="differs from frozen job"):
        validate_reuse_manifest(identity_drift)

    payload_drift = copy.deepcopy(manifest)
    payload_drift["bundles"][0]["remote_integrity"]["files"][0]["size"] += 1
    with pytest.raises(ValueError, match="allowed payload catalog digest mismatch"):
        validate_reuse_manifest(payload_drift)


def test_cli_materializes_once_and_check_requires_byte_identity(tmp_path):
    draft = build_reuse_manifest()
    catalog_path = _write_catalog(tmp_path, draft)
    output = tmp_path / "reuse_manifest.json"
    args = [
        "--checksum-catalog",
        str(catalog_path),
        "--output",
        str(output),
    ]

    assert main(args) == 0
    expected = build_reuse_manifest(checksum_catalog_path=catalog_path)
    assert output.read_bytes() == render_reuse_manifest(expected)
    with pytest.raises(FileExistsError):
        main(args)
    assert main([*args, "--check"]) == 0

    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match="not byte-identical"):
        main([*args, "--check"])


def test_cli_without_catalog_fails_before_creating_output(tmp_path):
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="unsafe bulk fetch is forbidden"):
        main(["--output", str(output)])
    assert not output.exists()
