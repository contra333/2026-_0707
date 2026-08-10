import copy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.fetch_fixed_readout_stage2_reuse as fetch
from scripts.build_fixed_readout_stage2_reuse_manifest import (
    CATALOG_SCHEMA_VERSION,
    DESTINATION,
    EXPECTED_INVENTORY_FILE_SHA256,
    HF_ROOT,
    build_reuse_manifest,
    validate_reuse_manifest,
)


def _content(bundle_id, logical_path):
    return f"{bundle_id}\0{logical_path}".encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _complete_manifest(tmp_path):
    draft = build_reuse_manifest()
    required = draft["path_contract"]["required_logical_paths"]
    rows = []
    for bundle in draft["bundles"]:
        bundle_id = bundle["bundle_id"]
        files = []
        for logical_path in required:
            payload = _content(bundle_id, logical_path)
            files.append(
                {
                    "logical_path": logical_path,
                    "sha256": _sha(payload),
                    "size": len(payload),
                }
            )
        by_path = {row["logical_path"]: row for row in files}
        rows.append(
            {
                "artifact_identity": copy.deepcopy(bundle["identity"]),
                "artifact_manifest_file_sha256": by_path[
                    "artifact_manifest.json"
                ]["sha256"],
                "bundle_checksums_file_sha256": by_path[
                    "bundle_checksums.sha256"
                ]["sha256"],
                "checkpoint_sha256": bundle_id,
                "files": files,
                "hf_uri": bundle["hf_uri"],
            }
        )
    catalog = {
        "bundles": rows,
        "hf_root": HF_ROOT,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_inventory_sha256": EXPECTED_INVENTORY_FILE_SHA256,
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    manifest = build_reuse_manifest(checksum_catalog_path=catalog_path)
    validate_reuse_manifest(manifest)
    return manifest


def _materialize_test_manifest(tmp_path, monkeypatch):
    manifest = _complete_manifest(tmp_path)
    destination = tmp_path / "download"
    destination.mkdir()
    manifest["destination"]["root"] = str(destination.resolve())
    manifest_path = tmp_path / "reuse_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def validate_with_test_destination(candidate, *, require_remote_catalog=True):
        calls.append(require_remote_catalog)
        frozen = copy.deepcopy(candidate)
        frozen["destination"]["root"] = DESTINATION["root"]
        validate_reuse_manifest(
            frozen, require_remote_catalog=require_remote_catalog
        )

    monkeypatch.setattr(fetch, "validate_reuse_manifest", validate_with_test_destination)
    return manifest_path, manifest, destination, calls


def _expected(manifest):
    return fetch._expected_files(manifest)


def _write_expected(destination, record):
    path = destination / record["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_id, logical_path = record["path"].split("/", 1)
    path.write_bytes(_content(bundle_id, logical_path))


def _plan_payload(manifest, destination, expected, existing=()):
    existing = set(existing)
    operations = []
    for relative, record in expected.items():
        operations.append(
            {
                "type": "operation",
                "action": "skip" if relative in existing else "download",
                "path": relative,
                "reason": "same size" if relative in existing else "missing locally",
                "size": record["size"],
            }
        )
    header = {
        "type": "header",
        "source": manifest["hf_root"],
        "dest": str(destination.resolve()),
        "timestamp": "2026-08-10T00:00:00+00:00",
        "summary": {
            "uploads": 0,
            "downloads": len(expected) - len(existing),
            "deletes": 0,
            "skips": len(existing),
            "total_size": sum(
                record["size"]
                for relative, record in expected.items()
                if relative not in existing
            ),
        },
    }
    return [header, *operations]


def _write_plan(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_prepare_writes_exact_5520_filter_and_resume_safe_plan_command(
    tmp_path, monkeypatch
):
    manifest_path, manifest, destination, validator_calls = (
        _materialize_test_manifest(tmp_path, monkeypatch)
    )
    expected = _expected(manifest)
    first = next(iter(expected.values()))
    _write_expected(destination, first)
    filter_path = tmp_path / "exact.filter"
    plan_path = tmp_path / "download.plan.jsonl"

    result = fetch.prepare(
        manifest_path,
        filter_path,
        plan_path,
        destination=destination,
        hf_cli="/opt/hf/bin/hf",
    )

    lines = filter_path.read_text(encoding="utf-8").splitlines()
    assert len(expected) == fetch.EXPECTED_FILE_COUNT == 5_520
    assert lines[:-1] == [f"+ {relative}" for relative in expected]
    assert lines[-1] == "- *"
    assert result["verified_existing_count"] == 1
    assert result["expected_file_count"] == 5_520
    assert result["command"] == [
        "/opt/hf/bin/hf",
        "buckets",
        "sync",
        HF_ROOT,
        str(destination.resolve()),
        "--filter-from",
        str(filter_path),
        "--plan",
        str(plan_path),
        "--no-delete",
        "--ignore-times",
    ]
    assert "--apply" not in result["command"]
    assert validator_calls == [True]
    with pytest.raises(FileExistsError):
        fetch.prepare(manifest_path, filter_path, plan_path, destination=destination)
    occupied_plan = tmp_path / "occupied.plan.jsonl"
    occupied_plan.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fetch.prepare(
            manifest_path,
            tmp_path / "unused.filter",
            occupied_plan,
            destination=destination,
        )


def test_prepare_rejects_tamper_extra_and_symlink(tmp_path, monkeypatch):
    manifest_path, manifest, destination, _ = _materialize_test_manifest(
        tmp_path, monkeypatch
    )
    first = next(iter(_expected(manifest).values()))
    path = destination / first["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        fetch.prepare(
            manifest_path, tmp_path / "a.filter", tmp_path / "a.plan", destination=destination
        )

    path.write_bytes(_content(*first["path"].split("/", 1)))
    extra = destination / "unexpected.bin"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected destination file"):
        fetch.prepare(
            manifest_path, tmp_path / "b.filter", tmp_path / "b.plan", destination=destination
        )
    extra.unlink()
    (destination / "link").symlink_to(path)
    with pytest.raises(ValueError, match="symlink is forbidden"):
        fetch.prepare(
            manifest_path, tmp_path / "c.filter", tmp_path / "c.plan", destination=destination
        )


def test_verify_plan_accepts_exact_downloads_and_verified_existing_skips(
    tmp_path, monkeypatch
):
    manifest_path, manifest, destination, _ = _materialize_test_manifest(
        tmp_path, monkeypatch
    )
    expected = _expected(manifest)
    existing = list(expected)[:2]
    for relative in existing:
        _write_expected(destination, expected[relative])
    plan_path = tmp_path / "plan.jsonl"
    _write_plan(
        plan_path,
        _plan_payload(manifest, destination, expected, existing),
    )

    result = fetch.verify_plan(
        manifest_path, plan_path, destination=destination
    )
    assert result["downloads"] == 5_518
    assert result["skips"] == 2
    assert result["uploads"] == result["deletes"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("upload", "forbidden HF sync action"),
        ("delete", "forbidden HF sync action"),
        ("extra", "extra HF sync plan path|operation count mismatch"),
        ("size", "HF sync plan size mismatch"),
        ("skip_missing", "unverified-existing skip"),
        ("source", "HF sync plan source mismatch"),
        ("destination", "HF sync plan destination mismatch"),
    ],
)
def test_verify_plan_rejects_action_path_size_and_header_drift(
    tmp_path, monkeypatch, mutation, message
):
    manifest_path, manifest, destination, _ = _materialize_test_manifest(
        tmp_path, monkeypatch
    )
    expected = _expected(manifest)
    rows = _plan_payload(manifest, destination, expected)
    if mutation in {"upload", "delete"}:
        rows[1]["action"] = mutation
    elif mutation == "extra":
        rows[1]["path"] = "extra/path"
    elif mutation == "size":
        rows[1]["size"] += 1
    elif mutation == "skip_missing":
        rows[1]["action"] = "skip"
    elif mutation == "source":
        rows[0]["source"] += "/wrong"
    else:
        rows[0]["dest"] += "/wrong"
    plan_path = tmp_path / "bad-plan.jsonl"
    _write_plan(plan_path, rows)
    with pytest.raises(ValueError, match=message):
        fetch.verify_plan(manifest_path, plan_path, destination=destination)


def test_verify_download_checks_all_5520_and_writes_deterministic_no_overwrite_receipt(
    tmp_path, monkeypatch
):
    manifest_path, manifest, destination, _ = _materialize_test_manifest(
        tmp_path, monkeypatch
    )
    expected = _expected(manifest)
    for record in expected.values():
        _write_expected(destination, record)
    first_receipt = tmp_path / "receipt-a.json"
    second_receipt = tmp_path / "receipt-b.json"

    result = fetch.verify_download(
        manifest_path, first_receipt, destination=destination
    )
    fetch.verify_download(manifest_path, second_receipt, destination=destination)
    assert result["status"] == "CHECKSUM_VERIFIED_EXACT_ALLOWLIST"
    assert result["file_count"] == 5_520
    assert first_receipt.read_bytes() == second_receipt.read_bytes()
    with pytest.raises(FileExistsError):
        fetch.verify_download(
            manifest_path, first_receipt, destination=destination
        )

    extra = destination / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected destination file"):
        fetch.verify_download(
            manifest_path, tmp_path / "receipt-extra.json", destination=destination
        )
    extra.unlink()
    first = next(iter(expected.values()))
    (destination / first["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        fetch.verify_download(
            manifest_path, tmp_path / "receipt-tamper.json", destination=destination
        )
