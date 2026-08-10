import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import pytest

import scripts.materialize_fixed_readout_stage2_bundle as materializer


BUNDLE_ID = "a" * 64


def _canonical_sha256(payload):
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(logical_path):
    return f"{BUNDLE_ID}\0{logical_path}".encode("utf-8")


def _synthetic_case(tmp_path, monkeypatch):
    logical_paths = sorted(
        ["artifact_manifest.json", "bundle_checksums.sha256"]
        + [f"arrays/group-{index // 20:02d}/{index:03d}.bin" for index in range(182)]
    )
    files = [
        {
            "logical_path": logical_path,
            "sha256": hashlib.sha256(_payload(logical_path)).hexdigest(),
            "size": len(_payload(logical_path)),
        }
        for logical_path in logical_paths
    ]
    by_path = {row["logical_path"]: row for row in files}
    digest_payload = {
        "artifact_manifest_file_sha256": by_path["artifact_manifest.json"]["sha256"],
        "bundle_checksums_file_sha256": by_path["bundle_checksums.sha256"]["sha256"],
        "files": files,
    }
    manifest = {
        "bundles": [
            {
                "assigned_host": "synthetic-host",
                "bundle_id": BUNDLE_ID,
                "hf_uri": f"hf://synthetic/{BUNDLE_ID}",
                "identity": {
                    "checkpoint_sha256": BUNDLE_ID,
                    "training_seed": 1,
                },
                "remote_integrity": {
                    **digest_payload,
                    "allowed_payload_catalog_sha256": _canonical_sha256(
                        digest_payload
                    ),
                    "selected_file_count": len(files),
                    "selected_total_size": sum(row["size"] for row in files),
                },
            }
        ],
        "path_contract": {"required_logical_paths": logical_paths},
        "retrieval_ready": True,
        "schema_version": "fixed_readout_stage2_reuse_manifest_v1",
    }
    manifest_path = tmp_path / "reuse_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = tmp_path / "source" / BUNDLE_ID
    for row in files:
        path = source.joinpath(*PurePosixPath(row["logical_path"]).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_payload(row["logical_path"]))
    monkeypatch.setattr(
        materializer,
        "validate_reuse_manifest",
        lambda candidate, require_remote_catalog=True: None,
    )
    return manifest_path, manifest, source, files


def _materialize(tmp_path, monkeypatch, *, suffix=""):
    manifest_path, manifest, source, files = _synthetic_case(tmp_path, monkeypatch)
    destination = tmp_path / f"destination{suffix}"
    receipt = tmp_path / f"receipt{suffix}.json"
    result = materializer.materialize_bundle(
        manifest_path,
        BUNDLE_ID,
        source,
        destination,
        receipt,
        _test_only_skip_committed_source_validation=True,
    )
    return result, manifest, source, files, destination, receipt


def test_materializes_exact_allowlist_while_ignoring_regular_source_extras(
    tmp_path, monkeypatch
):
    manifest_path, manifest, source, files = _synthetic_case(tmp_path, monkeypatch)
    (source / "unselected-production-array.npy").write_bytes(b"not selected")
    (source / "unselected" / "metadata.json").parent.mkdir()
    (source / "unselected" / "metadata.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "destination"
    receipt_path = tmp_path / "receipt.json"

    receipt = materializer.materialize_bundle(
        manifest_path,
        BUNDLE_ID,
        source,
        destination,
        receipt_path,
        _test_only_skip_committed_source_validation=True,
    )

    observed = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    expected = {
        PurePosixPath(BUNDLE_ID, row["logical_path"]).as_posix() for row in files
    }
    assert observed == expected
    assert "unselected-production-array.npy" not in observed
    assert receipt["status"] == "CHECKSUM_VERIFIED_EXACT_ONE_BUNDLE_COPY"
    assert receipt["file_count"] == materializer.EXPECTED_FILE_COUNT == 184
    assert receipt["total_size"] == sum(row["size"] for row in files)
    assert receipt["bundle_catalog_sha256"] == manifest["bundles"][0][
        "remote_integrity"
    ]["allowed_payload_catalog_sha256"]
    assert receipt["source"]["artifact_identity"] == manifest["bundles"][0][
        "identity"
    ]
    assert {row["copy_method"] for row in receipt["files"]} == {
        materializer.COPY_METHOD
    }
    assert receipt_path.read_bytes() == materializer._canonical_bytes(receipt)
    assert "timestamp" not in receipt


@pytest.mark.parametrize("tamper_kind", ["size", "same_size_sha"])
def test_rejects_tampered_allowlisted_source_before_creating_output(
    tmp_path, monkeypatch, tamper_kind
):
    manifest_path, _, source, files = _synthetic_case(tmp_path, monkeypatch)
    target = source.joinpath(*PurePosixPath(files[0]["logical_path"]).parts)
    original = target.read_bytes()
    target.write_bytes(b"x" if tamper_kind == "size" else b"x" * len(original))
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="source size mismatch|source SHA-256 mismatch"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            destination,
            tmp_path / "receipt.json",
            _test_only_skip_committed_source_validation=True,
        )
    assert not destination.exists()


def test_rejects_symlink_and_special_allowlisted_source_entries(tmp_path, monkeypatch):
    manifest_path, _, source, files = _synthetic_case(tmp_path, monkeypatch)
    first = source.joinpath(*PurePosixPath(files[0]["logical_path"]).parts)
    extra = source / "extra.bin"
    extra.write_bytes(first.read_bytes())
    first.unlink()
    first.symlink_to(extra)
    with pytest.raises(ValueError, match="symlink is forbidden"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            tmp_path / "symlink-destination",
            tmp_path / "symlink-receipt.json",
            _test_only_skip_committed_source_validation=True,
        )

    first.unlink()
    os.mkfifo(first)
    with pytest.raises(ValueError, match="not a regular file"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            tmp_path / "special-destination",
            tmp_path / "special-receipt.json",
            _test_only_skip_committed_source_validation=True,
        )


def test_post_copy_tamper_fails_without_publishing_destination(
    tmp_path, monkeypatch
):
    manifest_path, _, source, _ = _synthetic_case(tmp_path, monkeypatch)
    real_copyfile = materializer.shutil.copyfile
    calls = 0

    def corrupt_first_copy(source_path, destination_path, *, follow_symlinks=True):
        nonlocal calls
        result = real_copyfile(
            source_path, destination_path, follow_symlinks=follow_symlinks
        )
        if calls == 0:
            Path(destination_path).write_bytes(b"corrupted after copy")
        calls += 1
        return result

    monkeypatch.setattr(materializer.shutil, "copyfile", corrupt_first_copy)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="copied file size mismatch|SHA-256 mismatch"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            destination,
            tmp_path / "receipt.json",
            _test_only_skip_committed_source_validation=True,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.tmp-*"))


@pytest.mark.parametrize("populate", [False, True])
def test_rejects_any_existing_destination_without_overwrite(
    tmp_path, monkeypatch, populate
):
    manifest_path, _, source, _ = _synthetic_case(tmp_path, monkeypatch)
    destination = tmp_path / "destination"
    destination.mkdir()
    if populate:
        (destination / "owner-data.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="destination root already exists"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            destination,
            tmp_path / "receipt.json",
            _test_only_skip_committed_source_validation=True,
        )
    if populate:
        assert (destination / "owner-data.txt").read_text(encoding="utf-8") == "preserve"


def test_rejects_existing_receipt_before_copy(tmp_path, monkeypatch):
    manifest_path, _, source, _ = _synthetic_case(tmp_path, monkeypatch)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("owner receipt", encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(FileExistsError, match="receipt path already exists"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            destination,
            receipt,
            _test_only_skip_committed_source_validation=True,
        )
    assert receipt.read_text(encoding="utf-8") == "owner receipt"
    assert not destination.exists()


def test_rejects_destination_inside_git_worktree(tmp_path, monkeypatch):
    manifest_path, _, source, _ = _synthetic_case(tmp_path, monkeypatch)
    destination = materializer.ROOT / f".stage2-materializer-test-{tmp_path.name}"
    assert not destination.exists()
    with pytest.raises(ValueError, match="outside the Git worktree"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            destination,
            tmp_path / "receipt.json",
            _test_only_skip_committed_source_validation=True,
        )
    assert not destination.exists()


def test_rejects_relative_destination_without_creating_output(tmp_path, monkeypatch):
    manifest_path, _, source, _ = _synthetic_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="destination root must be absolute"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            Path("relative-destination"),
            tmp_path / "receipt.json",
            _test_only_skip_committed_source_validation=True,
        )


def test_production_path_requires_manifest_and_helper_at_committed_head(
    tmp_path, monkeypatch
):
    manifest_path, _, source, _ = _synthetic_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="manifest must be inside the Git worktree"):
        materializer.materialize_bundle(
            manifest_path,
            BUNDLE_ID,
            source,
            tmp_path / "destination",
            tmp_path / "receipt.json",
        )
