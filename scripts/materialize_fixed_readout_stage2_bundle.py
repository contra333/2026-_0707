#!/usr/bin/env python3
"""Materialize one exact Stage-2 bundle for distributed cache computation.

The source bundle may contain the full production payload.  This helper reads
and copies only the 184 paths frozen in the committed Stage-2 reuse manifest.
It never overwrites a destination, never mutates the source, and publishes the
verified one-bundle tree only after a complete temporary copy passes its
catalog checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from scripts.build_fixed_readout_stage2_reuse_manifest import (
        DEFAULT_OUTPUT,
        ROOT,
        sha256_file,
        validate_reuse_manifest,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_fixed_readout_stage2_reuse_manifest import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT,
        ROOT,
        sha256_file,
        validate_reuse_manifest,
    )


SCHEMA_VERSION = "fixed_readout_stage2_one_bundle_copy_receipt_v1"
EXPECTED_FILE_COUNT = 184
SCRIPT_RELATIVE_PATH = "scripts/materialize_fixed_readout_stage2_bundle.py"
VALIDATOR_RELATIVE_PATH = "scripts/build_fixed_readout_stage2_reuse_manifest.py"
COPY_METHOD = "shutil.copyfile(follow_symlinks=False)"
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _safe_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("allowlisted logical path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe allowlisted logical path: {value!r}")
    return value


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _validate_committed_sources(manifest_path: Path) -> tuple[str, list[dict[str, Any]]]:
    resolved_manifest = manifest_path.resolve()
    try:
        manifest_relative = resolved_manifest.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("production reuse manifest must be inside the Git worktree") from exc
    source_paths = (
        SCRIPT_RELATIVE_PATH,
        VALIDATOR_RELATIVE_PATH,
        manifest_relative,
    )
    rows: list[dict[str, Any]] = []
    for relative in source_paths:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"required committed source is absent: {relative}")
        working = path.read_bytes()
        try:
            committed = _git_bytes(relative)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"required source is not committed at HEAD: {relative}") from exc
        if working != committed:
            raise ValueError(f"required source differs from HEAD: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(working).hexdigest(),
                "size": len(working),
            }
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    return head, rows


def _load_frozen_bundle(
    manifest_path: Path,
    bundle_id: str,
    *,
    test_only_skip_committed_source_validation: bool,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"reuse manifest must be a regular file: {manifest_path}")
    if test_only_skip_committed_source_validation:
        source_validation = {
            "git_head": None,
            "sources": [],
            "status": "TEST_ONLY_BYPASSED",
        }
    else:
        git_head, sources = _validate_committed_sources(manifest_path)
        source_validation = {
            "git_head": git_head,
            "sources": sources,
            "status": "HEAD_MATCH",
        }
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("reuse manifest must contain a JSON mapping")
    validate_reuse_manifest(manifest, require_remote_catalog=True)
    if manifest.get("retrieval_ready") is not True:
        raise ValueError("reuse manifest is not retrieval-ready")
    requested_id = _require_hex64(bundle_id, "bundle_id")
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list) or any(
        not isinstance(row, Mapping) for row in bundles
    ):
        raise ValueError("reuse manifest bundles must be a list of mappings")
    matching = [row for row in bundles if row.get("bundle_id") == requested_id]
    if len(matching) != 1:
        raise ValueError("bundle_id must identify exactly one frozen reuse bundle")
    bundle = matching[0]
    integrity = bundle.get("remote_integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("selected bundle lacks remote-integrity metadata")
    files = integrity.get("files")
    if not isinstance(files, list):
        raise ValueError("selected bundle file catalog must be a list")
    path_contract = manifest.get("path_contract")
    if not isinstance(path_contract, Mapping):
        raise ValueError("reuse manifest lacks the frozen path contract")
    required_paths = path_contract.get("required_logical_paths")
    if not isinstance(required_paths, list):
        raise ValueError("reuse manifest lacks the frozen logical-path allowlist")
    safe_required = [_safe_logical_path(value) for value in required_paths]
    if any(not isinstance(row, Mapping) for row in files):
        raise ValueError("selected bundle file catalog entries must be mappings")
    file_paths = [_safe_logical_path(row.get("logical_path")) for row in files]
    if safe_required != sorted(set(safe_required)):
        raise ValueError("frozen logical-path allowlist must be sorted and unique")
    if file_paths != safe_required:
        raise ValueError("selected bundle file catalog differs from the frozen allowlist")
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"selected bundle must contain exactly {EXPECTED_FILE_COUNT} files")
    normalized_files: list[dict[str, Any]] = []
    for row in files:
        sha256 = _require_hex64(row.get("sha256"), f"SHA-256 for {row.get('logical_path')}")
        size = row.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("allowlisted file size must be a positive integer")
        normalized_files.append(
            {
                "logical_path": str(row["logical_path"]),
                "sha256": sha256,
                "size": size,
            }
        )
    digest_payload = {
        "artifact_manifest_file_sha256": integrity.get(
            "artifact_manifest_file_sha256"
        ),
        "bundle_checksums_file_sha256": integrity.get(
            "bundle_checksums_file_sha256"
        ),
        "files": normalized_files,
    }
    expected_catalog_sha = _require_hex64(
        integrity.get("allowed_payload_catalog_sha256"),
        "allowed payload catalog SHA-256",
    )
    if _canonical_sha256(digest_payload) != expected_catalog_sha:
        raise ValueError("selected bundle allowlisted catalog SHA-256 mismatch")
    expected_size = sum(row["size"] for row in normalized_files)
    if integrity.get("selected_file_count") != EXPECTED_FILE_COUNT:
        raise ValueError("selected bundle file-count metadata mismatch")
    if integrity.get("selected_total_size") != expected_size:
        raise ValueError("selected bundle byte-count metadata mismatch")
    selected = dict(bundle)
    selected["remote_integrity"] = dict(integrity)
    selected["remote_integrity"]["files"] = normalized_files
    return selected, hashlib.sha256(manifest_bytes).hexdigest(), source_validation


def _assert_outside_git(path: Path, *, label: str) -> None:
    repository = ROOT.resolve()
    if path == repository or repository in path.parents:
        raise ValueError(f"{label} must be outside the Git worktree")
    for ancestor in (path, *path.parents):
        marker = ancestor / ".git"
        is_repository = marker.is_file() or (
            marker.is_dir() and (marker / "HEAD").is_file()
        )
        if is_repository:
            raise ValueError(f"{label} must not be inside any Git worktree")


def _new_destination(path: str | Path, source_root: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("destination root must be absolute")
    if _lexists(raw):
        raise FileExistsError(f"destination root already exists: {raw}")
    parent = raw.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("destination parent must be an existing non-symlink directory")
    resolved = parent.resolve(strict=True) / raw.name
    _assert_outside_git(resolved, label="destination root")
    if resolved == source_root or source_root in resolved.parents:
        raise ValueError("destination root must not be inside the source bundle")
    return resolved


def _new_receipt_path(path: str | Path, source_root: Path, destination: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("receipt path must be absolute")
    if _lexists(raw):
        raise FileExistsError(f"receipt path already exists: {raw}")
    parent = raw.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("receipt parent must be an existing non-symlink directory")
    resolved = parent.resolve(strict=True) / raw.name
    if resolved == destination or destination in resolved.parents:
        raise ValueError("receipt must be outside the exact destination tree")
    if resolved == source_root or source_root in resolved.parents:
        raise ValueError("receipt must not mutate the source bundle")
    return resolved


def _source_file(source_root: Path, logical_path: str) -> Path:
    current = source_root
    parts = PurePosixPath(logical_path).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"allowlisted source entry is missing: {logical_path}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink is forbidden in allowlisted source path: {logical_path}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(mode):
                raise ValueError(f"non-directory source path component: {logical_path}")
        elif not stat.S_ISREG(mode):
            raise ValueError(f"allowlisted source entry is not a regular file: {logical_path}")
    return current


def _verify_source(
    source_root: Path, files: Sequence[Mapping[str, Any]]
) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    for row in files:
        logical_path = str(row["logical_path"])
        source = _source_file(source_root, logical_path)
        if source.stat(follow_symlinks=False).st_size != int(row["size"]):
            raise ValueError(f"allowlisted source size mismatch: {logical_path}")
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"allowlisted source SHA-256 mismatch: {logical_path}")
        verified[logical_path] = source
    return verified


def _expected_directories(bundle_id: str, files: Sequence[Mapping[str, Any]]) -> set[str]:
    expected: set[str] = set()
    for row in files:
        path = PurePosixPath(bundle_id, str(row["logical_path"]))
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            expected.add(parent.as_posix())
    return expected


def _verify_exact_tree(
    destination: Path,
    bundle_id: str,
    files: Sequence[Mapping[str, Any]],
) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("destination must be a non-symlink directory")
    observed_files: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for current, directories, filenames in os.walk(destination, followlinks=False):
        current_path = Path(current)
        if not stat.S_ISDIR(current_path.lstat().st_mode):
            raise ValueError(f"non-directory destination traversal entry: {current_path}")
        for name in directories:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink is forbidden in destination: {candidate}")
            if not stat.S_ISDIR(mode):
                raise ValueError(f"non-directory destination entry: {candidate}")
            observed_directories.add(candidate.relative_to(destination).as_posix())
        for name in filenames:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink is forbidden in destination: {candidate}")
            if not stat.S_ISREG(mode):
                raise ValueError(f"non-regular destination entry: {candidate}")
            observed_files[candidate.relative_to(destination).as_posix()] = candidate
    expected_files = {
        PurePosixPath(bundle_id, str(row["logical_path"])).as_posix(): row
        for row in files
    }
    if set(observed_files) != set(expected_files):
        unexpected = sorted(set(observed_files) - set(expected_files))
        missing = sorted(set(expected_files) - set(observed_files))
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"destination exact-file set mismatch: {detail}")
    expected_directories = _expected_directories(bundle_id, files)
    if observed_directories != expected_directories:
        unexpected = sorted(observed_directories - expected_directories)
        missing = sorted(expected_directories - observed_directories)
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"destination exact-directory set mismatch: {detail}")
    for relative, path in observed_files.items():
        row = expected_files[relative]
        if path.stat(follow_symlinks=False).st_size != int(row["size"]):
            raise ValueError(f"copied file size mismatch: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"copied file SHA-256 mismatch: {relative}")


def materialize_bundle(
    manifest_path: str | Path,
    bundle_id: str,
    source_bundle: str | Path,
    destination_root: str | Path,
    receipt_path: str | Path,
    *,
    _test_only_skip_committed_source_validation: bool = False,
) -> dict[str, Any]:
    """Copy one frozen bundle into a new exact tree and return its receipt."""
    manifest_path = Path(manifest_path)
    bundle, manifest_sha, source_validation = _load_frozen_bundle(
        manifest_path,
        bundle_id,
        test_only_skip_committed_source_validation=(
            _test_only_skip_committed_source_validation
        ),
    )
    source_raw = Path(source_bundle)
    if source_raw.is_symlink():
        raise ValueError("source bundle must not be a symlink")
    try:
        source_root = source_raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"source bundle is absent: {source_raw}") from exc
    if not source_root.is_dir():
        raise ValueError("source bundle must be an existing directory")
    if source_root.name != bundle_id:
        raise ValueError("source bundle directory name must equal bundle_id")
    destination = _new_destination(destination_root, source_root)
    receipt_output = _new_receipt_path(receipt_path, source_root, destination)
    files = bundle["remote_integrity"]["files"]

    # Preflight the complete allowlist before creating any temporary output.
    verified_sources = _verify_source(source_root, files)
    temporary_root: Path | None = None
    published = False
    try:
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        temporary_bundle = temporary_root / bundle_id
        for row in files:
            logical_path = str(row["logical_path"])
            target = temporary_bundle.joinpath(*PurePosixPath(logical_path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                verified_sources[logical_path], target, follow_symlinks=False
            )
        _verify_exact_tree(temporary_root, bundle_id, files)
        if _lexists(destination):
            raise FileExistsError(f"destination root appeared during copy: {destination}")
        temporary_root.rename(destination)
        published = True
        temporary_root = None
    finally:
        if not published and temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)

    # Verify the published tree again so the receipt binds the final paths.
    _verify_exact_tree(destination, bundle_id, files)
    catalog_sha = bundle["remote_integrity"]["allowed_payload_catalog_sha256"]
    file_receipts = [
        {
            "copy_method": COPY_METHOD,
            "logical_path": row["logical_path"],
            "sha256": row["sha256"],
            "size": row["size"],
        }
        for row in files
    ]
    receipt = {
        "bundle_catalog_sha256": catalog_sha,
        "bundle_id": bundle_id,
        "committed_source_validation": source_validation,
        "destination": {
            "bundle_directory": str(destination / bundle_id),
            "root": str(destination),
        },
        "file_count": len(file_receipts),
        "files": file_receipts,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "artifact_identity": bundle.get("identity"),
            "assigned_host": bundle.get("assigned_host"),
            "bundle_directory": str(source_root),
            "hf_uri": bundle.get("hf_uri"),
        },
        "status": "CHECKSUM_VERIFIED_EXACT_ONE_BUNDLE_COPY",
        "total_size": sum(row["size"] for row in files),
    }
    receipt["copied_file_catalog_sha256"] = _canonical_sha256(file_receipts)
    payload = _canonical_bytes(receipt)
    with receipt_output.open("xb") as handle:
        handle.write(payload)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one committed-manifest Stage-2 bundle into a new exact "
            "non-Git tree."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize_bundle(
        args.manifest,
        args.bundle_id,
        args.source_bundle,
        args.destination_root,
        args.receipt,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
