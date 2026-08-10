#!/usr/bin/env python3
"""Prepare and verify an exact, network-free Stage-2 artifact retrieval plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
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


FILTER_SCHEMA_VERSION = "fixed_readout_stage2_exact_filter_v1"
RECEIPT_SCHEMA_VERSION = "fixed_readout_stage2_download_receipt_v1"
EXPECTED_BUNDLE_COUNT = 30
EXPECTED_FILE_COUNT = 5_520


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"reuse manifest must be a regular file: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_reuse_manifest(manifest, require_remote_catalog=True)
    return manifest


def _outside_worktree(destination: Path) -> Path:
    if not destination.is_absolute():
        raise ValueError("destination must be absolute")
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError("destination must be outside the Git worktree")


def _expected_files(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    required = manifest["path_contract"]["required_logical_paths"]
    expected: dict[str, dict[str, Any]] = {}
    for bundle in manifest["bundles"]:
        bundle_id = bundle["bundle_id"]
        files = bundle["remote_integrity"]["files"]
        if [row["logical_path"] for row in files] != required:
            raise ValueError("bundle allowlist differs from required logical paths")
        for row in files:
            relative = PurePosixPath(bundle_id, row["logical_path"]).as_posix()
            if relative in expected:
                raise ValueError(f"duplicate expected retrieval path: {relative}")
            expected[relative] = {
                "path": relative,
                "sha256": row["sha256"],
                "size": row["size"],
            }
    if len(manifest["bundles"]) != EXPECTED_BUNDLE_COUNT:
        raise ValueError("retrieval requires exactly 30 bundles")
    if len(expected) != EXPECTED_FILE_COUNT:
        raise ValueError("retrieval requires exactly 5,520 files")
    return dict(sorted(expected.items()))


def _walk_destination(destination: Path) -> dict[str, Path]:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("destination must be an existing non-symlink directory")
    observed: dict[str, Path] = {}
    for current, directories, filenames in os.walk(destination, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *filenames]:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink is forbidden in destination: {candidate}")
            if name in directories and not stat.S_ISDIR(mode):
                raise ValueError(f"non-directory traversal entry: {candidate}")
        for name in filenames:
            candidate = current_path / name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError(f"non-regular file is forbidden: {candidate}")
            relative = candidate.relative_to(destination).as_posix()
            observed[relative] = candidate
    return observed


def _validate_existing(
    destination: Path,
    expected: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool,
) -> set[str]:
    observed = _walk_destination(destination)
    extras = sorted(set(observed) - set(expected))
    if extras:
        raise ValueError(f"unexpected destination file: {extras[0]}")
    if require_complete:
        missing = sorted(set(expected) - set(observed))
        if missing:
            raise ValueError(f"missing expected destination file: {missing[0]}")
    verified = set()
    for relative, path in observed.items():
        record = expected[relative]
        if path.stat().st_size != record["size"]:
            raise ValueError(f"existing file size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"existing file SHA-256 mismatch: {relative}")
        verified.add(relative)
    return verified


def _assert_destination_contract(manifest: Mapping[str, Any], destination: Path) -> Path:
    contract = manifest["destination"]
    if contract != {
        "create_automatically": False,
        "existing_file_policy": "verified_exact_resume_only",
        "kind": "absolute_non_git_directory",
        "mismatched_or_unexpected_existing_files": "reject",
        "must_be_outside_git_worktree": True,
        "root": contract["root"],
        "verified_existing_files": "skip_without_overwrite",
    }:
        raise ValueError("unsupported destination contract")
    resolved = _outside_worktree(destination)
    if str(resolved) != contract["root"]:
        raise ValueError("destination differs from the manifest-frozen root")
    return resolved


def render_exact_filter(expected: Mapping[str, Mapping[str, Any]]) -> bytes:
    lines = [f"+ {relative}" for relative in expected]
    lines.append("- *")
    return ("\n".join(lines) + "\n").encode("utf-8")


def prepare(
    manifest_path: str | Path,
    filter_path: str | Path,
    plan_path: str | Path,
    *,
    destination: str | Path | None = None,
    hf_cli: str = "hf",
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    destination_path = _assert_destination_contract(
        manifest, Path(destination or manifest["destination"]["root"])
    )
    expected = _expected_files(manifest)
    verified = _validate_existing(
        destination_path, expected, require_complete=False
    )
    filter_output = Path(filter_path)
    plan_output = Path(plan_path)
    if filter_output.resolve(strict=False) == plan_output.resolve(strict=False):
        raise ValueError("filter and plan paths must be distinct")
    if plan_output.exists() or plan_output.is_symlink():
        raise FileExistsError(f"plan path already exists: {plan_output}")
    if filter_output.resolve(strict=False).is_relative_to(destination_path):
        raise ValueError("filter file must be outside the retrieval destination")
    if plan_output.resolve(strict=False).is_relative_to(destination_path):
        raise ValueError("plan file must be outside the retrieval destination")
    payload = render_exact_filter(expected)
    with filter_output.open("xb") as handle:
        handle.write(payload)
    command = [
        hf_cli,
        "buckets",
        "sync",
        manifest["hf_root"],
        str(destination_path),
        "--filter-from",
        str(filter_output),
        "--plan",
        str(plan_output),
        "--no-delete",
        "--ignore-times",
    ]
    return {
        "command": command,
        "command_shell": shlex.join(command),
        "expected_file_count": len(expected),
        "filter_sha256": hashlib.sha256(payload).hexdigest(),
        "verified_existing_count": len(verified),
    }


def _load_plan(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("type") != "header":
        raise ValueError("HF sync plan must start with one header")
    if any(row.get("type") != "operation" for row in rows[1:]):
        raise ValueError("HF sync plan contains a non-operation row")
    return rows[0], rows[1:]


def verify_plan(
    manifest_path: str | Path,
    plan_path: str | Path,
    *,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    destination_path = _assert_destination_contract(
        manifest, Path(destination or manifest["destination"]["root"])
    )
    expected = _expected_files(manifest)
    verified_existing = _validate_existing(
        destination_path, expected, require_complete=False
    )
    header, operations = _load_plan(plan_path)
    if header.get("source") != manifest["hf_root"]:
        raise ValueError("HF sync plan source mismatch")
    if header.get("dest") != str(destination_path):
        raise ValueError("HF sync plan destination mismatch")
    if len(operations) != len(expected):
        raise ValueError("HF sync plan operation count mismatch")
    by_path: dict[str, dict[str, Any]] = {}
    for operation in operations:
        relative = operation.get("path")
        if relative in by_path:
            raise ValueError(f"duplicate HF sync plan path: {relative}")
        if relative not in expected:
            raise ValueError(f"extra HF sync plan path: {relative}")
        action = operation.get("action")
        if action not in {"download", "skip"}:
            raise ValueError(f"forbidden HF sync action: {action}")
        if operation.get("size") != expected[relative]["size"]:
            raise ValueError(f"HF sync plan size mismatch: {relative}")
        if action == "skip" and relative not in verified_existing:
            raise ValueError(f"unverified-existing skip in HF sync plan: {relative}")
        if action == "download" and relative in verified_existing:
            raise ValueError(f"HF sync plan would overwrite verified existing file: {relative}")
        by_path[relative] = operation
    if set(by_path) != set(expected):
        raise ValueError("HF sync plan does not cover the exact allowlist")
    summary = header.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("HF sync plan header summary is absent")
    expected_downloads = len(expected) - len(verified_existing)
    expected_summary = {
        "uploads": 0,
        "downloads": expected_downloads,
        "deletes": 0,
        "skips": len(verified_existing),
        "total_size": sum(
            row["size"]
            for relative, row in expected.items()
            if relative not in verified_existing
        ),
    }
    if dict(summary) != expected_summary:
        raise ValueError("HF sync plan summary mismatch")
    return {"expected_file_count": len(expected), **expected_summary}


def verify_download(
    manifest_path: str | Path,
    receipt_path: str | Path,
    *,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    destination_path = _assert_destination_contract(
        manifest, Path(destination or manifest["destination"]["root"])
    )
    expected = _expected_files(manifest)
    verified = _validate_existing(destination_path, expected, require_complete=True)
    if len(verified) != EXPECTED_FILE_COUNT:
        raise ValueError("download verification did not cover exactly 5,520 files")
    receipt_output = Path(receipt_path)
    if receipt_output.resolve(strict=False).is_relative_to(destination_path):
        raise ValueError("receipt must be outside the retrieval destination")
    catalog = list(expected.values())
    receipt = {
        "destination": str(destination_path),
        "file_count": len(catalog),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "CHECKSUM_VERIFIED_EXACT_ALLOWLIST",
        "total_size": sum(row["size"] for row in catalog),
        "verified_file_catalog_sha256": hashlib.sha256(
            _canonical_bytes(catalog)
        ).hexdigest(),
    }
    payload = _canonical_bytes(receipt)
    with receipt_output.open("xb") as handle:
        handle.write(payload)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the exact Stage-2 reuse retrieval."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--destination", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--filter", type=Path, required=True)
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--hf-cli", default="hf")
    plan_parser = subparsers.add_parser("verify-plan")
    plan_parser.add_argument("--plan", type=Path, required=True)
    download_parser = subparsers.add_parser("verify-download")
    download_parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.manifest,
            args.filter,
            args.plan,
            destination=args.destination,
            hf_cli=args.hf_cli,
        )
    elif args.command == "verify-plan":
        result = verify_plan(
            args.manifest, args.plan, destination=args.destination
        )
    else:
        result = verify_download(
            args.manifest, args.receipt, destination=args.destination
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
