"""Atomic small-artifact publication and checksum helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    checksum_path = destination.with_name(f"{destination.name}.sha256")
    checksum_temporary = checksum_path.with_name(
        f".{checksum_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        checksum_temporary.write_text(f"{sha256_file(destination)}  {destination.name}\n", encoding="utf-8")
        os.replace(checksum_temporary, checksum_path)
    finally:
        checksum_temporary.unlink(missing_ok=True)


def create_preserved_attempt_directory(
    artifact_root: str | Path,
    study_id: str,
    trial_id: str,
    attempt_id: str,
) -> Path:
    attempt_dir = Path(artifact_root) / study_id / "trials" / trial_id / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir
