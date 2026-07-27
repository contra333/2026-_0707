#!/usr/bin/env python3
"""Capture one approved server's immutable production-preflight record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.studies.artifacts import atomic_write_json
from oge.studies.preflight import capture_server_preflight

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host-id",
        required=True,
        choices=["curie", "lise", "precision_medicine"],
    )
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = capture_server_preflight(
        host_id=args.host_id,
        repository_root=REPOSITORY_ROOT,
        dependency_lock=args.dependency_lock,
        artifact_root=args.artifact_root,
        backup_root=args.backup_root,
    )
    atomic_write_json(args.output, record)
    print(json.dumps({"record_hash": record["record_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
