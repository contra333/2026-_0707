#!/usr/bin/env python3
"""Fail closed unless all approved server preflights are identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.studies.artifacts import atomic_write_json
from oge.studies.preflight import (
    compare_server_preflights,
    load_infrastructure_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFRASTRUCTURE = (
    REPOSITORY_ROOT
    / "configs/studies/wrn28_10_optimizer_hpo_v1_2/infrastructure.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infrastructure", type=Path, default=DEFAULT_INFRASTRUCTURE)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    infrastructure = load_infrastructure_config(args.infrastructure)
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.report
    ]
    freeze = compare_server_preflights(reports, infrastructure)
    atomic_write_json(args.output, freeze)
    print(freeze["freeze_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
