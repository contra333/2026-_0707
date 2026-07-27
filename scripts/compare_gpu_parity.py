#!/usr/bin/env python3
"""Compare three synthetic A5000/A6000 parity reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.studies.artifacts import atomic_write_json
from oge.studies.preflight import compare_parity_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.report
    ]
    comparison = compare_parity_arrays(reports)
    atomic_write_json(args.output, comparison)
    print(comparison["comparison_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
