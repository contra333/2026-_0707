#!/usr/bin/env python3
"""Run a bounded non-protected Metric Contract v1.2 cache smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.evaluation.runtime import write_bounded_smoke_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-split", default="id_validation", choices=["id_validation"])
    args = parser.parse_args()
    output = write_bounded_smoke_result(
        raw_artifact_path=args.raw_artifact,
        output_dir=args.output_dir,
        query_split=args.query_split,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
