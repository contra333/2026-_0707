#!/usr/bin/env python3
"""Run the bounded Card-13 discriminant--residual archive diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.analysis.discriminant_residual_diagnostic import run_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("configs/analysis/discriminant_residual_diagnostic_v1.json"),
    )
    parser.add_argument("--preflight-archive", type=Path, required=True)
    parser.add_argument("--nc1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-cap-seconds", type=float, default=4.0 * 60.0 * 60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_diagnostic(
        input_manifest_path=args.input_manifest,
        preflight_archive_path=args.preflight_archive,
        nc1_root=args.nc1_root,
        output_dir=args.output_dir,
        hard_cap_seconds=args.hard_cap_seconds,
    )
    print(f"{summary['status']}: Gate 2 INCONCLUSIVE_IMMUTABLE")


if __name__ == "__main__":
    main()
