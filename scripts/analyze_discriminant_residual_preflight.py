#!/usr/bin/env python3
"""Generate the one bounded Card-13 v11 historical ID-only CPU preflight v2."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.analysis.discriminant_residual_preflight import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path(
            "configs/analysis/discriminant_residual_preflight_v1.json"
        ),
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--v1-bundle-records",
        type=Path,
        required=True,
        help="Immutable v1 bundle_records.jsonl used for the strict per-fit diff.",
    )
    parser.add_argument(
        "--hard-cap-seconds",
        type=float,
        default=4.0 * 60.0 * 60.0,
        help="Stop as INCONCLUSIVE before starting more work after this cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(
        input_manifest_path=args.input_manifest,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        v1_bundle_records_path=args.v1_bundle_records,
        hard_cap_seconds=args.hard_cap_seconds,
        progress_callback=lambda message: print(message, flush=True),
    )
    print(
        f"{summary['status']}: Gate 2 "
        f"{summary.get('gate_2', {}).get('gate_status', 'INCONCLUSIVE')}"
    )


if __name__ == "__main__":
    main()
