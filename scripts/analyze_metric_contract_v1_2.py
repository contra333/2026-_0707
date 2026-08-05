#!/usr/bin/env python3
"""Generate the authoritative C1-C4 Metric Contract v1.2 local analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.analysis.metric_contract_v1_2 import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "configs/evaluation/wrn28_10_cifar10_metric_v1_2/"
            "checkpoint_inventory.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_805)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Generate validated tables and statistics without Matplotlib figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        aggregate_dir=args.aggregate_dir,
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        resamples=args.resamples,
        random_seed=args.random_seed,
        make_figures=not args.skip_figures,
    )
    print(
        "PASS: "
        f"{summary['seed_aggregate_count']} seed aggregates, "
        f"{summary['seed0_record_count']} seed-0 records, "
        f"{summary['association_record_count']} associations"
    )


if __name__ == "__main__":
    main()
