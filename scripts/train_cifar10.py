#!/usr/bin/env python3
"""Run the reproducible OpenOOD-aligned CIFAR-10 classifier training path."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.training import run_training_from_config


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root / "configs/training/cifar10_wrn28_10.yaml",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Explicit resolved-config override for bounded validation or extension.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_training_from_config(
        config_path=args.config,
        data_root=args.data_root,
        run_dir=args.run_dir,
        device=args.device,
        resume_from=args.resume,
        max_epochs=args.max_epochs,
    )
    print(
        f"Completed epoch {summary['completed_epoch']} with artifacts in {args.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
