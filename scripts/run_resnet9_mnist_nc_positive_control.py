#!/usr/bin/env python3
"""Run one frozen ResNet9/MNIST NC positive-control interpolation arm."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.validation import run_resnet9_nc_positive_control


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root
        / "configs/validation/resnet9_mnist_nc_positive_control_v1.yaml",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--coupled-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3141)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_resnet9_nc_positive_control(
        config_path=args.config,
        data_root=args.data_root,
        run_dir=args.run_dir,
        device=args.device,
        coupled_ratio=args.coupled_ratio,
        seed=args.seed,
        max_epochs=args.max_epochs,
        download=args.download,
        resume_from=args.resume,
    )
    print(
        f"completed ratio={summary['coupled_ratio']} epoch={summary['completed_epoch']} "
        f"artifacts={args.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
