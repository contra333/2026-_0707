#!/usr/bin/env python3
"""Run the reproducible OGE CIFAR-10 holdout classifier training path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# PyTorch requires this process-level setting before the first CuBLAS call when
# strict deterministic algorithms are enabled. It is harmless for ordinary
# nondeterministic configs and is recorded by the runner environment payload.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from oge.training import run_training_from_config


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--fork-from-prefix",
        type=Path,
        help=(
            "Start a new run from a zero-decay last.pt while preserving model, "
            "optimizer tensor, scheduler, RNG, and DataLoader-generator state."
        ),
    )
    parser.add_argument(
        "--defer-id-test",
        action="store_true",
        help="Do not evaluate or materialize ID-test results for study-selection runs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Explicit resolved-config override for bounded validation or extension.",
    )
    parser.add_argument(
        "--stop-after-epoch-boundary",
        type=int,
        help=(
            "Execution-only Task F runtime control: return normally after all "
            "artifacts for this completed epoch are committed."
        ),
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
        fork_from_prefix=args.fork_from_prefix,
        max_epochs=args.max_epochs,
        defer_id_test=args.defer_id_test,
        stop_after_epoch_boundary=args.stop_after_epoch_boundary,
    )
    if summary["status"] == "completed":
        print(
            f"Completed epoch {summary['completed_epoch']} with artifacts in {args.run_dir}"
        )
    else:
        print(
            f"Stopped normally at epoch {summary['completed_epoch']} boundary "
            f"with artifacts in {args.run_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
