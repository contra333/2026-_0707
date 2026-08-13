#!/usr/bin/env python3
"""Export one Card 13 Task F ID-only WRN feature tap."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from oge.feature_export import TASK_F_ID_SPLITS, export_task_f_from_files
from oge.models.wide_resnet import WRN_FEATURE_TAP_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Task F ID-only feature artifact from an explicit provenance-bearing "
            "checkpoint and an evaluation-transformed NPZ fixture/input."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-split", choices=TASK_F_ID_SPLITS, required=True)
    parser.add_argument("--depth-tap", choices=WRN_FEATURE_TAP_NAMES, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every-batches", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every_batches <= 0:
        raise ValueError("--progress-every-batches must be positive")
    started = time.monotonic()
    completed_batches = 0

    def report_progress(processed: int, total: int) -> None:
        nonlocal completed_batches
        completed_batches += 1
        if completed_batches % args.progress_every_batches != 0 and processed != total:
            return
        elapsed = max(time.monotonic() - started, 1e-12)
        print(
            f"export progress: {processed}/{total} images "
            f"({processed / elapsed:.1f} images/s)",
            file=sys.stderr,
            flush=True,
        )

    artifact = export_task_f_from_files(
        checkpoint_path=args.checkpoint,
        input_npz_path=args.input_npz,
        artifact_root=args.artifact_root,
        dataset_split=args.dataset_split,
        depth_tap=args.depth_tap,
        device=args.device,
        batch_size=args.batch_size,
        progress_callback=report_progress,
    )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
