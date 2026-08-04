#!/usr/bin/env python3
"""Extract a deterministic Metric Contract v1.2 raw checkpoint cache."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from oge.evaluation.extraction import extract_checkpoint_artifact


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=root / "configs/datasets/oge_cifar10_holdout_v1.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["id_train", "id_validation"],
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--smoke-samples-per-class", type=int)
    parser.add_argument("--protected-authorization")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = " ".join(shlex.quote(value) for value in sys.argv)
    path = extract_checkpoint_artifact(
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        dataset_keys=args.splits,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        smoke_samples_per_class=args.smoke_samples_per_class,
        protected_authorization=args.protected_authorization,
        smoke_only=args.smoke_only,
        extraction_command=command,
        repository_root=Path(__file__).parents[1],
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
