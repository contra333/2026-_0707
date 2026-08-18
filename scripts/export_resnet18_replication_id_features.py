#!/usr/bin/env python3
"""Export an endpoint-only ResNet-18 replication ID artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.feature_export import (
    RESNET18_REPLICATION_ID_SPLITS,
    export_resnet18_replication_from_files,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-split", choices=RESNET18_REPLICATION_ID_SPLITS, required=True
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    artifact = export_resnet18_replication_from_files(
        checkpoint_path=args.checkpoint,
        input_npz_path=args.input_npz,
        artifact_root=args.artifact_root,
        dataset_split=args.dataset_split,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
