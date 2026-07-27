#!/usr/bin/env python3
"""Regenerate or verify the frozen WRN-28-10 optimizer grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from oge.studies.protocol import (
    generate_grid_bundle,
    load_study_config,
    materialize_grid_bundle,
    verify_materialized_grid_bundle,
)
from oge.training import load_training_config, resolve_training_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Only committed membership files are read; CIFAR images are not opened.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write canonical files. Without this flag, verify checked-in bytes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = load_study_config(args.study_config)
    base = load_training_config(REPOSITORY_ROOT / study["base_training_config"])
    resolved = resolve_training_config(base, data_root=args.data_root, device="cpu")
    bundle = generate_grid_bundle(study, resolved)
    destination = REPOSITORY_ROOT / study["frozen_grid_dir"]
    if args.write:
        materialize_grid_bundle(bundle, destination)
    else:
        verify_materialized_grid_bundle(destination, bundle)
    print(bundle["manifest"]["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
