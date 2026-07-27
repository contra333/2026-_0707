#!/usr/bin/env python3
"""Freeze host-local shards from approved GPU UUID preflights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.studies.artifacts import atomic_write_json
from oge.studies.preflight import generate_host_shards, load_infrastructure_config
from oge.studies.protocol import load_materialized_grid_bundle, load_study_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
)
DEFAULT_INFRASTRUCTURE = (
    REPOSITORY_ROOT
    / "configs/studies/wrn28_10_optimizer_hpo_v1_2/infrastructure.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--infrastructure", type=Path, default=DEFAULT_INFRASTRUCTURE)
    parser.add_argument("--preflight-freeze", type=Path, required=True)
    parser.add_argument("--throughput-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = load_study_config(args.study_config)
    grid = load_materialized_grid_bundle(
        REPOSITORY_ROOT / study["frozen_grid_dir"]
    )
    infrastructure = load_infrastructure_config(args.infrastructure)
    preflight = json.loads(args.preflight_freeze.read_text(encoding="utf-8"))
    throughput = json.loads(
        args.throughput_decision.read_text(encoding="utf-8")
    )
    manifest = generate_host_shards(
        grid,
        preflight,
        infrastructure,
        throughput,
    )
    atomic_write_json(args.output, manifest)
    print(manifest["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
