#!/usr/bin/env python3
"""Freeze v1.2 C1-C4 roles and the deduplicated follow-up plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.studies.artifacts import atomic_write_json
from oge.studies.protocol import load_materialized_grid_bundle, load_study_config
from oge.studies.selection import build_followup_plan, freeze_grid_roles

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--grid-results", type=Path, required=True)
    parser.add_argument("--role-freeze-output", type=Path, required=True)
    parser.add_argument("--followup-plan-output", type=Path, required=True)
    return parser.parse_args()


def _load_grid_results(path: Path) -> list[dict[str, object]]:
    if path.is_dir():
        candidates = sorted(path.glob("trials/*/trial.json"))
        if not candidates:
            candidates = sorted(path.glob("*/trial.json"))
        if not candidates:
            raise ValueError("grid-results directory contains no trial.json records")
        return [
            json.loads(candidate.read_text(encoding="utf-8"))
            for candidate in candidates
        ]
    results = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(results, dict):
        results = results.get("records")
    if not isinstance(results, list):
        raise ValueError("grid results must be a JSON list or a mapping with records")
    return results


def main() -> int:
    args = parse_args()
    study = load_study_config(args.study_config)
    bundle = load_materialized_grid_bundle(
        REPOSITORY_ROOT / study["frozen_grid_dir"]
    )
    results = _load_grid_results(args.grid_results)
    role_freeze = freeze_grid_roles(results, bundle)
    followup_plan = build_followup_plan(role_freeze, bundle)
    atomic_write_json(args.role_freeze_output, role_freeze)
    atomic_write_json(args.followup_plan_output, followup_plan)
    print(role_freeze["freeze_hash"])
    print(followup_plan["plan_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
