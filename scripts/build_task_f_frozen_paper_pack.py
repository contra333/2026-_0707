#!/usr/bin/env python3
"""Build the frozen Task F paper-quality analysis pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.analysis.task_f_frozen_paper_pack import build_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-analysis", type=Path, required=True)
    parser.add_argument("--manifest-input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-git-sha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_pack(
        merged_path=args.merged_analysis,
        manifest_input_paths=args.manifest_input,
        output_dir=args.output_dir,
        generator_git_sha=args.generator_git_sha,
    )
    print(json.dumps({"status": "PASS", "coverage": manifest["coverage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
