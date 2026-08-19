#!/usr/bin/env python3
"""Build host-local geometry or the central ResNet-18 paper pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.analysis.resnet18_paper_pack import (
    build_host_geometry,
    build_paper_pack,
    manifest_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    geometry = subparsers.add_parser(
        "geometry-host",
        help="compute ID-only endpoint geometry from one host terminal",
    )
    geometry.add_argument("--id-terminal", required=True, type=Path)
    geometry.add_argument("--output", required=True, type=Path)

    build = subparsers.add_parser(
        "build",
        help="audit completed score artifacts and build the central reader pack",
    )
    build.add_argument("--evaluation-root", required=True, type=Path)
    build.add_argument("--training-root", required=True, type=Path)
    build.add_argument(
        "--geometry-json",
        required=True,
        action="append",
        type=Path,
        help="host geometry JSON; provide exactly three times",
    )
    build.add_argument("--wrn-pack", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--analysis-git-sha", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "geometry-host":
        result = build_host_geometry(
            id_terminal_path=args.id_terminal,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "host": result["host"],
                    "runs": len(result["rows"]),
                    "output": str(args.output.resolve()),
                    "output_identity_sha256": result["output_identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    if len(args.geometry_json) != 3:
        raise SystemExit("build requires exactly three --geometry-json inputs")
    result = build_paper_pack(
        evaluation_root=args.evaluation_root,
        training_root=args.training_root,
        geometry_paths=args.geometry_json,
        wrn_root=args.wrn_pack,
        output_root=args.output_dir,
        analysis_git_sha=args.analysis_git_sha,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "technical_status": result["technical_status"],
                "scientific_verdict": result["scientific_verdict"],
                "coverage": result["coverage"],
                "manifest_sha256": manifest_sha256(args.output_dir),
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
