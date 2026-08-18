#!/usr/bin/env python3
"""Run or merge the Task F classifier-insensitive geometry fast kill test."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from oge.analysis.task_f_classifier_insensitive_kill import (
    load_canonical_json,
    merge_host_results,
    run_host_analysis,
    write_canonical_json,
)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    host = subparsers.add_parser("host")
    host.add_argument("--host-id", required=True)
    host.add_argument("--alignment-root", required=True)
    host.add_argument("--feature-root", required=True)
    host.add_argument("--bridge-root", required=True)
    host.add_argument("--output", required=True)
    host.add_argument("--analysis-git-sha")
    host.add_argument("--cpu-workers", required=True, type=int)
    host.add_argument("--chunk-size", type=int, default=2048)
    host.add_argument("--min-free-mib", type=int, default=2048)
    host.add_argument("--force-cpu", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--host-json", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--analysis-git-sha")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "host":
        payload = run_host_analysis(
            host_id=args.host_id,
            alignment_root=args.alignment_root,
            feature_root=args.feature_root,
            bridge_root=args.bridge_root,
            analysis_git_sha=args.analysis_git_sha or _git_sha(),
            cpu_workers=args.cpu_workers,
            chunk_size=args.chunk_size,
            min_free_mib=args.min_free_mib,
            force_cpu=args.force_cpu,
        )
    else:
        if len(args.host_json) != 3:
            raise ValueError("merge requires exactly three --host-json inputs")
        payload = merge_host_results(
            [load_canonical_json(path) for path in args.host_json],
            analysis_git_sha=args.analysis_git_sha,
        )
    digest = write_canonical_json(Path(args.output), payload)
    print(json.dumps({"status": "PASS", "output": args.output, "sha256": digest}))


if __name__ == "__main__":
    main()
