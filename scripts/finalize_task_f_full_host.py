#!/usr/bin/env python3
"""Wait for Task F source exports, validate a host shard, and upload it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.training.task_f_source_finalizer import finalize_task_f_source_host


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--expected-finalizer-git-sha", required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-cli", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize_task_f_source_host(
        repository_root=REPOSITORY_ROOT,
        expected_finalizer_git_sha=args.expected_finalizer_git_sha,
        host_id=args.host_id,
        source_worktree=args.source_worktree,
        output_root=args.output_root,
        hf_cli=args.hf_cli,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
