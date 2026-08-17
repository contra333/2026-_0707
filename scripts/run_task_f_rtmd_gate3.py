#!/usr/bin/env python3
"""Run or collect the frozen Task F fresh-ID RtMD Gate 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.analysis.task_f_rtmd_gate_execution import (
    collect_gate3,
    run_gate3_host,
    verify_gate3_terminal,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    host = subparsers.add_parser("run-host")
    host.add_argument(
        "--host-id", choices=("curie", "lise", "precision_medicine"), required=True
    )
    host.add_argument("--worker-spec-root", type=Path, required=True)
    host.add_argument("--ledger", type=Path, required=True)
    host.add_argument("--expected-evaluation-git-sha", required=True)
    host.add_argument("--execution-git-sha", required=True)
    host.add_argument("--output-directory", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--host-output", type=Path, action="append", required=True)
    collect.add_argument("--output-directory", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--terminal-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run-host":
        destination = run_gate3_host(
            host_id=args.host_id,
            worker_spec_root=args.worker_spec_root,
            ledger_path=args.ledger,
            expected_evaluation_git_sha=args.expected_evaluation_git_sha,
            execution_git_sha=args.execution_git_sha,
            output_directory=args.output_directory,
        )
        print(json.dumps({"status": "PASS", "output": str(destination)}))
        return 0
    if args.command == "collect":
        destination = collect_gate3(
            host_outputs=args.host_output,
            output_directory=args.output_directory,
        )
        terminal = verify_gate3_terminal(destination)
        print(
            json.dumps(
                {
                    "status": terminal["status"],
                    "gate3_status": terminal["gate3_verdict"]["status"],
                    "rtmd_activated": terminal["gate3_verdict"]["activated"],
                    "output": str(destination),
                },
                sort_keys=True,
            )
        )
        return 0
    terminal = verify_gate3_terminal(args.terminal_directory)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "gate3_status": terminal["gate3_verdict"]["status"],
                "rtmd_activated": terminal["gate3_verdict"]["activated"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
