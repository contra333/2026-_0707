#!/usr/bin/env python3
"""Download and checksum-verify three small operational evaluation shards."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from oge.evaluation.aggregation import discover_operational_jobs

HOSTS = ("curie", "lise", "precision_medicine")
BUCKET = "contra333/ICLR_RUN"


def _parse_plan(text: str) -> dict:
    rows = [json.loads(row) for row in text.splitlines() if row.strip()]
    if not rows or rows[0].get("type") != "header":
        raise ValueError("HF sync output lacks a JSONL header")
    summary = rows[0].get("summary", {})
    if int(summary.get("deletes", -1)) != 0:
        raise ValueError("HF shard download plan contains deletes")
    invalid = [
        row
        for row in rows[1:]
        if row.get("type") == "operation"
        and row.get("action") not in {"download", "skip"}
    ]
    if invalid:
        raise ValueError("HF shard download plan contains a non-download action")
    return {"header": rows[0], "operations": rows[1:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-git-sha", required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--hf-cli", type=Path, required=True)
    args = parser.parse_args()
    if args.download_root.exists():
        raise FileExistsError(f"download root already exists: {args.download_root}")
    args.download_root.mkdir(parents=True)
    evidence = {}
    for host_id in HOSTS:
        source = (
            f"hf://buckets/{BUCKET}/evaluations/metric_contract_v1.2/"
            f"_operations/{args.evaluation_git_sha}/{host_id}"
        )
        destination = args.download_root / host_id
        dry_run = subprocess.run(
            [
                str(args.hf_cli),
                "buckets",
                "sync",
                source,
                str(destination),
                "--dry-run",
                "--no-delete",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = _parse_plan(dry_run.stdout)
        plan_path = args.download_root / f"{host_id}_download_plan.jsonl"
        subprocess.run(
            [
                str(args.hf_cli),
                "buckets",
                "sync",
                source,
                str(destination),
                "--plan",
                str(plan_path),
                "--no-delete",
            ],
            check=True,
        )
        _parse_plan(plan_path.read_text(encoding="utf-8"))
        subprocess.run(
            [str(args.hf_cli), "buckets", "sync", "--apply", str(plan_path)],
            check=True,
        )
        marker_path = destination / "REMOTE_COMPLETE.json"
        if not marker_path.is_file():
            raise ValueError(f"downloaded shard {host_id} lacks REMOTE_COMPLETE.json")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "REMOTE_VERIFIED" or marker.get("destination") != source:
            raise ValueError(f"downloaded shard {host_id} completion identity mismatch")
        evidence[host_id] = {
            "source": source,
            "dry_run_summary": parsed["header"]["summary"],
        }
    jobs = discover_operational_jobs([args.download_root])
    (args.download_root / "collection_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "CHECKSUM_VERIFIED",
                "evaluation_git_sha": args.evaluation_git_sha,
                "host_count": len(HOSTS),
                "checkpoint_job_count": len(jobs),
                "hosts": evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.download_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
