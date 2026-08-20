#!/usr/bin/env python3
"""Validate and summarize six completed ResNet9/MNIST interpolation arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.validation.resnet9_nc_summary import (
    render_resnet9_nc_summary_markdown,
    summarize_resnet9_nc_positive_control,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_resnet9_nc_positive_control(args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        render_resnet9_nc_summary_markdown(summary),
        encoding="utf-8",
    )
    print(f"{summary['verdict']}: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
