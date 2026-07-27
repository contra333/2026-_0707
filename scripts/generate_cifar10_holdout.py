#!/usr/bin/env python3
"""Generate the frozen OGE CIFAR-10 45k/5k/10k imglists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oge.data import generate_cifar10_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-train-imglist", type=Path, required=True)
    parser.add_argument("--openood-id-validation-imglist", type=Path, required=True)
    parser.add_argument("--openood-id-test-imglist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_cifar10_holdout(
        official_train_imglist=args.official_train_imglist,
        openood_id_validation_imglist=args.openood_id_validation_imglist,
        openood_id_test_imglist=args.openood_id_test_imglist,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

