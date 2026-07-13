#!/usr/bin/env python3
"""Run a bounded random-model MSP infrastructure smoke evaluation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
import yaml

from oge.data import build_openood_cifar10_loaders, load_dataset_config
from oge.evaluation import evaluate_msp_protocol
from oge.models import make_model


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", choices=["toy_cifar_cnn", "resnet18", "wrn28_10"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--id-max-samples", type=int, default=128)
    parser.add_argument("--ood-max-samples", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=root / "configs/datasets/openood_cifar10_v1_5.yaml",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=root / "configs/evaluation/msp_smoke.yaml",
    )
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    dataset_config = load_dataset_config(args.dataset_config)
    with args.evaluation_config.open("r", encoding="utf-8") as handle:
        evaluation_config = yaml.safe_load(handle)
    loaders = build_openood_cifar10_loaders(
        dataset_config,
        data_root=args.data_root,
        id_max_samples=args.id_max_samples,
        ood_max_samples=args.ood_max_samples,
    )
    model_config = {"name": args.model, "num_classes": 10}
    if args.model == "resnet18":
        model_config["variant"] = "cifar"
    model = make_model(model_config)
    resolved_config = {
        "dataset": dataset_config,
        "evaluation": evaluation_config,
        "model": model_config,
        "data_root": str(args.data_root),
        "id_max_samples": args.id_max_samples,
        "ood_max_samples": args.ood_max_samples,
    }
    evaluate_msp_protocol(
        model,
        loaders,
        resolved_config=resolved_config,
        output_dir=args.output_dir,
        model_name=args.model,
        model_is_random_or_untrained=True,
        device=args.device,
        oge_git_sha=_git_sha(),
    )
    print(f"Wrote infrastructure-only smoke artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
