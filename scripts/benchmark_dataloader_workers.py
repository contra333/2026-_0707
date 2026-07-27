#!/usr/bin/env python3
"""Benchmark one full training epoch at full local-GPU concurrency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

from oge.data import build_openood_cifar10_loaders
from oge.models import make_model
from oge.optimizers import make_optimizer
from oge.studies.artifacts import atomic_write_json
from oge.studies.preflight import WORKER_CANDIDATES
from oge.training import (
    load_training_config,
    resolve_training_config,
    seed_everything,
    train_one_epoch,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host-id",
        choices=["curie", "lise", "precision_medicine"],
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--gpus")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _child(args: argparse.Namespace) -> int:
    if args.data_root is None or args.workers not in WORKER_CANDIDATES:
        raise ValueError("child benchmark requires data root and a frozen worker count")
    raw = load_training_config(args.config)
    resolved = resolve_training_config(
        raw,
        data_root=args.data_root,
        device="cuda:0",
        max_epochs=1,
    )
    training = resolved["training"]
    generator = seed_everything(int(training["seed"]), deterministic=False)
    loaders = build_openood_cifar10_loaders(
        resolved["dataset"]["definition"],
        data_root=resolved["dataset"]["data_root"],
        batch_size=int(training["batch_size"]),
        num_workers=args.workers,
        pin_memory=bool(training["pin_memory"]),
        drop_last=bool(training["drop_last"]),
        persistent_workers=args.workers > 0,
        train_generator=generator,
        config_root=resolved["dataset"]["imglist_config_root"],
    )
    device = torch.device("cuda:0")
    model = make_model(resolved["model"]).to(device)
    optimizer = make_optimizer(model, resolved["optimizer"])
    criterion = nn.CrossEntropyLoss()
    # The first complete epoch warms filesystem caches and persistent workers.
    # Its metrics are discarded; only the second complete epoch is timed.
    train_one_epoch(
        model,
        loaders["id"][resolved["dataset"]["train_split"]],
        optimizer,
        criterion,
        device=device,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = train_one_epoch(
        model,
        loaders["id"][resolved["dataset"]["train_split"]],
        optimizer,
        criterion,
        device=device,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "epoch_wall_seconds": elapsed,
                "sample_count": metrics["sample_count"],
                "num_workers": args.workers,
            },
            sort_keys=True,
        )
    )
    return 0


def _parent(args: argparse.Namespace) -> int:
    if (
        args.host_id is None
        or args.data_root is None
        or not args.gpus
        or args.output is None
    ):
        raise ValueError("host-id, data-root, gpus, and output are required")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    gpu_tokens = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if not gpu_tokens or len(gpu_tokens) != len(set(gpu_tokens)):
        raise ValueError("--gpus must contain unique explicit parent-visible indices")
    candidates: dict[str, list[float]] = {
        str(workers): [] for workers in WORKER_CANDIDATES
    }
    child_details: dict[str, list[list[dict[str, object]]]] = {
        str(workers): [] for workers in WORKER_CANDIDATES
    }
    for workers in WORKER_CANDIDATES:
        for _ in range(args.repeats):
            processes = []
            started = time.perf_counter()
            for gpu in gpu_tokens:
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--child",
                            "--workers",
                            str(workers),
                            "--data-root",
                            str(args.data_root),
                            "--config",
                            str(args.config),
                        ],
                        cwd=REPOSITORY_ROOT,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            details = []
            for gpu, process in zip(gpu_tokens, processes):
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"worker benchmark failed on GPU {gpu}: {stderr.strip()}"
                    )
                details.append({"gpu": gpu, **json.loads(stdout)})
            host_elapsed = time.perf_counter() - started
            candidates[str(workers)].append(host_elapsed)
            child_details[str(workers)].append(details)
    record = {
        "host_id": args.host_id,
        "full_gpu_concurrency": True,
        "gpu_selectors": gpu_tokens,
        "repeats": args.repeats,
        "candidates": candidates,
        "child_details": child_details,
    }
    atomic_write_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
