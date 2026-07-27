#!/usr/bin/env python3
"""Run a synthetic WRN forward and one-step update parity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

import torch
from torch import nn

from oge.models import make_model
from oge.optimizers import make_optimizer
from oge.studies.artifacts import atomic_write_json
from oge.studies.preflight import ASSIGNMENT_SEED, capture_numerical_policy
from oge.training import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host-id",
        required=True,
        choices=["curie", "lise", "precision_medicine"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU parity probe requires CUDA")
    seed_everything(ASSIGNMENT_SEED, deterministic=False)
    model = make_model(
        {
            "name": "wrn28_10",
            "num_classes": 10,
            "depth": 28,
            "widen_factor": 10,
            "dropout_rate": 0.0,
            "init_policy": "msr_fan_in",
        }
    )
    initial_state_hash = _state_hash(model)
    target = torch.device(args.device)
    model = model.to(target)
    optimizer = make_optimizer(
        model,
        {
            "name": "sgd",
            "lr": 0.1,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 5e-4,
            "weight_decay_policy": "weights_only_no_bias_norm",
        },
    )
    inputs = torch.linspace(
        -1.0,
        1.0,
        steps=4 * 3 * 32 * 32,
        dtype=torch.float32,
    ).reshape(4, 3, 32, 32).to(target)
    labels = torch.tensor([0, 1, 2, 3], device=target)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits_before = model(inputs)
    loss = nn.functional.cross_entropy(logits_before, labels)
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits_after, features_after = model(inputs, return_features=True)
    torch.cuda.synchronize(target)
    report = {
        "host_id": args.host_id,
        "hostname": socket.gethostname(),
        "seed": ASSIGNMENT_SEED,
        "device": str(target),
        "gpu_name": torch.cuda.get_device_name(target),
        "initial_state_hash": initial_state_hash,
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "logits_before": logits_before.detach().cpu().tolist(),
        "logits_after": logits_after.detach().cpu().tolist(),
        "features_after": features_after.detach().cpu().tolist(),
        "numerical_policy": capture_numerical_policy(),
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"initial_state_hash": initial_state_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
