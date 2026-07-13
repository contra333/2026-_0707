"""Checkpoint and random-state helpers for epoch-boundary resume."""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_SCHEMA_VERSION = "1.0"
SNAPSHOT_SCHEMA_VERSION = "1.0"


def capture_rng_state(train_generator: torch.Generator) -> dict[str, object]:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
        "train_dataloader_generator": train_generator.get_state(),
    }


def restore_rng_state(state: dict[str, object], train_generator: torch.Generator) -> None:
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "train_dataloader_generator",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing fields: {sorted(missing)}")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict):
        raise ValueError("checkpoint NumPy RNG state must be a mapping")
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count does not match runtime")
        torch.cuda.set_rng_state_all(cuda_state)
    train_generator.set_state(state["train_dataloader_generator"])


def atomic_torch_save(payload: object, path: str | Path) -> None:
    """Write a torch artifact in the target directory and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("torch artifact must contain a mapping")
    return payload
