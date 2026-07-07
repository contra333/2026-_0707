"""Optimizer factory."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from oge.optimizers.adam_coupled_decoupled import AdamCoupledDecoupled
from oge.optimizers.sgd_coupled_decoupled import SGDCoupledDecoupled
from oge.optimizers.sgdw import SGDW
from oge.train_utils.param_groups import DEFAULT_WEIGHT_DECAY_POLICY, make_weight_decay_param_groups


def _supports(cls, name: str) -> bool:
    return name in inspect.signature(cls).parameters


def _torch_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    # PyTorch-version-safe handling for foreach/fused parity flags: tests and
    # callers can pass them when supported, and older versions silently omit them.
    return {k: v for k, v in kwargs.items() if _supports(cls, k)}


def _params(params_or_model, weight_decay: float, policy: str):
    if isinstance(params_or_model, nn.Module):
        return make_weight_decay_param_groups(params_or_model, weight_decay, policy)
    return params_or_model


def make_optimizer(params_or_model, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = config["name"].lower()
    policy = config.get("weight_decay_policy", DEFAULT_WEIGHT_DECAY_POLICY)
    lr = config["lr"]
    common_torch = {"foreach": False, "fused": False}

    if name == "sgd":
        wd = config.get("weight_decay", 0.0)
        params = _params(params_or_model, wd, policy)
        return torch.optim.SGD(params, lr=lr, momentum=config.get("momentum", 0.0), nesterov=config.get("nesterov", False), weight_decay=wd)
    if name == "sgdw":
        wd = config.get("weight_decay", 0.0)
        params = _params(params_or_model, wd, policy)
        return SGDW(params, lr=lr, momentum=config.get("momentum", 0.0), nesterov=config.get("nesterov", False), weight_decay=wd)
    if name == "adam":
        wd = config.get("weight_decay", 0.0)
        params = _params(params_or_model, wd, policy)
        kwargs = _torch_kwargs(torch.optim.Adam, common_torch)
        return torch.optim.Adam(params, lr=lr, betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)), eps=config.get("eps", 1e-8), weight_decay=wd, **kwargs)
    if name == "adamw":
        wd = config.get("weight_decay", 0.0)
        params = _params(params_or_model, wd, policy)
        kwargs = _torch_kwargs(torch.optim.AdamW, common_torch)
        return torch.optim.AdamW(params, lr=lr, betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)), eps=config.get("eps", 1e-8), weight_decay=wd, **kwargs)
    if name == "sgd_coupled_decoupled":
        total = config.get("total_weight_decay", 0.0)
        ratio = config.get("coupled_ratio", 0.5)
        params = _params(params_or_model, total, policy)
        if ratio == 0.0:
            return SGDW(params, lr=lr, momentum=config.get("momentum", 0.0), nesterov=config.get("nesterov", False), weight_decay=total)
        if ratio == 1.0:
            return torch.optim.SGD(params, lr=lr, momentum=config.get("momentum", 0.0), nesterov=config.get("nesterov", False), weight_decay=total)
        return SGDCoupledDecoupled(params, lr=lr, momentum=config.get("momentum", 0.0), nesterov=config.get("nesterov", False), total_weight_decay=total, coupled_ratio=ratio)
    if name == "adam_coupled_decoupled":
        total = config.get("total_weight_decay", 0.0)
        ratio = config.get("coupled_ratio", 0.5)
        params = _params(params_or_model, total, policy)
        betas = (config.get("beta1", 0.9), config.get("beta2", 0.999))
        if ratio == 0.0:
            kwargs = _torch_kwargs(torch.optim.AdamW, common_torch)
            return torch.optim.AdamW(params, lr=lr, betas=betas, eps=config.get("eps", 1e-8), weight_decay=total, **kwargs)
        if ratio == 1.0:
            kwargs = _torch_kwargs(torch.optim.Adam, common_torch)
            return torch.optim.Adam(params, lr=lr, betas=betas, eps=config.get("eps", 1e-8), weight_decay=total, **kwargs)
        return AdamCoupledDecoupled(params, lr=lr, betas=betas, eps=config.get("eps", 1e-8), total_weight_decay=total, coupled_ratio=ratio)
    raise ValueError(f"Unknown optimizer: {name!r}")
