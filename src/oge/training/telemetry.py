"""Non-mutating Task F optimizer-update telemetry and counterfactual helpers."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from oge.optimizers import make_optimizer


UPDATE_TELEMETRY_SCHEMA_VERSION = "task_f_update_telemetry_v1"


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().double()).item())


def _moment_summary(value: torch.Tensor | None) -> dict[str, float] | None:
    if value is None:
        return None
    flat = value.detach().double().reshape(-1)
    if flat.numel() == 0:
        return {"norm": 0.0, "mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0}
    return {
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "mean_abs": float(flat.abs().mean().item()),
        "rms": float(torch.sqrt(torch.mean(flat.square())).item()),
        "max_abs": float(flat.abs().max().item()),
    }


def _angular_update(before: torch.Tensor, after: torch.Tensor) -> float | None:
    left = before.detach().double().reshape(-1)
    right = after.detach().double().reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None
    cosine = torch.dot(left, right) / (left_norm * right_norm)
    return float(torch.acos(torch.clamp(cosine, -1.0, 1.0)).item())


def _layer_name(parameter_name: str) -> str:
    return parameter_name.rpartition(".")[0] or "<root>"


@dataclass
class _StepCapture:
    global_step: int
    parameters: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]


class UpdateTelemetryRecorder:
    """Observe selected optimizer steps without modifying gradients or state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        audit_steps: Sequence[int],
        include_batch_norm: bool,
        activation_residual_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        steps = tuple(audit_steps)
        if (
            any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
            or list(steps) != sorted(set(steps))
        ):
            raise ValueError("audit_steps must be unique increasing positive integers")
        self.model = model
        self.optimizer = optimizer
        self.audit_steps = frozenset(steps)
        self.include_batch_norm = bool(include_batch_norm)
        self.activation_residual_provider = activation_residual_provider
        self.records: list[dict[str, Any]] = []
        self._parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        self._parameter_groups: dict[int, int] = {}
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id not in self._parameter_names:
                    raise ValueError("telemetry optimizer contains a parameter outside the model")
                if parameter_id in self._parameter_groups:
                    raise ValueError("telemetry parameter appears in more than one optimizer group")
                self._parameter_groups[parameter_id] = group_index

    @classmethod
    def from_config(
        cls,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
        activation_residual_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> "UpdateTelemetryRecorder":
        if config.get("schema_version") != UPDATE_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported update telemetry schema_version")
        return cls(
            model=model,
            optimizer=optimizer,
            audit_steps=config.get("audit_steps", []),
            include_batch_norm=bool(config.get("include_batch_norm", True)),
            activation_residual_provider=activation_residual_provider,
        )

    def before_step(self, *, global_step: int) -> _StepCapture | None:
        if global_step not in self.audit_steps:
            return None
        parameters = {}
        gradients = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            parameters[name] = parameter.detach().clone()
            gradients[name] = parameter.grad.detach().clone()
        return _StepCapture(
            global_step=global_step,
            parameters=parameters,
            gradients=gradients,
        )

    def after_step(self, capture: _StepCapture) -> None:
        current = dict(self.model.named_parameters())
        for parameter_name in capture.parameters:
            before = capture.parameters[parameter_name]
            gradient = capture.gradients[parameter_name]
            parameter = current[parameter_name]
            after = parameter.detach()
            update = after - before
            before_flat = before.double().reshape(-1)
            update_flat = update.double().reshape(-1)
            parameter_norm = _norm(before)
            gradient_norm = _norm(gradient)
            update_norm = _norm(update)
            dot = float(torch.dot(update_flat, before_flat).item())
            update_parameter_ratio = (
                update_norm / parameter_norm if parameter_norm != 0.0 else None
            )
            update_weight_cosine = (
                dot / (update_norm * parameter_norm)
                if update_norm != 0.0 and parameter_norm != 0.0
                else None
            )
            radial_signed = dot / parameter_norm if parameter_norm != 0.0 else None
            radial_norm = abs(radial_signed) if radial_signed is not None else None
            tangential_norm = None
            if radial_signed is not None:
                tangential_norm = math.sqrt(max(update_norm**2 - radial_signed**2, 0.0))
            state = self.optimizer.state.get(parameter, {})
            exp_avg = state.get("exp_avg") if isinstance(state, Mapping) else None
            exp_avg_sq = state.get("exp_avg_sq") if isinstance(state, Mapping) else None
            group_index = self._parameter_groups[id(parameter)]
            group = self.optimizer.param_groups[group_index]
            self.records.append(
                {
                    "schema_version": UPDATE_TELEMETRY_SCHEMA_VERSION,
                    "record_type": "parameter_update",
                    "global_step": capture.global_step,
                    "parameter_name": parameter_name,
                    "layer_name": _layer_name(parameter_name),
                    "parameter_group_index": group_index,
                    "parameter_group_identity": f"optimizer_group_{group_index}",
                    "parameter_group_decay": {
                        key: float(group[key])
                        for key in (
                            "weight_decay",
                            "total_weight_decay",
                            "wd_coupled",
                            "wd_decoupled",
                        )
                        if key in group
                    },
                    "parameter_norm": parameter_norm,
                    "gradient_norm": gradient_norm,
                    "optimizer_update_norm": update_norm,
                    "update_parameter_norm_ratio": update_parameter_ratio,
                    "update_weight_cosine": update_weight_cosine,
                    "radial_update_signed_magnitude": radial_signed,
                    "radial_update_norm": radial_norm,
                    "tangential_update_norm": tangential_norm,
                    "angular_update_radians": _angular_update(before, after),
                    "adam_moments": {
                        "exp_avg": _moment_summary(exp_avg),
                        "exp_avg_sq": _moment_summary(exp_avg_sq),
                    },
                }
            )
        self.records.append(self._diagnostic_record(capture.global_step))

    def _diagnostic_record(self, global_step: int) -> dict[str, Any]:
        batch_norm = []
        if self.include_batch_norm:
            for name, module in self.model.named_modules():
                if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                    continue
                running_variance = module.running_var
                batch_norm.append(
                    {
                        "module_name": name,
                        "scale_norm": _norm(module.weight) if module.weight is not None else None,
                        "shift_norm": _norm(module.bias) if module.bias is not None else None,
                        "running_variance": (
                            _moment_summary(running_variance)
                            if running_variance is not None
                            else None
                        ),
                    }
                )
        extensions = (
            copy.deepcopy(dict(self.activation_residual_provider()))
            if self.activation_residual_provider is not None
            else {}
        )
        return {
            "schema_version": UPDATE_TELEMETRY_SCHEMA_VERSION,
            "record_type": "step_diagnostics",
            "global_step": global_step,
            "batch_norm": batch_norm,
            "activation_residual_diagnostics": extensions,
        }


def candidate_optimizer_update(
    *,
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    optimizer_config: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Evaluate one optimizer step on a clone, never on the live parameter/state."""

    if parameter.shape != gradient.shape:
        raise ValueError("counterfactual parameter and gradient shapes differ")
    candidate = nn.Parameter(parameter.detach().clone())
    candidate.grad = gradient.detach().clone()
    optimizer = make_optimizer([candidate], copy.deepcopy(dict(optimizer_config)))
    if optimizer_state is not None:
        optimizer.state[candidate] = copy.deepcopy(dict(optimizer_state))
    before = candidate.detach().clone()
    optimizer.step()
    return candidate.detach() - before


def counterfactual_candidate_updates(
    *,
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    candidates: Mapping[str, Mapping[str, Any]],
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if not candidates:
        raise ValueError("at least one counterfactual candidate is required")
    return {
        name: candidate_optimizer_update(
            parameter=parameter,
            gradient=gradient,
            optimizer_config=config,
            optimizer_state=optimizer_state,
        )
        for name, config in candidates.items()
    }
