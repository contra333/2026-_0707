"""Parameter-group helpers for optimizer weight-decay policy."""

from __future__ import annotations

from torch import nn

DEFAULT_WEIGHT_DECAY_POLICY = "weights_only_no_bias_norm"
_NORM_TYPES = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)
_DECAY_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


def make_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    policy: str = DEFAULT_WEIGHT_DECAY_POLICY,
) -> list[dict]:
    """Split model parameters into decay/no-decay groups.

    The default policy applies weight decay only to Conv/Linear weights and
    excludes all biases plus BatchNorm/LayerNorm parameters.
    """
    if policy != DEFAULT_WEIGHT_DECAY_POLICY:
        raise ValueError(f"Unsupported weight_decay_policy: {policy!r}")

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    handled: set[str] = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            handled.add(full_name)
            if param_name == "bias" or isinstance(module, _NORM_TYPES):
                no_decay.append(param)
            elif param_name == "weight" and isinstance(module, _DECAY_TYPES):
                decay.append(param)
            else:
                no_decay.append(param)

    # Include any unusual direct parameters not seen above, conservatively no-decay.
    for name, param in model.named_parameters():
        if param.requires_grad and name not in handled:
            no_decay.append(param)

    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
