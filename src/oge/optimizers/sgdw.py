"""Decoupled-weight-decay SGD."""

from __future__ import annotations

import torch


class SGDW(torch.optim.Optimizer):
    """SGD with decoupled weight decay.

    The momentum buffer is updated from gradients only; weight decay is applied
    directly as parameter shrinkage before the gradient step.
    """

    def __init__(self, params, lr: float, momentum: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False):
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr, momentum, and weight_decay must be non-negative")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a positive momentum")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.detach()
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf
                p.add_(d_p, alpha=-lr)
        return loss
