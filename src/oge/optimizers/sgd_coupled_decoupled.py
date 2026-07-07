"""SGD with mixed coupled and decoupled weight decay."""

from __future__ import annotations

import torch


class SGDCoupledDecoupled(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0, total_weight_decay: float = 0.0, coupled_ratio: float = 0.5, nesterov: bool = False):
        if not 0.0 <= coupled_ratio <= 1.0:
            raise ValueError("coupled_ratio must be in [0, 1]")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a positive momentum")
        wd_c = coupled_ratio * total_weight_decay
        wd_d = (1.0 - coupled_ratio) * total_weight_decay
        defaults = dict(lr=lr, momentum=momentum, total_weight_decay=total_weight_decay, coupled_ratio=coupled_ratio,
                        wd_coupled=wd_c, wd_decoupled=wd_d, nesterov=nesterov, weight_decay=total_weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            wd_c, wd_d, nesterov = group["wd_coupled"], group["wd_decoupled"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd_d != 0:
                    p.mul_(1 - lr * wd_d)
                d_p = p.grad.detach()
                if wd_c != 0:
                    d_p = d_p.add(p.detach(), alpha=wd_c)
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
