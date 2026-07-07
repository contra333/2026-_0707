"""Adam with mixed coupled and decoupled weight decay."""

from __future__ import annotations

import math
import torch


class AdamCoupledDecoupled(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, total_weight_decay: float = 0.0, coupled_ratio: float = 0.5):
        if not 0.0 <= coupled_ratio <= 1.0:
            raise ValueError("coupled_ratio must be in [0, 1]")
        wd_c = coupled_ratio * total_weight_decay
        wd_d = (1.0 - coupled_ratio) * total_weight_decay
        defaults = dict(lr=lr, betas=betas, eps=eps, total_weight_decay=total_weight_decay, coupled_ratio=coupled_ratio,
                        wd_coupled=wd_c, wd_decoupled=wd_d, weight_decay=total_weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            beta1, beta2 = group["betas"]
            wd_c, wd_d = group["wd_coupled"], group["wd_decoupled"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd_d != 0:
                    p.mul_(1 - lr * wd_d)
                grad = p.grad.detach()
                if wd_c != 0:
                    grad = grad.add(p.detach(), alpha=wd_c)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
