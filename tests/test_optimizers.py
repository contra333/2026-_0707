import inspect

import pytest
import torch

from oge.optimizers import AdamCoupledDecoupled, SGDCoupledDecoupled, SGDW, make_optimizer


def torch_opt_kwargs(cls):
    # Version-safe parity flags: foreach/fused are passed only when supported by
    # the installed PyTorch version, so tests run on older PyTorch releases too.
    sig = inspect.signature(cls)
    return {k: False for k in ("foreach", "fused") if k in sig.parameters}


def param(value=(1.0, -2.0, 3.0), grad=(0.1, -0.2, 0.3)):
    p = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
    p.grad = torch.tensor(grad, dtype=torch.float64)
    return p


def step(opt):
    opt.step()


def assert_close(a, b):
    torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("momentum,nesterov", [(0.0, False), (0.9, True)])
def test_sgd_parity(momentum, nesterov):
    p1, p2 = param(), param()
    cfg = {"name": "sgd", "lr": 0.1, "momentum": momentum, "nesterov": nesterov, "weight_decay": 5e-4}
    opt1 = make_optimizer([p1], cfg)
    opt2 = torch.optim.SGD([p2], lr=0.1, momentum=momentum, nesterov=nesterov, weight_decay=5e-4)
    step(opt1); step(opt2)
    assert_close(p1, p2)


def test_sgdw_plain_manual_update():
    p = param()
    old = p.detach().clone()
    grad = p.grad.detach().clone()
    opt = make_optimizer([p], {"name": "sgdw", "lr": 0.1, "momentum": 0.0, "nesterov": False, "weight_decay": 0.01})
    step(opt)
    expected = old * (1 - 0.1 * 0.01) - 0.1 * grad
    torch.testing.assert_close(p.detach(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("momentum,nesterov", [(0.0, False), (0.9, True)])
def test_sgdw_zero_weight_decay_matches_sgd(momentum, nesterov):
    p1, p2 = param(), param()
    opt1 = make_optimizer([p1], {"name": "sgdw", "lr": 0.1, "momentum": momentum, "nesterov": nesterov, "weight_decay": 0.0})
    opt2 = torch.optim.SGD([p2], lr=0.1, momentum=momentum, nesterov=nesterov, weight_decay=0.0)
    step(opt1); step(opt2)
    assert_close(p1, p2)


def test_sgdw_momentum_buffer_is_grad_only():
    p = param(value=(10.0, -10.0), grad=(0.25, -0.5))
    opt = SGDW([p], lr=0.1, momentum=0.9, weight_decay=0.1)
    step(opt)
    buf = opt.state[p]["momentum_buffer"]
    torch.testing.assert_close(buf, torch.tensor((0.25, -0.5), dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_adam_parity_and_no_decoupled_weight_decay_kwarg():
    p1, p2 = param(), param()
    cfg = {"name": "adam", "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 1e-4}
    opt1 = make_optimizer([p1], cfg)
    opt2 = torch.optim.Adam([p2], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, **torch_opt_kwargs(torch.optim.Adam))
    assert isinstance(opt1, torch.optim.Adam)
    assert "decoupled_weight_decay" not in opt1.defaults or opt1.defaults["decoupled_weight_decay"] is False
    step(opt1); step(opt2)
    assert_close(p1, p2)


def test_adamw_parity():
    p1, p2 = param(), param()
    opt1 = make_optimizer([p1], {"name": "adamw", "lr": 5e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 1e-4})
    opt2 = torch.optim.AdamW([p2], lr=5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, **torch_opt_kwargs(torch.optim.AdamW))
    step(opt1); step(opt2)
    assert_close(p1, p2)


@pytest.mark.parametrize("ratio,cls", [(0.0, torch.optim.AdamW), (1.0, torch.optim.Adam)])
def test_adam_coupled_decoupled_endpoints(ratio, cls):
    p1, p2 = param(), param()
    cfg = {"name": "adam_coupled_decoupled", "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "total_weight_decay": 5e-4, "coupled_ratio": ratio}
    opt1 = make_optimizer([p1], cfg)
    opt2 = cls([p2], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, **torch_opt_kwargs(cls))
    assert isinstance(opt1, cls)
    step(opt1); step(opt2)
    assert_close(p1, p2)


def test_adam_coupled_decoupled_mixed_internal_check():
    p = param(value=(2.0, -3.0), grad=(0.4, -0.6))
    opt = make_optimizer([p], {"name": "adam_coupled_decoupled", "lr": 1e-3, "total_weight_decay": 0.2, "coupled_ratio": 0.5})
    old = p.detach().clone()
    step(opt)
    assert isinstance(opt, AdamCoupledDecoupled)
    expected_grad = p.grad.detach() + old * 0.1
    torch.testing.assert_close(opt.state[p]["exp_avg"], expected_grad * 0.1, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ratio,ref_name", [(0.0, "sgdw"), (1.0, "sgd")])
def test_sgd_coupled_decoupled_endpoints(ratio, ref_name):
    p1, p2 = param(), param()
    cfg = {"name": "sgd_coupled_decoupled", "lr": 0.1, "momentum": 0.9, "nesterov": True, "total_weight_decay": 5e-4, "coupled_ratio": ratio}
    opt1 = make_optimizer([p1], cfg)
    opt2 = make_optimizer([p2], {"name": ref_name, "lr": 0.1, "momentum": 0.9, "nesterov": True, "weight_decay": 5e-4})
    step(opt1); step(opt2)
    assert_close(p1, p2)


def test_sgd_coupled_decoupled_mixed_internal_check():
    p = param(value=(2.0, -3.0), grad=(0.4, -0.6))
    opt = make_optimizer([p], {"name": "sgd_coupled_decoupled", "lr": 0.1, "momentum": 0.9, "nesterov": False, "total_weight_decay": 0.2, "coupled_ratio": 0.5})
    old = p.detach().clone()
    step(opt)
    assert isinstance(opt, SGDCoupledDecoupled)
    expected_buf = p.grad.detach() + old * 0.1
    torch.testing.assert_close(opt.state[p]["momentum_buffer"], expected_buf, rtol=1e-12, atol=1e-12)


def test_interpolation_optimizers_respect_group_no_decay():
    for opt_cls, kwargs in (
        (SGDCoupledDecoupled, {"lr": 0.1, "momentum": 0.0, "total_weight_decay": 0.2, "coupled_ratio": 0.5}),
        (AdamCoupledDecoupled, {"lr": 0.1, "total_weight_decay": 0.2, "coupled_ratio": 0.5}),
    ):
        decay_p = param(value=(2.0,), grad=(0.0,))
        no_decay_p = param(value=(2.0,), grad=(0.0,))
        opt = opt_cls(
            [
                {"params": [decay_p], "weight_decay": 0.2},
                {"params": [no_decay_p], "weight_decay": 0.0},
            ],
            **kwargs,
        )
        step(opt)
        assert no_decay_p.item() == pytest.approx(2.0)
        assert decay_p.item() != pytest.approx(2.0)
