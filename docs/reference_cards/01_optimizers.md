# Reference Card 01: Optimizers

This card is the implementation reference for optimizer semantics in this repository.

## Shared parameter-group policy

Default `weight_decay_policy`: `weights_only_no_bias_norm`.

- Apply configured weight decay to Conv and Linear weights, including a Linear layer named `classifier`.
- Apply `weight_decay=0` to bias parameters, BatchNorm parameters, and LayerNorm parameters.
- Optimizers must be testable on toy `torch.nn.Parameter` tensors without a model or dataset.

## Optimizer semantics

### `sgd`

Uses `torch.optim.SGD` directly. PyTorch SGD `weight_decay` is coupled L2 weight decay. Plain SGD is represented by `momentum=0.0` and `nesterov=false`.

### `sgdw`

Uses custom `SGDW`. It must not use `torch.optim.SGD(weight_decay=...)` to implement decoupled weight decay.

- Weight decay is not added to gradients.
- Weight decay is not added to the momentum buffer.
- Decoupled weight decay is applied directly to the parameter by shrinkage.

Plain update:

```text
parameter_new = parameter_old * (1 - lr * weight_decay) - lr * grad
```

### `adam`

Uses `torch.optim.Adam` directly. Do not pass or enable `decoupled_weight_decay=True`, even when the installed PyTorch version supports it. Weight decay is coupled L2 penalty.

### `adamw`

Uses `torch.optim.AdamW` directly. Weight decay is decoupled.

### `sgd_coupled_decoupled`

Accepts `total_weight_decay` and `coupled_ratio`:

```text
wd_coupled = coupled_ratio * total_weight_decay
wd_decoupled = (1 - coupled_ratio) * total_weight_decay
```

Factory endpoints:
- `coupled_ratio=0.0`: return custom `SGDW`.
- `coupled_ratio=1.0`: return `torch.optim.SGD`.
- Intermediate ratios: return custom `SGDCoupledDecoupled`.

For intermediate ratios, only `wd_coupled` enters gradient/momentum; only `wd_decoupled` is applied as parameter shrinkage.

### `adam_coupled_decoupled`

Accepts `total_weight_decay` and `coupled_ratio`:

```text
wd_coupled = coupled_ratio * total_weight_decay
wd_decoupled = (1 - coupled_ratio) * total_weight_decay
```

Factory endpoints:
- `coupled_ratio=0.0`: return `torch.optim.AdamW`.
- `coupled_ratio=1.0`: return `torch.optim.Adam`.
- Intermediate ratios: return custom `AdamCoupledDecoupled`.

For intermediate ratios, only `wd_coupled` enters gradients and Adam moments; only `wd_decoupled` is applied as parameter shrinkage.
