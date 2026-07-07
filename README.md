# Optimizer Geometry Experiments

Initial PyTorch codebase for ICLR 2027 optimizer-geometry experiments.

This first step intentionally includes only:
- optimizer semantics and factory code
- a reference card for optimizer behavior
- optimizer and parameter-group unit tests

It intentionally does **not** include model architectures, datasets, training loops, CIFAR training, or GPU training.

## Test

```bash
pytest tests/test_optimizers.py tests/test_param_groups.py -q
```
