# Optimizer Geometry Experiments

Initial PyTorch codebase for ICLR 2027 optimizer-geometry experiments.

The repository currently contains:

- optimizer semantics and factory code;
- optimizer and parameter-group unit tests;
- a model API contract for logits and penultimate features;
- implemented `toy_cifar_cnn`, `resnet18`, and `wrn28_10` model endpoints;
- an architecture implementation plan for future research backbones.

The `toy_cifar_cnn` model is only for API smoke testing. `resnet18` and `wrn28_10` are implemented research backbones. VGG-16 and ConvNeXt-Tiny remain documented as planned backbones and should not be treated as implemented until code and tests add them.

This repository intentionally does **not** yet include dataset loaders, training loops, checkpointing, OOD detectors, CIFAR training scripts, or GPU training scripts.

## References

- Optimizer semantics: `docs/reference_cards/01_optimizers.md`
- Architecture API and planning: `docs/reference_cards/02_architectures.md`
- Architecture implementation checklist: `docs/reference_cards/03_architecture_implementation_checklist.md`

## Test

```bash
pytest tests/test_model_api.py tests/test_optimizers.py tests/test_param_groups.py -q
```
