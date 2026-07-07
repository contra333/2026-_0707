# Optimizer Geometry Experiments

Initial PyTorch codebase for ICLR 2027 optimizer-geometry experiments.

The repository currently contains:

- optimizer semantics and factory code;
- optimizer and parameter-group unit tests;
- a model API contract for logits and penultimate features;
- a temporary toy/API model fixture;
- an architecture implementation plan for future research backbones.

The current model fixture is only for API smoke testing. VGG-16, WideResNet-28-10, ResNet18, and ConvNeXt-Tiny are documented as planned research backbones and should not be treated as implemented unless the code and tests actually add them.

This repository intentionally does **not** yet include dataset loaders, training loops, checkpointing, OOD detectors, CIFAR training scripts, or GPU training scripts.

## References

- Optimizer semantics: `docs/reference_cards/01_optimizers.md`
- Architecture API and planning: `docs/reference_cards/02_architectures.md`
- Architecture implementation checklist: `docs/reference_cards/03_architecture_implementation_checklist.md`

## Test

```bash
pytest tests/test_model_api.py tests/test_optimizers.py tests/test_param_groups.py -q
```
