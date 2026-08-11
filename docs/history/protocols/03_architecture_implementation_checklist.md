# Reference Card 03: Architecture Implementation Checklist

> **Status: completed historical task checklist.** The first architecture implementation has been merged. Keep this file as provenance for that task and as a record of its validation contract. Do not use its one-time allowed scope as the specification for a new task. New bounded work must be defined in a GitHub Issue following `docs/WORKFLOW.md`.

This checklist is for the Codex CLI implementation PR that will run on a torch/GPU-capable server. It is not permission to add training infrastructure.

## Allowed scope for the first real implementation PR

The first real architecture implementation PR may implement only:

- `toy_cifar_cnn`
- `resnet18`
- `wrn28_10`

It may leave these as documented planned backbones:

- `vgg16`
- `convnext_tiny`

## Files to create or update

Expected implementation files:

- `src/oge/models/toy_cnn.py`
- `src/oge/models/resnet.py`
- `src/oge/models/wide_resnet.py`
- `src/oge/models/factory.py`
- `src/oge/models/__init__.py`
- `tests/test_model_api.py`

Do not create dataset loaders, training loops, checkpointing code, OOD detectors, GPU scripts, feature dumps, or large files.

## Factory names

`make_model(config)` should accept these names once implemented:

- `toy_cifar_cnn`
- `resnet18`
- `wrn28_10`
- `vgg16` only after implementation
- `convnext_tiny` only after implementation

`make_model(config)` must fail loudly when `name` is missing or unknown. It must not default to a toy model.

## Forward API requirements

Every implemented model must satisfy:

```python
logits = model(x)
logits, features = model(x, return_features=True)
```

Required behavior:

- `logits.shape == [B, num_classes]`.
- `features.shape == [B, feature_dim]`.
- `features` is the penultimate representation immediately before the final classifier.
- `return_features=True` does not change logits.
- The final classifier is exposed as `model.classifier`.
- Research backbones use native feature dimensions and must not add arbitrary projection layers to force a common `feature_dim`.
- CIFAR/ImageNet-style changes are selected by explicit `variant`, not by dataset name.

## Tests that must pass

The implementation PR must update or add endpoint tests covering:

- factory accepts each implemented name;
- missing `name` fails loudly;
- unknown `name` fails loudly;
- logits shape for every implemented model;
- feature shape for every implemented model;
- `model(x)` logits match `model(x, return_features=True)[0]`;
- `model.classifier` has the expected input and output dimensions;
- parameter-group behavior still excludes bias, BatchNorm, and LayerNorm from weight decay.

## Forbidden additions

Do not add:

- dataset loaders;
- training loops;
- checkpointing;
- OOD detectors;
- GPU training scripts;
- CIFAR training scripts;
- feature dumps;
- pretrained weight downloads;
- arbitrary projection heads for research backbones;
- implicit dataset-to-architecture selection.

## Server validation command

Run this exact command on the implementation server:

```bash
pytest tests/test_model_api.py tests/test_optimizers.py tests/test_param_groups.py -q
```

If torch or another dependency is unavailable, do not claim pytest success. Report the environment limitation and any static checks that were actually run.
