# Reference Card 02: Architectures

This card is the implementation reference for model architecture semantics in this repository. It is intentionally a planning/specification document: it does **not** mean that all listed research backbones are already implemented.

The research path is:

```text
optimizer / training rule
→ penultimate representation geometry
→ detector score behavior
→ reliability under ID ambiguity and OOD shift
```

Because the experiment measures geometry, architecture-side choices that alter the penultimate representation are experimental variables. They must be explicit, documented, and selected through config.

## Global model API

Every model factory endpoint must support the same forward API:

```python
logits = model(x)
logits, features = model(x, return_features=True)
```

For batch size `B`, configured class count `num_classes`, and the model's exposed `feature_dim`:

- `logits.shape == [B, num_classes]`
- `features.shape == [B, feature_dim]`
- `features` is the penultimate representation immediately before the final classifier.
- `return_features=True` must not change logits. `model(x)` and `model(x, return_features=True)[0]` must be numerically identical for the same input and model state.
- The final classifier must be exposed as `model.classifier`.

## Registry and implementation status

| Registry name | Intended file path | Intended class name | Role | Status | Reference source |
| --- | --- | --- | --- | --- | --- |
| `toy_cifar_cnn` | `src/oge/models/toy_cnn.py` | `ToyCifarCNN` | Toy/API smoke-test fixture only | Implemented | Local API contract |
| `resnet18` | `src/oge/models/resnet.py` | `ResNet18` | Research backbone | Implemented | He et al., Deep Residual Learning for Image Recognition |
| `wrn28_10` | `src/oge/models/wide_resnet.py` | `WideResNet` | Primary CIFAR research backbone | Implemented | Zagoruyko & Komodakis, Wide Residual Networks |
| `vgg16` | `src/oge/models/vgg.py` | `VGG16` | Research backbone | Planned | Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition |
| `convnext_tiny` | `src/oge/models/convnext.py` | `ConvNeXtTiny` | Modern ConvNet research backbone | Planned | Liu et al., A ConvNet for the 2020s |

## Current toy fixture status

The repository contains `src/oge/models/toy_cnn.py` with class `ToyCifarCNN` and factory name `toy_cifar_cnn`. Treat it only as a toy/API fixture for testing the model API. It is **not** a research backbone and must not be used as evidence for optimizer-geometry conclusions.

## Per-architecture specifications

### `toy_cifar_cnn`

- **Intended file path:** `src/oge/models/toy_cnn.py`.
- **Intended class name:** `ToyCifarCNN`.
- **Role:** toy fixture for API smoke tests only.
- **Implementation status:** implemented toy/API fixture.
- **Reference source:** local model API tests and this reference card.
- **Allowed variants:** none unless explicitly documented later.
- **Input assumption:** CIFAR-like `[B, 3, 32, 32]` inputs for smoke testing.
- **Penultimate feature definition:** activation immediately before `model.classifier`.
- **Classifier exposure rule:** final linear classifier must be `model.classifier`.
- **Feature_dim policy:** configurable feature dimension is acceptable for the toy fixture because it is not a research backbone.
- **Common pitfalls:** do not describe it as a research model; do not use it as the default when a config omits `model.name`; do not let dataset names silently select it.

### `resnet18`

- **Intended file path:** `src/oge/models/resnet.py`.
- **Intended class name:** `ResNet18`.
- **Role:** research backbone.
- **Implementation status:** implemented.
- **Reference source:** He et al., Deep Residual Learning for Image Recognition.
- **Allowed variants:** explicit `variant` values only: `cifar` or `imagenet`. Dataset name must not choose the variant implicitly. The `cifar` variant uses a 3x3 stride-1 stem with no max-pool. The `imagenet` variant uses a 7x7 stride-2 stem followed by 3x3 stride-2 max-pool. Both variants use adaptive global average pooling before `model.classifier`.
- **Penultimate feature definition:** global-average-pooled output after the final residual stage and before the classifier.
- **Classifier exposure rule:** final `Linear(feature_dim, num_classes)` must be `model.classifier`.
- **Feature_dim policy:** use the native feature width implied by the selected explicit variant. Do not add projection layers solely to force a shared feature dimension.
- **Common pitfalls:** silently switching to a CIFAR stem when `dataset=cifar10`; returning pre-pool feature maps as `features`; hiding the final linear layer under a nonstandard attribute; adding an undocumented projection head.

### `wrn28_10`

- **Intended file path:** `src/oge/models/wide_resnet.py`.
- **Intended class name:** `WideResNet`.
- **Role:** primary CIFAR research backbone.
- **Implementation status:** implemented.
- **Reference source:** Zagoruyko & Komodakis, Wide Residual Networks.
- **Allowed variants:** exactly WRN depth/widen settings requested in config; the registry endpoint `wrn28_10` means depth 28 and widen factor 10.
- **Penultimate feature definition:** global-average-pooled output after the final WRN block and final BN/ReLU, immediately before the classifier.
- **Classifier exposure rule:** final `Linear(feature_dim, num_classes)` must be `model.classifier`.
- **Feature_dim policy:** native `feature_dim = 64 * widen_factor`, so `wrn28_10` has `feature_dim = 640` unless a future reference-card update explicitly changes this.
- **Common pitfalls:** omitting the final BN/ReLU before pooling; returning flattened spatial maps; using a projection to match ResNet dimensions; accepting ambiguous WRN names without explicit depth and widen factor.

### `vgg16`

- **Intended file path:** `src/oge/models/vgg.py`.
- **Intended class name:** `VGG16`.
- **Role:** research backbone.
- **Implementation status:** planned.
- **Reference source:** Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition.
- **Allowed variants:** explicit `variant` values only, such as `cifar` or `imagenet`, after the implementation PR documents classifier structure and pooling behavior.
- **Penultimate feature definition:** activation immediately before the final classifier linear layer. A CIFAR variant should expose a single final linear classifier over pooled convolutional features unless a multi-layer classifier is explicitly documented.
- **Classifier exposure rule:** the final linear layer that maps penultimate features to logits must be `model.classifier`.
- **Feature_dim policy:** use the native explicit-variant feature width. Do not insert arbitrary projection layers to force a common width.
- **Common pitfalls:** using the original ImageNet multi-layer classifier without documenting which activation is penultimate; making `features` refer to the convolutional module instead of the penultimate vector; implicit dataset-based variant selection.

### `convnext_tiny`

- **Intended file path:** `src/oge/models/convnext.py`.
- **Intended class name:** `ConvNeXtTiny`.
- **Role:** modern ConvNet research backbone.
- **Implementation status:** planned.
- **Reference source:** Liu et al., A ConvNet for the 2020s.
- **Allowed variants:** explicit `variant` values only; `convnext_tiny` must document any CIFAR/ImageNet stem or resolution adaptation before implementation.
- **Penultimate feature definition:** pooled normalized representation immediately before the classifier head.
- **Classifier exposure rule:** final classifier head must be exposed as `model.classifier`; if a reference implementation uses `head`, wrap or alias the final classifier without changing logits.
- **Feature_dim policy:** use the native ConvNeXt-Tiny feature width for the selected explicit variant. Do not add arbitrary projections to match other backbones.
- **Common pitfalls:** losing the LayerNorm before the head; applying weight decay to LayerNorm parameters; relying on a TorchVision feature extractor instead of supporting `return_features=True` directly; making resolution or stem changes implicit.

## Anti-footgun rules

- `make_model(config)` must not silently default to a toy model when `name` is missing. Missing or unknown names must fail loudly.
- Dataset names must not implicitly change architecture.
- CIFAR/ImageNet-style architectural variants must be selected explicitly through `variant`.
- Real research backbones must not add arbitrary projection layers just to force a common `feature_dim`.
- Any architecture-side change that changes penultimate geometry must be explicit in config and documentation.
- ConvNeXt uses LayerNorm. The current parameter-group policy excludes LayerNorm from weight decay, and that behavior must be preserved unless a later optimizer reference card changes it.
- TorchVision `create_feature_extractor` may be used as a reference during implementation, but final project models must directly support `return_features=True`.
