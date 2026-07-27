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
| `vit_small` | `src/oge/models/vit.py` | `ViTSmall` | Modern transformer research backbone | Planned | Dosovitskiy et al., An Image is Worth 16x16 Words; Lee, Lee, and Song, Vision Transformer for Small-Size Datasets |

## Research lineup decision (2026-07-23)

This section fixes which backbones and datasets form the optimizer-geometry
study lineup, at which protocol level, and in which execution order. Grid
values, role rules, and seed policy are normative in
[`07_optimizer_comparison_hpo_protocol.md`](07_optimizer_comparison_hpo_protocol.md)
protocol v1.2; literature anchors are in
[`10_optimizer_grid_literature_anchors.md`](10_optimizer_grid_literature_anchors.md).
This is a planning decision: rows marked "planned" have no implementation,
and no lineup run has been executed.

| Priority | Backbone | Dataset | Variation axis covered | Protocol level | Status |
| --- | --- | --- | --- | --- | --- |
| 1 (main) | `wrn28_10` | CIFAR-10 | main setting: BN + residual + wide CNN, OOD-literature standard | Full v1.2: grid 36 + `C1`-`C4` seeds 0-2 + both pair controls | implemented |
| 2 | `resnet18` (`cifar` variant) | CIFAR-10 | depth/width regime; direct OpenOOD v1.5 comparability | Reduced: grid 36 + `C1`/`C3` seeds 0-2 | implemented |
| 3a | `vgg16` (`cifar` variant, BatchNorm) | CIFAR-10 | no residual connections (plain CNN), Neural-Collapse-literature lineage | Reduced | planned |
| 3b | `resnet18` (`cifar` variant) | CIFAR-100 | class-count/dataset axis at fixed architecture | Reduced, CIFAR-100 validity floor `0.60` | implemented (dataset contract pending) |
| 4 | `vit_small` | CIFAR-10 | LayerNorm + attention + no convolutional prior; strongest documented optimizer sensitivity | Reduced; grid provisional pending a bounded pilot | planned |

Lineup rationale and constraints:

- WRN-28-10 stays the main setting: it is the CIFAR OOD-literature standard
  backbone, has the repository's only completed long-run baseline, and its
  640-dimensional penultimate feature is the geometry reference point.
- `resnet18` on CIFAR-10 is the cheapest replication and aligns with the
  OpenOOD v1.5 training recipe family, enabling sanity comparison against
  published benchmark numbers.
- `vgg16` isolates the residual-connection axis while keeping BatchNorm and
  the convolutional prior fixed; the lineup uses the BatchNorm variant so
  that normalization is not confounded with the residual axis.
- `vit_small` isolates normalization (LayerNorm), token mixing (attention),
  and recipe era simultaneously; it is deliberately last because ViT
  from-scratch training on CIFAR is the highest-risk recipe (see the pilot
  requirements below).
- Execution order is 1 -> 2 -> 3a/3b -> 4. Later rows must not block
  earlier rows; each variation runs under its own bounded execution Issue.

ViT pilot requirements (before any ViT grid is frozen):

- The current training schema hard-pins `multistep` scheduling with no
  warmup; from-scratch ViT training is expected to need warmup plus cosine
  decay and possibly gradient clipping. Any such change is a versioned
  training-schema extension for the ViT arm only, decided in the ViT
  implementation Issue — it must not silently alter the CNN arms.
- The pinned small-dataset ViT recipe (card 10) also uses label smoothing
  and heavy augmentation, both of which conflict with this project's fixed
  unsmoothed-cross-entropy and flip/crop-only contract and are known to
  change penultimate geometry. The pilot must decide, and record, which
  recipe elements are adopted, with the default being the project contract
  (accepting lower ViT accuracy) rather than the reference recipe.
- Fallbacks if the from-scratch pilot fails to reach the CIFAR-10 validity
  floor: (a) ViT/DeiT fine-tuned from a pretrained checkpoint, reported as
  a separately labeled regime because pretraining changes initial geometry;
  or (b) `convnext_tiny` adapted to CIFAR as the modern-architecture axis.

Deferred, unchanged by this decision: `convnext_tiny` from scratch on
ImageNet-200 remains the later modern-extension option from the planning
discussion; the held-out `Muon` optimizer arm remains out of scope. This
lineup differs from the 2026-07-23 offline planning note (kept outside
version control) in two recorded ways: `vgg16`/CIFAR-10 is added as the
residual-axis control, and ViT from scratch is preferred over ViT
fine-tuning for the modern axis — with the fine-tuning fallback retained —
because from-scratch training keeps optimizer identity as the only
representation-shaping training rule.

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
- **Reference source:** Zagoruyko & Komodakis, *Wide Residual Networks*, and the
  authors' official `szagoruyko/wide-residual-networks` code at commit
  `ae6d0d0561484172790c7a63c8ce6ade5a5a2914`. The projection-shortcut
  comparison uses `pytorch/resnet.py`; the dropout placement and option use
  `models/wide-resnet.lua`; the weight initialization uses `models/utils.lua`
  and `pytorch/utils.py`.
- **Initialization semantics (frozen 2026-07-27):** the model config field
  `init_policy` selects the policy and its only supported value is
  `msr_fan_in`. Under that policy:
  - every `Conv2d` weight is drawn from `normal(0, sqrt(2 / fan_in))` with
    `fan_in = kernel_height * kernel_width * in_channels`, implemented as
    `kaiming_normal_(mode="fan_in", nonlinearity="relu")` because the ReLU gain
    is `sqrt(2)`. This includes the 1x1 projection shortcuts;
  - every `BatchNorm2d` weight is `1` and bias is `0`;
  - the `classifier` bias is `0` and its weight keeps the framework default.

  This reproduces both pinned official implementations, which agree:
  `models/utils.lua` `MSRinit` computes `n = kW*kH*nInputPlane` and draws
  `normal(0, sqrt(2/n))`, and `pytorch/utils.py` calls `kaiming_normal_` with no
  mode argument, whose PyTorch default is `fan_in`. `FCinit` in
  `models/utils.lua` likewise zeroes only the fully connected bias.

  `fan_in` and `fan_out` coincide for every convolution inside a WRN group and
  differ only at the stem and the three channel-expanding blocks, where the
  standard deviations differ by up to a factor of `sqrt(10)`. BatchNorm makes
  the forward function invariant to convolution weight scale, so the visible
  effect is on effective learning rate and feature-norm growth rather than on
  accuracy. Because those are mechanism variables for this project,
  `init_policy` is materialized into the resolved config and therefore enters
  the canonical scientific config hash; two runs differing only in
  initialization can never share a config identity.

  History: runs before 2026-07-27, including the Issue #14 SGD seed-0 baseline,
  used `mode="fan_out"`, which matched neither official implementation. Those
  checkpoints and their validation report are retained unchanged as historical
  provenance and are excluded from protocol v1.2 results.
- **Block semantics:** use the pre-activation basic block with two 3x3
  convolutions and ReLU activations. Compute
  `preactivated = relu1(bn1(x))`. A same-shape identity shortcut carries the
  original `x`; a channel- or stride-changing 1x1 projection consumes
  `preactivated`. The residual branch is
  `conv1(preactivated) -> bn2 -> relu2 -> dropout -> conv2`.
- **Allowed variants:** exactly WRN depth/widen settings requested in config;
  the registry endpoint `wrn28_10` means depth 28 and widen factor 10.
  `dropout_rate` is a required explicit model-config value in
  `[0.0, 1.0)`; it does not select a different registry endpoint.
- **Dropout presets:** the main preset `configs/models/wrn28_10.yaml` uses
  `dropout_rate: 0.0`. The appendix ablation
  `configs/models/wrn28_10_dropout.yaml` uses `dropout_rate: 0.3`. Dropout
  is applied after the second BN/ReLU and before the second 3x3 convolution in
  every basic block; a zero rate is represented by an identity operation.
- **Penultimate feature definition:** global-average-pooled output after the final WRN block and final BN/ReLU, immediately before the classifier.
- **Classifier exposure rule:** final `Linear(feature_dim, num_classes)` must be `model.classifier`.
- **Feature_dim policy:** native `feature_dim = 64 * widen_factor`, so `wrn28_10` has `feature_dim = 640` unless a future reference-card update explicitly changes this.
- **Common pitfalls:** projecting the raw block input instead of the
  preactivated tensor; applying dropout outside the documented residual-branch
  position; omitting explicit `dropout_rate`; omitting the final BN/ReLU
  before pooling; returning flattened spatial maps; using a projection to
  match ResNet dimensions; accepting ambiguous WRN names without explicit
  depth and widen factor; changing `init_policy` without expecting a new
  scientific config identity.

#### Audited deviations from the official WRN implementation (2026-07-27)

Architecture-side audit of `src/oge/models/wide_resnet.py` against the pinned
paper and repository. Data, augmentation, and split deviations are out of scope
here and belong to [`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md)
and [`05_training_protocol.md`](05_training_protocol.md).

| Item | Verdict |
| --- | --- |
| Fixed 8x8 average pooling vs `AdaptiveAvgPool2d((1, 1))` | **Identical.** The CIFAR path reaches an 8x8 final feature map, so the two are the same operation. The adaptive form is the more general spelling. |
| `return_features=True` and `model.classifier` | **Pure API addition.** The computation graph is unchanged and `tests/test_model_api.py` asserts logits parity between the two call forms. |
| `dropout_rate` presets `0.0` and `0.3` | **Identical.** Both are conditions reported in the paper. |
| Projection shortcut consumes the pre-activated tensor | **Identical.** `pytorch/resnet.py` computes `F.conv2d(o1, ...)` on the pre-activated tensor. |
| Dropout after the second BN/ReLU, before the second 3x3 convolution | **Identical.** Matches `models/wide-resnet.lua`. |
| Convolution initialization | **Corrected 2026-07-27** from `fan_out` to `fan_in`. Both official implementations use fan_in; the previous value matched neither. See the initialization semantics above. |
| BatchNorm unit scale and zero shift; classifier bias zeroed, weight at framework default | **Identical** to `MSRinit` and `FCinit`. |
| Weight decay applied to Conv/Linear weights only, excluding bias and BatchNorm | **Deliberate deviation, and part of the experimental design.** The paper and OpenOOD decay all parameters. Decaying BatchNorm `gamma` suppresses channels, which would contaminate the coupled-versus-decoupled comparison this project runs. Fixed by `weights_only_no_bias_norm` in [`01_optimizers.md`](01_optimizers.md). Accuracy is therefore not expected to match published OpenOOD numbers exactly. |
| Checkpoint, resume, and provenance contracts | **Pure addition.** No effect on the training computation. |

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
- **Lineup note (2026-07-23):** the research lineup uses the CIFAR variant with BatchNorm (VGG-16-BN-style layout) so that the residual-connection axis is isolated while normalization stays fixed across the CNN arms. The implementation PR must document the classifier structure it adopts.

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

### `vit_small`

- **Intended file path:** `src/oge/models/vit.py`.
- **Intended class name:** `ViTSmall`.
- **Role:** modern transformer research backbone (LayerNorm + attention axis
  of the research lineup).
- **Implementation status:** planned; additionally gated on the ViT pilot
  requirements in the lineup section above.
- **Reference source:** Dosovitskiy et al., *An Image is Worth 16x16 Words*,
  for the architecture; Lee, Lee, and Song, *Vision Transformer for
  Small-Size Datasets*, and its pinned official repository (see
  `docs/sources.lock.yaml`) for the CIFAR from-scratch recipe anchor.
- **Allowed variants:** explicit config only. The registry endpoint must
  require explicit `patch_size`, `embed_dim`, `depth`, `num_heads`, and
  `mlp_ratio`; a CIFAR configuration is expected to use `patch_size=4` on
  32x32 inputs. Exact dimensions are frozen in the implementation Issue and
  recorded here when decided. Dataset names must not select dimensions
  implicitly.
- **Penultimate feature definition:** the token representation after the
  final transformer block and final LayerNorm that directly feeds
  `model.classifier`. Whether that vector is the class token or mean-pooled
  patch tokens must be chosen explicitly in the implementation Issue,
  recorded here, and exposed identically through `return_features=True`.
- **Classifier exposure rule:** the final `Linear(feature_dim, num_classes)`
  head must be `model.classifier`.
- **Feature_dim policy:** native `embed_dim` of the configured variant; no
  projection layers to match CNN widths.
- **Weight-decay grouping:** LayerNorm parameters and biases are already
  excluded by `weights_only_no_bias_norm`. Positional embeddings and the
  class token are not `Conv/Linear` weights, so the current conservative
  default assigns them no decay; the implementation Issue must verify and
  test this rather than assume it.
- **Common pitfalls:** taking features before the final LayerNorm; letting
  the class-token-versus-pooling choice differ between `model(x)` and
  `return_features=True`; applying weight decay to positional embeddings or
  the class token; silently importing the reference recipe's label
  smoothing or augmentation, which the project contract excludes; training
  from scratch without warmup and concluding architecture instability from
  what is a schedule artifact.

## Anti-footgun rules

- `make_model(config)` must not silently default to a toy model when `name` is missing. Missing or unknown names must fail loudly.
- Dataset names must not implicitly change architecture.
- CIFAR/ImageNet-style architectural variants must be selected explicitly through `variant`.
- Real research backbones must not add arbitrary projection layers just to force a common `feature_dim`.
- Any architecture-side change that changes penultimate geometry must be explicit in config and documentation. Weight initialization is such a change: it is carried by the `init_policy` model-config field, is materialized into the resolved config, and enters the canonical scientific config hash.
- Initialization must not be silently retuned to a framework or TorchVision default. `wrn28_10` follows the pinned official `fan_in` policy; `resnet18` keeps `fan_out`, which matches the TorchVision reference implementation rather than a pinned author repository, and that difference between the two backbones is intentional and recorded.
- ConvNeXt uses LayerNorm. The current parameter-group policy excludes LayerNorm from weight decay, and that behavior must be preserved unless a later optimizer reference card changes it.
- TorchVision `create_feature_extractor` may be used as a reference during implementation, but final project models must directly support `return_features=True`.
