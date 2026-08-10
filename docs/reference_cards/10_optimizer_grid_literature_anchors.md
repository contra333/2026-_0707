# Reference Card 10: Optimizer Grid and Architecture Lineup Literature Anchors

## Purpose, authority, and evidence boundary

This card maps every numeric choice in protocol v1.2
([`07_optimizer_comparison_hpo_protocol.md`](07_optimizer_comparison_hpo_protocol.md))
and the historical v1.2 planning lineup
([`02_architectures.md`](02_architectures.md)) to pinned external sources or
to an explicitly labeled project judgment. It also records how this project
sits relative to the closest prior work found in the 2026-07-23 literature
pass.

This card is evidence mapping, not protocol. Card 07 v1.2 is normative for
grid values, roles, and seeds; card 02 is normative for the lineup. If this
card and card 07 ever disagree on a number, card 07 wins and the discrepancy
must be fixed as a documentation bug.

Literature values inform the grids; they do not validate them. No source
below demonstrates that these grids contain any optimizer's competitive
region under this repository's exact recipe (200-epoch multistep, no warmup,
`weights_only_no_bias_norm`, unsmoothed cross entropy, flip/crop-only
augmentation). Transfer of published hyperparameters across schedule,
architecture, and epoch budget is an assumption, and card 07's boundary-hit
rule exists precisely because that assumption can fail.

All external claims cited here were checked at abstract/README/source-file
level on 2026-07-23 unless the entry says otherwise; page-level quotes for
the full texts remain to be captured when the paper draft is written. Source
keys refer to `docs/sources.lock.yaml`.

## Source keys used by this card

Existing keys: `wide_resnet_paper`, `wide_resnet_official_repository`,
`openood_v1_5_paper`, `openood_v1_5_repository`, `adam_paper`,
`decoupled_weight_decay_paper`, `neural_collapse_paper`,
`mahalanobis_ood_paper`, `knn_ood_paper`.

Keys added by this decision: `optimizer_comparison_protocol_paper` (Choi et
al.), `optimizer_neural_collapse_2026_paper` (Zhao et al.),
`optimization_ood_detection_sensitivity_paper` (Abdelzad et al.),
`optimizer_ood_generalization_paper` (Naganuma et al.),
`outlier_features_paper` (He et al.), `vit_paper` (Dosovitskiy et al.),
`vit_early_convolutions_paper` (Xiao et al.), `small_dataset_vit_paper`
(Lee, Lee, and Song), `small_dataset_vit_repository` (pinned official
implementation).

## Grid design principles

1. **Comparisons are a function of the tuning protocol.** Choi et al.
   (`optimizer_comparison_protocol_paper`) show that empirical optimizer
   rankings are highly sensitive to the hyperparameter search space and
   budget. v1.2 therefore gives every optimizer the same grid shape
   (`3 LR x 4 WD`), the same budget (12 seed-0 cells), and the same
   selection rules, while letting the numeric ranges be
   optimizer-appropriate.
2. **Log-scale placement around a literature center.** Each learning-rate
   triplet is a x3 log ladder centered on the canonical value for that
   optimizer family; each weight-decay set is a deliberately broad
   project-judgment bracket with at most four points, favoring interpretable
   round values over dense
   coverage. A 12-cell grid is a landscape instrument, not a fine
   optimizer: it must produce accuracy spread (for `C3`/`C4` matching and
   the descriptive scatter plots) as well as near-optimal cells.
3. **Historical-design continuity is provenance, not evidence.** Values
   retained from the earlier random-search design were originally planning
   anchors; the external 64-run outcomes are not used. Current numeric
   support comes from the pinned sources below or is labeled project
   judgment.
4. **Zero-decay endpoints are scientific, not tuning, cells.** The `SGD`
   and `Adam` zero columns exist because unchecked feature/parameter norm
   growth is a central mechanism variable for this project's geometry and
   raw-versus-normalized detector questions (`outlier_features_paper`,
   `optimizer_neural_collapse_2026_paper`).

## SGD grid anchors

Grid: `lr in {0.03, 0.1, 0.3}`, `wd in {0, 1e-4, 5e-4, 1e-3}`, momentum
`0.9`, nesterov, coupled L2.

| Value | Justification | Sources |
| --- | --- | --- |
| `lr=0.1` | Canonical WRN CIFAR recipe and OpenOOD v1.5 CIFAR recipe; identical to the repository's completed Issue #14 baseline | `wide_resnet_paper`, `openood_v1_5_paper` |
| `lr=0.03`, `lr=0.3` | Approximate x3 log steps around the center; `0.3` probes the high-LR regime and `0.03` moves toward the accuracy region where adaptive methods may overlap for C3 | project judgment |
| `wd=5e-4` | Canonical WRN and OpenOOD v1.5 CIFAR weight decay | `wide_resnet_paper`, `openood_v1_5_paper` |
| `wd=1e-4` | Common lighter CNN decay; interior log point between the zero endpoint and the canonical value | project judgment (community practice) |
| `wd=1e-3` | Strong-decay side of the canonical value | project judgment |
| `wd=0` | No-explicit-decay mechanism endpoint | `optimizer_neural_collapse_2026_paper` (weight decay's role in collapse), `outlier_features_paper` |

The repository's fixed schedule (200 epochs, `multistep [60, 120, 160]`,
`gamma=0.2`, batch 128) is itself the WRN paper training protocol, which is
why the SGD center cell doubles as a published-recipe reproduction.

## Adam grid anchors

Grid: `lr in {3e-4, 1e-3, 3e-3}`, `wd in {0, 1e-4, 5e-4, 1e-3}` (coupled L2
via `torch.optim.Adam`), `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.

| Value | Justification | Sources |
| --- | --- | --- |
| `lr=1e-3` | Adam's canonical default step size | `adam_paper` |
| `lr=3e-4`, `lr=3e-3` | Approximate x3 log ladder around the canonical default; `3e-3` deliberately probes a possible stability edge | project judgment |
| `wd=1e-4` | Scale-level reading of L2 values reported competitive for Adam-with-L2 in the decoupled-weight-decay paper's CIFAR study | `decoupled_weight_decay_paper`; project judgment |
| `wd=5e-4` | Canonical CNN L2 magnitude, shared numerically with the SGD column for cross-optimizer readability | project judgment (community practice) |
| `wd=1e-3` | Strong coupled-L2 endpoint | project judgment |
| `wd=0` | Pure-adaptive endpoint and primary norm-growth mechanism cell | `outlier_features_paper`; project judgment |

Honesty note: the decoupled-weight-decay paper reports its Adam/AdamW
CIFAR-10 hyperparameter studies as heatmap figures (26 2x64d ResNet, 100
epochs for the sensitivity figures), and the exact grid points were not
numerically extractable in this pass; only the normalized-weight-decay
statement below was extracted as a number. The Adam L2 column is therefore
anchored by a scale-level reading of that paper plus explicit project
judgment, not by an exact published grid.

## AdamW grid anchors

Grid: `lr in {3e-4, 1e-3, 3e-3}` (identical to Adam for shared cells and
comparability), `wd in {1e-4, 1e-3, 1e-2, 1e-1}` (decoupled, PyTorch
`AdamW`), `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.

| Value | Justification | Sources |
| --- | --- | --- |
| `lr` triplet | Same as Adam: shared-cell requirement plus matched step-size scale across the coupling pair | card 07 v1.2 pair-control design |
| `wd=1e-2` | Interior point below the derived competitive region | derivation below; project judgment |
| `wd=1e-1` | Upper bracket of the derived competitive region for this 200-epoch horizon | derivation below |
| `wd=1e-3`, `wd=1e-4` | Weak-decay tail; both double as Adam-shared pair-control cells | card 07 v1.2; project judgment |
| no `wd=0` cell | `AdamW(wd=0)` is algorithmically identical to `Adam(wd=0)`; the Adam zero column covers the endpoint for both | card 07 v1.2 |

### Normalized weight decay derivation (AdamW upper-region bracket)

The decoupled-weight-decay paper (`decoupled_weight_decay_paper`) defines
normalized weight decay via `lambda = lambda_norm * sqrt(b / (B * T))` with
batch size `b`, training-set size `B`, and epochs `T`, and reports
`lambda_norm` around `0.025`-`0.05` as the observed optimal scale for its
CIFAR-10 experiments.

For this project's fixed recipe (`b=128`, `B=45000` ID train, `T=200`):

```text
sqrt(128 / (45000 * 200)) ≈ 3.77e-3
lambda ≈ 0.025..0.05 * 3.77e-3 ≈ 9.4e-5 .. 1.9e-4
```

That `lambda` multiplies the schedule multiplier directly in the paper's
formulation, whereas PyTorch `AdamW` applies per-step decay
`lr_t * weight_decay`. Equating the initial-step decay gives the
PyTorch-unit conversion `weight_decay ≈ lambda / lr_0`:

```text
lr_0 = 1e-3  ->  weight_decay ≈ 0.094 .. 0.189
lr_0 = 3e-3  ->  weight_decay ≈ 0.031 .. 0.063
```

So the derived competitive region for this horizon is roughly
`weight_decay in [0.031, 0.189]` depending on learning rate, which the grid
brackets with `1e-2` and `1e-1`. Caveats, recorded deliberately: the paper's
experiments used a Shake-Shake-regularized ResNet with cosine/restart
schedules, not WRN-28-10 with multistep; the `lambda_norm` optimum is
figure-level, not a guarantee; and the conversion ignores schedule-shape
differences. This derivation justifies bracketing, not a point estimate,
and card 07 already anticipates a possible boundary hit at `1e-1`.

## Effective per-step decay consistency

For coupled SGD, per-step L2 shrinkage is approximately
`lr * wd * parameter` (momentum accumulation aside); for decoupled AdamW it
is exactly `lr_t * wd * parameter`. Cross-checking canonical recipes in
those units:

```text
SGD   (0.1,  5e-4)  ->  5e-5 per step   (WRN / OpenOOD canonical)
AdamW (1e-3, 5e-2)  ->  5e-5 per step   (typical modern AdamW recipe scale)
```

The two canonical recipes coincide in per-step decay, which is a useful
sanity signal that the SGD and AdamW grids are centered on comparable
regularization scales. Grid coverage in these units: SGD spans `0` to
`3e-4` per step (`0.3 x 1e-3`), AdamW spans `3e-8` to `3e-4` per step. As
card 07 states, equal per-step numeric decay never
implies equal effective regularization across update rules — momentum and
adaptive preconditioning interact with decay differently
(`decoupled_weight_decay_paper`, `optimizer_neural_collapse_2026_paper`).

## Shared-cell rationale (pair controls)

The Adam/AdamW pair cells `(lr=1e-3, wd=1e-4)` and `(lr=3e-3, wd=1e-3)`
were chosen because: both are cells of both v1.2 grids (no extra seed-0
budget); together they span a x3 learning-rate step and a x10 decay step, so
the coupling comparison is observed at two distinct regularization scales;
and `(1e-3, 1e-4)` is a weak coupled/decoupled comparison point. The
planning note's illustrative pair `(3e-3, 5e-4)` was not adopted because `5e-4` is
not an AdamW grid value; forcing it in would have broken the AdamW decade
ladder or cost a 13th cell.

## Historical v1.2 architecture-lineup anchors

This section preserves the 2026-07-23 planning rationale. It is not the active
minimum paper lineup. Research Contract v2 now fixes WRN-28-10/CIFAR-10 as main
and ResNet-18/CIFAR-100 as replication; see
[`12_fixed_readout_intervention_protocol_v2.md`](12_fixed_readout_intervention_protocol_v2.md).

| Lineup row | Anchor and comparability | Sources |
| --- | --- | --- |
| `wrn28_10` / CIFAR-10 (main) | Recipe is the WRN paper protocol itself; WRN-class backbones are standard in the CIFAR OOD-detection literature | `wide_resnet_paper`, `wide_resnet_official_repository`, `mahalanobis_ood_paper`, `knn_ood_paper` |
| `resnet18` / CIFAR-10, CIFAR-100 | OpenOOD v1.5 trains CIFAR ResNet-18 with SGD momentum `0.9`, `lr=0.1` (cosine), `wd=5e-4`, batch 128, 100 epochs; the SGD center cell matches those values, so published benchmark numbers remain a sanity reference. Comparability caveats: this project uses 200-epoch multistep, not 100-epoch cosine | `openood_v1_5_paper`, `openood_v1_5_repository` |
| `vgg16` / CIFAR-10 | Plain (non-residual) CNNs including VGG are part of the original Neural Collapse evidence base, so the residual-axis control stays comparable to that literature | `neural_collapse_paper` |
| `vit_small` / CIFAR-10 | ViTs are documented as "sensitive to the choice of optimizer (AdamW vs. SGD), optimizer hyperparameters, and training schedule length", making ViT the strongest-contrast lineup member for an optimizer-effect study; from-scratch CIFAR feasibility is anchored by the small-size-datasets ViT work | `vit_paper`, `vit_early_convolutions_paper`, `small_dataset_vit_paper`, `small_dataset_vit_repository` |

### Pinned ViT reference recipe and its conflicts with the project contract

Defaults of `main.py` in the pinned `small_dataset_vit_repository` (commit
`54bd796acd28fa11deac21a97b14b154379b34e7`): AdamW, `lr=1e-3`,
`weight_decay=5e-2`, 100 epochs, batch 128, 10 warmup epochs with cosine
decay, label smoothing `0.1`, and heavy augmentation (AutoAugment/RandAug
options, CutMix, MixUp, Random Erasing). Reported CIFAR accuracy for their
small-ViT variant (SL-ViT): `94.53%` CIFAR-10, `76.92%` CIFAR-100.

Conflicts with this project's fixed contract, to be resolved by the ViT
pilot (card 02): warmup and cosine scheduling are outside the current
training schema; label smoothing conflicts with unsmoothed cross entropy;
heavy augmentation conflicts with the flip/crop-only transform contract.
Default resolution is to keep the project contract and accept lower ViT
accuracy, because label smoothing and strong augmentation are themselves
geometry-shaping training rules that would confound the optimizer effect.

### Provisional ViT grid candidates (not authorized; pilot-gated)

- `AdamW`/`Adam`: same learning-rate triplet `{3e-4, 1e-3, 3e-3}` (contains
  the reference `1e-3`); AdamW decade ladder unchanged — `1e-2`/`1e-1`
  bracket the reference `5e-2`.
- `SGD`: candidate downward shift to `{0.01, 0.03, 0.1}` with warmup,
  reflecting documented SGD difficulty on ViTs
  (`vit_early_convolutions_paper`); to be frozen only after the pilot.

## Historical prior-work positioning (2026-07-23 pass; superseded for v2)

| Prior work | What it covers | What this project adds beyond it |
| --- | --- | --- |
| Zhao et al. 2026, `optimizer_neural_collapse_2026_paper` | Closest known prior work: optimizer choice and coupled-versus-decoupled weight decay govern Neural Collapse emergence (reports that NC cannot emerge under decoupled decay in adaptive optimizers, i.e., AdamW; momentum accelerates NC; introduces an NC0 diagnostic) | No OOD detection at all. This project connects the same training-rule axes to feature-space OOD detector reliability (raw versus L2-normalized readouts), under accuracy-matched comparisons, on an OpenOOD v1.5-aligned benchmark. The v1.2 AdamW decay column and both coupling pair controls directly probe the regime their result concerns; agreement or disagreement with their NC claim is reportable either way |
| Abdelzad, Czarnecki, and Salay 2020, `optimization_ood_detection_sensitivity_paper` | Early direct evidence that OOD-detection performance and even detector rankings (e.g., Mahalanobis versus MSP) are sensitive to the training optimizer | Predates the modern detector panel and OpenOOD evaluation standard; no geometry mechanism, no accuracy matching, no multi-seed matched design. Serves as motivation that the effect exists (abstract-level reading; full-text audit pending) |
| Naganuma et al. 2023, `optimizer_ood_generalization_paper` | Large-scale evidence that adaptive optimizers underperform non-adaptive ones on out-of-distribution *generalization* (accuracy under shift) | Different task: OOD generalization is not OOD *detection*. Cited to delimit the claim boundary, not as detection evidence |
| He et al. 2024, `outlier_features_paper` | Adam produces stronger outlier features (higher kurtosis) than SGD; normalization layers amplify outlier features | Mechanism-adjacent: supports treating feature-norm/outlier statistics as optimizer-sensitive, but targets quantization/training stability, not OOD detection or collapse geometry |

Historical positioning summary: the optimizer-to-geometry link and the
optimizer-to-OOD-sensitivity link each existed separately in this pass. The
former wording about an “L2-normalization repair” is no longer an active
novelty claim: Mahalanobis++ and later geometry-based Mahalanobis work occupy
that ground. Card 12 owns the current gap statement: shared-prefix
coupling-rule intervention, detector-formula-linked score-overlap diagnosis,
selective gap attenuation, and second-regime replication.

## Unverified assumptions and decision log

- Exact numeric grids behind the decoupled-weight-decay paper's CIFAR
  heatmaps were not extracted; the normalized-weight-decay scale statement
  is the only number imported, and its transfer to WRN-28-10 with multistep
  scheduling is an assumption.
- The claimed detector-ranking flip in Abdelzad et al. and the
  Adam-versus-SGD kurtosis claim in He et al. were read at abstract level
  on 2026-07-23; quote-level verification happens at paper-writing time.
- Expected accuracy overlap between low-decay/low-LR SGD cells and tuned
  adaptive cells (needed for `C3`) is a literature-informed expectation,
  not a guarantee; card 07 defines the unresolved-`C3` fallback.
- Community-practice justifications (`SGD wd=1e-4`, `Adam wd=5e-4`) are
  labeled project judgment; no single citation pins them.
- The Zhao et al. 2026 entry was found during the 2026-07-23 search pass;
  its experimental grids were not yet audited and its claims are cited from
  the abstract. It must be re-audited before the related-work section is
  written.
- Whether the ViT arm keeps the project contract (default) or imports
  recipe elements is pilot-gated and will be recorded in card 02 when
  decided.
