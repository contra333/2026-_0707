# Project status

Last updated: 2026-08-19
Fast-kill analysis HEAD: `ecba28ef22fc4b8893119f5224880876ecbd76df`
Frozen inspection-pack generator HEAD: `d0245f2ad46d2a42c2086b760bb1b6012ce2f3ef`
Plan B freeze documentation commit:
`9c88a22357669b7210a53f78d7a43f36126fdcdc`
ResNet seed-first reader-pack source/merge commits:
`8973c74edde73375b2db02eb3858757e65a6b470` /
`7aaf6206b1bebe6139f3fcc7b9817c6ef0b92f78`

## Current phase

Task F의 50-run training, ID-only geometry/alignment, protected one-shot
evaluation, bounded score recovery, central aggregation이 완료됐다. 14-context
classifier-insensitive geometry fast kill gate도 완료됐고 판정은 **FAIL**이다.
14/14 context가 기술 검증을 통과했지만 `rho > 1`은 0/14였다. Candidate Main은
종료하고, 현재 **Raw-MD pair-instability Plan B**를 논문 본체로 유지한다.

Prospective ResNet-18/CIFAR-10 endpoint replication도 20/20 checkpoint의
ID-only fitting과 승인된 six-OOD one-shot 평가까지 완료됐다. 기술 판정은 `PASS`,
동결한 과학 판정은 **`PARTIAL`**이다. Raw-MD 방향, Churn, large-versus-small
context, Marginal localization, small-cell ID guardrail은 재현됐지만 large-context
Near에서 `abs(Delta RMD) < abs(Delta Raw MD)`가 실패했다. Card 13 Section 15.3에
따라 architecture-general claim은 열리지 않으며 현재 ICLR main-paper promotion은
중단한다. 새 LR/WD grid나 mechanism으로 구조하지 않는다.
후속 seed-first reader pack과 post-result ID-only endpoint geometry
diagnostic도 완료됐으며 기존 `PARTIAL` 판정은 변경되지 않았다.

Current source of interpretation:

- [`paper/task_f_result_analysis.md`](paper/task_f_result_analysis.md)
- [`reference_cards/13_active_paper_protocol.md`](reference_cards/13_active_paper_protocol.md)
- [`paper/intervention_supporting_theory_outline.md`](paper/intervention_supporting_theory_outline.md)

과거 preflight, execution, PR chronology는
[`history/validation/task_f_execution_chronology_through_20260818.md`](history/validation/task_f_execution_chronology_through_20260818.md)에
보존했다.

## Terminal identity

- Training execution SHA: `9eb3c1fa56d880ea5220badac7bc71ba75786d22`
- Fresh-ID/evaluation bridge SHA: `2a22a651001e6466d067493e0966656c79219081`
- Protected recovery SHA: `b4b19f915d6272af345fc0d2146967b73620b9c2`
- Protected terminal:
  `/mnt/drive/lab1/oge/artifacts/task_f_protected/06c61f6f/central_b4b19f9/PROTECTED_COMPLETE.json`
- Terminal file SHA-256:
  `f618ffedbe94af49d69f1456dd6a324086ee0297b91dfb5c3b3b0e9e37cb1521`
- Terminal status: `PASS`; 360 score contexts, 2,106 seed-pair records,
  594 paired aggregates.

### ResNet-18 replication terminal

- Evaluation source SHA: `9538b0d34acf153451183223a88b3f3d98d9d7d1`
- Numerical recovery SHA: `9dd22cc62b434e0253e4fd6966c8c685a6edef64`
- Central terminal identity:
  `f0492271d31d13a1d8b4774303ba22485c701844e46bf37a2947743b5343721f`
- Independent validation identity:
  `5b58c3fa85a3c62542f181139a8e3e777941ab00e70acec52a817adbb3622656`
- Technical status: `PASS`; 20 runs, 10 paired seed-contexts, six OOD datasets
- Scientific verdict: `PARTIAL`; only `near_rmd_gap_smaller` failed
- Verified bundle:
  `hf://buckets/contra333/ICLR_RUN/aggregate/resnet18_cifar10_replication_v3_evaluation_20260819/f0492271d31d13a1d8b4774303ba22485c701844e46bf37a2947743b5343721f/`
- Bundle `SHA256SUMS` SHA-256:
  `a67e9bad5324b06df43d62153fe73aa4fa5f70e6b0c39d78b7fe6c8be51992c0`

## Completed analysis

The read-only post-result extraction verified:

- 360 protected score contexts;
- 660 geometry fits;
- 657 alignment records;
- 50/50 telemetry files;
- sibling init/data-stream identity before every C-D, D-Z, C-Z comparison;
- four Adam endpoint cells, all six OOD datasets, Raw/L2 MD/RMD/Marginal;
- primary time/depth formation, flip-burden structure, fixed geometry panel;
- within-fixed-LR WD contrasts and `WD x coupling` difference-in-differences;
- primary Marginal spectral allocation gate.
- 14/14 endpoint classifier-insensitive fast-kill contexts; all identity,
  checksum, projection-reconstruction, and required precision checks passed.

Fast-kill cell medians were `0.00795` for the primary anchor, `0.00443` for
high-LR/high-WD, `0.01005` for the all-ID-guardrail-PASS low-LR/low-WD cell,
and `0.00473` for low-LR/high-WD. Every seed had `rho < 1`, so the proposed
classifier-insensitive carrier is not supported under the frozen statistic.

Compact tracked result:
[`results/task_f_result_analysis_v1.json`](../results/task_f_result_analysis_v1.json),
SHA-256 `bbdb3327a77797e4a26f1e4175201081d22190e2fe28b40b6e940958edfc5a33`.

Detailed external bundle:
`hf://buckets/contra333/ICLR_RUN/aggregate/task_f_result_analysis_20260818/ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed/`.
The merged JSON SHA-256 is
`ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed`;
the portable `SHA256SUMS` file SHA-256 is
`e4c1bf9363321097bd3f3c3beae9c0183e33f8ddf37ac5873e17011bc07a3ad4`.
The bundle contains the detailed CSV/figures and both Evidence Packs; raw
checkpoints and feature/score arrays remain in the recorded server roots.

Fast-kill bundle:
`hf://buckets/contra333/ICLR_RUN/aggregate/task_f_classifier_insensitive_kill_20260818/bcebc1a002555d14e526d5734de8c8b1b31dc7372c4aef7ce3b637411e3908e/`.
The merged file SHA-256 is
`90f4fc447a32b14748ef992878ca8ba1e87e8a148f6810677e4819d2d56a4b27`;
the bundle `SHA256SUMS` SHA-256 is
`d45d077673ffcded30144ee23e776f79ae8285f71739a7318c52cb0af7c05c11`.

Seed-first paper figure/table pack:
`hf://buckets/contra333/ICLR_RUN/aggregate/task_f_paper_pack_20260818/fa2b1535af55b74c64a873734afd11eca9e1ebf01ef50b99d934fac86da61a82/`.
The generator was merged in PR #128 at
`aefb7363dcd30a6c7637c5b545b50b69323cbfd4`; the pack is bound to analysis
commit `1a329474fbf4df00996f204a8e598cfb2c537d5a` and source result SHA-256
`ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed`.
It contains seed-first Near/Far tables and PDF/SVG/300-dpi PNG for Figures
1--4 plus appendix geometry and negative-gate panels.

Frozen result-inspection and paper-quality pack:
`hf://buckets/contra333/ICLR_RUN/aggregate/task_f_frozen_paper_pack_20260819/8e6b8ab9fe52cfd8f2f7fb33681de4e1f10c8d674fba3597dd6427f6e59759dd/`.
It contains 12 Figure/table rendering triplets (PDF/SVG/300-dpi PNG), 29 seed-first CSV
tables, a manifest, and portable `SHA256SUMS`. The manifest SHA-256 is the
hash-addressed suffix above and the generator is bound to clean commit
`d0245f2ad46d2a42c2086b760bb1b6012ce2f3ef`. Independent regeneration and
the HF round trip reproduced all 67 pack files byte-for-byte. The pack was built only
from the completed merged analysis and compact server manifest exports; it
loaded no checkpoint, protected example, feature array, or score array.
Figure 1의 reader-QA 수정은 gray seed 연결선, vertical interval, No-decay
reference를 제거하고 AdamW/Mixed/Adam을 circle/square/triangle로 중복 부호화했다.
Raw seed는 작고 투명하게, mean은 같은 shape의 outline marker로 표시한다. 별도
Figure 1 supplement는 네 LR×WD context에서 AdamW/Adam의 absolute Raw-MD
AUROC를 같은 seed-first 표현으로 보여 준다. Figure 2는 같은 네 context에서
AdamW Raw MD, Adam Raw MD, paired `Adam − AdamW`의 dataset별 seed mean을 세
heatmap으로 분리하고 두 Raw-MD panel은 공통 0--1 scale을 쓴다. 기존 수치 결과는
변경하지 않았다. Figure 2 supplement는 SGDW/SGDM의 여섯 dataset Raw MD 절대값과
18개 exact-pair effect를 분리하고 Near/Far 표에 No-decay와 Raw MD/RMD/L2-MD를
보존한다. Figure 3A는 네 LR×WD context의 RMD/L2-MD recovery matrix이며 Figure 3B는
primary context의 세 readout 절대 AUROC다. Cross-LR context는 pool하지 않고 SGDM
reversal은 optimizer-family boundary로만 해석한다.

ResNet-18 seed-first PARTIAL-boundary reader pack:
`hf://buckets/contra333/ICLR_RUN/aggregate/resnet18_paper_pack_20260819/d8320eb640caa8ea9d6504752ff3d18d54b7e845728cedcd329b4150b1b4b39d/`.
The manifest SHA-256 is the hash-addressed suffix. It contains 9 Figure
triplets (R1--R6, with R2A/R2B, and S1--S2), seed/summary CSVs, protocol and
numerical tables, a unified provenance manifest, and portable checksums.
Coverage is 20 runs, 10 C--D pairs, six datasets, 84 score arrays per pair,
and 20 host-local endpoint geometry records. The maximum Gain/Loss/Churn
identity residual is `1.11e-16`; duplicate builds and the HF round trip were
byte-identical. Geometry is explicitly post-result ID-only exploratory and
does not alter the frozen `PARTIAL` verdict.

No checkpoint was loaded. No training, protected inference, detector refit,
band ablation, clipping, or new detector was run.

## Current evidence

### Controlled primary result

At Adam `LR=1e-3, WD=1e-4`, coupled-minus-decoupled Raw MD:

| Region | AdamW | Adam | DeltaAUROC | PairOrderChurn |
| --- | ---: | ---: | ---: | ---: |
| Near | 0.5922 | 0.4178 | -0.1743 | 0.3486 |
| Far | 0.6709 | 0.3874 | -0.2835 | 0.3863 |

Primary ID guardrail is Accuracy `PASS`, NLL `FAILED`, ECE `PASS`; it is a
multidimensional ID/OOD Pareto result, not a same-ID result.

### Local context and readout boundaries

- Raw-MD C-D sign is negative in all four Adam cells, but magnitude is
  context-dependent.
- Cross-LR absolute differences are descriptive because high/low LR groups do
  not share the same stream.
- Same-LR cross-WD contrasts are controlled. Low-LR Raw-MD `WD x coupling` DiD
  is `-0.0912/-0.0797` Near/Far; high-LR is `+0.0004/+0.0618`.
- RMD attenuation repeats more consistently than L2 signed-gap attenuation.
- L2-MD substantially improves absolute AUROC, but does not reduce every local
  signed gap.
- Marginal is the main adverse score component across all four Adam cells.
  OOD-side dominance is strong in the primary/high-WD contexts but not universal.
- SGDM has the opposite C-D sign and remains an optimizer-family boundary, not
  a causal family main effect.

### Formation and geometry

- Primary effect is early detectable and amplified through epoch 120, then
  maintained.
- It is small at stage1/2 and large at stage3/penultimate.
- Large-effect contexts generally have larger norm, condition, concentration,
  and effective-rank differences than the small-effect cell.
- The pattern is concordant but not monotone; unique geometry mediation is not
  established.
- Primary raw/L2 component theorem applicability is 30/30 `NOT_APPLICABLE`.
- Decoupled Marginal spectral reconstruction passed five seeds; coupled raw
  spectral attribution is `NOT AVAILABLE` because the fit is inapplicable.
- RtMD Gate 3 remains `FAILED_INAPPLICABLE`; the slot is closed.

## Active paper framing

Working title:

> **Training-Rule-Induced Pair-Ranking Multiplicity in Mahalanobis OOD
> Detection**

The frozen paper question is:

> Holding architecture, data, objective, initialization, and minibatch order
> fixed, how much can changing only the decay-coupling rule reorganize the
> same ID--OOD pair ordering of a protocol-fixed, branch-refitted Mahalanobis
> detector; in which score component, epoch, and depth does that sensitivity
> appear?

The current supported claim is a controlled decay-coupling case study, not a
general optimizer-recipe map. Raw-MD pair-ranking non-invariance, Gain/Loss
cancellation, predominantly Marginal score localization, time/depth formation,
and local-context boundaries are the five body results. `Fixed detector
parameters`, `same ID performance`, `Marginal causal mediator`, and universal
coupling-harm language are prohibited.

## Active workstreams

1. **WRN paper figure/table packs -- COMPLETE.** PR #128's original pack and
   the 2026-08-19 frozen inspection pack are both remotely verified. The new
   pack adds the frozen six-dataset alpha small multiples, four-context
   heatmap, absolute Raw/RMD/L2 recovery, Gain/Loss/Churn, fixed geometry,
   time/depth concordance, FPR95, and top-10 trace-share supplement. All
   macros are formed within seed before Near/Far aggregation; the inferential
   unit remains the training seed.
2. **ResNet-18/CIFAR-10 replication -- ENDPOINT COMPLETE; `PARTIAL`.** The
   strict deterministic v3 training completed 20/20 runs at
   `e2f6845e88b22bc0783c5fda58186f9930083ef7`. After the owner's separate
   endpoint authorization, PR #137's evaluator loaded all 20 epoch-200
   checkpoints, exported ID-train/validation features, fitted branch-specific
   Raw MD/RMD/Marginal/L2 readouts from ID-train only, computed ID-test
   Accuracy/NLL/ECE, and then performed the frozen six-OOD one-shot evaluation.
   All large features remain host-local.

   Technical coverage passed 20 runs, 10 C--D seed pairs, two contexts, and all
   six OOD datasets. Both contexts passed all three ID guardrails. Raw-MD C--D
   was negative in Near/Far for both contexts; large-context Churn was
   `0.3244/0.3617`, the small gaps were smaller, and Marginal accounting had
   the adverse sign and exceeded 50% of the total. The one frozen failure was
   large-context Near RMD attenuation: `abs(Delta RMD)=0.1426` versus
   `abs(Delta Raw MD)=0.0356`. The resulting gate is `PARTIAL`.

   A false numerical stop in the old score verifier was corrected in PR #139
   by separating score-scale and AUROC tolerances. Recovery reused only stored
   score arrays, verified their SHA-256 identities, and performed no second
   inference or refit. Independent reconstruction reproduced the terminal
   verdict. The canonical terminal, validation identity, and shared bundle are
   recorded above and in Card 13 Section 15.7.

   The post-result reader pack is also complete. PR #143 added a fail-closed
   CPU analysis CLI, recomputed AUROC/FPR95/Gain/Loss/Churn and exact score
   accounting from stored score arrays, and computed endpoint geometry from
   existing host-local `id_train` features. It produced no new endpoint and
   did not change the verdict. ResNet epoch/depth formation remains
   `NOT_AVAILABLE_BY_DESIGN`.

The WRN broad LR/WD grid, CIFAR-100, additional optimizers, and new mechanisms
are not parallel rescue workstreams. After ResNet `FULL`, the default is to
stop training and write; after `PARTIAL` or `FAIL`, do not open a rescue grid.

## Not active

- no further protected evaluation or checkpoint inference beyond the completed
  authorized ResNet endpoint;
- no reopened RtMD or replacement detector;
- no `S_perp` causal attribution;
- no causal LR conclusion from current Task F;
- no claim that stronger WD monotonically amplifies coupling damage;
- no architecture-general conclusion after the replication gate returned
  `PARTIAL`;
- no classifier-insensitive-geometry main claim; the Section 14 gate failed;
- no reverse affine rescue, small-singular subspace search, normalization rescue,
  projected OOD score, or trajectory expansion for this failed gate.
- no ResNet rescue LR/WD grid, CIFAR-100 extension, replacement mechanism, or
  architecture-general claim after the frozen `PARTIAL` verdict.
