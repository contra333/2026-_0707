# Project status

Last updated: 2026-08-18
Repository HEAD used for analysis: `e0f35285f0edbc5f88077cc2e3a7f136e42554d7`

## Current phase

Task F의 50-run training, ID-only geometry/alignment, protected one-shot
evaluation, bounded score recovery, central aggregation이 완료됐다. 현재 단계는 새
실험 실행이 아니라 **완료된 artifact의 post-result analysis와 paper claim gate**다.

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

The research-program question is:

> Which optimizer-side training choices create representations that are
> sensitive or incompatible with a fixed Raw Mahalanobis readout, and how does
> that geometry form during training?

Task F currently supports a narrower claim:

> A controlled decay-coupling/WD intervention can change representation
> geometry and exact Raw-MD pair ordering, with the adverse movement localized
> predominantly to the Marginal score channel and amplified over time/depth.

Do not claim that Task F has already mapped optimizer recipes generally or
identified a unique spectrum or `S_perp` mediator.

## Next decisions

1. Finalize the main/supplement figure allocation and the geometry
   `concordant/mixed/discordant` adjudication.
2. Decide whether the current paper remains the narrow controlled case study or
   adopts the broader optimizer-origin RQ.
3. For the narrow paper, run ResNet-18/CIFAR-10 focal replication next.
4. For the broad RQ, first preregister a common-init/common-stream paired LR
   bridge/factorial; current cross-LR data cannot supply that causal contrast.
5. Keep additional adaptive optimizers, ConvNeXt/ViT, and a broad phase map
   after those gates.

## Not active

- no new protected evaluation or checkpoint inference;
- no reopened RtMD or replacement detector;
- no `S_perp` causal attribution;
- no causal LR conclusion from current Task F;
- no claim that stronger WD monotonically amplifies coupling damage;
- no architecture-general conclusion before replication.
