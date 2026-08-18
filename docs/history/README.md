# Historical archive index

This directory is read-only provenance, not current research authority. The
active experiment contract is
[`../reference_cards/13_active_paper_protocol.md`](../reference_cards/13_active_paper_protocol.md).
The complete pre-diet tree is recoverable from commit `8e59fb8` and annotated
tag `archive/pre-repo-diet-20260811`; Git history was not rewritten.

## Externalized Metric Contract analysis

The former generated directory `docs/analysis/metric_contract_v1_2_c1_c4/`
is preserved as this immutable ZIP:

```text
hf://buckets/contra333/ICLR_RUN/archives/repository_diet/metric_contract_v1_2_c1_c4/31c32bbf7fa9d09d21dd6663dbb651c35782366de3f020d16e4403acd8ac86f9/metric_contract_v1_2_definitions_and_c1_c4_results.zip
```

- ZIP bytes: `5,001,469`
- ZIP SHA-256: `31c32bbf7fa9d09d21dd6663dbb651c35782366de3f020d16e4403acd8ac86f9`
- contained `output_manifest.json` SHA-256:
  `e0509aeb4288915f0b5222d6c5ec9fcace03b71777060b091c5902b1adb721b1`
- archive validation: local `unzip -t`, remote size, and temporary-download
  SHA-256 all matched before the Git copy was removed

The package can be regenerated with `scripts/analyze_metric_contract_v1_2.py`,
but regenerated tables and figures belong in ignored `artifacts/` or an
external directory, not under `docs/`.

## Task F result-analysis bundle

The detailed post-result tables, figures, merged JSON, and two Evidence Packs
are preserved outside Git at:

```text
hf://buckets/contra333/ICLR_RUN/aggregate/task_f_result_analysis_20260818/ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed/
```

- merged JSON SHA-256:
  `ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed`
- bundle `SHA256SUMS` SHA-256:
  `e4c1bf9363321097bd3f3c3beae9c0183e33f8ddf37ac5873e17011bc07a3ad4`
- active interpretation:
  [`../paper/task_f_result_analysis.md`](../paper/task_f_result_analysis.md)

## Historical records

- [`protocols/`](protocols/): superseded experiment and implementation plans
- [`validation/`](validation/): completed or closed execution records
- [`validation/task_f_execution_chronology_through_20260818.md`](validation/task_f_execution_chronology_through_20260818.md):
  pre-consolidation Task F execution and PR chronology
- [`validation/intervention_supporting_theory_outline_pre_result_20260818.md`](validation/intervention_supporting_theory_outline_pre_result_20260818.md):
  prospective pre-result manuscript/theory outline, including closed gates
- historical radial Stage-2: closed `FAILED`; not an active launch dependency
- historical component attribution: `PASS` discovery evidence; not fresh
  decay-coupling confirmation

## Owner-supplied local files

Eight former root-level untracked files were moved without content changes to:

```text
/home/contra333/2026여름방학실험코드/2026-_0707_local_history/pre_repo_diet_8e59fb8/
```

| File | SHA-256 |
| --- | --- |
| `0806 추가실험 계획 및 논문뼈대.zip` | `1f331a2e91f1bd5732c50a47ab0beb0ed717305680128a47363c3267bdf72c7a` |
| `0806_논문뼈대 수정본.md` | `b2a86645b71d7c09b48db82c9d35a9d2bf81cde474ed3490d453ebf9dbe50d2c` |
| `0806_논문뼈대 수정본.md:Zone.Identifier` | `b952d24cf701bc4e7a7d21e1ae85c611824e80579a89836c0cb49b193446ccfe` |
| `0807_논문뼈대 v2.md` | `9b7d07a6d6912819a89604ad96bf7c80ecd4811415e45baeafaa8a6407c26588` |
| `7월 23일 문제 정의 및 논문 뼈대.md` | `bf9c5fcf008343a9d5ebdafd3cb505951917d115548db01d1dcd2c6bbb0b65ea` |
| `7월 30일 평가 metric 정의 조사.md` | `980da9d79494604bfd1178b6980998565db43fb48468bd387ace4379dff7795d` |
| `OOD_metric_audit_handoff.zip` | `cf43472c0c16419465d631cd1a0ae004a7d3eeb1401a09f086e33901d5820b5a` |
| `metric_contract_v1_2_definitions_and_c1_c4_results.zip` | `31c32bbf7fa9d09d21dd6663dbb651c35782366de3f020d16e4403acd8ac86f9` |
