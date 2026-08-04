# Issue 49 Seed 1/2 Role and Pair Follow-up Execution

## Scope and final verdict

Issue #49 executed only the frozen 30-row follow-up plan: seed 1/2 role
replication, the missing Adam pair rows, and the six SGDW pair rows. It did
not extend the grid or access ID-test, OOD, geometry/Neural Collapse, feature,
or detector evidence.

Final verdict on 2026-08-04: **PASS**. All 30 production rows completed epoch
200, passed the independent artifact gate, and were uploaded with zero
deletes. The three live remote listings retain their final
`REMOTE_COMPLETE.json` markers. The 12 reused seed-0 records and 30 new
records form 42 unique `(config_hash, seed)` identities across 14
configurations. Per-seed values, mean/sample-SD summaries, and paired-control
differences are committed and checksum-sealed.

## Frozen identities

| Item | Value |
| --- | --- |
| Production Git SHA | `3556841340e6f6b92782af045ed4a468e6e271bd` |
| Role-freeze hash | `fdf67c1184abc489542ca64cad2410ff38aa816acb1e9e5289d60461600373fa` |
| Follow-up-plan hash | `3a3b00dbcf0ee3dc20c0959013665bf4243a2644e21f4552a831e0fdaed69264` |
| Execution ID | `role_pair_followup_20260730` |
| Execution plan | `configs/studies/wrn28_10_optimizer_hpo_v1_2/followup_execution.yaml` |
| Aggregate | `results/wrn28_10_optimizer_hpo_v1_2/followup_20260730/aggregate.json` |
| Aggregate hash | `a403d23b747ba6b927211dd3a9acc7526fabcbba1daf9b56ad735219191fc856` |
| Pull Request | #50 |

The execution-plan validator confirms all 30 scheduled trial IDs exactly
once, host counts 13/10/7, every optimizer family on every host, all 14
configurations rotating across the three hosts over seeds 0/1/2, and every
declared matched-pair co-location.

## Host completion and upload evidence

The host state, artifact validation, and Hugging Face marker were re-read on
2026-08-04. `file_count` below is the verified tree before the final marker;
the remote listing has one additional `REMOTE_COMPLETE.json`.

| Host | Rows | Terminal gate | Dry-run uploads/deletes | Verified files + marker | Bytes before marker | Marker Xet hash |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `curie` | 13/13 | `REMOTE_VERIFIED` | 283 / 0 | 283 + 1 | 24,826,205,304 | `3bfb3a0651266a21b1d516960c6da44d33888613c5ef3c611954bbc1ebdbcc1c` |
| `lise` | 10/10 | `REMOTE_VERIFIED` | 220 / 0 | 220 + 1 | 18,984,786,670 | `ae8ed3bb8b925b970287f70785a293212161e2f5807b57b80626b1c0b96357e0` |
| `precision_medicine` | 7/7 | `REMOTE_VERIFIED` | 157 / 0 | 157 + 1 | 13,435,325,201 | `21a9c93754a54c8d91dda5e59f93048385ed61f64155888b9837bf6b275f60dd` |

The verification records require epoch 200, completed trial status, matching
config/seed/Git identities, valid `last.pt` and `best_val.pt` checksums,
deferred ID-test state, and absence of protected evaluation fields. The final
aggregate builder additionally re-hashes every downloaded `trial.json`
against each host's `artifact_validation.json`, requires that validation to
equal the supervisor state, and rejects missing or duplicate identities.

The two small aggregate files were uploaded without delete to
`hf://buckets/contra333/ICLR_RUN/aggregate/role_pair_followup_20260730/`.
The saved plan contained two uploads, zero downloads, zero deletes, and
112,522 bytes. A fresh download passed the committed SHA-256 sidecar and was
byte-identical to the Git worktree artifact.

## Aggregate policy and counts

- Primary metric source: epoch-200 `last.pt` `final_validation`.
- Seeds: 0, 1, and 2 retained individually.
- Dispersion: sample standard deviation (`ddof=1`).
- Pair difference: decoupled minus coupled (`AdamW - Adam`, `SGDW - SGD`).
- Counts: 12 reused seed-0 records + 30 new records = 42 unique identities.
- Configuration count: 14, each containing exactly seeds 0/1/2.
- Protected evaluation: deferred throughout Issue #49.

## Per-configuration validation summaries

| Labels | Optimizer/config | Seed 0/1/2 accuracy | Accuracy mean +/- sample SD | NLL mean +/- sample SD |
| --- | --- | --- | ---: | ---: |
| `pair:adam_adamw` | `adam/0ac334244797` | 0.8676/0.8632/0.8766 | 0.869133 +/- 0.006830 | 0.430632 +/- 0.008838 |
| `role:adam:C4` | `adam/3e2bac6d488d` | 0.9392/0.9402/0.9400 | 0.939800 +/- 0.000529 | 0.339832 +/- 0.008812 |
| `pair:adam_adamw` | `adam/a8d0e1a8e585` | 0.9384/0.9324/0.9312 | 0.934000 +/- 0.003857 | 0.354341 +/- 0.015550 |
| `role:adam:C1`, `role:adam:C3` | `adam/b701f780ad99` | 0.9450/0.9430/0.9412 | 0.943067 +/- 0.001901 | 0.392714 +/- 0.014233 |
| `role:adam:C2` | `adam/db137e7d3c71` | 0.9410/0.9344/0.9374 | 0.937600 +/- 0.003305 | 0.281451 +/- 0.018139 |
| `pair:adam_adamw`, `role:adamw:C4` | `adamw/4039a31891b5` | 0.9378/0.9444/0.9442 | 0.942133 +/- 0.003754 | 0.446793 +/- 0.040096 |
| `pair:adam_adamw`, `role:adamw:C3` | `adamw/5742764ef3bd` | 0.9466/0.9410/0.9404 | 0.942667 +/- 0.003420 | 0.439322 +/- 0.037612 |
| `role:adamw:C1` | `adamw/61d7d1f20bc5` | 0.9564/0.9550/0.9568 | 0.956067 +/- 0.000945 | 0.283629 +/- 0.012360 |
| `pair:sgd_sgdw:A`, `role:sgd:C1` | `sgd/188f2e248d51` | 0.9582/0.9568/0.9572 | 0.957400 +/- 0.000721 | 0.192548 +/- 0.005658 |
| `role:sgd:C4` | `sgd/7aa36ee28bf5` | 0.9384/0.9388/0.9376 | 0.938267 +/- 0.000611 | 0.379650 +/- 0.008756 |
| `pair:sgd_sgdw:B`, `role:sgd:C2` | `sgd/86af0abedd46` | 0.9582/0.9574/0.9520 | 0.955867 +/- 0.003372 | 0.209093 +/- 0.016364 |
| `role:sgd:C3` | `sgd/f315d560c588` | 0.9458/0.9488/0.9478 | 0.947467 +/- 0.001528 | 0.302660 +/- 0.013788 |
| `pair:sgd_sgdw:B` | `sgdw/2f7b24bb35f8` | 0.9430/0.9414/0.9388 | 0.941067 +/- 0.002120 | 0.324190 +/- 0.027587 |
| `pair:sgd_sgdw:A` | `sgdw/99f6ea026047` | 0.9496/0.9514/0.9516 | 0.950867 +/- 0.001102 | 0.239820 +/- 0.006792 |

## Paired-control differences

| Pair/cell | Direction | Accuracy difference seed 0/1/2 | Mean +/- sample SD | NLL difference mean +/- sample SD |
| --- | --- | --- | ---: | ---: |
| Adam/AdamW, `lr=0.001`, `wd=0.0001` | `AdamW - Adam` | +0.0082/+0.0086/+0.0092 | +0.008667 +/- 0.000503 | +0.084980 +/- 0.036217 |
| Adam/AdamW, `lr=0.003`, `wd=0.001` | `AdamW - Adam` | +0.0702/+0.0812/+0.0676 | +0.073000 +/- 0.007219 | +0.016160 +/- 0.035172 |
| SGD/SGDW, cell A | `SGDW - SGD` | -0.0086/-0.0054/-0.0056 | -0.006533 +/- 0.001793 | +0.047272 +/- 0.012422 |
| SGD/SGDW, cell B | `SGDW - SGD` | -0.0152/-0.0160/-0.0132 | -0.014800 +/- 0.001442 | +0.115097 +/- 0.021523 |

These are descriptive pair controls. Equal numeric learning rate and weight
decay do not imply equal effective regularization across coupled and
decoupled optimizers. Pair-only configurations are not promoted to primary
role evidence.

## Validation evidence

Pre-production evidence retained from the launch commit:

- local focused suite: **PASS**, `44 passed`;
- local full suite: **PASS**, `187 passed, 1 expected local CUDA warning`;
- every server: `pip check`, full `187 passed`, holdout hashes, numerical
  policy, host-shard dry-run, and one-epoch actual-data smoke: **PASS**.

Final aggregation and regression validation on 2026-08-04:

- server state and artifact-validation reread over fresh SSH: **PASS**, 30/30;
- live HF `REMOTE_COMPLETE.json` listing and Xet hash parity: **PASS**, 3/3;
- host `trial.json` checksum parity and 42-row unique identity gate: **PASS**;
- focused follow-up/selection/orchestration suite: **PASS**, `47 passed`;
- complete CPU regression suite: **PASS**, `190 passed, 1 expected local CUDA warning`;
- aggregate HF dry-run: **PASS**, two uploads, zero deletes;
- aggregate remote list, fresh download, SHA-256, and byte parity: **PASS**;
- `git diff --check`: **PASS**.

The CUDA warning is the recorded local driver/runtime mismatch. It is not a
server CUDA result and does not invalidate the CPU regression suite.

## Interpretation boundary

- Seed 0 selected the frozen roles, so role summaries retain selection bias
  toward seed 0.
- No seed was dropped, replaced, or rerun to improve a mean.
- With `n=3`, the reported sample SDs are descriptive and are not strong
  significance evidence.
- This completion authorizes the later protected-evaluation task only for the
  already frozen role config/seed identities. It does not itself contain
  ID-test, OOD, geometry/NC, feature, or detector results.

## Preserved setup observations

- The first deployment command used an incorrectly expanded full SHA. The
  `git cat-file` gate rejected it before checkout, smoke, or production. The
  real SHA was read with `git rev-parse HEAD` and all hosts were then deployed
  successfully.
- The first `precision_medicine` production-launch SSH attempt returned
  `Connection refused`. No process started in that attempt. The fixed host
  assignment was retained; a later connection succeeded and the detached
  supervisor was verified after reconnect.
