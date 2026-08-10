# Stage 2 existing-artifact mechanism gate execution

Date: 2026-08-10

## Outcome

The checksum-bound Stage 2 production execution reached a fail-closed
`FAILED` result. The 30-bundle, 210-variant raw-kNN cache completed and passed
the committed full-cache `status` rehash. The subsequent full evidence build
stopped on the first bundle because the prespecified exact targeted-witness
oracle failed for `radial_l2_mahalanobis:cifar100`.

No evidence directory was published. The frozen reducer was then invoked on
the expected evidence paths and produced the intentional six-file preflight
`FAILED` package: no candidate was selected and
`scientific_launch_allowed=false`. Stage 3 is not authorized. This is neither
`NO_GO` nor evidence that radial invariance is scientifically false.

The production failure is a numerical-oracle boundary. In exact arithmetic,
positive sample-wise radial scaling is removed by row L2 normalization. In the
observed float64 path, normalization-level differences of at most about
`1e-16` were amplified by the tied-covariance precision calculation and broke
exact score ties. The frozen policy requires zero weak-rank disagreement and
AUROC drift at most `1e-12`, so the result must remain `FAILED`; its thresholds
were not relaxed after observing protected discovery data.

This is protected discovery execution evidence. It is not fresh confirmation,
a causal training-rule result, or cross-regime replication.

## Frozen identities

| Item | Value |
| --- | --- |
| Tooling Git SHA | `b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03` |
| Research Contract SHA-256 | `84d032016ef1bb479192197b4393d7906c4d83ea1ba377c3fadf49e2697b3c91` |
| Reuse manifest SHA-256 | `f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc` |
| Candidate registry SHA-256 | `8808a588b05eb73eadef81b7370dc1c485df0817efc7e4b0f9870b5d9025fd2a` |
| Gate policy SHA-256 | `7e62c9dfd6b6f72b90caf43fd2b35c83fef674b6c85d99fabcf1cb179bc7299d` |
| Remote catalog SHA-256 | `a142194187147288d2be630f0e858eb92e38323f154c8a23d2be13a5654d9111` |
| Verified file-catalog SHA-256 | `1691467f5a054ac29fb2e6a18068e965d63ebc17f856b40eb074e7bb0e86f410` |
| Plan SHA-256 | `94ee66ae95882efa86e8711689393771e489f1465c997afca9fd0349f7fa198d` |
| Plan ID | `4b2679c117cf8225a24b1ca6eeedfd16bdd2a10dc46612041d5a1bf216a5d5aa` |
| Analysis ID | `fa7285b4b82ef5839f87ba69f6c9efd94d1f5693710e58fd9e5639e03a9cd837` |
| Numeric runtime fingerprint | `205ec0f79fbba67b0d2f3d7676e76d8ca53c5dc30e5c860530eefceda82e1632` |
| Numeric runtime record | `cdf593f3d222d04768cd18a3edf47f7b6afced15b84d414f752b64118376fec1` |
| Python | `/home/ghjin/0707_exp/issue57_eval_exec_venv/bin/python`; CPython 3.11.9 |
| Packages | NumPy 1.26.4; SciPy 1.17.1; scikit-learn 1.5.2 |
| Query / bank / concurrency | `1024 / 45000 / 24` |
| Source root | `/home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/reuse` |
| Relocated baseline root | `/home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/baseline` |

All recorded BLAS/OpenMP runtime libraries reported one numeric thread. The
fixed environment excluded ambient user site packages and
`LD_LIBRARY_PATH`.

## Capacity and execution plan

- The selected synthetic benchmark used query chunk 1,024 and bank chunk
  45,000. Three timings were `2.4769189842`, `2.4707663460`, and
  `2.4528580578` seconds; median `2.4707663460` seconds.
- Benchmark artifact SHA-256:
  `460ae119c724a8561054c1304f7a88889ea69a2ad506aa9d9ce4489e2fed2c18`.
- The benchmark's linear single-worker estimate was 23.04 hours. It was used
  only for host capacity planning, not as scientific evidence.
- One initial launch-guard attempt matched its own safety process check and
  exited before starting a supervisor or writing cache artifacts. The next
  invocation started one supervisor. There was no resume invocation and no
  manual per-bundle replay.
- Source mutation, destination overwrite, deletion, checkpoint reevaluation,
  feature re-extraction, and upload were `NONE`.

## Cache completion

| Field | Observed |
| --- | ---: |
| Bundle count | 30 |
| Variant count | 210 |
| Cache file count rehashed remotely | 2,730 |
| Completion records | 30 |
| Materialization receipts | 30 |
| Completion catalog SHA-256 | `d8b7cf858eaa626625eccae8a40e0a433d2de5f97dee08408ed1da7d75f587d4` |
| Supervisor run status | `PASS`; 30 completed, zero failed |
| Committed `status` result | `PASS` |
| Committed `status` SHA-256 | `e263f7e130c8f2d4e15147143f7385c721633f7bd6807064f6b312d6da036135` |
| Staging root | present, real directory, empty |
| Scientific gate at cache boundary | `NOT_RUN` |

The committed status invocation exited 0 with empty stderr. Local archive
verification recomputed the status-to-completion catalog and every
completion-to-receipt SHA link. It did not claim to rehash the 2,730 cache
files locally; that array/file-tree boundary is attested by the committed
remote `status` execution.

## Evidence build failure

The full command used the frozen reuse manifest, the original retrieval root,
the verified cache, the relocated five-file v1.2 baseline, query chunk 1,024,
and bank chunk 45,000. It did not enable `--compute-missing-cache`.

| Field | Observed |
| --- | --- |
| Exit code | `1` |
| Stdout | empty |
| Stderr bytes | 1,088 |
| Stderr SHA-256 | `34c39a41223a62ed3c3680dbaa71e4cfaefdf17935808ae17a2f5f8aa1371f94` |
| Published evidence directory | absent |
| Failing assertion | `targeted witness invariance failed: radial_l2_mahalanobis:cifar100` |
| First bundle | `09442d81872d37e49f213f59d4687b6e86e375282e92a87346ad77f88c731cf0` |
| Bundle identity | Adam, seed 0, config `b701f780ad99d469949a985b0472b8d1f26be0545e531d92895b7559ca142328` |

Because atomic evidence publication did not occur, the expected 720 model
rows and 360 diagnostic rows do not exist and were not supplied to the
reducer.

## Exact-runtime numerical diagnosis

A separate read-only diagnostic used the same frozen Lise Python, packages,
thread environment, first bundle, full ID-test population, and CIFAR-100. It
was diagnostic only and did not create or repair evidence.

| Quantity | Observed |
| --- | ---: |
| Train / all-query shape | `45000 x 640` / `163660 x 640` |
| Compared ID-test + CIFAR-100 scores | 19,000 |
| Normalized train max absolute difference | `5.551115123125783e-17` |
| Normalized query max absolute difference | `1.1102230246251565e-16` |
| Covariance max absolute difference | `8.131516293641283e-20` |
| Precision max absolute difference | `1.192034687846899e-06` |
| Numerical covariance rank | `640` before and after |
| Condition number | about `15752.9181` before and after |
| Score max absolute / relative difference | `8.894858183339238e-10` / `1.1256828212304002e-13` |
| Score `np.allclose(atol=1e-12, rtol=1e-10)` | `true` |
| Weak-rank disagreement | `0.00031578947368421053` = 6/19,000 |
| Base / witness AUROC | `0.8691396499999999` / `0.8691396555555555` |
| AUROC absolute drift | `5.5555555711350735e-09` |
| Exact ties | reference 3; witness 0 |

The score relation passed the frozen mixed absolute/relative closeness check,
but exact weak-rank equality and the `1e-12` AUROC-drift requirement failed.
This is consistent with one half pair-credit changing among 90 million
ID--OOD pairs. It must not be converted into a `PASS` by post-outcome rounding
or tolerance changes.

## Scientific gate decision

The reducer was invoked on the expected evidence paths after atomic evidence
publication had failed. Its preflight therefore emitted a checksum-bound
`FAILED` package with four intentionally empty JSONL outputs.

| Field | Value |
| --- | --- |
| Status | `FAILED` |
| Decision analysis ID | `ba37abf0028371d219b84b22e8f2418cf4aac3fa502524595b345699e1d39eab` |
| Input analysis ID | `null` |
| Selected / full-data candidate | `null` / `null` |
| Scientific launch allowed | `false` |
| Hard validity | `false` |
| Reducer reason code | `ARTIFACT_CHECKSUM_FAILED` |
| Preflight trusted | `false` |
| Seed / LODO / LOCO selection counts | `0 / 0 / 0` |
| Near / far support counts | `0 / 0` |
| Candidate ranking | empty |

The reducer reason describes the immediate missing evidence checksum input.
The preceding evidence stderr records the root cause: the required targeted
witness oracle failed before atomic evidence publication. No scientific
estimand or candidate ranking was reduced.

## Reducer package

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `contrast_metrics.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `dataset_summaries.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `cv_folds.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `diagnostic_summaries.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `gate_decision.json` | 1 object | `8f966e6d9df466301f78ee61303a314982a4c6ba889b00af55555fb4d9edc708` |
| `checksums.sha256` | 5 bindings | `a3d3778362666c4e821e1dd113ca2399e59f398274c0c4b94f50c38f48d36b04` |

The reducer's actual exit code was 1, as required for `FAILED`; stderr was
empty. `preconfirmation_freeze.json` was not created.

## Archive and remote synchronization

The immutable execution metadata was copied without overwrite to:

```text
/home/contra333/2026여름방학실험코드/fixed_readout_stage2_results/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03
```

The archive contains plans and invocation records, benchmarks, 30 receipts,
30 completion records and their logs, and the six gate files. The absent
evidence directory and its failed invocation are explicit catalog state.

| Field | Value |
| --- | --- |
| Archive status | `VERIFIED_FAILED_EXECUTION` |
| Archived files / bytes | 127 / 3,842,991 |
| Before/after remote catalog | byte-identical |
| Transfer catalog SHA-256 | `755150663c47168c2f0371a78050d718c4918d4814021397bf33c0c29e71a9fb` |
| Completion-to-receipt chain | `PASS` |
| Gate checksum verification | `PASS` |
| Owner historical untracked files | preserved |

The raw 19.6 GB reuse population, 210-variant cache, and materialized bundles
remain on Lise and were not duplicated into this small local metadata archive.

## Commands actually run

The exact scientific subprocess argument vectors, fixed environment, start/end
times, exit codes, stdout/stderr sizes, and hashes are preserved under the
archive's `payload/plans/*.invocation.json` records. The principal commands
were:

```bash
/home/ghjin/0707_exp/issue57_eval_exec_venv/bin/python \
  /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/scripts/supervise_fixed_readout_stage2_cache_host.py run \
  --plan /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/plans/cache_host_plan_q1024_b45000_c24.json

/home/ghjin/0707_exp/issue57_eval_exec_venv/bin/python \
  /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/scripts/supervise_fixed_readout_stage2_cache_host.py status \
  --plan /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/plans/cache_host_plan_q1024_b45000_c24.json

/home/ghjin/0707_exp/issue57_eval_exec_venv/bin/python \
  /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/scripts/extract_fixed_readout_stage2_model_metrics.py \
  --reuse-manifest /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/configs/evaluation/fixed_readout_stage2/reuse_manifest.json \
  --retrieval-root /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/reuse \
  --output-directory /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/evidence \
  --cache-root /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/cache \
  --parity-baseline-root /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/baseline \
  --query-chunk-size 1024 --bank-chunk-size 45000

/home/ghjin/0707_exp/issue57_eval_exec_venv/bin/python \
  /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/scripts/reduce_fixed_readout_stage2_gate.py \
  --selection-manifest /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/evidence/selection_manifest.json \
  --model-metrics /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/evidence/model_metrics.jsonl \
  --diagnostic-metrics /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/evidence/diagnostic_metrics.jsonl \
  --input-checksums /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/evidence/checksums.sha256 \
  --candidate-registry /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/configs/evaluation/fixed_readout_stage2/candidate_registry.json \
  --gate-policy /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/repo/configs/evaluation/fixed_readout_stage2/gate_policy.json \
  --output-dir /home/ghjin/0707_exp/fixed_readout_stage2/b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03/gate
```

The archive verifier performed two independent remote catalogs around six
no-overwrite `rsync -aR` transfers and then recomputed every transferred file
size/SHA, completion catalog, receipt link, and gate checksum.

## Stage 3 implication

Research Contract v2 Stage 3 is stopped. A retry of the same frozen command,
candidate replacement, tolerance relaxation, or use of partial evidence to
rescue the result is not allowed. Any remediation requires an explicitly
versioned contract/tooling revision, a new committed SHA and analysis ID, and
fresh output paths. The current cache and failure artifacts remain immutable
historical evidence.

## Post-execution validation

| Check | Result |
| --- | --- |
| Policy and reducer focused suite | `PASS`; 69 tests in 41.86 s |
| `git diff --check` | `PASS` |
| Markdown trailing-whitespace, final-newline, and relative-link check | `PASS` |
| Local archive before/after catalog equality | `PASS` |
| Local 127-file size/SHA and completion-to-receipt verification | `PASS` |
| Lise repository HEAD and gate-decision SHA recheck | `PASS` |

The Lise recheck observed repository HEAD
`b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03`, gate status `FAILED`,
`scientific_launch_allowed=false`, absent evidence directory, and gate-decision
SHA-256 `8f966e6d9df466301f78ee61303a314982a4c6ba889b00af55555fb4d9edc708`.

## Explicitly not run

- checkpoint reevaluation, feature re-extraction, or new protected-dataset
  traversal;
- a successful 720/360 evidence package or scientific candidate reduction;
- shared-prefix fork implementation or `preconfirmation_freeze.json`;
- fresh prefix training or protected OOD confirmation;
- comparable-ID equivalence, causal decay-coupling, or cross-regime claims;
- ResNet-18/CIFAR-100 replication;
- Hugging Face upload, source mutation, overwrite, or deletion.
