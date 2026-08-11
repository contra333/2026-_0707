# Issue 53 Metric Contract v1.2 Runtime Validation on Curie

## Scope and verdict

Issue #53 implements the evaluation runtime frozen by
`docs/reference_cards/11_metric_contract_v1_2.md`. The implementation includes
checkpoint-to-feature extraction, metric fitting/scoring, explicit numerical
status, checkpoint-centric artifacts, canonical registry entries, and
tiny-fixture/parity/failure/invariance tests.

Final bounded verdict on 2026-08-04: **PASS** at clean Git commit
`14609a4d0c7498c77d59b0e5dbd825f341961e28`. Curie reproduced the complete
238-test CPU suite and extracted/evaluated one completed WRN-28-10
`last.pt` checkpoint using only class-balanced subsets of `id_train` and
`id_validation`. No ID-test, OOD, or 9k compatibility split was accessed or
produced.

This is implementation and infrastructure evidence, not a protected research
result. Full checkpoint evaluation, multi-server sharding, Hugging Face upload,
seed aggregation, plots, and paper conclusions remain outside Issue #53.

## Environment

| Item | Observed value |
| --- | --- |
| Host | `curie` |
| Clean worktree | `/home/ghjin/0707_exp/2026-_0707-issue53-validation` |
| Git commit | `14609a4d0c7498c77d59b0e5dbd825f341961e28` |
| Python | `3.11.9` |
| PyTorch | `2.5.1+cu121` |
| TorchVision | `0.20.1+cu121` |
| CUDA runtime | `12.1` |
| NVIDIA driver | `580.119.02` |
| GPU | `NVIDIA RTX A5000` |
| Runtime | `/home/ghjin/miniconda3/envs/oge-wrn-v1.2-a2/candidate-venv/bin/python` |

All four A5000s reported zero compute processes immediately before the final
smoke. GPU 0 was assigned with `CUDA_VISIBLE_DEVICES=0`.

## Checkpoint and bounded data identity

| Item | Value |
| --- | --- |
| Checkpoint role | `last` confirmatory primary |
| Checkpoint epoch | `200` |
| Checkpoint SHA-256 | `9a4e3a76aa795903df5d98bbcbbedfc2592c36de7a529286eae1762e9e12c62b` |
| Training Git SHA | `3556841340e6f6b92782af045ed4a468e6e271bd` |
| Training config hash | `61d7d1f20bc593cc2d106f24f0343084f56f811c8ebc63c856eaced34c50b65f` |
| Training seed | `2` |
| Extraction splits | `id_train`, `id_validation` only |
| Smoke selection | first 50 samples per CIFAR-10 class, restored to original imglist order |
| Samples | 500 train + 500 validation; exactly 50 for each label 0 through 9 |
| Protected authorization | none |

The smoke selector is a validation-only CLI option. It does not change the
production 45k/5k/10k membership or the full evaluation contract.

## Commands and observed results

The clean Curie checkout was created from the pushed task branch, then checked
at the exact commit above. Commands below used the clean checkout.

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/src" \
  /home/ghjin/miniconda3/envs/oge-wrn-v1.2-a2/candidate-venv/bin/python \
  -m pytest -q
```

Result: **PASS**, `238 passed in 11.51s`.

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=0 \
  /home/ghjin/miniconda3/envs/oge-wrn-v1.2-a2/candidate-venv/bin/python \
  scripts/extract_metric_features.py \
  --checkpoint /home/ghjin/0727ICLR실험/issue49_artifacts/3556841/curie/production/wrn28_10_optimizer_hpo_v1_2__followup__fdf67c1184ab__role_pair_followup_20260730__curie/trials/followup-adamw-61d7d1f20bc5-seed-2/attempts/followup-adamw-61d7d1f20bc5-seed-2-attempt-001/run/checkpoints/last.pt \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --artifact-root /home/ghjin/0707_exp/issue53_artifacts/14609a4d0c7498c77d59b0e5dbd825f341961e28/raw \
  --splits id_train id_validation \
  --device cuda:0 --batch-size 128 --num-workers 0 \
  --smoke-samples-per-class 50 --smoke-only
```

Result: **PASS**. The manifest records the clean evaluation SHA,
`git_dirty=false`, deterministic first-batch ordinary-forward logit parity,
float32 raw arrays, checkpoint/config/seed/role identity, ordered sample
digests, membership identity, and no protected authorization.

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/src" \
  /home/ghjin/miniconda3/envs/oge-wrn-v1.2-a2/candidate-venv/bin/python \
  scripts/evaluate_metric_cache.py \
  --raw-artifact /home/ghjin/0707_exp/issue53_artifacts/14609a4d0c7498c77d59b0e5dbd825f341961e28/raw/9a4e3a76aa795903df5d98bbcbbedfc2592c36de7a529286eae1762e9e12c62b \
  --output-dir /home/ghjin/0707_exp/issue53_artifacts/14609a4d0c7498c77d59b0e5dbd825f341961e28/metrics \
  --query-split id_validation
```

Result: **PASS**. Independent readback confirmed:

- raw checksum manifest: **PASS**, 15 files;
- metric checksum manifest: **PASS**, 19 files;
- 18 detector/logit score arrays: shape `[500]`, finite, and individually
  SHA-256 matched;
- ViM author `DIM=320` plus appendix `DIM={64,128,256,320}`;
- NECO author pipeline (`StandardScaler -> PCA`, `neco_dim=100`) executed;
- ID metrics, training geometry, held-out feature norm/NC4 control, kNN/LID,
  TwoNN, covariance spectra, and Neural Collapse diagnostics serialized;
- `smoke_only=true`, `protected_result=false`, and no protected split data.

`git diff --check` passed before the code commit. The clean Curie checkout
remained unmodified after tests and smoke execution.

## Definition and parity coverage

- Tiny fixtures cover ECE, tie-group AURC, AUROC edge cases, temperature
  scaling/fallback, 1D Gaussian detectors, simplex NC1-NC4, diagonal-spectrum
  rank definitions, and ViM no-extra-centering covariance.
- Failure fixtures cover zero/near-zero norms, non-finite inputs, all-zero or
  invalid covariance spectra, single-class/error-free misclassification
  targets, and duplicate nearest neighbors. Outcomes use explicit
  `success`, `degenerate`, or `failed` status instead of silent success clamps.
- Invariance fixtures cover sample/class permutation, deterministic neighbor
  tie-breaking, and batch-equivalent computations against float64 CPU
  references.
- ViM, NECO, FD-Shifts AURC, and DADApy TwoNN are compared with independent
  primitives or formula fixtures keyed to the pinned commits in
  `docs/sources.lock.yaml`. `scikit-learn` was available on Curie and exercised
  for ViM covariance and NECO StandardScaler/PCA parity.
- `dadapy` was not installed on Curie, so a direct upstream-package import
  comparison was **NOT_RUN**. The pinned base-estimator OLS formula fixture
  passed instead. No dependency was added solely for this smoke.

## Preserved pre-commit observations

Two dirty-worktree rehearsals were retained under
`/home/ghjin/0707_exp/issue53_artifacts/` and were not overwritten:

1. A 20-per-class (200-sample) training subset correctly failed ViM fitting
   because it cannot support the frozen WRN residual dimension `DIM=320`.
   The final smoke therefore uses 50 per class (500 samples); the production
   path remains the full 45k fit split.
2. The first 50-per-class rehearsal found constant WRN feature channels.
   Rejecting zero variance was inconsistent with author-compatible
   `StandardScaler`, which assigns scale 1 to constant channels. The runtime
   was corrected to accept that defined transform while still rejecting
   non-finite or non-positive actual scaler values. Focused and full tests were
   rerun before the clean-SHA smoke.

These rehearsals are implementation diagnostics, not research results.

## Deliberately not run

- Protected `id_test_primary=10k`, all near/far OOD datasets, and the 9k
  compatibility-only ID list: **NOT_RUN** by Issue scope.
- Full multi-checkpoint/seed evaluation, Hugging Face upload, aggregation,
  figures, and scientific analysis: **NOT_RUN** by Issue scope.
- An actual-data GDA-ClassDensity smoke: **NOT_RUN** in the bounded small-N
  run because the frozen per-class full-covariance fit is deliberately costly
  and singular at this sample size. The 1D/class-density and adaptive-jitter
  behavior is covered by synthetic tests. Full GDA execution remains for the
  protected production evaluation.
- Desktop/local pytest: **NOT_RUN** because the local Python 3.12.3 environment
  has no `pytest` module. Curie's pinned runtime ran the focused and complete
  suites instead.

## Interpretation boundary

The clean smoke establishes that the committed runtime loads a real completed
WRN checkpoint, preserves deterministic sample identities, produces finite
non-protected metrics, and writes checksum-addressed artifacts. It does not
establish full-dataset runtime, OOD performance, optimizer differences,
multi-seed stability, or paper conclusions. Those claims require a separate
protected execution Issue after review and merge of Issue #53.
