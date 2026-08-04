# Issue #57 Metric Contract v1.2 production execution

Date: 2026-08-04  
Issue: [#57](https://github.com/contra333/2026-_0707/issues/57)  
Pull request: [#58](https://github.com/contra333/2026-_0707/pull/58)

## Outcome

The frozen WRN-28-10/CIFAR-10 Metric Contract v1.2 population completed on
the three approved hosts. Exactly 60 authorized checkpoint jobs reached
checkpoint-level `REMOTE_VERIFIED`: `curie=20`, `lise=20`, and
`precision_medicine=20`. No `excluded_pair_only` identity, 9k compatibility
ID split, or OOD-validation split was evaluated.

All three checksum-verified operational shards were read back on Curie. The
central aggregate contains 30 training identities, 10 configurations, seeds
`{0,1,2}`, and separate `last` primary and `best_val` control rows. All 95,160
per-checkpoint scalar records and all 31,720 seed aggregates have status
`success`; the aggregate has zero non-success records. This is execution and
measurement evidence, not a paper conclusion.

## Frozen identities

| Item | Value |
| --- | --- |
| Metric definition | `metric_contract_v1.2` |
| Scientific evaluator Git SHA | `c38b09694be88aa74de0741b39e9d3ba0d6ff61a` |
| Supervisor terminal-state hotfix | `1af220bbdcac99f4c762613a7556ce2e1901e8da` |
| Inventory hash | `35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c` |
| Dataset-policy hash | `f80984d28ef9a7550633fb3aea514f783f3db1fa655a6d3f49850b10857e6afc` |
| Protected authorization | `issue-57-protected-metric-v1.2` |
| Checkpoint roles | `last` primary; `best_val` control |
| ID split counts | `id_train=45,000`, `id_validation=5,000`, `id_test=10,000` |
| Authorized OOD counts | CIFAR-100 9,000; TIN 7,793; MNIST 70,000; SVHN 26,032; Texture 5,640; Places365 35,195 |

The three source checkouts remained clean at the scientific evaluator SHA.
After the production pilot, a supervisor-only defect was found: a completed
result passed `job_id` once positionally and once as a keyword while updating
the state ledger. The evaluator had already completed local checksum
verification and Hugging Face readback before that exception. Commit
`1af220b` removes only the duplicate terminal-state argument and adds a
regression test. The hotfix package was installed into a host-local isolated
execution environment while the clean evaluator checkout and every child
evaluation process remained at `c38b096`. Consequently every scalar record
has one scientific `evaluation_git_sha`, while the orchestration correction
remains separately attributable. Completed bundles and upload controls were
resumed without recomputation or deletion.

## Implementation and validation

The production layer provides:

- protected single-checkpoint evaluation with checkpoint, inventory, dataset,
  split, and authorization validation before traversal;
- exact chunked Mahalanobis, GDA, kNN, LID, and TwoNN computation;
- checkpoint-local float64 fitting and canonical scalar/intermediate output;
- source-local concurrent supervision with attempt preservation and verified
  resume;
- dry-run-first, zero-delete Hugging Face synchronization and readback;
- checksum-verified operational-shard collection and deterministic seed
  aggregation.

The final local focused check passed `3 passed`. The final local complete CPU
suite passed `266 passed, 1 warning`; the warning was a local CUDA-driver
initialization warning in a CPU temperature-scaling test and did not fail the
suite. `git diff --check` passed.

Before execution, the isolated host environments passed the complete
`c38b096` suite:

| Host | Result before launch | Result after supervisor hotfix package |
| --- | --- | --- |
| Curie | `265 passed in 12.77s` | `265 passed in 12.91s` |
| Lise | `265 passed in 12.41s` | `265 passed in 11.15s` |
| precision_medicine | `265 passed in 17.87s` | `265 passed in 17.61s` |

The host-local execution environments inherited only the pinned runtime
dependencies and installed the task package separately. Shared training
runtimes and source checkpoints were not modified.

## Production pilot

The final scientific-SHA pilot used Curie GPU UUID
`GPU-b7682b36-2ce0-802b-8bb1-e7668087ecde` and job
`eval-followup-adamw-61d7d1f20bc5-seed-2-last`, checkpoint SHA-256
`9a4e3a76aa795903df5d98bbcbbedfc2592c36de7a529286eae1762e9e12c62b`.

It completed all nine executable splits, produced 1,586 successful scalar
records, verified 57 raw-artifact files and 559 production payload files, and
uploaded 560 files totaling 1,122,929,478 bytes. The artifact-manifest SHA-256
is `7140ee4f914cc9cab773e2cce2450cb196d3b3bcf9788166e93f494c2faddc60`.
The upload control reached `REMOTE_VERIFIED`.

Preserved earlier attempts document three pre-production integration defects:

1. dataset OOD entries initially lacked `expected_count`;
2. split-key validation compared dictionary insertion order after JSON sorting;
3. the legacy host Hugging Face CLI represented an empty listing as
   `(empty)` rather than JSON.

Each failure occurred in its own evaluation-SHA artifact namespace. Nothing
was deleted or silently reused. The final pilot ran after these evaluator and
upload fixes at `c38b096`; the later `1af220b` change affects only supervisor
terminal-state bookkeeping as described above.

## Three-host execution evidence

| Host | GPUs | Assigned | Remote verified | Failed | Host operational upload |
| --- | ---: | ---: | ---: | ---: | --- |
| Curie | 4 | 20 | 20 | 0 | `REMOTE_VERIFIED` |
| Lise | 2 | 20 | 20 | 0 | `REMOTE_VERIFIED` |
| precision_medicine | 4 | 20 | 20 | 0 | `REMOTE_VERIFIED` |

Every successful bundle is stored under the frozen checkpoint-centric
destination:

```text
hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2/<checkpoint_sha256>/
```

Host operational evidence is stored at:

```text
hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2/
  _operations/c38b09694be88aa74de0741b39e9d3ba0d6ff61a/{curie,lise,precision_medicine}/
```

The collector downloaded 65 files from each host shard with zero deletes.
Their reported sizes were 38,234,225 bytes (Curie), 38,233,475 bytes (Lise),
and 38,234,465 bytes (precision_medicine). Collection status was
`CHECKSUM_VERIFIED`, with three hosts and exactly 60 checkpoint jobs.

## Aggregate and Hugging Face result

The aggregate summary reports:

| Field | Value |
| --- | ---: |
| Checkpoint jobs | 60 |
| Training identities | 30 |
| Configurations | 10 |
| Per-checkpoint scalar records | 95,160 |
| Successful per-checkpoint scalar records | 95,160 |
| Seed aggregate records | 31,720 |
| Successful seed aggregate records | 31,720 |
| Non-success seed aggregates | 0 |
| Detector rank-concordance records | 40 |
| Successful rank-concordance records | 40 |
| `last` scalar records | 47,580 |
| `best_val` scalar records | 47,580 |

The checksum-valid aggregate contains:

- `aggregate.json` — population, role, seed, count, and checkpoint index;
- `per_checkpoint_records.jsonl` — all checkpoint/seed scalar records;
- `seed_aggregates.jsonl` — seed values, mean, and sample SD (`ddof=1`);
- `detector_rank_concordance.jsonl` — successful Kendall tau-b records;
- artifact manifest and manifest SHA-256 sidecar.

It was uploaded as six files totaling 158,808,698 bytes and independently
verified at:

```text
hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2/
  _aggregates/35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c/
  c38b09694be88aa74de0741b39e9d3ba0d6ff61a/
```

The same aggregate was copied to the local non-Git analysis directory:

```text
/home/contra333/2026여름방학실험코드/issue57_metric_results/
  c38b09694be88aa74de0741b39e9d3ba0d6ff61a/aggregate/
```

The local manifest sidecar and every listed payload SHA-256 were recomputed and
matched. This directory is the analysis handoff; it is not committed to the
repository.

## Commands actually run

Representative commands, with host-specific absolute roots and GPU UUIDs
expanded at execution time, were:

```bash
TMPDIR=/tmp PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  -m pytest -q tests/test_metric_evaluation_supervisor_v1_2.py

TMPDIR=/tmp PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  -m pytest -q

env -u PYTHONPATH <isolated-execution-python> \
  scripts/supervise_metric_evaluation_host.py run \
  --host-id <host> \
  --expected-git-sha c38b09694be88aa74de0741b39e9d3ba0d6ff61a \
  --source-root <frozen-source-key=absolute-root> \
  --data-root <verified-openood-root> \
  --artifact-root <host-bundle-root> \
  --state-root <host-state-root> \
  --gpus <frozen-live-idle-uuid-list> \
  --hf-cli <isolated-hf-1.26.0-cli> \
  --batch-size 128 --num-workers 4 --blas-threads 4

env -u PYTHONPATH <isolated-execution-python> \
  scripts/collect_metric_evaluation_shards.py \
  --evaluation-git-sha c38b09694be88aa74de0741b39e9d3ba0d6ff61a \
  --download-root <new-operational-shard-root> \
  --hf-cli <isolated-hf-1.26.0-cli>

env -u PYTHONPATH <isolated-execution-python> \
  scripts/aggregate_metric_evaluation_results.py \
  --operational-root <checksum-verified-operational-shard-root> \
  --output-dir <new-aggregate-root> \
  --hf-cli <isolated-hf-1.26.0-cli> \
  --upload-control-root <new-aggregate-upload-control-root>
```

## Remaining boundary

Production measurement, upload, readback, and seed aggregation are complete.
No plots, optimizer-effect interpretation, hypothesis test, restoration-effect
claim, or paper table has yet been accepted. Those are the next local analysis
phase and must preserve the `last` primary versus `best_val` control boundary.
DDU/spectral-normalization training remains outside Metric Contract v1.2.
