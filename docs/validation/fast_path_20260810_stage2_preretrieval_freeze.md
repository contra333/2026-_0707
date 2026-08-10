# Stage 2 pre-retrieval freeze validation

Date: 2026-08-10

## Outcome

The fixed-readout Stage 2 discovery population and decision procedure were
frozen before any selected raw score, feature, or logit array was retrieved or
opened. The freeze binds the existing Metric Contract v1.2 `last.pt` /
`confirmatory_primary` population to an exact remote checksum catalog, an
exact selective-download allowlist, a non-Git destination, two eligible
radial-normalization candidates, and a prespecified scientific gate.

This record authorizes selective reuse only. It is not a Stage 2 scientific
result and supplies no causal, mechanism, or focal-channel conclusion.

## Frozen identities and scope

| Item | Value |
| --- | --- |
| Source checkpoint inventory file SHA-256 | `39c89b30aecbea0984cafd1ac75cf8f8676808ee12b4728fb4fb21e845f6e7a1` |
| Source inventory hash | `35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c` |
| Dataset-policy hash | `f80984d28ef9a7550633fb3aea514f783f3db1fa655a6d3f49850b10857e6afc` |
| Scientific evaluator Git SHA | `c38b09694be88aa74de0741b39e9d3ba0d6ff61a` |
| Research Contract v2 file SHA-256 | `84d032016ef1bb479192197b4393d7906c4d83ea1ba377c3fadf49e2697b3c91` |
| Candidate registry SHA-256 | `8808a588b05eb73eadef81b7370dc1c485df0817efc7e4b0f9870b5d9025fd2a` |
| Gate policy SHA-256 | `7e62c9dfd6b6f72b90caf43fd2b35c83fef674b6c85d99fabcf1cb179bc7299d` |
| Remote checksum catalog SHA-256 | `a142194187147288d2be630f0e858eb92e38323f154c8a23d2be13a5654d9111` |
| Reuse manifest SHA-256 | `f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc` |
| Full remote size-plan SHA-256 | `70a8201dc67e439bdff945c12c20b2b3a1cf88911882fc8ee774c5d2d61eb31d` |
| Metadata-only sync-plan SHA-256 | `c80f3c900481c85616411d7de960d116ff92346e8e759db6cca075f79a20dce2` |
| Population | 10 unique configurations x seeds `{0,1,2}` = 30 `last.pt` bundles |
| OOD environments | CIFAR-100, TinyImageNet, MNIST, SVHN, Textures, Places365 |
| Exact allowlist | 184 files per bundle; 5,520 files; 19,620,378,841 bytes |
| Frozen remote root | `hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2` |
| Frozen local destination | `/home/contra333/2026여름방학실험코드/fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1` |

The catalog was built from downloaded bundle metadata and the size-only HF
sync plan. For each selected bundle, `artifact_manifest.json` identity and
`bundle_checksums.sha256` were cross-checked, and every allowlisted path was
bound to its remote byte size and SHA-256. The five local v1.2 scalar-aggregate
baseline files were also checked against the hashes and sizes embedded in the
reuse manifest.

## Frozen analysis policy

- Eligible candidates are exactly Mahalanobis-Raw to Mahalanobis++ and
  kNN-Raw K=50 to kNN-L2 K=50; priority only breaks an exact selection tie.
- Energy-T1 is the feature-only exact null. Formula invariants and prespecified
  non-invariant witnesses must pass before scientific selection.
- Integrity and scalar parity use all 30 bundles. Candidate selection uses
  seeds 1 and 2; seed 0 is post-selection sensitivity because it contributed
  to the historical role freeze. Selection noise is the linear q90 of the ten
  config-wise absolute seed-1/seed-2 differences; seed 0 cannot enter a margin.
- Each of the six OOD datasets remains a separate environment. The gate also
  requires near/far support, seed stability, LODO, LOCO, practical margins,
  and no practical reversal.
- Any identity, checksum, scalar-parity, formula-reconstruction, or exact-null
  failure yields `FAILED`. No qualifying candidate yields `NO_GO`; unstable
  support yields `INCONCLUSIVE`; only one fully eligible candidate yields
  `PASS`.
- Null-control drift is a hard validity check and does not enter the candidate
  ranking score. Legacy C3/C4 and C1 Adam--AdamW directions are reported
  descriptive contrasts and cannot affect eligibility or the final decision.
- The policy fixes tie credit, all nine raw-to-transform pair-order transition
  masses, corrected/harmed mass, LODO fold recomputation, LOCO held-out
  median-pair attenuation, seed-view margins, and decision precedence.

## Retrieval and write-safety contract

- The fetch helper is network-free: it emits an exact include filter and HF
  plan, then separately validates that the plan contains only download/skip
  operations for the 5,520 frozen paths.
- The destination is outside the repository and every Git worktree. The tool
  rejects symlinks, unexpected files, size/hash mismatches, uploads, deletes,
  and overwrites.
- An interrupted transfer may resume only after every existing file is
  independently size- and checksum-verified; verified files are skipped.
- The current mechanism script is deliberately labeled a selective
  bundle/pair **evidence extractor**, not the scientific gate reducer. It
  rejects files outside the allowlist, validates all 30 bundle identities
  before extraction, checks selected stored-score formulas, provides an exact
  raw-kNN cache, binds cache output to manifest/policy/code hashes, and writes
  reports and hash sidecars without overwrite. `--skip-raw-knn` is explicitly
  `NOT_RUN`; every extractor output reports the scientific gate as `NOT_RUN`.

## Validation evidence

| Check | Result |
| --- | --- |
| Generated catalog byte identity (`--check`) | `PASS`; SHA-256 `a142194187147288d2be630f0e858eb92e38323f154c8a23d2be13a5654d9111` |
| Generated reuse manifest byte identity (`--check`) | `PASS`; SHA-256 `f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc` |
| Stage 2 focused tests | `46 passed in 16.89s` |
| Complete local CPU suite | `327 passed in 23.95s` |
| `git diff --check` | `PASS` |

## Commands actually run

```bash
TMPDIR=/tmp PYTHONPATH=.:src .venv/bin/python \
  scripts/collect_fixed_readout_stage2_remote_catalog.py \
  --metadata-root /tmp/fixed_readout_stage2_metadata_20260810_360c109 \
  --sync-plan /tmp/fixed_readout_stage2_remote_sizes_20260810_360c109.plan.jsonl \
  --output configs/evaluation/fixed_readout_stage2/remote_checksum_catalog.json \
  --check

TMPDIR=/tmp PYTHONPATH=.:src .venv/bin/python \
  scripts/build_fixed_readout_stage2_reuse_manifest.py \
  --checksum-catalog \
    configs/evaluation/fixed_readout_stage2/remote_checksum_catalog.json \
  --output configs/evaluation/fixed_readout_stage2/reuse_manifest.json \
  --check

TMPDIR=/tmp PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_fixed_readout_stage2_policy.py \
  tests/test_fixed_readout_stage2_remote_catalog.py \
  tests/test_fixed_readout_stage2_reuse_manifest.py \
  tests/test_fetch_fixed_readout_stage2_reuse.py \
  tests/test_fixed_readout_stage2_mechanism.py

TMPDIR=/tmp PYTHONPATH=.:src .venv/bin/python -m pytest -q

git diff --check
```

## Explicitly not run at this freeze

- selected raw score, feature, logit, or sample-ID payload retrieval;
- production scalar-parity or formula-reconstruction analysis;
- production raw-kNN computation;
- Stage 2 all-configuration aggregation, candidate selection, or gate decision;
- GPU/server execution, protected dataset traversal, checkpoint inference,
  model training, or Hugging Face mutation/upload;
- Stage 3 shared-prefix confirmation or Stage 4 replication.

The metadata-only working copies used to construct the committed checksum
catalog remain outside Git. They are procedural inputs, not research evidence.

The exact pre-retrieval status is therefore:

```text
pre-retrieval integrity and selective evidence-extractor tooling: PASS
Stage 2 scientific mechanism gate: NOT_RUN
Stage 3 launch authorization: not established
```
