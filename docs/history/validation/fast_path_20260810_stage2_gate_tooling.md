# Stage 2 gate retrieval and tooling validation

Date: 2026-08-10

## Outcome

The complete preselected Stage 2 payload was retrieved to the frozen non-Git
destination and verified byte-for-byte against the committed allowlist. The
one-host exact raw-kNN cache worker, complete candidate/diagnostic evidence
extractor, and fail-closed scientific reducer were then implemented and
validated without parsing a selected production NumPy array.

This record validates retrieval integrity and execution tooling only. It is
not a Stage 2 scientific result. Raw-kNN production computation, evidence
extraction, candidate reduction, and the scientific gate decision remain
`NOT_RUN` until this tooling is committed and executed in one locked numeric
runtime.

## Retrieved payload identity

| Item | Value |
| --- | --- |
| Frozen reuse-manifest SHA-256 | `f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc` |
| Retrieval status | `CHECKSUM_VERIFIED_EXACT_ALLOWLIST` |
| File count | 5,520 |
| Total bytes | 19,620,378,841 |
| Verified file-catalog SHA-256 | `1691467f5a054ac29fb2e6a18068e965d63ebc17f856b40eb074e7bb0e86f410` |
| Retrieval receipt SHA-256 | `20d8ba250065dabb542db6bfe4b0c222f5802aad17f6c3ff9621e376bd4178fa` |
| Exact transfer-filter SHA-256 | `01f5e40c499c3c6bbf722954ebef20ab805b4c70d345f247e7e137a08f9ab55f` |
| Exact transfer-plan SHA-256 | `4be3f33464a216676f0426d6b7e3c03f769c12b0dc06af1908d673b9f457826d` |
| Destination | `/home/contra333/2026여름방학실험코드/fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1` |

The retrieval helper copied and hashed bytes only. No selected `.npy` file was
loaded with NumPy, and no distributional statistic or scientific result was
computed during retrieval verification.

## Frozen execution outputs

The committed execution path is designed to produce the following exact
population before the reducer may issue a decision:

- 30 checkpoint bundles x 7 exact K=50 raw-kNN cache variants;
- 720 candidate evidence rows: 30 bundles x 6 OOD datasets x 2 eligible
  raw-to-transform candidates x 2 states;
- 360 diagnostic evidence rows: 30 bundles x 6 OOD datasets x CTM/Pure
  Residual;
- 12 descriptive diagnostic summaries: 6 OOD datasets x CTM/Pure Residual;
- one checksum-bound decision package with status `PASS`, `NO_GO`,
  `INCONCLUSIVE`, or `FAILED`.

CTM and Pure Residual are hard-isolated from candidate selection and ranking.
CTM must reproduce its stored score and the v1.2 scalar baseline. Pure
Residual is a v2 diagnostic defined as negative ViM residual with the
authoritative DIM 320/640 split; it is checked against both stored residual
and full ViM arrays but has no v1.2 scalar detector baseline. Energy-T1 is the
exact feature-only null and remains bitwise unchanged by feature-only
transforms.

## Numeric and provenance controls

- The supervisor freezes one committed repository SHA, one 30-bundle reuse
  manifest, one Python launcher, package versions, BLAS/threadpool identity,
  module origins, chunk sizes, and all task roots in a deterministic plan.
- Production entry points reject working-tree scientific sources that differ
  from `HEAD`. The reducer also verifies the evidence worker's committed code
  sources and evidence Git SHA before a scientific launch can be allowed.
- All numeric thread controls are fixed to one thread per worker; ambient user
  site packages and `LD_LIBRARY_PATH` are excluded.
- Cache identities bind the selected bundle, source allowlist, analysis ID,
  runtime fingerprint, algorithm, K, transformation, and chunks. Resume is
  allowed only after checksum and manifest verification.
- Concurrent workers stage into a derived same-filesystem sibling outside the
  recursively validated shared cache tree. The supervisor freezes that staging
  root and the plan-output path, then revalidates all six execution roots and
  the output as pairwise disjoint during plan, write, run, and status.
- A full evidence build may relocate the five v1.2 baseline files through an
  explicit `--parity-baseline-root`. Only location is portable: the exact
  logical names, sizes, and SHA-256 values remain authoritative, and the root
  must contain exactly five real non-symlink regular files.
- Evidence and reducer destinations must be fresh, outside Git and every input
  tree, and have real non-symlink parent chains. Existing paths, including
  dangling symlinks, fail before protected arrays are read.
- The exhaustive kNN backend uses float64 squared Euclidean distances and a
  lexicographic `(distance, sample_id)` boundary repair, making tied K-neighbor
  selection independent of chunk boundaries.

The fresh disjoint-output, one-supervisor operational contract prevents two
writers from targeting the same destination. The final directory publication
uses a checked `os.rename`; it is not a Linux-specific atomic `NOREPLACE`
primitive against a malicious concurrent process that races to create the
same path between the check and rename. This residual P2 limitation is
accepted only under the recorded single-supervisor constraint.

## Independent audit corrections

Independent synthetic review found and closed the following production
blockers before commit:

1. output destinations could be redirected through a dangling symlink or
   nested under a protected input tree;
2. the final evidence population did not yet preserve separate Pure Residual
   and CTM diagnostic rows;
3. recomputed BLAS/eigendecomposition outputs were incorrectly being treated
   as requiring bitwise hashes rather than tolerance-based full-array parity;
4. the reducer did not independently validate CTM scalar parity against its
   v1.2 baseline fields;
5. the reducer did not yet bind the evidence manifest to the exact committed
   worker code sources and analysis Git SHA;
6. a reducer `FAILED` artifact still returned shell exit code zero;
7. concurrent workers could mistake another worker's normal in-progress cache
   temporary directory for corruption;
8. frozen absolute baseline paths were not portable to the locked execution
   host;
9. the derived staging root and plan output were not yet frozen and checked
   against every supervisor root, allowing a bad plan path to mutate a source
   tree before later rejection.

Regression tests now cover each fail-closed boundary. Full-array numerical
parity uses the prespecified `atol=1e-12`, `rtol=1e-10`; hashes bind provenance
and immutable inputs rather than replacing a numerical-tolerance check.

## Validation evidence

| Check | Result |
| --- | --- |
| Retrieval receipt and exact allowlist verification | `PASS`; 5,520/5,520 files and 19,620,378,841/19,620,378,841 bytes |
| Six execution-script `py_compile` check | `PASS` |
| CLI construction/help smoke checks | `PASS` |
| Stage 2 focused integration suite | `PASS`; 175 tests in 63.88 s |
| Complete local CPU suite | `PASS`; 456 tests in 79.34 s |
| Independent production-boundary audit | `PASS`; no open P0/P1 |
| `git diff --check` | `PASS` |

The final timings and exact committed tooling SHA are recorded by the tooling
commit and the subsequent execution record; test fixtures and synthetic
benchmark data are not research evidence.

## Commands actually run

```bash
sha256sum \
  /home/contra333/2026여름방학실험코드/fixed_readout_stage2_retrieval_90b3ced.receipt.json \
  /home/contra333/2026여름방학실험코드/fixed_readout_stage2_retrieval_90b3ced.filter \
  /home/contra333/2026여름방학실험코드/fixed_readout_stage2_retrieval_90b3ced.plan.jsonl

.venv/bin/python -m py_compile \
  scripts/benchmark_fixed_readout_stage2_raw_knn.py \
  scripts/extract_fixed_readout_stage2_model_metrics.py \
  scripts/fixed_readout_stage2_raw_knn_backend.py \
  scripts/materialize_fixed_readout_stage2_bundle.py \
  scripts/reduce_fixed_readout_stage2_gate.py \
  scripts/supervise_fixed_readout_stage2_cache_host.py

PYTHONPATH=.:src .venv/bin/python \
  scripts/supervise_fixed_readout_stage2_cache_host.py --help
PYTHONPATH=.:src .venv/bin/python \
  scripts/extract_fixed_readout_stage2_model_metrics.py --help
PYTHONPATH=.:src .venv/bin/python \
  scripts/reduce_fixed_readout_stage2_gate.py --help

TMPDIR=/tmp PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_fixed_readout_stage2_policy.py \
  tests/test_fixed_readout_stage2_remote_catalog.py \
  tests/test_fixed_readout_stage2_reuse_manifest.py \
  tests/test_fetch_fixed_readout_stage2_reuse.py \
  tests/test_fixed_readout_stage2_mechanism.py \
  tests/test_fixed_readout_stage2_raw_knn_backend.py \
  tests/test_fixed_readout_stage2_model_metrics.py \
  tests/test_fixed_readout_stage2_gate_reducer.py \
  tests/test_materialize_fixed_readout_stage2_bundle.py \
  tests/test_supervise_fixed_readout_stage2_cache_host.py

TMPDIR=/tmp PYTHONPATH=.:src .venv/bin/python -m pytest -q

git diff --check
```

## Explicitly not run at this tooling boundary

- selected production `.npy` parsing or any scientific statistic from those
  arrays;
- production raw-kNN cache construction;
- production 720-row candidate or 360-row diagnostic evidence extraction;
- Stage 2 reduction, focal selection, or scientific gate decision;
- checkpoint inference, protected dataset traversal, GPU work, model training,
  shared-prefix branching, fresh seeds, or CIFAR-100 replication;
- Hugging Face upload or mutation of any source artifact.

The exact status at this boundary is therefore:

```text
retrieval integrity and Stage 2 execution tooling: PASS
Stage 2 scientific mechanism gate: NOT_RUN
Stage 3 launch authorization: not established
```
