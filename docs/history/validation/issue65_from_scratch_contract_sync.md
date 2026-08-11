# Issue #65: from-scratch paired-trajectory contract synchronization

Date: 2026-08-11

Branch: `docs/issue-65-paired-trajectory-contract`

Status: documentation and contract synchronization complete; fresh research
execution `NOT_RUN`

## Scope completed

- Replaced Card 13 v3 with the active paired-trajectory Card 13 v4 while
  retaining the card number.
- Replaced the shared-prefix main experiment with coupled, decoupled, and
  zero-decay runs trained from epoch 0 using the same initialization and data
  stream.
- Froze the Adam 2 x 2 LR/nominal-WD design, 36-run allocation, nine-run SGDM
  control, checkpoint/time/depth design, epoch-200 primary endpoint, and
  secondary `best_val`/epoch-300 controls.
- Added the concrete update -> representation geometry -> score component ->
  exact pair-order -> channel-matched attenuation analysis chain.
- Synchronized the manuscript outline, project context/status, README,
  architecture/training/metric cards, literature anchors, source lock, local
  draft manifest, and document contract tests.
- Kept Card 12, its failed radial Stage-2 evidence, v3 historical discovery,
  historical validation records, and `fork_from_prefix` implementation
  unchanged. The fork is now optional follow-up infrastructure.

No code, training configuration, protected dataset, remote artifact, server
run, GPU job, checkpoint, feature cache, or research result was created or
modified by this Issue.

## Validation

### Passed

```text
TMPDIR=/tmp/oge_issue65_pytest PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/pytest -q -s \
  tests/test_research_contract_v4_docs.py \
  tests/test_research_contract_v2_docs.py
10 passed in 0.08s

PYTHONPATH=.:src TMPDIR=/tmp/oge_issue65_full_path \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q -s
477 passed, 1 warning in 81.71s

.venv/bin/python -m compileall -q src scripts tests
PASS

git diff --check
PASS
```

The one pytest warning is a local CUDA driver/runtime warning emitted while a
CPU test checks CUDA availability. It did not fail or skip the suite.

### Failed environment invocations and resolution

The first focused pytest invocation used the repository `.venv` without an
explicit temporary directory and failed before running tests because pytest's
capture temporary file disappeared (`FileNotFoundError`). Re-running with a
task-specific `/tmp` directory and capture disabled passed.

The first full-suite invocation omitted `PYTHONPATH=.:src` and stopped during
collection with four `ModuleNotFoundError: scripts` errors. Re-running the
same suite with the repository import path explicitly set passed all 477
tests.

### Not available

The Issue checklist named `scripts/validate_repo.py`, but no such tracked file
exists in this repository (`git ls-files scripts` and `rg --files` confirmed
its absence). It was therefore `NOT_RUN`, not reported as passed. The complete
pytest suite, compileall, document-link tests, source-lock tests, and
`git diff --check` supply the available local validation.

## Remaining work and evidence boundary

- v4 runner/config support for the expanded snapshots, multi-depth feature
  taps, update logging, run manifests, and analysis artifacts is not yet
  implemented.
- ID/practical margins, onset thresholds, spectral-band boundaries, power/MDE,
  multiplicity, and protected-OOD go/no-go rules require a pre-execution
  addendum.
- All fresh training, protected OOD evaluation, and replication/scale arms are
  `NOT_RUN`.
- The preserved untracked historical drafts and ZIPs were not edited, moved,
  deleted, staged, or uploaded.
