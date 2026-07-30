# Issue 49 Seed 1/2 Role and Pair Follow-up Execution

## Scope and current verdict

Issue #49 executes only the frozen 30-row follow-up plan: seed 1/2 role
replication, the missing Adam pair rows, and the six SGDW pair rows. It does
not extend the grid or access ID-test, OOD, geometry/Neural Collapse, feature,
or detector evidence.

Current verdict on 2026-07-30: **IN_PROGRESS**. All implementation, local
validation, three-host preflight, host-shard dry-runs, and one-epoch smokes
passed. The 30 production rows are running as detached server processes.
Terminal results, Hugging Face uploads, 42-row aggregation, and multi-seed
statistics remain `NOT_RUN` until the production rows complete.

## Frozen identities

| Item | Value |
| --- | --- |
| Production Git SHA | `3556841340e6f6b92782af045ed4a468e6e271bd` |
| Role-freeze hash | `fdf67c1184abc489542ca64cad2410ff38aa816acb1e9e5289d60461600373fa` |
| Follow-up-plan hash | `3a3b00dbcf0ee3dc20c0959013665bf4243a2644e21f4552a831e0fdaed69264` |
| Execution ID | `role_pair_followup_20260730` |
| Execution plan | `configs/studies/wrn28_10_optimizer_hpo_v1_2/followup_execution.yaml` |
| Pull Request | #50 |

The execution-plan validator confirms all 30 scheduled trial IDs exactly once,
host counts 13/10/7, every optimizer family on every host, all 14
configurations rotating across the three hosts over seeds 0/1/2, and every
declared matched-pair co-location.

## Host assignment and external paths

| Host | GPU/runtime | New rows | Concurrency | Production artifact root | HF destination |
| --- | --- | ---: | ---: | --- | --- |
| `curie` | RTX A5000 x4 | 13 | 4 | `/home/ghjin/0727ICLR실험/issue49_artifacts/3556841/curie/production` | `servers/curie/role_pair_followup_20260730` |
| `lise` | RTX A5000 x2 | 10 | 2 | `/home/ghjin/0727ICLR실험/issue49_artifacts/3556841/lise/production` | `servers/lise/role_pair_followup_20260730` |
| `precision_medicine` | RTX A6000 x4 | 7 | 4 | `/mnt/drive/lab1/oge/artifacts/issue49/3556841/precision_medicine/production` | `servers/precision_medicine/role_pair_followup_20260730` |

The separate `control/` sibling on each host stores the supervisor PID, atomic
state, log, future upload dry-run/plan, and completion evidence. It is not a
training-artifact source.

## Validation evidence before production

Local WSL:

- focused execution/selection/orchestration/supervisor tests: **PASS**,
  `44 passed`;
- full regression suite: **PASS**, `186 passed, 1 warning`; the warning is
  the recorded local CUDA driver/runtime mismatch and is not a server CUDA
  result;
- `git diff --check`: **PASS**.

Each department server at the production SHA:

- clean detached Git identity: **PASS**;
- `pip check`: **PASS**;
- full suite: **PASS**, `187 passed` on each host;
- committed 45k/5k/10k membership and 60,001-row manifest regeneration:
  **PASS**, byte-identical;
- Python 3.11.9, PyTorch 2.5.1+cu121, TorchVision 0.20.1+cu121, CUDA 12.1,
  cuDNN 90100: **PASS**;
- common numerical policy: **PASS**; FP32, matmul precision `highest`, CUDA
  matmul TF32 disabled, cuDNN TF32 enabled, cuDNN benchmark/deterministic and
  deterministic algorithms disabled;
- host-shard dry-run: **PASS**, exact counts `curie=13`, `lise=10`,
  `precision_medicine=7`;
- one-row/one-epoch follow-up smoke: **PASS** on all three hosts,
  `smoke_only_completed`, 1/1, valid checkpoints, ID-test deferred.

## Detached production launch

Production launched around 2026-07-30 14:51 KST with
`nohup setsid flock -n`. Fresh SSH connections observed each supervisor with
`PPID=1`, its own session ID, and `TTY=?`, so the process lifetime does not
depend on the launching SSH, WSL computer, or Codex session.

| Host | Launcher PID | First observed state | First active child count |
| --- | ---: | --- | ---: |
| `curie` | `1122705` | `RUNNING` | 4 |
| `lise` | `1513090` | `RUNNING` | 2 |
| `precision_medicine` | `3986916` | `RUNNING` | initialization pending at first snapshot |

Later fresh connections observed epoch 1 on all active `curie` and `lise`
children and epoch 2 on all four `precision_medicine` children. The
intermittent precision SSH refusal therefore did not stop its detached
supervisor or training processes.

The supervisor will upload only after all assigned rows are epoch-200
`completed` and pass the independent artifact gate. It creates a whole-tree
SHA-256 manifest, records a no-delete dry run, saves and applies an HF sync
plan, verifies the remote path/size listing, records Xet hashes, and uploads
`REMOTE_COMPLETE.json` last. Training or integrity failure prevents upload.

## Preserved setup observations

- The first deployment command used an incorrectly expanded full SHA. The
  `git cat-file` gate rejected it before checkout, smoke, or production. The
  real SHA was read with `git rev-parse HEAD` and all hosts were then deployed
  successfully.
- The first `precision_medicine` production-launch SSH attempt returned
  `Connection refused`. No process started in that attempt. The fixed host
  assignment was retained; a later connection succeeded and the detached
  supervisor was verified after reconnect.

## Completion boundary

The following remain `NOT_RUN` at this snapshot:

- 30/30 terminal epoch-200 integrity validation;
- the three HF uploads and `REMOTE_COMPLETE` markers;
- the combined 12 reused plus 30 new record set;
- per-seed values, mean/sample-SD, and paired differences;
- final context/status synchronization and PR readiness;
- every protected evaluation and all metric/feature implementation.

Do not interpret GPU activity or partial histories as a completed experiment.
The final report must replace this section with terminal and remote evidence or
an explicit failure/blocker.
