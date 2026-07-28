# AGENTS.md

## Prompt-only owner fast path

The only fast-path trigger is the exact lowercase prompt prefix `fast path`.
A prompt triggers the fast path only when its first non-empty line starts with
the exact words `fast path` and the next character is whitespace or the line
ends. The task text follows that prefix.

Before choosing the workflow, apply this case-sensitive conceptual regular
expression to the first non-empty line:

```text
^fast path(?:[ \t]|$)
```

Do not use a broad string-prefix check. In particular, a colon immediately
after `path` fails the expression and must use the standard workflow.

| First non-empty line | Fast path |
| --- | --- |
| `fast path do the task` | yes |
| `fast path` | yes; ask for the missing task |
| `fast path: do the task` | no |
| `FAST PATH do the task` | no |
| `/fast do the task` | no |
| `fast-path do the task` | no |

`FAST PATH`, `fast path:`, `/fast`, `fast-path`, and `FAST:` are not aliases
and do not trigger this workflow. Do not require a shell command, slash
command, executable, alias, environment variable, configuration toggle, or
separate activation step.

For one bounded fast-path task:

- do not create a GitHub Issue, task branch, or Pull Request;
- work directly on `main`, first confirming that local `main` can be safely
  fast-forwarded to `origin/main`;
- read the core context documents and every relevant reference card, code path,
  and test, but do not require an active Issue;
- allow the explicitly requested scope to include documentation,
  configuration, scripts, core code, tests, and durable research protocol;
- update the relevant authoritative repository documents in the same task when
  research semantics or workflow rules change;
- run validation in proportion to the risk of the change;
- stage only the task files, commit intentionally, and push `main`;
- complete explicitly requested server synchronization, training, transfer,
  and Hugging Face upload steps without another project-workflow confirmation;
- preserve unrelated files and report exact `PASS`, `FAILED`, `NOT_RUN`, and
  `BLOCKED` evidence.

The authorization covers the named task and the steps needed to complete it,
including follow-up corrections during that unfinished task. It expires when
the task is complete; a new unrelated task requires a new `fast path` prefix.
If the prompt contains only `fast path` without a task, ask for the task rather
than creating persistent fast-path state.

The prefix does not silently authorize unrequested deletion, force-push,
artifact overwrite, protected-split access, secret disclosure, or termination
of another user's process. Destructive or protected actions must be explicit
in the task and remain subject to the applicable safety and evidence rules.
Never fabricate validation or research evidence.

## Mandatory reading order

For the standard Issue workflow, before planning or editing, read:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/WORKFLOW.md`
3. `docs/STATUS.md`
4. the active GitHub Issue
5. every reference card listed in that Issue
6. relevant code and tests

For a triggered fast-path task, read items 1-3 and every reference card, code
path, and test relevant to the explicit task; items 4-5 are not required.

If no active Issue is identified and the prompt does not trigger the fast path,
do not start a repository-changing task unless the change satisfies every
condition in the **Trivial documentation exception** below. Otherwise, ask for
or locate the bounded task specification first.

## Project scope

This repository is for optimizer-geometry experiments. The repository is the source of truth for code, research semantics, task scope, and validation evidence.

Use:

- `docs/reference_cards/01_optimizers.md` for optimizer semantics;
- `docs/reference_cards/02_architectures.md` for model and penultimate-feature semantics;
- `docs/reference_cards/04_openood_v1_5_protocol.md` for the OpenOOD v1.5-aligned
  CIFAR-10 data and OOD-evaluation protocol;
- `docs/reference_cards/05_training_protocol.md` for classifier training,
  checkpoint, resume, and run-artifact semantics;
- `docs/reference_cards/06_feature_ood_detectors.md` for durable feature-based
  OOD detector naming, fitting, scoring, numerical-stability, and variant
  semantics;
- `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md` for the
  WRN-28-10 four-optimizer HPO, comparison, seed, checkpoint, and study
  provenance protocol;
- `docs/reference_cards/08_raw_feature_artifact_contract.md` for deterministic
  checkpoint-feature artifacts, provenance, checksums, and protected-split
  authorization;
- `docs/reference_cards/09_core_representation_metrics.md` for the frozen
  confirmatory geometry, logit-control, and low-complexity detector panel;
- `docs/reference_cards/10_optimizer_grid_literature_anchors.md` for the
  literature and project-judgment anchors behind protocol v1.2 grids and the
  architecture lineup;
- future task-specific reference cards for other durable experiment protocols.

`docs/reference_cards/03_architecture_implementation_checklist.md` is historical context for the first architecture implementation. Its one-time task scope is not the current active task.

## Instruction priority

When instructions conflict, use this order:

1. verified official semantics or primary reference sources named by the project;
2. durable project reference cards;
3. the active GitHub Issue and its acceptance criteria;
4. this file and workflow rules;
5. existing code behavior;
6. ad hoc prompt wording.

Do not silently resolve a material conflict. Stop the affected work and report the conflicting sources.

The fast-path prefix changes the workflow, not the source-of-truth priority.
When a fast-path task explicitly changes a durable rule, update the relevant
reference card or repository policy in the same direct-main task before
implementation relies on the new rule.

## Scope control

- Modify only files and behaviors allowed by the active Issue or explicitly
  named by the triggered fast-path task.
- Do not perform adjacent refactors, renames, formatting sweeps, or API changes
  unless required by the active Issue or explicitly requested by the
  fast-path task.
- Do not add a new research variable implicitly.
- Architecture and dataset variants must be explicit in configuration.
- Do not treat toy fixtures, smoke tests, or partial runs as research evidence.
- Do not add training, dataset, OOD, geometry, or GPU infrastructure unless the
  active Issue allows it or the fast-path task explicitly requests it.

The first two rules do not require an Issue for a triggered fast-path task or
for a change that satisfies the **Trivial documentation exception**. A trivial
documentation change must remain strictly limited to the stated correction.

## Existing safety rules

- If PyTorch official optimizer semantics, the optimizer reference card, and implementation requirements conflict, stop and report instead of resolving silently.
- Architecture implementations must follow `docs/reference_cards/02_architectures.md`.
- If architecture documentation, the active Issue, code, and reference source conflict, stop and report instead of resolving silently.
- All optimizers must use the shared parameter-group builder.

## Trivial documentation exception

A documentation-only correction may be committed directly to `main` without a GitHub Issue, task branch, or Pull Request only when **all** of the following are true:

- it only fixes an obvious typo, grammar, punctuation, formatting error, or broken documentation link;
- it does not change research semantics, implementation instructions, project scope, validated status, acceptance criteria, commands, configuration behavior, or experimental policy;
- it does not modify Python code, tests, configuration files, dependencies, GitHub templates, or generated artifacts;
- the intended correction is unambiguous and small enough to inspect directly;
- the commit message clearly identifies it as a documentation correction.

A change to `AGENTS.md`, `docs/WORKFLOW.md`, `docs/STATUS.md`, or a reference card is **not** trivial when it changes a rule, workflow, project phase, implementation status, or research meaning. When uncertain, use an Issue, branch, and Pull Request.

## Branch and Pull Request policy

- Do not push standard task work directly to `main`, except for a triggered
  fast-path task or a change that satisfies every condition in the **Trivial
  documentation exception**.
- Use one bounded branch and Pull Request per Issue when practical.
- For a standard Issue task, commit server fixes and validation changes to the
  same task branch. For a fast-path task, commit them directly to `main`.
- Link the Pull Request to the Issue with `Closes #<issue-number>` when merge should close it.
- Do not mark acceptance criteria complete without evidence.

## Validation reporting

At the end of work, report:

1. changed files;
2. acceptance criteria addressed;
3. exact commands actually run;
4. passed and failed checks;
5. requested checks not run and why;
6. environment limitations;
7. server validation still required;
8. generated artifacts and their locations;
9. remaining unverified assumptions.

Never claim that a test, training run, GPU check, or external behavior succeeded unless it was actually run and observed.

## Context synchronization

Do not maintain an independent edited copy of repository context documents when
the repository is accessible. External ChatGPT Projects or local context
folders may contain temporary snapshots, but the repository remains
authoritative. Decisions made in Chat or Work must be recorded in an Issue,
reference card, status document, Pull Request, or the direct-main commit and
affected repository artifacts of a triggered fast-path task before
implementation relies on them.

For a triggered fast-path task, the direct-main commit and any updated
reference card, status document, configuration, validation record, or run
artifact provide that authoritative repository record; an Issue or Pull
Request is not additionally required.
