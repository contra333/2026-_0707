# AGENTS.md

## Mandatory reading order

Before planning or editing, read:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/WORKFLOW.md`
3. `docs/STATUS.md`
4. the active GitHub Issue
5. every reference card listed in that Issue
6. relevant code and tests

If no active Issue is identified, do not start a repository-changing task. Ask for or locate the bounded task specification first.

## Project scope

This repository is for optimizer-geometry experiments. The repository is the source of truth for code, research semantics, task scope, and validation evidence.

Use:

- `docs/reference_cards/01_optimizers.md` for optimizer semantics;
- `docs/reference_cards/02_architectures.md` for model and penultimate-feature semantics;
- future reference cards for other durable experiment protocols.

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

## Scope control

- Modify only files and behaviors allowed by the active Issue.
- Do not perform adjacent refactors, renames, formatting sweeps, or API changes unless required by the Issue.
- Do not add a new research variable implicitly.
- Architecture and dataset variants must be explicit in configuration.
- Do not treat toy fixtures, smoke tests, or partial runs as research evidence.
- Do not add training, dataset, OOD, geometry, or GPU infrastructure unless the active Issue allows it.

## Existing safety rules

- If PyTorch official optimizer semantics, the optimizer reference card, and implementation requirements conflict, stop and report instead of resolving silently.
- Architecture implementations must follow `docs/reference_cards/02_architectures.md`.
- If architecture documentation, the active Issue, code, and reference source conflict, stop and report instead of resolving silently.
- All optimizers must use the shared parameter-group builder.

## Branch and Pull Request policy

- Do not push task work directly to `main`.
- Use one bounded branch and Pull Request per Issue when practical.
- Server fixes and validation changes should be committed to the same task branch.
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

Do not maintain an independent edited copy of repository context documents when the repository is accessible. External ChatGPT Projects or local context folders may contain temporary snapshots, but the repository remains authoritative. Decisions made in Chat or Work must be recorded in an Issue, reference card, status document, or Pull Request before implementation relies on them.