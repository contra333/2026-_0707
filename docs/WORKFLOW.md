# AI-Assisted Research Workflow

## Purpose

This document defines how research reasoning, implementation, server validation, and GitHub history are connected. The repository is the common handoff point between Chat, Work, desktop Codex, server Codex CLI, and human review.

The workflow is not a requirement to use every tool for every task. Use the smallest path that preserves research meaning, implementation scope, and validation evidence.

## Core principle

```text
Discuss and decide
→ record the decision once
→ implement within an explicit task scope
→ validate in the real environment
→ merge evidence with the code
→ interpret the result
```

Do not rewrite the same task independently for several AI tools. In the
standard workflow, the active GitHub Issue and repository documents are the
shared task contract.

Two direct-main paths are available:

- a prompt-only owner fast path triggered by the exact lowercase prefix
  `fast path`;
- a trivial documentation correction that satisfies every condition in the
  **Trivial documentation maintenance** section.

Neither path requires an Issue, task branch, or Pull Request.

## Roles

### Chat

Use Chat to:

- clarify the research question and hypothesis;
- compare interpretations;
- identify confounders and alternative explanations;
- discuss experimental meaning;
- interpret completed results.

Chat output is not authoritative until relevant decisions are recorded in the repository or active Issue.

### Work

Use Work when a task benefits from longer research or a finished artifact, such as:

- literature review;
- experiment-plan synthesis;
- comparison tables;
- reports, paper sections, or result summaries.

Work is optional for small implementation tasks. Its decisions must be reconciled into repository documents or the active Issue.

### Reference cards

Reference cards contain durable rules reused across tasks. Examples include:

- optimizer semantics;
- parameter-group policy;
- penultimate feature definition;
- architecture variants;
- future training or evaluation protocols.

Create or update a reference card only when the rule should apply beyond the current task. One-time instructions belong in an Issue.

### GitHub Issue

A GitHub Issue is the authoritative specification for one bounded task. It should state:

- goal and research reason;
- documents to read first;
- allowed and forbidden scope;
- acceptance criteria;
- required validation;
- expected outputs;
- explicitly excluded work.

Use one Issue per independently reviewable unit of work. Keep it open until the associated change is merged or the task is abandoned.

An Issue is not required for a triggered fast-path task or for a trivial
documentation correction that satisfies every condition in the **Trivial
documentation maintenance** section. If a correction changes meaning, status,
scope, instructions, commands, or policy and the prompt does not trigger the
fast path, it is not trivial and requires the normal Issue workflow.

### Desktop Codex

Use desktop Codex for repository-aware work that can be developed or checked locally:

- code and test drafts;
- configuration and documentation changes;
- static checks and available CPU tests;
- diff inspection and focused fixes.

Desktop Codex must read `AGENTS.md`, the active Issue, and referenced documents
before editing, unless it is performing a triggered fast-path task or making a
trivial documentation correction permitted by `AGENTS.md`. A fast-path task
still requires the core context and every task-relevant reference card, code
path, and test. Desktop Codex must not claim server or GPU validation that it
did not run.

### Server Codex CLI

Use server Codex CLI for environment-dependent implementation and validation:

- installed PyTorch and TorchVision behavior;
- CUDA and GPU execution;
- dataset paths and loaders;
- training and evaluation smoke tests;
- long-running experiments;
- checkpoint and artifact handling.

Server changes must be committed to the same task branch and included in the
same Pull Request whenever possible. For a triggered fast-path task, synchronize
each requested server to the exact direct-main commit and commit any requested
server fixes back to `main`.

### Pull Request

The Pull Request combines:

- actual code and document changes;
- links to the task Issue;
- acceptance-criteria status;
- commands actually run;
- environment information;
- failed or skipped validation;
- remaining limitations.

A Pull Request is not complete merely because code exists. It must make the
evidence and unverified parts visible. A Pull Request is not required for a
triggered fast-path task.

## Standard workflow cycle

### 1. Discuss the research decision

Use Chat to determine what is being tested, which variables change, which variables remain fixed, and what result would support or weaken the hypothesis.

### 2. Research or plan when needed

Use Work only when literature review, multi-document synthesis, or a finished plan materially improves the task. Small changes may move directly from Chat to an Issue.

### 3. Update durable semantics when necessary

Ask whether a decision should apply to future tasks.

- If yes, update or add a reference card.
- If no, keep it in the active Issue or in the explicit scope and repository
  artifacts of a triggered fast-path task.
- If uncertain in the standard workflow, start in the Issue and promote it to
  a reference card after reuse becomes clear.

### 4. Create the task Issue

Create an Issue using `.github/ISSUE_TEMPLATE/research_task.md`.

The Issue is the common instruction for all subsequent agents. Avoid creating separate, divergent prompts for desktop Codex and server Codex CLI.

Skip this step for a triggered fast-path task or for a trivial documentation
correction that meets every condition in the **Trivial documentation
maintenance** section.

### 5. Create a task branch and draft the change

Use a branch named after the Issue, for example:

```text
feat/issue-12-cifar-training
fix/issue-18-adamw-checkpoint
analysis/issue-25-geometry-metrics
```

Desktop Codex or another repository-aware agent implements only the Issue scope, adds tests, and records local validation.

A task branch is not used for a triggered fast-path task and is not required
for a permitted trivial documentation correction.

### 6. Validate on the department server

Check out the same branch on the server. Run the Issue's required commands in the real environment. Fix environment-dependent problems on the same branch, commit them, and push them.

Do not run a full experiment when the Issue requests only a smoke test. Do not claim a test passed when it was skipped or blocked.

### 7. Open or update the Pull Request

Use `.github/pull_request_template.md`. Link the Issue with `Closes #<issue-number>` when merge should close it.

Include desktop and server validation separately. State the exact environment and unverified items.

A Pull Request is not required for a triggered fast-path task or for a
permitted trivial documentation correction.

### 8. Review, merge, and close

Review the diff against:

1. reference cards;
2. Issue scope and acceptance criteria;
3. tests and server evidence;
4. research-variable isolation.

Merge only after required validation is satisfied or the exception is explicitly accepted. Close the Issue if GitHub does not close it automatically.

### 9. Interpret results and start the next cycle

Use Work to assemble result artifacts when useful, then use Chat to examine
conclusions, confounders, and the next experiment. In the standard workflow,
record the resulting task as a new Issue rather than extending a completed
Issue indefinitely. A new fast-path task instead requires a new `fast path`
prefix.

## Minimal paths

Not every task needs the full nine-stage route.

### Prompt-only owner fast path

The only trigger is the exact lowercase prompt prefix `fast path`. The first
non-empty line must start with those exact words, followed by whitespace or
end-of-line. The task text follows the prefix.

Apply this case-sensitive conceptual regular expression to the first non-empty
line before choosing a workflow:

```text
^fast path(?:[ \t]|$)
```

Do not use a broad string-prefix check. A colon immediately after `path` fails
the expression and uses the standard Issue workflow.

| First non-empty line | Fast path |
| --- | --- |
| `fast path do the task` | yes |
| `fast path` | yes; ask for the missing task |
| `fast path: do the task` | no |
| `FAST PATH do the task` | no |
| `/fast do the task` | no |
| `fast-path do the task` | no |

```text
fast path seed=0 run을 서버별로 분배하고 main에 반영한 뒤
학습과 Hugging Face 업로드까지 끝내
```

`FAST PATH`, `fast path:`, `/fast`, `fast-path`, and `FAST:` do not trigger
this workflow. There is no shell command, alias, executable, environment
variable, configuration toggle, or persistent activation state.

The fast path is a per-task workflow authorization:

- skip Issue creation, task-branch creation, Pull Request, and merge;
- work directly on a safely synchronized `main`;
- allow the explicit task to change documentation, configuration, scripts,
  core code, tests, and durable research protocol;
- update any affected authoritative repository documents in the same task;
- run validation proportional to the change;
- stage only task files, commit intentionally, and push `main`;
- synchronize requested servers to the exact commit SHA;
- perform explicitly requested training, transfer, and Hugging Face upload
  steps without an extra project-workflow confirmation;
- report exact `PASS`, `FAILED`, `NOT_RUN`, and `BLOCKED` evidence.

```text
fast path <bounded task>
→ inspect and fast-forward main
→ edit only the task scope
→ validate proportionally
→ commit and push main
→ synchronize requested servers to the same SHA
→ complete requested training, transfer, and upload
→ report evidence
```

The authorization remains active for follow-up corrections needed to finish
the same unfinished task. It expires at completion, and a new unrelated task
requires a new `fast path` prefix. If no task follows the prefix, ask for the
task without creating persistent state.

The prefix does not infer unrequested destructive actions, protected-split
access, secret disclosure, artifact overwrite, force-push, or termination of
another user's process. Such actions must be explicit and remain subject to
the applicable safety and evidence rules. Unrelated worktree files must be
preserved and excluded from staging.

For multi-server execution, use `git pull --ff-only origin main` and verify the
exact commit SHA before starting. Record the Git SHA, resolved configuration,
seed, host/GPU, environment, and artifact path. For Hugging Face transfer,
confirm the source and remote path, inspect a dry run, upload without
unrequested deletion, and verify the remote listing.

### Trivial documentation maintenance

A documentation-only correction may be committed directly to `main` without an Issue, task branch, or Pull Request only when **all** of the following are true:

- it only fixes an obvious typo, grammar, punctuation, formatting error, or broken documentation link;
- it does not change research semantics, implementation instructions, project scope, validated status, acceptance criteria, commands, configuration behavior, or experimental policy;
- it does not modify Python code, tests, configuration files, dependencies, GitHub templates, or generated artifacts;
- the intended correction is unambiguous and small enough to inspect directly;
- the commit message clearly identifies it as a documentation correction.

```text
Direct inspection
→ direct documentation commit to main
```

A change to `AGENTS.md`, this workflow, `docs/STATUS.md`, or a reference card is not trivial when it changes a rule, project phase, implementation status, or research meaning. When uncertain, use the normal Issue, branch, and Pull Request path.

### Non-trivial documentation or test change

```text
Chat or direct inspection
→ Issue
→ Desktop Codex
→ Pull Request
→ merge
```

### GPU-dependent implementation

```text
Chat
→ Issue
→ Desktop Codex draft
→ Server Codex CLI validation/fix
→ Pull Request
→ merge
```

### Literature-driven experiment design

```text
Chat
→ Work
→ reference-card update if durable
→ Issue
→ implementation and validation
→ Pull Request
→ Work/Chat interpretation
```

## Issue versus reference card

Use this test:

> Should this rule still be mandatory for a different task next month?

- No: put it in the Issue.
- Yes: put it in a reference card and link that card from the Issue.

Examples:

- "Add a CIFAR-10 loader in this task" → Issue.
- "All experiment runs store config, seed, commit SHA, and environment" → durable training-protocol reference card.
- "Do not implement OOD evaluation in this PR" → Issue.
- "Penultimate features are taken immediately before `model.classifier`" → architecture reference card.

## Prompt pattern for repository-aware agents

Use a short prompt that points to the shared contract:

```text
Read AGENTS.md, docs/PROJECT_CONTEXT.md, docs/WORKFLOW.md,
docs/STATUS.md, the active GitHub Issue, and every reference card
listed in that Issue.

Work only within the Issue's allowed scope. Do not silently resolve
conflicts between the Issue, reference cards, code, and official
semantics. Report the conflict.

After editing, report changed files, acceptance criteria addressed,
commands actually run, failures, skipped checks, and server validation
still required.
```

For a trivial documentation correction permitted by `AGENTS.md`, identify the exact correction, verify that it changes no meaning or policy, and keep the direct `main` commit limited to that correction.

For an owner-authorized direct-main task, use:

```text
fast path <bounded task and requested completion steps>
```

Do not replace the prefix with a terminal command or a separate activation
prompt.

## Context use in ChatGPT Projects and local folders

Preferred method:

1. clone or connect the repository;
2. let the AI read the current files from that repository;
3. identify the active Issue unless the task triggers the fast path or is a
   permitted trivial documentation correction;
4. avoid maintaining separate edited copies.

When a ChatGPT Project cannot read GitHub directly, upload temporary snapshots of the context files listed in `docs/PROJECT_CONTEXT.md`. Refresh them after relevant repository changes. Do not treat the uploaded copy as the master document.

For a local Codex project, keep these documents inside the cloned repository. There is normally no need to copy them to a second local context folder. `AGENTS.md` serves as the entry point and links to the rest.

## Completion rule

For a standard task, a cycle is complete when:

- the Issue scope has been implemented or explicitly rejected;
- required validation has been run and reported;
- the Pull Request is merged or closed;
- repository status is updated when the project phase changed;
- new research questions are placed in a new Issue.

For a permitted trivial documentation correction, the maintenance action is complete when the narrow correction is committed to `main` and directly inspected to confirm that no meaning, status, or policy changed.

For a triggered fast-path task, the cycle is complete when the scoped change
is validated proportionally, committed and pushed to `main`, every explicitly
requested server/action is completed or truthfully reported as
`FAILED`/`NOT_RUN`/`BLOCKED`, and unrelated files remain untouched.
