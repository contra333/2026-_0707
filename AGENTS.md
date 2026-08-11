# Repository operating rules

## Personal defaults

- 기본 답변은 한국어로 한다. 고유명사, API, 코드, 명령어는 필요한 경우
  영어 원문을 유지한다.
- Windows와 WSL은 별도 환경으로 취급하고 경로 문법을 섞지 않는다.
- 기존 사용자 파일, dirty worktree, 실행 중인 process를 임의로 변경하지
  않는다.

## Source of truth

역할은 중복하지 않는다.

- `docs/PROJECT_CONTEXT.md`: 변하지 않는 연구 목표와 문서 지도
- `docs/STATUS.md`: 현재 완료 상태, blocker, 바로 다음 작업
- `docs/reference_cards/13_active_paper_protocol.md`: 현재 논문 실험 계약의
  유일한 권위
- `docs/paper/intervention_supporting_theory_outline.md`: 사람이 읽는 논문 서사;
  수치와 실행 규칙은 active protocol을 따른다.
- 그 밖의 reference card: 해당 component를 실제로 변경할 때만 읽는 구현 계약
- `docs/history/`: 읽기 전용 과거 기록; 현재 동작이나 상태의 권위가 아님

외부 논문과 공식 구현은 `docs/sources.lock.yaml`에 기록한다. Local document는
그 파일에 다시 등록하지 않는다.

## Reading order

코드, config, schema, training, evaluation 또는 server 작업은 다음만 읽는다.

1. `docs/PROJECT_CONTEXT.md`
2. `docs/STATUS.md`
3. 논문 실험을 바꾸는 경우 active paper protocol
4. 변경할 component의 reference card, code, tests
5. standard workflow라면 active GitHub Issue

모든 reference card와 historical validation log를 선제적으로 읽지 않는다.
문서-only 작업은 대상 문서와 직접 연결된 source만 읽는다.

## Workflow choice

### Documentation-only direct main

Markdown 또는 외부-source lock만 바꾸며 code, test, experiment config, schema,
generated artifact, protected data, server state를 건드리지 않는 작업은 연구 의미가
바뀌더라도 Issue/branch/PR 없이 `main`에 직접 commit하고 push할 수 있다.

- 연구 설계는 active paper protocol 한 곳에만 기록한다.
- 실행 상태가 실제로 바뀐 경우에만 `STATUS.md`도 수정한다.
- README, PROJECT_CONTEXT, 다른 card, test에 같은 문장을 복제하지 않는다.
- prose 문구 존재 여부를 고정하는 test나 별도 validation report를 만들지 않는다.
- 검증은 local Markdown link, 변경한 YAML parse, `git diff --check`로 제한한다.

### Standard Issue workflow

Python, test, config, schema, dependency, generated-output policy 또는 runtime behavior를
바꾸면 하나의 bounded Issue와 branch/PR을 사용한다. 변경한 code path에 맞는 focused
test를 먼저 실행한다. Shared API/runtime을 바꾸거나 production launch 전일 때만 전체
CPU suite를 요구한다.

### Owner fast path

첫 non-empty line이 case-sensitive 정규식 `^fast path(?:[ \t]|$)`와 일치하면
Issue/branch/PR 없이 `main`에서 요청 범위를 끝까지 수행한다. `fast path:`,
`FAST PATH`, `/fast`, `fast-path`는 해당하지 않는다. 이 prefix는 deletion,
force-push, protected access, overwrite 또는 다른 사용자의 process 종료를 자동 승인하지
않는다.

## Scientific boundaries

- 계획, fixture, smoke test, partial run을 연구 결과로 표현하지 않는다.
- checkpoint/seed 단위 결과를 먼저 검증하고 그 뒤 aggregate한다.
- `last.pt`와 `best_val.pt`, ID-positive와 OpenOOD-compatible metric convention을
  섞지 않는다.
- protected OOD 결과를 training, fitting, checkpoint selection에 사용하지 않는다.
- PyTorch 공식 optimizer semantics, active reference card, 구현이 충돌하면 멈추고
  충돌을 보고한다.
- architecture와 dataset variant는 config에 명시한다.

## Artifacts and history

- 재생성 가능한 대형 표, 그림, checksum catalog는 Git에 commit하지 않는다.
- 생성물은 ignored `artifacts/` 또는 repository 밖에서 만들고, 공유할 결과는
  hash-addressed HF artifact로 보존한다.
- Git에는 generator, 작은 fixture test, 요약, artifact URI와 SHA만 둔다.
- `docs/history/`는 수정하지 않는다. 과거 tree 복구는 tag와 Git history를 사용한다.
- history rewrite와 force-push는 명시적 별도 승인 없이는 금지한다.

## Editing and safety

- 검색은 `rg`/`rg --files`, tracked edit은 `apply_patch`를 우선한다.
- unrelated refactor, formatting sweep, 새 연구 변수를 추가하지 않는다.
- deletion 전에 정확한 대상과 복구 지점을 확인한다. 사용자 home, repository root,
  unresolved glob을 destructive target으로 사용하지 않는다.
- stage는 명시적 task file만 하고 unrelated/untracked file을 포함하지 않는다.
- server/GPU/HF 작업은 실제 관찰한 host, SHA, status만 보고한다.

## Completion report

결과를 먼저 말하고 다음만 간단히 보고한다.

- 변경 범위와 commit/PR 또는 direct-main SHA
- 실제 실행한 검증과 PASS/FAILED
- 요청 범위 중 실행하지 못한 항목과 이유
- 연구 실행이라면 artifact 위치와 남은 검증

관련 없는 `NOT_RUN` 목록이나 과거 검증 내역을 반복하지 않는다.
