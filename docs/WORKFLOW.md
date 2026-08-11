# Working workflow

이 문서는 사람이 보는 짧은 작업 안내다. Agent의 정확한 운영 규칙은
[`AGENTS.md`](../AGENTS.md)를 따른다.

## 기본 원칙

```text
질문과 결정을 정한다
→ 한 권위 문서에 한 번 기록한다
→ 필요한 범위만 구현한다
→ 위험에 비례해 검증한다
→ 결과와 다음 작업을 갱신한다
```

같은 결정을 README, STATUS, 여러 card, test에 반복하지 않는다. Historical 문서는
현재 상태를 설명하기 위해 고치지 않는다.

## 작업 경로

### 문서-only

논문 framing, active protocol, paper outline, 문장과 링크만 바뀌고 code/config/schema,
generated artifact, protected data 또는 외부 실행을 건드리지 않으면 direct-main을
사용한다.

```text
main 동기화 → 대상 문서 수정 → link/YAML/diff 검사 → commit → push
```

연구 설계는 `reference_cards/13_active_paper_protocol.md`만 바꾼다. 실제 완료 상태나
blocker가 달라졌을 때만 `STATUS.md`도 바꾼다.

### 코드·config·실행

Runtime behavior, API, test, schema, experiment config, dependency 또는 artifact policy가
바뀌면 bounded GitHub Issue와 branch/PR을 사용한다.

```text
Issue → 관련 source만 읽기 → 구현 → focused validation → PR → merge
```

전체 CPU suite는 shared runtime/API 변경이나 production launch 전에 실행한다.
GPU/server 검증은 해당 환경이 필요한 변경에만 수행한다.

### Fast path

Owner가 정확한 `fast path` prefix를 사용하면 `AGENTS.md` 규칙에 따라 direct-main으로
처리한다. 범위와 안전 경계는 일반 작업과 같다.

## Research execution

연구 실행은 다음 경계를 지킨다.

- config/seed/checkpoint/dataset identity를 실행 전에 고정한다.
- smoke test와 partial output은 readiness evidence일 뿐 연구 결과가 아니다.
- server별 completion을 확인한 뒤 checkpoint별 결과를 검증하고 seed aggregate를
  만든다.
- protected OOD는 fitting이나 model selection에 사용하지 않는다.
- 실패 결과도 보존하며 post-hoc detector나 checkpoint로 구조하지 않는다.

## Artifact policy

- Git: code, compact config, active contracts, small fixtures, result summary, URI/SHA
- HF: checkpoint, feature/score bundle, 생성된 대형 표·그림, checksum-addressed archive
- local ignored `artifacts/`: 재생성 가능한 작업 출력
- `docs/history/`: 읽기 전용 과거 기록

대형 생성물을 Git으로 되돌리지 않는다. HF 업로드는 새 hash-addressed 경로만 사용하고
기존 artifact를 덮어쓰거나 삭제하지 않는다.

## Completion

작업이 끝났다는 말은 terminal 상태와 실제 검증이 있을 때만 사용한다. Handoff에는
변경 결과, commit/PR, 실행한 검사, artifact URI, 남은 blocker만 적는다.
