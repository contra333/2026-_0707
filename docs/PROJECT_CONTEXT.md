# Project context

## Research objective

이 저장소는 다음 연결을 연구한다.

```text
training rule
→ representation geometry over time and depth
→ fixed Mahalanobis score components
→ ID/OOD pair ordering and reliability
```

현재 논문은 같은 initialization과 data stream에서 coupled, decoupled,
zero-decay training을 처음부터 비교한다. 알려진 MD--Marginal--RMD와
size--stretch 분해는 측정도구이며 새로운 detector나 decomposition으로 주장하지 않는다.

## Active sources

- [`STATUS.md`](STATUS.md): 지금 완료된 것, blocker, 다음 작업
- [`13_active_paper_protocol.md`](reference_cards/13_active_paper_protocol.md):
  실행 가능한 논문 실험 계약의 유일한 권위
- [`intervention_supporting_theory_outline.md`](paper/intervention_supporting_theory_outline.md):
  교수님과 연구자가 읽는 논문 서사
- [`sources.lock.yaml`](sources.lock.yaml): 외부 논문과 공식 구현 provenance
- [`history/README.md`](history/README.md): 과거 protocol, validation, 외부 artifact 지도

Optimizer, architecture, dataset, detector 또는 metric 구현을 실제로 바꿀 때만 해당
reference card를 추가로 읽는다. Historical validation log는 현재 작업의 필수 입력이
아니다.

## Stable boundaries

- Repository code/config/active protocol이 구현과 실험 의미의 source of truth다.
- Chat, Notion, copied Markdown은 토론과 표현을 위한 interface이며 실행 전에 active
  protocol과 합치시킨다.
- Completed historical grid와 component analysis는 discovery evidence다. Fresh paired
  trajectory를 대신하지 않는다.
- Fixture, smoke test, planned experiment, running process는 research result가 아니다.
- Generated tables, figures, checkpoints, feature/score arrays는 Git 밖의 hash-addressed
  artifact로 보존한다.
