# Project context

## Research objective

이 저장소는 다음 연결을 연구한다.

```text
training rule
→ representation geometry over time and depth
→ fixed Mahalanobis score components
→ ID/OOD pair ordering and reliability
```

현재 논문은 fixed OOD readout의 **pair-ranking multiplicity**를 중심 문제로 둔다.
같은 initialization과 data stream에서 coupled, decoupled, zero-decay training을
처음부터 비교하고, decay coupling만 바꾸었을 때 동일 ID--OOD pair ordering이
same-policy seed variation보다 더 많이 바뀌는지 검정한다. Within-class-whitened
class-mean span `S`와 residual complement `S-perp`에서 Raw MD와 Marginal이
공유하는 residual 항을 RMD가 상쇄한다는 조건부 정리는 그 multiplicity가 어느
score channel에서 생기고 어떤 readout이 왜 덜 민감한지를 설명하는 이론적
interface다.

Raw/L2와 MD/RMD의 2 x 2 비교는 normalization과 residual cancellation의
상호작용을 진단한다. 정리는 각 transform 내부에는 적용되지만, L2가 raw fit의
`S-perp`만 부분 제거한다는 주장은 정리에서 자동으로 나오지 않으며 직접 component
검정이 필요하다.

한 branch와 fit 안에서 Gaussian score가 sample로부터 읽는 중심 interface는
`q_perp=||P_S-perp x||^2`와 `P_S x`다. 다른 기하 지표는 이 interface와
branch-specific fit이 왜 움직였는지 설명하거나 대안 경로를 검사한다. Full-rank
whitening은 ID-train의 `mean(q_perp)=d-dim(S)`를 고정하지만 `Var(q_perp)`와 tail은
고정하지 않는다.

Fisher/LDA의 discriminant subspace, 알려진 MD--Marginal--RMD 관계,
size--stretch, spectrum--allocation, radial/angular dynamics는 선행 지식 또는
측정도구다. 중심 논문은 optimizer leaderboard가 아니며, OOD instability의 존재
자체를 최초로 보였다고 주장하지 않는다. 새 기여 후보는 통제된 same-init 개입,
pair-level transition accounting, score-component 귀속, 그리고 fixed-total-decay
`alpha in {0, 0.5, 1}` 확인 설계의 결합이다. 단, Card 13 v10은 residual tail과
novelty gate를 모두 통과할 때만 평가하는 하나의
조건부 Residual-t Mahalanobis (`RtMD`) method slot을 사전등록한다. Score
accounting만으로 완전한 causal mediation을 주장하지 않는다.

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
