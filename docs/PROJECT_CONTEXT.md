# Project context

## Research objective

이 저장소는 다음 연결을 연구한다.

```text
training rule
-> representation geometry over time and depth
-> fixed Mahalanobis score components
-> ID/OOD pair ordering and reliability
```

상위 연구 질문은 **어떤 optimizer-side training choice가 fixed Raw MD에 민감한
representation을 만들며, 그 차이가 학습 중 언제 어디서 형성되는가**이다. 여기서
`Mahalanobis-hostile`은 보편적 속성명이 아니라 고정 benchmark에서 absolute
AUROC/FPR95, paired Delta, Gain/Loss/Churn으로 운영적으로 정의한 결과다.

완료된 Task F는 이 넓은 질문 전체의 답이 아니라 decay coupling과 local WD
context를 통제한 첫 mechanism case study다. 현재 가장 강하게 확립된 연결은 다음과
같다.

```text
controlled decay-coupling intervention
-> exact pair-ranking multiplicity
-> Raw-MD sensitivity
-> Marginal/global OOD-side score localization
```

Zero, AdamW, Adam은 공통 initialization과 data stream에서 병렬로 학습된 sibling
arms다. `C-D`, `D-Z`, `C-Z`는 이 identity가 seed별로 일치할 때만 controlled
contrast로 취급한다. 서로 다른 LR/WD cell의 차이는 local context heterogeneity를
보여주는 descriptive comparison이다. 다만 같은 LR sibling group 안의 두 WD
level은 identity가 일치하므로 branch별 WD contrast와 `WD x coupling`
difference-in-differences는 controlled contrast다. High/low LR group은 data-stream
identity가 다르므로 cross-LR 차이를 LR causal effect나 dose response로 해석하지
않는다. SGDM sign reversal도 optimizer-family boundary이지 family main effect의
인과 추정이 아니다.

Raw/L2와 MD/RMD의 2 x 2 비교는 normalization과 global-Marginal cancellation의
서로 다른 attenuation을 진단한다. `MD = RMD + Marginal`과 저장 score replacement
accounting은 score-level localization이다. 이것만으로 causal 또는 unique mediation을
주장하지 않는다.

Within-class-whitened class-mean span `S`와 residual complement `S-perp`에 관한
정리는 명시된 numerical applicability 조건에서만 supporting theory다. Primary
Raw/L2 component fit은 coupled와 decoupled를 합쳐 30/30 `NOT_APPLICABLE`이므로,
현재 headline은 `S_perp` attribution이 아니다. 저장된 pseudoinverse MD, Marginal,
RMD와 alignment 숫자는 유효한 score/diagnostic evidence지만 theorem-valid component
mediation으로 승격하지 않는다.

Fisher/LDA discriminant subspace, RMD, L2 normalization, size--stretch,
spectrum--allocation, radial/angular dynamics는 선행 지식 또는 측정 도구다. 중심
논문은 optimizer leaderboard가 아니며 OOD instability의 존재 자체를 최초로
보였다고 주장하지 않는다. 새 기여 후보는 controlled same-init intervention, exact
Gain/Loss/Churn accounting, aggregate AUROC가 숨기는 pair reorganization, Marginal
중심 score localization, 그리고 time/depth formation evidence의 결합이다. Primary의
OOD-side dominance는 모든 Adam context에서 동일하지 않으므로 local-context 결과로
한정한다. 더 넓은 `optimizer-side origin` headline은 cross-context geometry/score
concordance와 common-stream LR factorial이 통과한 뒤에만 승격한다.
`RtMD` slot은 Gate 3 `FAILED_INAPPLICABLE`로 닫혔으며 현재 논문의 기여가 아니다.

## Active sources

- [`STATUS.md`](STATUS.md): 지금 완료된 것, blocker, 다음 작업
- [`13_active_paper_protocol.md`](reference_cards/13_active_paper_protocol.md):
  실행 가능한 논문 실험 계약의 유일한 권위
- [`task_f_result_analysis.md`](paper/task_f_result_analysis.md): 완료된 Task F의
  결과 해석, claim boundary, 기존 artifact 추가 분석에 대한 단일 기준
- [`intervention_supporting_theory_outline.md`](paper/intervention_supporting_theory_outline.md):
  교수님과 연구자가 읽는 논문 서사
- [`sources.lock.yaml`](sources.lock.yaml): 외부 논문과 공식 구현 provenance
- [`history/README.md`](history/README.md): 과거 protocol, validation, 외부 artifact 지도

Notion은 연구 토론과 의사결정 interface다. 실행 의미와 현재 결과의 claim boundary는
위 repository 문서와 일치시킨다. Historical validation log는 provenance가 필요할
때만 사용한다.

## Stable boundaries

- Repository code/config/active protocol이 구현과 실험 의미의 source of truth다.
- Card 13의 사전등록 estimand와 판단 규칙은 결과에 맞춰 수정하지 않는다.
- Completed historical grid와 component analysis는 discovery evidence다. Fresh paired
  trajectory를 대신하지 않는다.
- Fixture, smoke test, planned experiment, running process는 research result가 아니다.
- Generated tables, figures, checkpoints, feature/score arrays는 Git 밖의 hash-addressed
  artifact로 보존한다.
