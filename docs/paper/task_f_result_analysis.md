# Task F result analysis

Last updated: 2026-08-18
Fast-kill analysis HEAD: `ecba28ef22fc4b8893119f5224880876ecbd76df`

이 문서는 완료된 Task F를 논문 질문에 맞춰 해석하는 단일 기준이다. Card 13의
사전등록 estimand를 바꾸지 않으며, Evidence Pack 두 개는 원자료 inventory와
provenance로 유지한다.

표시의 뜻은 다음과 같다.

- **사실:** terminal, manifest, protocol에 직접 기록된 값
- **재계산:** 저장 score/feature/fit에서 결정론적으로 다시 계산한 값
- **해석:** 관찰과 일치하지만 인과 또는 유일 매개로 확립되지 않은 설명
- **미확립:** 현재 자료로는 주장할 수 없는 내용

## 1. 결론부터

### 현재 논문의 중심

상위 연구 질문은 “어떤 optimizer-side training choice가 Raw MD에 민감한
representation을 만드는가?”이다. 그러나 Task F가 지금 직접 답한 범위는 더 좁다.

```text
same initialization and data stream
-> controlled decay-coupling / within-LR WD contrasts
-> pair Gain/Loss/Churn and Raw-MD change
-> predominantly Marginal score localization
-> time/depth and covariance-geometry concordance
```

**사실:** 네 Adam cell 모두 coupled-minus-decoupled Raw-MD DeltaAUROC가 음수다.
그 크기는 near에서 `-0.0218`부터 `-0.1780`, far에서 `-0.0787`부터 `-0.2835`로
context마다 다르다. Primary의 large effect는 단일 cell 우연으로만 보기는 어렵지만,
현재 설계는 LR main effect를 인과적으로 식별하지 않는다.

### Candidate Main fast kill 판정: FAIL

**사실:** Card 13 Section 14의 사전 고정 statistic을 14개 C--D endpoint pair에
적용했다. 14/14가 source identity, checksum, projection reconstruction과 numerical
validation을 통과했지만 `rho > 1`은 0/14였다. Cell별 결과는 다음과 같다.

| Cell | `rho > 1` | median `rho` | range |
| --- | ---: | ---: | ---: |
| `LR=1e-3, WD=1e-4` anchor | 0/5 | 0.00795 | 0.00780--0.00816 |
| `LR=1e-3, WD=1e-3` | 0/3 | 0.00443 | 0.00426--0.00448 |
| `LR=3e-4, WD=1e-4` all-PASS cell | 0/3 | 0.01005 | 0.00954--0.01015 |
| `LR=3e-4, WD=1e-3` | 0/3 | 0.00473 | 0.00454--0.00478 |

Centered classifier rowspace rank는 모든 context에서 9였다. Maximum projection-energy
reconstruction error는 `5.53e-8`이었고, anchor seed 0 GPU float32--CPU float64
`rho` relative difference는 `1.25e-8`로 판정 부호가 유지됐다. 경계
`abs(log(rho)) < 0.05` context는 없었다.

**판정:** coverage와 precision criterion은 통과했지만 anchor, all-PASS cell,
high-WD support criterion이 모두 실패했다. Frozen statistic 아래에서 held-out
D-to-C non-affine residual은 차원당 classifier-insensitive complement가 아니라
classifier-sensitive rowspace에 훨씬 더 집중됐다. 따라서
`training rule -> classifier-insensitive deformation -> OOD readout` Candidate Main은
채택하지 않는다.

**해석 경계:** 이 결과는 모든 종류의 classifier-insensitive geometry가 존재하지
않는다는 명제가 아니다. 이번 논문에 필요하다고 사전 지정한 carrier가 관찰되지
않았다는 빠른 기각이다. Supporting logit diagnostics는 저장했지만 새로운
equivalence threshold로 gate를 구조하지 않았다. 양방향 affine refit, 작은 singular
subspace, normalization, trajectory, projected OOD score를 후속 rescue로 실행하지
않고 아래 Raw-MD pair-instability Plan B를 유지한다.

**재계산:** 같은 LR 안의 WD contrast는 실제 sibling identity가 일치한다. Low-LR에서
WD를 `1e-4 -> 1e-3`로 바꾸면 coupled branch Raw MD가 near/far
`-0.0962/-0.0779` 이동한 반면 decoupled branch는 `-0.0050/+0.0018`이었다.
따라서 `WD x coupling` DiD는 `-0.0912/-0.0797`이다. High-LR의 DiD는
`+0.0004/+0.0618`로 같지 않다. WD가 coupling effect를 보편적으로 단조 증폭한다는
주장은 기각된다.

**해석:** small-effect low-LR/low-WD cell에서는 covariance difference도 상대적으로
작고, 나머지 large-effect cells에서는 norm, condition, spectral concentration,
effective rank 차이가 더 크다. 이는 optimizer-side geometry formation과 Raw-MD
sensitivity의 concordance를 지지한다. 다만 high-LR 두 WD cell의 geometry와 net
AUROC가 단조 정렬되지 않으므로 geometry의 유일 mediation은 미확립이다.

### 논문 headline gate

- **현재 사용:** controlled optimizer-side case study of Raw-MD pair-ranking
  multiplicity and score/geometry formation.
- **조건부 승격:** common-stream LR factorial까지 통과하면 “which optimizer-side
  recipes create Raw-MD-incompatible representations?”를 main RQ로 승격한다.
- **사용 금지:** Task F만으로 optimizer family, LR, WD의 전체 causal landscape를
  규명했다는 문장.

## 2. 입력과 검산

| 항목 | 회수량 | 판정 |
| --- | ---: | --- |
| protected score contexts | 360 | PASS |
| ID-only geometry fits | 660 | PASS |
| sibling alignment records | 657 | PASS |
| update telemetry files | 50 | PASS; 550 selected-step rows |
| endpoint seed rows | 1,512 | C-D, D-Z, C-Z; 5 readouts x 6 datasets |
| formation seed rows | 960 | five epochs and four depths |
| primary spectral contexts | 10 | D 5 PASS; C 5 NOT_APPLICABLE |

모든 endpoint sibling은 seed별 `initialization_sha256`와 `data_stream_sha256`가
일치할 때만 계산했다. Sample-level `MD = RMD + Marginal` residual과 replacement
accounting residual은 검산을 통과했다. Pair transition은 C++ `O(N log N)` 구현을
작은 brute-force oracle과 대조했다.

## 3. Endpoint sibling matrix

아래 값은 six-dataset seed-paired result를 Near/Far로 macro 평균한 것이다. `D`는
AdamW, `C`는 Adam이며 Delta는 항상 `C-D`다.

| Adam cell | Region | D Raw MD | C Raw MD | Delta | Churn | ID guardrail |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LR 1e-3, WD 1e-4 | Near | 0.5922 | 0.4178 | -0.1743 | 0.3486 | Acc PASS, NLL FAIL, ECE PASS |
|  | Far | 0.6709 | 0.3874 | -0.2835 | 0.3863 | Pareto result |
| LR 1e-3, WD 1e-3 | Near | 0.5928 | 0.4148 | -0.1780 | 0.3810 | Acc FAIL, NLL PASS, ECE PASS |
|  | Far | 0.6559 | 0.4133 | -0.2426 | 0.3755 | Pareto result |
| LR 3e-4, WD 1e-4 | Near | 0.7303 | 0.7085 | -0.0218 | 0.2313 | all PASS |
|  | Far | 0.7866 | 0.7079 | -0.0787 | 0.2317 | comparable-ID cell |
| LR 3e-4, WD 1e-3 | Near | 0.7252 | 0.6122 | -0.1130 | 0.3168 | Acc PASS, NLL FAIL, ECE PASS |
|  | Far | 0.7884 | 0.6300 | -0.1585 | 0.3005 | Pareto result |

**판정:** tested cells에서는 Raw-MD fragility와 coupling deterioration이
heterogeneous하게 반복된다. Cross-LR absolute 차이는 기술적 연관이다. Primary를
“same ID performance”로 부르지 않는다.

### Zero arm: 같은 AUROC와 같은 behavior는 다르다

| Cell | Region | D-Z DeltaAUROC | D-Z Churn |
| --- | --- | ---: | ---: |
| LR 1e-3, WD 1e-4 | Near / Far | +0.0014 / +0.0018 | 0.1897 / 0.1540 |
| LR 1e-3, WD 1e-3 | Near / Far | +0.0013 / -0.0132 | 0.1862 / 0.1547 |
| LR 3e-4, WD 1e-4 | Near / Far | +0.0048 / +0.0022 | 0.1708 / 0.1378 |
| LR 3e-4, WD 1e-3 | Near / Far | -0.0002 / +0.0040 | 0.1717 / 0.1379 |

**재계산:** Zero와 AdamW의 net AUROC는 거의 같아도 `0.14-0.19` 수준의 pair
ordering이 바뀐다. Gain과 Loss의 상쇄를 “representation이 같다”로 해석할 수 없다.

### 새로 확인한 within-LR WD contrast

Delta는 `WD 1e-3 - WD 1e-4`다. DiD는 high-WD C-D에서 low-WD C-D를 뺀 값이다.

| LR context | Branch | Near Delta / Churn | Far Delta / Churn | Coupling x WD DiD Near / Far |
| --- | --- | ---: | ---: | ---: |
| 1e-3 | coupled | -0.0003 / 0.3183 | +0.0399 / 0.2975 | +0.0004 / +0.0618 |
| 1e-3 | decoupled | -0.0007 / 0.1877 | -0.0219 / 0.1521 |  |
| 3e-4 | coupled | -0.0962 / 0.2939 | -0.0779 / 0.2872 | -0.0912 / -0.0797 |
| 3e-4 | decoupled | -0.0050 / 0.1707 | +0.0018 / 0.1366 |  |

**사실/재계산:** 이 비교는 같은 LR sibling group 안에서 init/stream identity가
일치하므로 controlled contrast다. Low-LR에서는 stronger WD의 adverse movement가
주로 coupled branch에 나타난다. High-LR에서는 같은 패턴이 반복되지 않으며 SVHN은
오히려 coupled AUROC가 `+0.1546` 이동한다. “WD 증가가 coupling damage를 단조
증폭한다”는 미확립이 아니라 현재 자료와 맞지 않는다.

## 4. Readout과 score localization

### RMD와 L2는 method novelty가 아니라 probe다

| Cell | Readout | Near C/D/Delta/Churn | Far C/D/Delta/Churn |
| --- | --- | ---: | ---: |
| 1e-3, 1e-4 | Raw RMD | 0.8806/0.8908/-0.0102/0.1286 | 0.9006/0.9111/-0.0105/0.1154 |
|  | L2 MD | 0.8466/0.8716/-0.0250/0.1686 | 0.9435/0.9334/+0.0101/0.0868 |
| 1e-3, 1e-3 | Raw RMD | 0.8430/0.8929/-0.0499/0.1618 | 0.8433/0.9146/-0.0713/0.1593 |
|  | L2 MD | 0.8222/0.8763/-0.0541/0.1918 | 0.9248/0.9393/-0.0144/0.0984 |
| 3e-4, 1e-4 | Raw RMD | 0.8894/0.8916/-0.0021/0.1070 | 0.9140/0.9171/-0.0031/0.0911 |
|  | L2 MD | 0.8459/0.8798/-0.0339/0.1350 | 0.9123/0.9345/-0.0222/0.0844 |
| 3e-4, 1e-3 | Raw RMD | 0.8858/0.8891/-0.0033/0.1241 | 0.9165/0.9117/+0.0048/0.1040 |
|  | L2 MD | 0.8792/0.8790/+0.0002/0.1416 | 0.9509/0.9341/+0.0169/0.0788 |

**판정:** RMD attenuation은 네 cell에서 비교적 반복된다. L2-MD의 absolute AUROC는
두 branch 모두 크게 회복하지만 signed-gap attenuation은 cell-dependent다. 특히
low-LR/low-WD Near에서 L2 gap `-0.0339`는 Raw gap `-0.0218`보다 작지 않다.
따라서 “RMD와 L2가 언제나 coupling sensitivity를 줄인다”가 아니라, RMD는 더
일관된 cancellation probe이고 L2는 absolute recovery가 강한 nonlinear refit이다.

### MD = RMD + Marginal replacement accounting

| Cell | Region | Delta | phi RMD | phi Marginal | Marginal share | phi ID | phi OOD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3, 1e-4 | Near | -0.1743 | -0.0405 | -0.1338 | 0.767 | +0.2164 | -0.3908 |
|  | Far | -0.2835 | -0.0371 | -0.2464 | 0.869 | +0.1725 | -0.4559 |
| 1e-3, 1e-3 | Near | -0.1780 | -0.0667 | -0.1113 | 0.625 | +0.3575 | -0.5356 |
|  | Far | -0.2426 | -0.0667 | -0.1759 | 0.725 | +0.3057 | -0.5483 |
| 3e-4, 1e-4 | Near | -0.0218 | +0.0105 | -0.0323 | 1.481 | -0.0335 | +0.0116 |
|  | Far | -0.0787 | +0.0101 | -0.0888 | 1.128 | -0.0307 | -0.0480 |
| 3e-4, 1e-3 | Near | -0.1130 | -0.0255 | -0.0875 | 0.774 | +0.0051 | -0.1181 |
|  | Far | -0.1585 | -0.0216 | -0.1369 | 0.864 | +0.0027 | -0.1611 |

Share는 `abs(mean phi_Marginal) / abs(mean Delta)`다. Primary dataset별 값은
`0.693-0.940`; seed별 ratio 평균도 별도로 보존했다.

**판정:** Marginal은 네 cell 모두에서 adverse Raw-MD movement의 주된 score
component다. Share가 1을 넘는 low-LR/low-WD cell은 Marginal deterioration과 RMD
improvement가 상쇄됐다는 뜻이지 100%를 넘는 causal mediation이 아니다. Primary와
두 high-WD contexts에서는 OOD-side negative motion이 지배하지만, low-LR/low-WD
Near에서는 `phi_ID=-0.0335`, `phi_OOD=+0.0116`이다. 따라서 OOD-side dominance를
Adam-family 전체 법칙으로 쓰지 않는다.

## 5. Pair multiplicity는 소수 sample 현상인가

Primary Raw-MD C-D에서 OOD sample별 churn burden의 seed 평균 median은 dataset에
따라 `0.293-0.580`이다. Top 10% sample이 전체 churn의 `17.3-25.2%`를 운반한다.
이는 매우 작은 sample 집합 하나가 현상 전체를 독점하는 모양이 아니다. Cross-seed
flip-burden Spearman은 `0.30-0.59`, top-10% Jaccard는 `0.09-0.17`로 일부 구조는
반복되지만 동일한 hard sample set으로 고정되지 않는다. Pair/sample을 독립 반복으로
검정하지 않으며 통계 단위는 seed다.

## 6. Geometry concordance

| Cell | Norm C-D | Global rank D -> C | Global condition D -> C | Global eff. rank D -> C | Top-10 trace D -> C |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-3, 1e-4 | +2.43 | 640 -> 512.6 | 8.21e4 -> 1.09e8 | 16.82 -> 11.09 | 0.889 -> 0.956 |
| 1e-3, 1e-3 | +5.33 | 640 -> 374.3 | 7.67e4 -> 6.28e8 | 16.78 -> 9.74 | 0.891 -> 0.973 |
| 3e-4, 1e-4 | +0.84 | 640 -> 640 | 1.32e5 -> 1.90e6 | 13.21 -> 12.28 | 0.940 -> 0.953 |
| 3e-4, 1e-3 | +5.20 | 640 -> 640 | 1.34e5 -> 7.90e8 | 13.23 -> 10.01 | 0.940 -> 0.981 |

**재계산:** small Raw-MD effect cell은 norm/effective-rank/concentration 차이도 가장
작다. Large-effect cells는 더 큰 concentration과 condition difference를 보인다.
그러나 high-LR high-WD는 primary보다 condition/rank 차이가 더 큰데 net Raw-MD
gap은 더 크지 않다. 판정은 `concordant but not monotone`이다.

Primary C-D alignment에서 raw principal-angle seed-mean profile은
`1.41, 1.91, 2.03, 2.25, 2.36, 2.47, 2.69, 2.97, 3.11` degrees,
chordal distance는 `0.1784`, held-out-ID affine normalized Frobenius residual은
`0.2447`이다. L2 profile은 `0.58-1.78` degrees, chordal `0.0943`, affine residual
`0.2349`다. 모든 primary alignment component status는 `INAPPLICABLE`이므로 이
숫자는 diagnostic이며 theorem-valid cross-branch attribution이 아니다.

## 7. Formation

| Axis | Point | Near Delta / Churn | Far Delta / Churn |
| --- | --- | ---: | ---: |
| epoch | 10 | -0.0739 / 0.2546 | -0.1811 / 0.2947 |
|  | 60 | -0.1072 / 0.2977 | -0.2148 / 0.3294 |
|  | 120 | -0.1771 / 0.3416 | -0.2914 / 0.3742 |
|  | 160 | -0.1781 / 0.3504 | -0.2892 / 0.3845 |
|  | 200 | -0.1743 / 0.3486 | -0.2835 / 0.3863 |
| depth | stage1 | +0.0032 / 0.1407 | -0.0182 / 0.0857 |
|  | stage2 | -0.0313 / 0.2138 | -0.0268 / 0.1271 |
|  | stage3 | -0.1540 / 0.3188 | -0.3065 / 0.3621 |
|  | penultimate | -0.1743 / 0.3486 | -0.2835 / 0.3863 |

**사실:** effect는 epoch 10부터 검출되고 epoch 120까지 증폭된 뒤 유지된다. Depth는
stage1에서 거의 없고 stage3/penultimate에서 크다. 표현은 `early detectable, later
amplified and deep`가 정확하다.

Raw Marginal share는 epoch 10부터 Near/Far `0.926/0.968`이고 epoch 200에는
`0.767/0.869`다. OOD-side negative motion도 epoch 10부터 존재하며 60-120 사이 크게
증가한다. Primary endpoint의 norm, total trace, effective-rank difference도 같은 구간에
커진다. 이것은 formation concordance이지 optimizer update의 frozen-state mediation은
아니다. Telemetry는 actual branch-state witness다. Step 70,400 parameter norm은
Zero/AdamW/mixed/Adam이 `819.8/821.0/105.2/99.2`로 크게 갈라진다.

## 8. Spectral allocation gate

저장 global covariance/precision과 protected feature로 Marginal band `A/P/Q`를
재구성했다. Decoupled 다섯 seed는 stored Marginal max residual `1.91e-10` 이하로
PASS했다. Coupled raw fit은 다섯 seed 모두 원래부터 `NOT_APPLICABLE`이며 reconstruction
또한 안정적 cross-branch attribution에 사용할 수 없었다.

따라서:

- decoupled branch 내부 spectral allocation은 prior-work-informed diagnostic으로 보존;
- coupled-minus-decoupled band mediation은 **NOT AVAILABLE**;
- `tail band -> Marginal gap` 또는 `tail = S_perp` 주장은 하지 않음;
- band-removal AUROC, clipping, 새 detector score는 실행하지 않음.

이 결과는 spectrum 분석을 본문 mechanism의 최종 고리로 승격하기보다 geometry
appendix/supporting lens로 두라는 gate다.

## 9. 새 논리 뼈대의 타당성

“어떤 optimizer-side choice가 Mahalanobis-sensitive representation을 만드는가?”는
선행연구와 구분되는 좋은 상위 질문이다. 선행연구가 이미 점유한 것은 RMD, L2
normalization, spectrum-allocation identity와 representation-side MD failure다. 이
논문의 차별점은 fixed architecture/data/objective에서 training-side choice를 같은
init/stream으로 통제하고, geometry가 형성되어 exact pair behavior로 전달되는 과정을
추적하는 데 있다.

다만 현재 evidence가 unlock한 문장은 다음 정도다.

> A controlled optimizer-side change can move a representation and its fixed Raw-MD
> pair ordering, with the effect carried predominantly by the Marginal score channel
> and amplified over training and depth.

아직 unlock하지 못한 문장은 다음이다.

> We identify which optimizer recipes generally create Mahalanobis-hostile
> representations.

후자는 causal LR comparison, architecture replication, 더 넓은 recipe support가
필요하다. `Mahalanobis-hostile`도 dataset-independent label이 아니라 frozen benchmark의
operational outcome으로 정의한다.

Adam coupled/decoupled update 차이를 `-eta lambda (P_t-I) theta`로 쓰는 설명은
frozen-moment local heuristic일 뿐 exact Adam dynamics가 아니다. Coupled decay는
gradient뿐 아니라 first/second moment history와 denominator를 함께 바꾼다. Jacobian을
통한 feature-covariance drift도 plausible interpretation이지 현재 theorem이 아니다.
이 이론은 motivation/appendix에만 두고 telemetry 및 cross-cell concordance를 넘어서
인과 결론으로 사용하지 않는다.

## 10. 수정된 논문 구조

1. **Open problem:** representation-side MD failure는 알려졌지만 optimizer-side origin과
   formation은 덜 통제되어 있다.
2. **Controlled design:** Zero/AdamW/Adam sibling과 within-LR WD contrasts.
3. **Primary phenomenon:** absolute Raw MD, Gain/Loss/Churn, ID/OOD Pareto boundary.
4. **Local context interaction:** four-cell C-D 반복과 controlled WD x coupling; cross-LR는
   descriptive.
5. **Score localization:** RMD/L2 probes, exact MD decomposition, Marginal 중심 accounting;
   OOD-side pattern의 context boundary.
6. **Formation:** early detectable/later amplified time trajectory와 deep-layer formation,
   update telemetry.
7. **Geometry:** fixed panel의 concordance와 non-monotonic caveat; spectrum은 prior lens.
8. **Boundaries:** SGDM reversal, primary applicability failure, single architecture.
9. **Replication/future factorial:** 현재 contribution과 분리해 preregistered next test로 제시.

## 11. 다음 의사결정

### 기존 artifact로 완료된 것

- four-cell C-D/D-Z/C-Z endpoint matrix와 exact pair accounting;
- within-LR WD contrast와 WD x coupling DiD;
- cross-cell Marginal 및 ID/OOD-side localization;
- flip-burden concentration과 seed consistency;
- fixed geometry panel, alignment, time/depth, telemetry;
- spectral allocation gate: D PASS, C NOT_APPLICABLE.

### 아직 현재 artifact에서 정리할 수 있는 것

- main figure 후보의 seed interval/guardrail annotation과 six-dataset supplement 배치;
- geometry panel을 `concordant/mixed/discordant` 사전 기준으로 최종 adjudication;
- alpha=0.5를 새 broad RQ가 아닌 primary interpolation evidence로 재배치;
- SGDM의 동일 score-localization 표를 boundary supplement로 정리.

### 새 실험 gate

현재 narrow paper를 우선하면 ResNet-18/CIFAR-10 focal replication이 먼저다. 넓은
optimizer-origin headline을 선택하면 그 전에 seed별 모든 LR arm이 공통 init/stream을
공유하는 paired LR bridge/factorial이 필요하다. Additional adaptive optimizer,
ConvNeXt/ViT, broad phase map은 그 다음이다.

## 12. Claim ledger

| Claim | Status | 이유 |
| --- | --- | --- |
| coupling-only intervention changes exact pair ordering | SUPPORTED | four Adam cells; verified sibling identity |
| effect magnitude depends on local LR/WD context | SUPPORTED descriptively/locally | four cells plus within-LR WD DiD |
| LR causes the absolute baseline difference | NOT ESTABLISHED | cross-LR streams differ |
| stronger WD universally worsens coupled Raw MD | CONTRADICTED | high-LR interaction differs, SVHN reverses |
| Raw-MD movement is predominantly Marginal | SUPPORTED | all four cells; cancellation may yield share >1 |
| OOD-side motion always dominates | PARTIALLY SUPPORTED | primary/high-WD yes; low-LR low-WD Near no |
| RMD attenuation repeats | SUPPORTED in tested Adam cells | absolute performance remains high; high-LR/high-WD gap is larger |
| L2 always attenuates signed gap | NOT SUPPORTED | low-LR/low-WD Near counterexample |
| L2 substantially recovers absolute AUROC | SUPPORTED | all tested cells have high absolute L2-MD AUROC |
| geometry difference tracks effect magnitude | PARTIALLY SUPPORTED | broad concordance, no monotone mediation |
| coupled spectral band uniquely carries the gap | NOT AVAILABLE | coupled raw fit NOT_APPLICABLE |
| S_perp uniquely mediates the effect | NOT ESTABLISHED | primary raw/L2 component theorem 30/30 NOT_APPLICABLE |
| optimizer-side origin is established generally | NOT YET SUPPORTED | requires paired LR factorial and replication |

## 13. Provenance

### Tracked outputs

- `results/task_f_result_analysis_v1.json`, SHA-256
  `bbdb3327a77797e4a26f1e4175201081d22190e2fe28b40b6e940958edfc5a33`
- `results/task_f_result_analysis_v1.json.sha256`
- `scripts/analyze_task_f_existing_artifacts.py`
- `scripts/task_f_pair_accounting.cpp`
- `scripts/summarize_task_f_result_analysis.py`
- `scripts/plot_task_f_result_analysis.py`

### External detailed output

- HF bundle:
  `hf://buckets/contra333/ICLR_RUN/aggregate/task_f_result_analysis_20260818/ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed/`
- merged JSON inside the bundle: `task_f_result_analysis_merged.json`
- merged SHA-256:
  `ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed`
- CSV/figure manifest inside the bundle: `detail/extraction_manifest.json`
- manifest SHA-256:
  `4052cac248685d0fc5fb8e9f6db29d01b889140fcdcf492b858df7ef319c5ef6`
- snapshot `SHA256SUMS` SHA-256:
  `e4c1bf9363321097bd3f3c3beae9c0183e33f8ddf37ac5873e17011bc07a3ad4`
- original Evidence Pack SHA-256:
  `6296464e210668d2c56e5ca08e52438ab4de4e97dad452c5ccfd2c96f61a1712`
- gap-fill Evidence Pack SHA-256:
  `c25eb90180a572bb918419c960e3ff183e2960e715d420e6de928f44ee9374dd`

Portable restore:

```bash
hf buckets sync \
  hf://buckets/contra333/ICLR_RUN/aggregate/task_f_result_analysis_20260818/ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed/ \
  task_f_result_analysis_20260818
cd task_f_result_analysis_20260818
sha256sum -c SHA256SUMS
```

Source roots:

- curie/lise protected:
  `/home/ghjin/0707_exp/task_f_protected_06c61f6f_artifacts`
- curie/lise fresh ID:
  `/home/ghjin/0707_exp/task_f_fresh_eval_2a22a651_artifacts`
- curie/lise training:
  `/home/ghjin/0707_exp/task_f_full_9eb3c1fa_artifacts`
- precision_medicine protected:
  `/mnt/drive/lab1/oge/artifacts/task_f_protected/06c61f6f`
- precision_medicine fresh ID:
  `/mnt/drive/lab1/oge/artifacts/task_f_fresh_eval/2a22a651`
- precision_medicine training:
  `/mnt/drive/lab1/oge/artifacts/task_f_full/9eb3c1fa`

Read-only extraction command pattern:

```bash
python scripts/analyze_task_f_existing_artifacts.py host \
  --host HOST --protected-root PROTECTED --fresh-root FRESH \
  --training-root TRAINING --pair-library /tmp/libtask_f_pair_accounting.so \
  --output /tmp/task_f_result_analysis_HOST_spectral.json --include-spectral

python scripts/analyze_task_f_existing_artifacts.py merge \
  --inputs HOST_JSONS --output task_f_result_analysis_merged.json

python scripts/summarize_task_f_result_analysis.py \
  --input task_f_result_analysis_merged.json \
  --compact-output results/task_f_result_analysis_v1.json \
  --detail-dir DETAIL_DIR \
  --git-sha e0f35285f0edbc5f88077cc2e3a7f136e42554d7 \
  --artifact-uri HF_BUNDLE_URI \
  --artifact-manifest-sha256 4052cac248685d0fc5fb8e9f6db29d01b889140fcdcf492b858df7ef319c5ef6 \
  --artifact-checksums-sha256 e4c1bf9363321097bd3f3c3beae9c0183e33f8ddf37ac5873e17011bc07a3ad4
```

이번 분석은 checkpoint loading, training, protected inference, detector refitting,
band-removal score, clipping, 새 detector를 실행하지 않았다. 기존 score/feature/fit과
telemetry의 결정론적 summary만 계산했다.

## 14. Classifier-insensitive fast kill artifact

- analysis code SHA: `ecba28ef22fc4b8893119f5224880876ecbd76df`
- merged canonical JSON SHA-256:
  `90f4fc447a32b14748ef992878ca8ba1e87e8a148f6810677e4819d2d56a4b27`
- merged payload identity:
  `bcebc1a002555d14e526d5734de8c8b1b31dc7372c4aef7ce3b637411e3908e9`
- bundle `SHA256SUMS` SHA-256:
  `d45d077673ffcded30144ee23e776f79ae8285f71739a7318c52cb0af7c05c11`
- HF bundle:
  `hf://buckets/contra333/ICLR_RUN/aggregate/task_f_classifier_insensitive_kill_20260818/bcebc1a002555d14e526d5734de8c8b1b31dc7372c4aef7ce3b637411e3908e/`

Host-local execution used the live idle-GPU check and scheduled 6/2/6 contexts
on curie/lise/precision_medicine. Only the small canonical host JSON files were
collected. The run loaded no checkpoint, performed no feature inference or
affine/detector refit, and accessed no protected ID/OOD data.
