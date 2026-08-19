# Task F result analysis

Last updated: 2026-08-19
Fast-kill analysis HEAD: `ecba28ef22fc4b8893119f5224880876ecbd76df`
Frozen inspection-pack generator HEAD: `f3dd41cbb4a0eb1fde570fa8f7555348c5eb62a1`

이 문서는 완료된 Task F를 논문 질문에 맞춰 해석하는 단일 기준이다. Card 13의
사전등록 estimand를 바꾸지 않으며, Evidence Pack 두 개는 원자료 inventory와
provenance로 유지한다.

표시의 뜻은 다음과 같다.

- **사실:** terminal, manifest, protocol에 직접 기록된 값
- **재계산:** 저장 score/feature/fit에서 결정론적으로 다시 계산한 값
- **해석:** 관찰과 일치하지만 인과 또는 유일 매개로 확립되지 않은 설명
- **미확립:** 현재 자료로는 주장할 수 없는 내용

## 1. Reader-first summary

### 동결한 논문 질문

> Architecture, data, objective, initialization, minibatch order를 고정하고
> decay-coupling rule만 바꿀 때, protocol-fixed, branch-refitted Mahalanobis
> detector의 동일 ID--OOD pair ordering은 얼마나 바뀌며 그 민감성은
> score의 어느 부분과 학습의 언제·어느 depth에서 나타나는가?

`Protocol-fixed, branch-refitted`는 각 branch가 같은 ID-only fitting 절차를
따르되 자신의 class means과 covariance를 다시 fit한다는 뜻이다.
동일한 detector parameter를 공유한 실험이 아니다.

### 다섯 가지 결과

| Finding | 사람이 읽을 결론 | 핵심 근거 | Claim boundary |
| --- | --- | --- | --- |
| 1. Controlled non-invariance | 감쇠 coupling 방식만 바꿔도 Raw MD가 동일 pair를 다르게 정렬한다. | 네 Adam cell의 C--D가 모두 음수; primary Near/Far `-0.1743/-0.2835` | 보편적 optimizer law이 아님 |
| 2. AUROC hides multiplicity | 최종 AUROC가 같아도 내부 pair decision은 다를 수 있다. | Zero--AdamW net AUROC는 거의 0이지만 Churn은 `0.14--0.19` | Churn 자체를 새 detector로 주장하지 않음 |
| 3. Score localization | Raw-MD adverse movement는 모든 channel의 일괄 붕괴가 아니라 주로 Marginal에 위치한다. | Primary dataset별 Marginal share `69--94%`; RMD는 상대적으로 안정 | Exact accounting이지 causal mediation이 아님 |
| 4. Formation | 민감성은 endpoint 우연이 아니라 학습과 depth를 따라 형성된다. | epoch 10부터 검출, 120까지 증폭; stage1/2보다 stage3/penultimate에서 큰 효과 | Telemetry는 same-state causal audit가 아님 |
| 5. Context boundary | Coupling sensitivity의 크기와 부호는 local optimization context에 의존한다. | low-LR WD×coupling DiD `-0.0912/-0.0797`, high-LR `+0.0004/+0.0618`; SGDM reversal | WRN cross-LR causal effect, monotone WD law를 주장하지 않음 |

### Reader figure map

| Figure | 이 그림으로 답할 질문 | Evidence |
| --- | --- | --- |
| Figure 1 | Primary에서 coupling 강도에 따라 dataset별 Raw MD가 어떻게 움직이는가? | AdamW alpha=0, Mixed alpha=0.5, Adam alpha=1의 six-dataset 2x3 dot plots; raw seeds와 mean diamond |
| Figure 2 | Raw-MD C--D가 LR/WD context와 dataset에 따라 어떻게 달라지는가? | dataset x four-Adam-context heatmap; seed-paired DeltaAUROC |
| Figure 3 | Raw MD 실패 뒤 RMD/L2-MD가 어느 절대 수준까지 회복하는가? | C와 D 각각의 Raw MD/RMD/L2-MD 절대 AUROC와 seed variation |
| Figure 4 | Raw-MD net change는 어떤 Gain/Loss 교환으로 만들어지는가? | dataset별 Gain, Loss, Churn; 같은 seed의 C--D pair transition |
| Figure 5 | OOD 변화와 함께 움직이는 endpoint geometry는 무엇인가? | norm, effective rank, within trace, CDNV, NC0--NC3의 paired effect |
| Figure 6 | Raw-MD와 geometry 차이가 언제, 어느 depth에서 형성되는가? | epoch/depth trajectory를 나란히 둔 concordance panel |
| Supplement 1 | AUROC 회복이 FPR95에서도 보이는가? | Raw MD/RMD/L2-MD의 절대 FPR95 |
| Supplement 2 | effective-rank 감소와 spectral concentration 증가는 일치하는가? | top-10 trace share와 별도 CSV table |

기존 4-point `geometry_effect_concordance` scatter는 적은 context에서 regression을
암시하므로 본문 그림으로 쓰지 않는다. Geometry, alignment, SGDM,
spectral applicability, `S_perp`, classifier-insensitive fast kill은 appendix에서
진단 또는 negative boundary로 보존한다.

### 2026-08-19 frozen inspection pack에서 직접 확인한 결과

모든 평균과 불확실성의 통계 단위는 training seed다. Figure에는 raw seed를 숨기지
않고, 같은 seed의 C--D를 먼저 계산했다. Table은 mean, sample SD, two-sided paired
90% Student-t interval을 함께 보존한다. 아래 표의 `C / D`는 Adam / AdamW의 seed
평균이고 Churn은 Raw MD의 C--D 비교다.

| OOD dataset | Raw MD C / D | RMD C / D | L2-MD C / D | Raw C--D | Raw Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | 0.4348 / 0.6112 | 0.8707 / 0.8794 | 0.8287 / 0.8584 | -0.1765 | 0.3458 |
| TinyImageNet | 0.4009 / 0.5731 | 0.8904 / 0.9021 | 0.8644 / 0.8847 | -0.1722 | 0.3514 |
| MNIST | 0.2360 / 0.3834 | 0.8987 / 0.8992 | 0.9613 / 0.8907 | -0.1474 | 0.3067 |
| SVHN | 0.3394 / 0.8675 | 0.9024 / 0.9209 | 0.9700 / 0.9850 | -0.5281 | 0.5581 |
| Textures | 0.5645 / 0.8282 | 0.8956 / 0.9146 | 0.9389 / 0.9598 | -0.2637 | 0.3304 |
| Places365 | 0.4097 / 0.6043 | 0.9058 / 0.9098 | 0.9038 / 0.8981 | -0.1947 | 0.3498 |

Primary ID Accuracy는 C `0.93344 +/- 0.00180`, D `0.94156 +/- 0.00330`으로
평균 차이가 `-0.00812`다. 같은 조건에서 Raw MD C--D는 Near `-0.1743`, Far
`-0.2835`다. 따라서 여기서 사용할 담백한 기술은 **Accuracy 차이는 작고 Raw MD
변화는 훨씬 크다**이다. Accuracy가 완전히 같다는 전제나 NLL 하나로 전체 현상을
재정의하지 않는다.

Coupled branch에서 RMD-Raw 회복은 dataset별 `+0.3311--+0.6627`,
L2-MD-Raw 회복은 `+0.3744--+0.7252`다. 특히 SVHN은 Raw MD C/D가
`0.3394/0.8675`까지 갈라지지만 RMD와 L2-MD는 두 branch 모두 `0.90` 이상이다.
이는 representation에 OOD 정보가 전부 사라진 현상보다는 **Raw-MD readout에
특이적인 실패와 회복**이라는 기술과 일치한다. 다만 이것만으로 covariance
concentration이 유일한 원인이라고 확정하지 않는다.

Raw-MD Churn은 여섯 dataset 모두 `0.307--0.558`이며 Loss가 Gain보다 크다.
즉 AUROC 하락은 일부 sample 하나의 단순 오차가 아니라 많은 ID--OOD pair의
ordering 교환으로 나타난다. Primary endpoint geometry에서는 C--D effective rank
`-5.7304`, top-10 trace share `+0.06734`, log feature norm `+0.15197`, log within
trace `+0.20959`, CDNV `-0.04972`가 함께 관찰된다. NC0 `-1.84175`, NC2
`+0.26690`, NC3 `-0.11337`의 paired 90% interval은 0을 지나지 않지만 NC1
`+0.00430`은 0을 지난다. Geometry는 이처럼 축을 정해 방향과 formation을
대조하며, 모든 지표를 넣은 회귀나 unique mediation으로 해석하지 않는다.

재현 가능한 bundle은 다음 위치에 있다.

`hf://buckets/contra333/ICLR_RUN/aggregate/task_f_frozen_paper_pack_20260819/c80194480ab557a68b2306fd0c25c5cef7c5533da83b283500375fb0ee9faa99/`

Bundle에는 8개 Figure의 PDF/SVG/300-dpi PNG, 20개 CSV table, manifest,
`SHA256SUMS`가 있다. Manifest SHA-256은 bundle suffix와 같은
`c80194480ab557a68b2306fd0c25c5cef7c5533da83b283500375fb0ee9faa99`다.
동일 입력/코드로 독립 재생성한 44개 Figure/Table 파일과 manifest는 byte-identical
했다. 이 과정은 checkpoint, example, feature array, score array를 읽지 않은
manifest-only post-result analysis다.

### 해석 금지선

- Primary를 `same ID performance`로 부르지 않는다. Accuracy/ECE는 PASS,
  NLL은 improvement direction으로 FAIL이다.
- Marginal은 adverse score movement의 localization이지 causal mediator가 아니다.
- Current WRN cross-LR difference는 stream을 공유하지 않으므로 descriptive다.
- Geometry panel은 `concordant but not monotone`이며 unique mediator를
  정하지 않는다.
- Section 14 carrier는 기술 검증을 통과한 14/14 context에서
  `rho > 1`이 0/14였다. 재정의하여 rescue하지 않는다.

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

## 9. 논문 판정

선행연구가 이미 점유한 것은 RMD, L2 normalization,
spectrum--allocation identity, size--stretch, representation-side MD failure,
그리고 training detail이 detector ranking에 영향을 준다는 범위다. 이
논문의 단위는 더 구체적이다.

```text
matched decay-coupling intervention
-> exact same-pair Gain/Loss/Churn
-> RMD/Marginal score-response localization
-> matched epoch/depth formation
```

다만 현재 evidence가 unlock한 문장은 다음 정도다.

> A matched decay-coupling intervention can reorganize the pair ordering of a
> protocol-fixed, branch-refitted Raw Mahalanobis detector; in the tested WRN
> contexts, adverse movement is predominantly localized to Marginal and is
> amplified over training and depth.

아직 unlock하지 못한 문장은 다음이다.

> We identify which optimizer recipes generally create Mahalanobis-hostile
> representations.

현재 Plan B는 후자로 승격하지 않는다. ResNet replication은 현재의
controlled pattern을 architecture 밖에서 검증하는 것이지 broad recipe map을
만드는 실험이 아니다.

Adam coupled/decoupled update 차이를 `-eta lambda (P_t-I) theta`로 쓰는 설명은
frozen-moment local heuristic일 뿐 exact Adam dynamics가 아니다. Coupled decay는
gradient뿐 아니라 first/second moment history와 denominator를 함께 바꾼다. Jacobian을
통한 feature-covariance drift도 plausible interpretation이지 현재 theorem이 아니다.
이 이론은 motivation/appendix에만 두고 telemetry 및 cross-cell concordance를 넘어서
인과 결론으로 사용하지 않는다.

### 9.1 Prospective ResNet-18 replication: `PARTIAL`

Card 13 Section 15에서 결과를 보기 전에 동결한 20-run
ResNet-18/CIFAR-10 gate를 완료했다. 모든 checkpoint identity, ID-only fit,
sample order, protected coverage, pair-accounting, numerical reconstruction이
통과했으며, 독립 재계산도 같은 판정을 냈다. 아래 값은 WRN과 동일하게 seed 안에서
dataset을 먼저 Near/Far 평균한 뒤 다섯 seed를 요약한 C--D다.

| Context | Region | C Raw MD | D Raw MD | Delta Raw MD [90% interval] | Churn | Delta RMD | `phi_RMD` | `phi_Marginal` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| large, LR 1e-3 | Near | 0.5137 | 0.5492 | -0.0356 [-0.0478, -0.0233] | 0.3244 | -0.1426 | +0.0211 | -0.0567 |
|  | Far | 0.5005 | 0.6763 | -0.1758 [-0.1902, -0.1614] | 0.3617 | -0.1302 | +0.0090 | -0.1848 |
| small, LR 3e-4 | Near | 0.6037 | 0.6167 | -0.0130 [-0.0232, -0.0027] | 0.2763 | -0.0015 | +0.0392 | -0.0522 |
|  | Far | 0.6464 | 0.7437 | -0.0973 [-0.1287, -0.0660] | 0.2552 | +0.0075 | +0.0369 | -0.1342 |

여섯 dataset의 Raw-MD C--D mean은 모두 음수였다.

| Context | CIFAR-100 | TinyImageNet | MNIST | SVHN | Texture | Places365 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large | -0.0314 | -0.0397 | -0.0116 | -0.4307 | -0.2088 | -0.0521 |
| small | -0.0062 | -0.0198 | -0.0800 | -0.1888 | -0.0987 | -0.0219 |

**재현된 부분:** large와 small의 Near/Far Raw-MD 방향, 최소 4/5 seed 방향,
large Churn `>=0.10`, small absolute gap의 상대적 감소, Marginal의 adverse sign과
50% 초과 accounting, small-cell Accuracy/NLL/ECE guardrail이 모두 통과했다.
Small-cell absolute paired NLL mean은 margin `0.08`에 대해 `0.07799`로
통과했으며 interval은 descriptive다.

**깨진 부분:** large-context Near에서
`abs(Delta RMD)=0.1426 > abs(Delta Raw MD)=0.0356`였다. 이 RMD 평균은 seed 2와
3의 Near delta `-0.3532/-0.3500` 때문에 커졌고 나머지 세 seed는
`-0.0016/-0.0063/-0.0021`이었다. 이는 계산 경계가 아니라 architecture/seed
heterogeneity다. Far에서는 RMD gap `0.1302`가 Raw-MD gap `0.1758`보다 작아
동결 조건을 통과했다.

따라서 15개 expanded boolean check 중 `near_rmd_gap_smaller` 하나만 실패했고 과학 판정은
**`PARTIAL`**이다. 이는 Raw-MD pair non-invariance 자체의 실패가 아니지만,
“RMD attenuation까지 포함한 WRN의 complete pattern이 architecture 밖에서
재현됐다”는 주장을 잠근다. 사전등록에 따라 architecture-general claim과 현재 ICLR
main-paper promotion을 열지 않으며 LR/WD grid, CIFAR-100, 새 mechanism으로
rescue하지 않는다. WRN result는 그 실험 범위 안의 결과로 남는다.

## 10. 동결된 본문 구조

1. **Introduction:** aggregate equivalence는 pair-behavior equivalence가 아님.
2. **Related Work:** geometry, normalization, experiment reliability, optimizer/NC.
3. **Problem Setup:** parallel siblings, branch refit, Gain/Loss/Churn,
   RMD/Marginal accounting.
4. **Controlled Design:** WRN four cells, seeds, ID guardrails, protected
   population, inferential unit.
5. **Results:** controlled non-invariance, pair multiplicity, score
   localization, formation, context/negative boundaries.
6. **ResNet Replication Boundary:** `PARTIAL` result와 RMD seed heterogeneity;
   architecture-general claim이 열리지 않은 이유.
7. **Discussion and Limitations:** ID Pareto, causal-LR 금지, WRN-scope claim.
8. **Conclusion:** matched intervention에서 확립한 연결만 요약.

## 11. 실행 라우팅

현재 artifact로 four-cell endpoint/pair accounting, within-LR WD/DiD,
Marginal·ID/OOD-side accounting, time/depth, geometry/alignment, telemetry,
SGDM, spectral applicability과 두 negative mechanism gate가 완료됐다.

두 병렬 engineering 작업은 완료됐다.

1. Detailed CSV에서 seed-first macro, paired 90% interval, main
   figure/table pack을 생성하고 PR #128로 병합했다.
2. Card 13 Section 15의 ResNet-18/CIFAR-10 planner, provenance, endpoint ID
   artifact, pending evaluator, pilot와 approval packet을 PR #129로 병합했다.

ResNet v3 pilot, 20-run main training, ID export/fitting, ID-test guardrails,
protected endpoint evaluation과 독립 검산까지 완료됐다. Technical status는
`PASS`, frozen scientific verdict는 `PARTIAL`이다. Production evaluator는 PR
#137, score-scale verifier correction은 PR #139에 병합됐다.

WRN broad LR/WD grid는 진행하지 않는다. ResNet `FULL` 이후에도
기본값은 새 학습을 멈추고 논문을 완성하는 것이다. `PARTIAL` 또는
`FAIL`을 LR grid, new optimizer, 새 mechanism으로 rescue하지 않는다.

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
| optimizer-side origin is established generally | NOT SUPPORTED | Plan B is a controlled decay-coupling case study |
| classifier-insensitive carrier explains the result | REJECTED FOR THIS PAPER | frozen gate had `rho > 1` in 0/14 contexts |
| architecture-general pattern | NOT UNLOCKED | prospective ResNet gate was `PARTIAL`; Near RMD attenuation failed |

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

### Seed-first paper figure/table pack

- generator analysis commit:
  `1a329474fbf4df00996f204a8e598cfb2c537d5a`
- generator merge commit:
  `aefb7363dcd30a6c7637c5b545b50b69323cbfd4`
- paper-pack manifest SHA-256:
  `fa2b1535af55b74c64a873734afd11eca9e1ebf01ef50b99d934fac86da61a82`
- source merged-result SHA-256:
  `ec0d235f3e85ba60635998b919b15b24ec6987efd20c7e43f09893881c9c24ed`
- HF bundle:
  `hf://buckets/contra333/ICLR_RUN/aggregate/task_f_paper_pack_20260818/fa2b1535af55b74c64a873734afd11eca9e1ebf01ef50b99d934fac86da61a82/`
- local ignored copy:
  `artifacts/task_f_paper_pack_1a329474/`

The pack contains deterministic seed rows, summary intervals, Tables 1--3,
Figures 1--4, and appendix geometry/negative-gate outputs. It performs no new
checkpoint inference, detector fitting, or protected access.

### ResNet-18 endpoint replication

- evaluation source SHA: `9538b0d34acf153451183223a88b3f3d98d9d7d1`
- score-verifier/recovery SHA: `9dd22cc62b434e0253e4fd6966c8c685a6edef64`
- production plan SHA-256:
  `3bc0348684702973a16d3b6f0c8fe84ee24a565b425e1149ff214d19af090df2`
- protected authorization SHA-256:
  `1c51e7cb81d0530659d99b2013c671244221baaa3d347e3ae1f64dbf542cf42d`
- central terminal identity:
  `f0492271d31d13a1d8b4774303ba22485c701844e46bf37a2947743b5343721f`
- central terminal file SHA-256:
  `f1a043a56adc54522550e14f2b8d4a26222488fdc9e51569ba9b99222431bab0`
- independent validation identity:
  `5b58c3fa85a3c62542f181139a8e3e777941ab00e70acec52a817adbb3622656`
- independent validation file SHA-256:
  `c1c587b623d05607fca3795e60c1c35fa3fd44ee9a09a681458dd9e65f276fe3`
- bundle `SHA256SUMS` SHA-256:
  `a67e9bad5324b06df43d62153fe73aa4fa5f70e6b0c39d78b7fe6c8be51992c0`
- verified shared bundle:
  `hf://buckets/contra333/ICLR_RUN/aggregate/resnet18_cifar10_replication_v3_evaluation_20260819/f0492271d31d13a1d8b4774303ba22485c701844e46bf37a2947743b5343721f/`

The bundle contains terminal/validation JSON, the ten paired score records and
score arrays, execution plans, authorization, recovery records, and portable
checksums. Large checkpoints and exported features remain on the execution
hosts. The numerical recovery did not rerun inference or detector fitting.

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

## 14. Classifier-insensitive fast kill: negative appendix result

Card 13 Section 14의 사전 고정 statistic을 14개 C--D endpoint pair에
적용했다. 14/14가 identity, checksum, projection reconstruction,
precision validation을 통과했지만 `rho > 1`은 0/14였다.

| Cell | `rho > 1` | median `rho` | range |
| --- | ---: | ---: | ---: |
| `LR=1e-3, WD=1e-4` anchor | 0/5 | 0.00795 | 0.00780--0.00816 |
| `LR=1e-3, WD=1e-3` | 0/3 | 0.00443 | 0.00426--0.00448 |
| `LR=3e-4, WD=1e-4` all-PASS | 0/3 | 0.01005 | 0.00954--0.01015 |
| `LR=3e-4, WD=1e-3` | 0/3 | 0.00473 | 0.00454--0.00478 |

Centered classifier rowspace rank는 모든 context에서 9였다. Maximum
projection-energy reconstruction error는 `5.53e-8`, anchor seed 0 GPU
float32--CPU float64 `rho` relative difference는 `1.25e-8`이었고 경계
context는 없었다. 따라서 이 논문에 사전 지정한
classifier-insensitive carrier는 지지되지 않는다. 이것은 모든 가능한
classifier-insensitive geometry의 부재 정리가 아니며, 이 결과를 이용해
다른 subspace, threshold, normalization으로 rescue하지 않는다.

Artifact identity:

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
