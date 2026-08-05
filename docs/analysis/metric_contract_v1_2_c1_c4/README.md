# C1–C4 Metric Contract v1.2 로컬 분석

## 기술 요약

checksum 검증을 통과한 Metric Contract v1.2 중앙 aggregate만 사용하여 C1–C4의 10개 unique configuration과 각 3개 training seed를 정리했다. Confirmatory primary인 `last.pt`에서 C1의 ID-test accuracy는 SGD 0.9541 ± 0.0013, ADAM 0.9385 ± 0.0017, ADAMW 0.9525 ± 0.0018이다. 모든 `mean ± SD`는 seed 0·1·2의 산술평균과 sample standard deviation (`ddof=1`)이며 confidence interval이 아니다.

OOD 결과는 CIFAR-100, TinyImageNet, MNIST, SVHN, Textures, Places365를 끝까지 분리했다. 아래의 dataset별 최고 AUROC는 C1 canonical detector panel을 기술적으로 훑은 결과이며, 새로운 primary detector를 사후 선택한 것이 아니다. Geometry–OOD 상관은 unique config가 10개이고 seed 0가 role 선택에 사용되었으므로 모두 exploratory association으로만 해석한다.

## 고정된 role configuration

| Role | Optimizer | LR | WD | Status |
| --- | --- | --- | --- | --- |
| C1 | SGD | 0.1 | 0.0005 | selected |
| C1 | ADAM | 0.0003 | 0 | selected |
| C1 | ADAMW | 0.003 | 0.1 | selected |
| C2 | SGD | 0.3 | 0.0001 | selected |
| C2 | ADAM | 0.0003 | 0.001 | selected |
| C2 | ADAMW | — | — | protocol_defined_absent |
| C3 | SGD | 0.1 | 0 | selected |
| C3 | ADAM | 0.0003 | 0 | selected |
| C3 | ADAMW | 0.001 | 0.0001 | selected |
| C4 | SGD | 0.3 | 0 | selected |
| C4 | ADAM | 0.0003 | 0.0001 | selected |
| C4 | ADAMW | 0.003 | 0.001 | selected |

Adam C1과 C3은 같은 scientific configuration이다. 역할 해석을 위해 두 위치에 모두 표시하지만 association 분석에는 한 번만 포함한다. AdamW C2는 `NA — protocol absent`이며 어떤 값도 대입하지 않았다.

## C1–C4 ID·geometry 비교

논문용 ID 표는 [`last` primary](tables/id/paper_last.md)와 [`best_val` control](tables/id/paper_best_val.md), geometry 표는 [`last` primary](tables/geometry/paper_last.md)와 [`best_val` control](tables/geometry/paper_best_val.md)이다. 그림은 각 seed 값과 mean ± sample SD를 함께 보인다. Notion의 `Text & Markdown` 가져오기용 primary 비교 문서는 [`notion_c1_c4_primary_comparison.md`](notion_c1_c4_primary_comparison.md)이다.

- [`last.pt` role comparison](figures/role_summary_last.svg)
- [`best_val.pt` role comparison](figures/role_summary_best_val.svg)

calibration sensitivity, feature-norm quantile, classwise summary, Neural Collapse diagnostic, spectrum, LID, TwoNN을 포함한 중앙 aggregate의 모든 scalar는 `tables/id/`와 `tables/geometry/`에 저장했다. Main/context 배치는 Card 11을 따르며 원본 aggregate의 tier 값은 수정하지 않았다. Seed-matched optimizer delta는 추론검정 없이 `tables/deltas/`에 별도로 제공한다.

## OOD dataset별 결과

| OOD dataset | C1 optimizer | Detector | AUROC mean ± SD |
| --- | --- | --- | --- |
| CIFAR-100 | SGD | `detector/knn_l2_k50` | 0.905050 ± 0.001195 |
| TinyImageNet | SGD | `detector/knn_l2_k50` | 0.923844 ± 0.001346 |
| MNIST | SGD | `detector/neco_author_std_dim100` | 0.961736 ± 0.004277 |
| SVHN | SGD | `detector/mahalanobis_pp` | 0.984284 ± 0.002994 |
| Textures | SGD | `detector/mahalanobis_pp` | 0.966999 ± 0.002912 |
| Places365 | ADAMW | `detector/relative_mahalanobis_raw` | 0.929226 ± 0.002549 |

`tables/ood/` 아래 각 dataset 디렉터리에는 `last`와 `best_val`을 분리한 canonical 11-detector 논문 표, 19-detector 전수 3-seed scalar 표, seed=0 표가 있다. AUROC, FPR@95, AUPR-In, AUPR-Out을 raw sample 수준에서 dataset 간 pooling하지 않았다. Near/far/overall macro는 `tables/cross_dataset_macro_audit/`에만 격리했으며 위 dataset별 결론에는 사용하지 않았다. Detector의 본문·부록 배치는 [`detector_panel.md`](tables/detector_panel.md)에 고정했다.

## Geometry–OOD association은 exploratory

각 OOD dataset과 checkpoint에서 primary coefficient는 10개 unique configuration mean의 Spearman correlation이다. 강건성 계수와 95% CI는 30개 seed-level 관측을 사용한다. 10,000회 configuration-block bootstrap에서는 config 10개를 복원추출하고 선택된 각 config의 seed 3개를 함께 유지한 뒤, replicate 안에서 raw value를 다시 rank 변환해 Spearman coefficient를 계산했다. Two-sided permutation은 10개 config-mean block을 10,000회 섞었다. Benjamini–Hochberg correction은 dataset·checkpoint·endpoint별 16×11 가설군에 적용했다. Fixed random seed는 `20260805`이다. FPR association은 `−FPR@95`를 사용하므로 양의 상관은 항상 더 좋은 OOD 성능을 뜻한다.

아래에는 dataset별 `last.pt` AUROC matrix에서 absolute config-mean correlation이 가장 큰 한 셀을 위치 확인용으로 표시한다. `seed-level cluster 95% CI`는 바로 앞의 seed-level coefficient에 대한 구간이며 config-mean coefficient의 구간이 아니다. 이 표는 confirmatory evidence나 causal result가 아니다.

| OOD dataset | Geometry | Detector | config-mean ρ | seed-level ρ | seed-level cluster 95% CI | q |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | NC4 | `detector/mahalanobis_raw` | 0.915 | 0.871 | [0.523, 0.966] | 0.141 |
| TinyImageNet | NC4 | `detector/mahalanobis_raw` | 0.891 | 0.831 | [0.464, 0.950] | 0.238 |
| MNIST | SW participation ratio | `detector/knn_l2_k50` | 0.976 | 0.891 | [0.607, 0.932] | 0.009 |
| SVHN | RankMe | `detector/ctm_prototype_cosine` | 0.927 | 0.898 | [0.583, 0.946] | 0.023 |
| Textures | RankMe | `detector/knn_l2_k50` | 0.927 | 0.917 | [0.604, 0.954] | 0.047 |
| Places365 | CDNV | `detector/neco_author_std_dim100` | -0.830 | -0.757 | [-0.917, -0.366] | 0.339 |

각 dataset은 `figures/associations/` 아래 독립된 AUROC와 FPR heatmap을 갖는다. AUPR-In/Out association은 같은 절차로 계산해 `tables/associations/`의 appendix CSV에 보존했으며 heatmap panel로 승격하지 않았다. Dataset을 합친 correlation은 없다. Config 수, seed-0 selection bias, multiplicity, 공통 checkpoint provenance 때문에 q-value가 작더라도 인과나 일반화 주장을 하지 않는다.

## Metric 정의와 Methods 연결

[`methods_crosswalk.csv`](tables/methods_crosswalk.csv)와 [`methods_crosswalk.md`](tables/methods_crosswalk.md)는 formula ID, artifact key, checkpoint/split, direction, reporting tier, transform, Card 11 문장을 연결한다. 논문에 바로 옮길 English 문단은 [`METHODS_READY_EN.md`](METHODS_READY_EN.md)에 있다. 핵심 규칙은 다음과 같다.

- `last.pt`: confirmatory primary; `best_val.pt`: 별도 deployment control.
- 모든 3-seed 값: seed 0·1·2 산술평균 ± sample SD (`ddof=1`).
- AUROC: ID positive; FPR@95: inclusive linear 5th-percentile ID threshold.
- Geometry: 정의가 별도 transform을 명시하지 않으면 raw ID-training feature.
- seed=0 표: audit layer이며 독립적인 confirmatory replication이 아님.

## 검증·한계·결과 경계

분석 시작 시 모든 input hash, 60 checkpoint job, 30 training identity, 10 configuration, 31,720개 successful seed aggregate, 95,160개 successful per-checkpoint scalar, seed=0 exact parity를 다시 검증했다. 전수 정리는 중앙 aggregate에 있는 scalar만 대상으로 한다. Raw score, feature array, 전체 spectrum, class-pair matrix와 그 밖의 checkpoint-bundle intermediate는 내려받거나 재계산하지 않았으며 **미포함**이다.

따라서 이 보고서는 재현 가능한 descriptive result와 exploratory association을 제공한다. Optimizer causality, `n=3`에 기반한 통계적 우월성, universal best detector, cross-dataset generalization을 확립하지 않는다.

## 다음 사용 단계

Metric 정의를 바꾸지 않은 채 dataset별 표에서 논문 본문과 부록의 최종 배치를 선택할 수 있다. Confirmatory inference, 새 detector selection, DDU/SN ablation, 추가 protected population은 별도로 사전 명시해야 한다.
