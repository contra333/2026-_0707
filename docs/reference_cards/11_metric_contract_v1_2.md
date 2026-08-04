# Reference Card 11: Metric Definition Dictionary v1.2

## 0. 목적, 범위, 권위

이 문서는 학습된 classifier를 평가할 때 **무슨 값을 계산하는지**, 그
값을 논문에 **어떤 수학적 정의로 기록하는지**, 구현이 그 정의와
**일치하는지 어떻게 검증하는지**를 한 곳에 고정한다.

실행 가능한 범위는 다음 하나다.

```text
model: WRN-28-10
dataset: CIFAR-10
feature endpoint: model(x, return_features=True)의 640차원 raw penultimate feature
definition_version: metric_contract_v1.2
implementation_status: frozen definition; runtime implementation pending
```

ImageNet-200과 OpenOOD의 9,000-image ID compatibility 평가는 이 버전의
실행 범위가 아니다. OpenOOD에서는 공개 OOD datalist membership, near/far
grouping, deterministic preprocessing만 사용한다. 이 문서는 metric 이름,
수식, reporting tier, artifact key, 실패 상태와 검증 oracle의 권위다.
인접한 데이터·feature·detector 세부 규칙은
[`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md),
[`06_feature_ood_detectors.md`](06_feature_ood_detectors.md),
[`08_raw_feature_artifact_contract.md`](08_raw_feature_artifact_contract.md),
[`09_core_representation_metrics.md`](09_core_representation_metrics.md)와
함께 읽되, 서로 다른 수식이나 이름을 허용하지 않는다. Pinned paper와
repository는 [`docs/sources.lock.yaml`](../sources.lock.yaml)에 기록한다.

이 문서가 정의를 동결했다는 사실은 구현이나 실험 결과가 존재한다는
뜻이 아니다. protected split 접근, feature extraction, detector 실행과
연구 결과 생성은 별도 Issue의 명시적 승인이 필요하다.

## 1. 공통 평가 계약

### 1.1 표기

| 기호 | 의미 |
| --- | --- |
| `N`, `K`, `p` | sample 수, class 수, feature 차원; 현재 `K=10`, `p=640` |
| `z_i in R^p` | sample `i`의 raw penultimate feature |
| `ell_i in R^K` | raw classifier logit |
| `y_i`, `yhat_i` | true label, classifier prediction |
| `I_c`, `N_c` | class `c`의 sample index set과 크기 |
| `mu_c` | raw feature의 class mean |
| `mu_G` | class-balanced global mean `(1/K) sum_c mu_c` |
| `m_c` | centered class mean `mu_c - mu_G` |
| `W`, `b` | classifier weight `[K,p]`와 bias `[K]` |
| `s(x)` | 큰 값일수록 ID-like인 detector score |

Class statistics는 달리 쓰지 않는 한 deterministic `id_train` raw feature로
계산한다.

### 1.2 split registry

| logical role | physical config key | membership | 허용 용도 |
| --- | --- | ---: | --- |
| `id_train` | `id_train` | 45,000 | training, detector fit, class/covariance/reference-bank fit, primary geometry |
| `id_validation` | `id_validation` | 5,000 | checkpoint/HPO selection, temperature fit, held-out geometry control |
| `id_test_primary` | `id_test` | 10,000 | 승인 후 최종 ID 성능과 모든 v1.2 OOD metric의 ID side |
| compatibility provenance | `id_test_openood` | 9,000 | membership 보존만; v1.2에서는 추출·평가하지 않음 |

`id_test_primary`, OOD test, geometry, detector 결과는 classifier 또는
checkpoint 선택에 사용하지 않는다. `ood_validation_tin`도 selection이나
detector tuning에 사용하지 않는다.

### 1.3 checkpoint와 reporting tier

| checkpoint | 역할 | 보고 방식 |
| --- | --- | --- |
| `last.pt` | optimizer에서 terminal geometry/OOD로 이어지는 confirmatory primary | 모든 main table의 primary |
| `best_val.pt` | validation-selected deployment control | 별도 control 열/표 |

한 checkpoint에서 fit한 scaler, PCA, covariance, class mean, feature bank,
temperature 또는 ViM `alpha`를 다른 checkpoint에 재사용하지 않는다.

Reporting tier는 다음 다섯 값만 쓴다.

- `primary`: main claim과 main table.
- `control`: `best_val.pt` 또는 logit-only deployment control.
- `appendix`: 사전 고정 sensitivity 또는 scaled audit value.
- `exploratory`: mechanism 탐색; confirmatory claim 금지.
- `excluded`: 정의는 설명하지만 v1.2 artifact를 만들지 않음.

### 1.4 dtype, tolerance, tie, 상태

- raw cache는 card 08에 따라 float32로 저장할 수 있지만 fitting,
  covariance, eigendecomposition, distance reference calculation은 float64다.
- normalization 전 `norm == 0`과 `0 < norm <= epsilon_norm`을 세며
  `epsilon_norm = 1e-12`다.
- exact-zero 또는 non-finite input은 affected metric `failed`; near-zero는
  `degenerate`다. clamp로 나눗셈을 계속해도 상태를 `success`로 바꾸지 않는다.
- class/distance tie의 deterministic tie-break는 smallest class index와
  smallest sample ID다. Metric threshold의 tie는 해당 정의의 inclusive
  comparison을 그대로 적용한다.
- covariance eigenvalue tolerance는
  `tol = lambda_max * p * eps_float64`다. `lambda < -tol`이면 `failed`,
  `-tol <= lambda < 0`이면 0으로 clip하고 count를 저장한다.
- 모든 scalar는 `success | degenerate | failed` 상태와 reason code를 가진다.
  의미 없는 scalar는 0으로 대체하지 않고 JSON `null`과 non-finite-free
  serialization을 사용한다.
- checkpoint/seed별 값을 먼저 계산한다. seed aggregate는 arithmetic mean과
  sample standard deviation (`correction=1`)을 사용하며 per-seed 값도 남긴다.

### 1.5 공통 result record

모든 결과는 최소한 다음을 기록한다.

```text
metric_name, formula_id, definition_version, reporting_tier
checkpoint_role, checkpoint_sha256, completed_epoch, training_seed
dataset_protocol, membership_manifest_sha256, fit_split, query_split
feature_endpoint, transform, score_direction
fit_dtype, evaluation_dtype, sample_count, class_counts
fixed_hyperparameters, runtime_diagnostics, source_commit
artifact_key, status, reason_codes, smoke_only
```

Per-sample score, per-class vector, pair matrix, full spectrum 또는 raw norm처럼
aggregate를 재계산하는 데 필요한 중간값도 함께 저장한다.

## 2. ID 성능, calibration, failure ranking

### 2.1 Accuracy `[ID-1]`

**무엇을 재는가.** Top-1 classifier correctness의 비율이다.

$$
\operatorname{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}
\mathbf 1\{\arg\max_c \ell_{ic}=y_i\}.
$$

- tier/split: primary, `id_test_primary`; `last.pt` primary와 `best_val.pt`
  control을 분리한다.
- 방향/key: 클수록 좋음; `id/accuracy`.
- 저장: `num_correct`, `N`, per-class correct/count, predictions.
- 실패: empty input, label 범위 오류, non-finite logits.
- 검증: hand-count와 sample/class permutation invariance.
- 논문 표준 문장: “We report top-1 accuracy on the frozen 10,000-image
  CIFAR-10 test split.”

### 2.2 Negative Log-Likelihood `[ID-2]`

$$
\operatorname{NLL}=-\frac{1}{N}\sum_{i=1}^{N}
\log\frac{\exp(\ell_{i y_i})}{\sum_c\exp(\ell_{ic})}.
$$

- tier/split: primary, raw `id_test_primary` logits.
- 방향/key: 작을수록 좋음; `id/nll_raw`.
- hyperparameter: mean reduction, stable `logsumexp`, no label smoothing.
- 저장: per-sample NLL, mean, count.
- 실패: Accuracy와 동일.
- 검증: direct float64 `logsumexp`와 equality; logit common-offset invariance.
- 논문 표준 문장: “NLL is the mean negative log-probability assigned to the
  true class using unscaled logits.”

### 2.3 Expected Calibration Error `[ID-3]`

`p_i=softmax(ell_i)`, `g_i=max_c p_ic`, `yhat_i=argmax_c p_ic`로 둔다.
`M`개의 equal-width bin은 `B_1=[0,1/M]`,
`B_m=((m-1)/M,m/M]` (`m>1`)다.

$$
\operatorname{ECE}_M=\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\right|.
$$

Empty bin의 기여는 0이다.

- tier: `M=15` primary; `M in {10,30}` appendix sensitivity.
- split/transform: raw `id_test_primary` logits, no fitted parameter.
- 방향/keys: 작을수록 좋음; `id/ece_raw_m15`, `id/ece_raw_m10`,
  `id/ece_raw_m30`.
- 저장: bin edges, count, accuracy, mean confidence, weighted gap.
- 실패: confidence가 `[0,1]` 밖, sample count 0, non-finite probability.
- 검증: boundary confidence `0`, `1/M`, `1`; perfect one-hot fixture; sample
  permutation invariance.
- 논문 표준 문장: “ECE uses 15 fixed equal-width confidence bins; 10- and
  30-bin values are prespecified sensitivity analyses.”

### 2.4 Validation-fitted temperature scaling `[ID-4]`

Checkpoint마다 `id_validation`에서 scalar positive temperature를 fit한다.

$$
\hat T=\arg\min_{T>0}\operatorname{NLL}
\left(\operatorname{softmax}(\ell_{val}/T),y_{val}\right),
\qquad T=\exp(\theta).
$$

Frozen optimizer:

```yaml
initial_log_T: 0.0
dtype: float64
optimizer: torch.optim.LBFGS
lr: 0.01
max_iter: 200
max_eval: 250
history_size: 100
tolerance_grad: 1.0e-7
tolerance_change: 1.0e-9
line_search_fn: strong_wolfe
accepted_temperature: [0.01, 100.0]
accepted_nll_increase: 1.0e-12
fallback: np.geomspace(0.01, 100.0, 2001)
```

LBFGS가 예외를 내거나, objective/gradient/T가 non-finite이거나, T가 범위
밖이거나, validation NLL이 `T=1`보다 `1e-12` 초과 증가하면 fallback을
사용한다. Ascending fallback grid에서 `np.argmin`의 첫 index를 택한다.

- tier: `best_val.pt` deployment control; `last.pt`에도 독립 fitting한
  confirmatory calibration control을 저장할 수 있으나 두 T를 공유하지 않는다.
- evaluation: `id_test_primary`에 `ell/T`를 적용해 NLL와 ECE-M15를 계산한다.
- 방향/keys: `id/temperature`, `id/nll_ts`, `id/ece_ts_m15`.
- 저장: T, optimizer steps/evaluations, val/test before/after NLL/ECE,
  `fallback_used`.
- 실패: fallback 전체가 non-finite이면 `failed`. Argmax 또는 accuracy가
  scaling 전후 달라지면 `failed`.
- source: `gpleiss/temperature_scaling@ce1154ec...`; positive log-T와
  fallback은 project-constrained variant다.
- 검증: official code synthetic parity, positive T, argmax invariance,
  grid-minimum hand fixture.
- 논문 표준 문장: “A single positive temperature was fitted on ID validation
  by float64 NLL minimization and applied once to the frozen ID test logits.”

### 2.5 Misclassification AUROC `[ID-5]`

$$
t_i=\mathbf 1\{\hat y_i=y_i\},\qquad
g_i=\max_c\operatorname{softmax}(\ell_i)_c.
$$

AUROC는 `t_i=1`을 positive, raw MSP를 score로 계산한다.

- tier/split: primary failure-ranking control, `id_test_primary`.
- 방향/key: 클수록 correct/incorrect ranking이 좋음;
  `id/misclassification_auroc_msp_raw`.
- 실패: all-correct 또는 all-wrong이면 binary class가 하나뿐이므로
  `degenerate`와 null scalar를 저장한다.
- 검증: two-score rank fixture, score tie의 half-credit, temperature 미사용.
- 논문 표준 문장: “Misclassification AUROC treats correct predictions as the
  positive class and ranks them using raw maximum softmax probability.”

### 2.6 AUGRC `[ID-6]`

Binary error에 대해 Accuracy와 위 AUROC를 사용한다.

$$
\operatorname{AUGRC}=
(1-\operatorname{AUROC}_{correct})\,a(1-a)
+\frac{1}{2}(1-a)^2,
$$

여기서 `a=Accuracy`다.

- tier/direction/key: primary, 작을수록 좋음; `id/augrc_msp_raw`.
- 저장: accuracy, AUROC, 두 additive term.
- 실패: Misclassification AUROC가 degenerate이면 AUGRC도 degenerate.
- source: FD-Shifts selective-classification evaluation.
- 검증: perfect accuracy에서 0; direct formula equality.
- 논문 표준 문장: “AUGRC combines residual error prevalence and raw-MSP
  ranking quality using its closed-form binary-error expression.”

### 2.7 AURC `[ID-7]`

Confidence가 같은 sample을 하나의 threshold group으로 취급한다. 모든
sample을 accept한 `(coverage=1, risk=error rate)`에서 시작해 confidence가
낮은 group부터 reject하여 coverage를 0까지 내린다. Non-empty accepted set의
risk는 accepted error mean이며 coverage 0 endpoint는 직전 risk를 반복한다.

$$
\operatorname{AURC}=-\operatorname{Trapezoid}
\left(\{R_j\},\{C_j\}\right),\qquad C_0=1>C_1>\cdots>C_J=0.
$$

- tier/direction/key: appendix, 작을수록 좋음; `id/aurc_msp_raw`.
- confidence: raw MSP; temperature scaling 금지.
- scale: canonical scalar는 `[0,1]` unscaled area. FD-Shifts의 display
  `x1000`은 metadata일 뿐 별도 primary metric이 아니다.
- 저장: unique thresholds, coverage, risk, group counts, display scale.
- 실패: empty input; all confidence equal은 정상이며 AURC=error rate.
- source: `IML-DKFZ/fd-shifts@c4467aec...`.
- 검증: identical-confidence fixture, all-equal fixture, pinned code 결과를
  1000으로 나눈 값과 parity.
- 논문 표준 문장: “AURC is the unscaled trapezoidal area under the tie-grouped
  raw-MSP risk-coverage curve.”

## 3. Logit-based OOD controls

모든 score는 raw logit에서 계산하고 temperature를 적용하지 않는다.

### 3.1 Maximum Softmax Probability `[LOGIT-1]`

$$s_{MSP}(x)=\max_c\operatorname{softmax}(\ell(x))_c.$$

- 의미/tier: confidence-only OOD baseline, primary control.
- 방향/key: 클수록 ID-like; `ood_score/msp`.
- fit/hyperparameter: 없음.
- 저장/검증: probabilities, prediction, score; official
  `hendrycks/error-detection@276d605b...`와 parity.
- 논문 표준 문장: “MSP is the largest raw-logit softmax probability.”

### 3.2 Maximum Logit Score `[LOGIT-2]`

$$s_{MLS}(x)=\max_c\ell_c(x).$$

- 의미/tier: softmax normalization 없는 logit magnitude control, primary.
- 방향/key: 클수록 ID-like; `ood_score/max_logit`.
- 실패: non-finite logits.
- 검증: direct maximum; class permutation invariance.
- 논문 표준 문장: “Maximum Logit Score is the largest unscaled classifier
  logit.”

### 3.3 Energy-T1 `[LOGIT-3]`

$$s_{Energy-T1}(x)=\log\sum_c\exp(\ell_c(x)).$$

이는 `T=1`에서 conventional energy의 음수이며 이미 ID-like다.

- tier/key: primary control; `ood_score/energy_t1`.
- hyperparameter: `T=1`; sweep 또는 calibration T 사용 금지.
- 저장: stable logsumexp와 max-logit component.
- source/검증: `wetliu/energy_ood@77f3c09b...`와 synthetic parity.
- 논문 표준 문장: “We use the ID-oriented negative energy score
  `logsumexp(logits)` at fixed temperature one.”

### 3.4 Predictive Entropy `[LOGIT-X]`

$$H(x)=-\sum_c p_c(x)\log p_c(x).$$

Entropy는 클수록 uncertain/OOD-like이므로 공통 ID-like score 방향과 반대다.
v1.2에서는 `excluded`이며 artifact key가 없다. 부호를 바꾼 새 detector를
조용히 추가하지 않는다. 수식은 다른 논문의 entropy와 구별하기 위해서만
기록한다.

## 4. Gaussian distance and density detectors

### 4.1 공통 tied covariance `[GAUSS-0]`

$$
\mu_c=\frac{1}{N_c}\sum_{i\in I_c}z_i,\qquad
r_i=z_i-\mu_{y_i},\qquad
\Sigma_W=\frac{1}{N}\sum_i r_ir_i^\top.
$$

Float64 `EmpiricalCovariance(assume_centered=True)`를 residual matrix에 fit해
`precision_`을 사용한다. Denominator는 `1/N`; explicit jitter나
`np.linalg.inv`는 사용하지 않는다. Covariance, precision, eigen diagnostics,
condition number를 저장한다.

### 4.2 Mahalanobis `[GAUSS-1]`

$$
d_c(z)=(z-\mu_c)^\top\Sigma_W^\dagger(z-\mu_c),\qquad
s_{Maha}(z)=-\min_c d_c(z).
$$

- 의미/tier: raw feature가 가장 가까운 ID class Gaussian에 얼마나
  가까운지, primary.
- fit/query/transform: `id_train`; `id_test_primary`와 각 OOD; raw.
- 방향/key: 클수록 ID-like; `detector/mahalanobis_raw`.
- 제외: input perturbation, multi-layer ensemble, logistic regression.
- 실패: precision/score non-finite 또는 covariance backend failure.
- source/검증: OpenOOD `3c35632e...`; direct float64 quadratic fixture.
- 논문 표준 문장: “Mahalanobis is the negative minimum class-conditional
  quadratic distance under a tied biased within-class covariance.”

### 4.3 Marginal Mahalanobis `[GAUSS-2]`

$$
\mu_0=\frac1N\sum_i z_i,\quad
\Sigma_0=\frac1N\sum_i(z_i-\mu_0)(z_i-\mu_0)^\top,
\quad s_{Marginal}(z)=-(z-\mu_0)^\top\Sigma_0^\dagger(z-\mu_0).
$$

- 의미/tier: label을 무시한 global Gaussian geometry, diagnostic.
- key: `detector/marginal_mahalanobis_raw`; 약어 `MMD` 금지.
- 저장/실패/검증: `GAUSS-1`과 동일하되 `Sigma_0` identity를 별도 저장.
- 논문 표준 문장: “Marginal Mahalanobis is the negative distance to a
  single label-agnostic ID Gaussian.”

### 4.4 Relative Mahalanobis `[GAUSS-3]`

$$
s_{RMD}(z)=\max_c\left[d_0(z)-d_c(z)\right].
$$

- 의미/tier: class-conditional proximity에서 global proximity를 빼는
  primary contrastive distance.
- transform/key: raw; `detector/relative_mahalanobis_raw`.
- 주의: `Sigma_W`와 `Sigma_0`는 서로 다른 covariance다.
- 검증: 저장된 `d_0`, 모든 `d_c`로 score를 재구성.
- 논문 표준 문장: “Relative Mahalanobis subtracts the class-conditional
  distance from the marginal distance and takes the largest class contrast.”

### 4.5 Mahalanobis++ `[GAUSS-4]`

Fit과 query 모두 normalization 전에 zero policy를 적용한 뒤

$$\hat z=z/\|z\|_2$$

로 바꾸고 class mean과 tied covariance를 normalized `id_train`에서 다시
fit한다.

$$
s_{Maha++}(z)=-\min_c(\hat z-\hat\mu_c)^\top
\hat\Sigma_W^\dagger(\hat z-\hat\mu_c).
$$

- 의미/tier: radial variation을 제거한 primary Mahalanobis.
- key: `detector/mahalanobis_pp`; legacy 결과명은 metadata alias만 허용.
- 금지: query만 normalize하거나 raw covariance 재사용.
- 저장: exact/near-zero counts, normalized mean/covariance identity.
- source: `mueller-mp/maha-norm@53de550b...`.
- 검증: unit-norm fit/query, reference score parity, zero/near-zero fixture.
- 논문 표준 문장: “Mahalanobis++ refits the class means and tied covariance
  after sample-wise L2 normalization of both ID-train and query features.”

### 4.6 Relative Mahalanobis++ `[GAUSS-5]`

Normalized space에서 class와 marginal Gaussian을 모두 다시 fit한다.

$$s_{RMD++}(z)=\max_c[\hat d_0(z)-\hat d_c(z)].$$

- tier/key: primary; `detector/relative_mahalanobis_pp`.
- 실패/검증: `GAUSS-4` zero policy와 `GAUSS-3` component reconstruction.
- 논문 표준 문장: “Relative Mahalanobis++ computes both conditional and
  marginal distances in the same L2-normalized feature space.”

### 4.7 GDA-ClassDensity `[GAUSS-6]`

SN-off checkpoint에 class-wise full unbiased covariance를 fit한다.

$$
\Sigma_c=\frac{1}{N_c-1}\sum_{i\in I_c}
(z_i-\mu_c)(z_i-\mu_c)^\top,
$$

$$
s_{GDA}(z)=\log\sum_{c=1}^{K}\pi_c
\mathcal N(z;\mu_c,\Sigma_c+\epsilon I),
\qquad \pi_c=N_c/N.
$$

`epsilon`은 card 06의 shared first-success ladder
`0`, `finfo(float64).tiny`, `10^-308,...,10^-1`에서 고른다.

- 의미/tier: feature-space class density, appendix measurement.
- transform/key: raw; `detector/gda_class_density`.
- 방향: full log density가 클수록 ID-like.
- 저장: class counts/priors, means, covariance denominator, selected shared
  jitter, per-class log density, logsumexp.
- 실패: ladder 전체 Cholesky 실패, empty/singleton class, non-finite density.
- source: DDU official density kernel `f597744c...`; current score 이름은
  GDA-ClassDensity다. `DDU`는 future SN ablation에만 허용한다.
- 검증: 1D two-class hand density, shared jitter order, logsumexp parity.
- 논문 표준 문장: “For SN-off checkpoints we report GDA-ClassDensity, a
  class-prior-weighted mixture of class-wise full-covariance Gaussian feature
  densities; we do not label this readout DDU.”

## 5. Neighbor, prototype, and subspace detectors

### 5.1 kNN-L2 `[SUB-1]`

$$
\hat z=\frac{z}{\|z\|_2+10^{-10}},\qquad
s_{kNN}(z)=-d^2_{(50)}(\hat z,\{\hat z_i:i\in id\_train\}).
$$

Zero/near-zero validation은 denominator epsilon보다 먼저 수행한다.

- 의미/tier: normalized ID bank의 local support, primary.
- fit/query: complete `id_train` bank; test/OOD query에는 self exclusion 없음.
- hyperparameter/key: exact squared-L2 `K=50`; `detector/knn_l2_k50`.
- 방향: 작고 가까운 50th distance일수록 score가 커서 ID-like.
- 저장: 50th distance, neighbor sample ID, K, exact-search flag.
- source: `deeplearning-wisc/knn-ood@2afb2bbe...`.
- 검증: brute-force exact distance, batch invariance, tie/sample-ID order.
- 논문 표준 문장: “kNN-L2 is the negative squared distance to the 50th
  nearest L2-normalized ID-training feature.”

### 5.2 Nearest Class Prototype `[SUB-2]`

$$s_{NCP}(z)=-\min_c\|z-\mu_c\|_2.$$

- 의미/tier: covariance 없는 radial+angular prototype diagnostic.
- transform/key: raw; `detector/ncp_euclidean_raw`.
- 저장: nearest class/distance, feature/prototype norms, angle.
- 검증: direct Euclidean fixture; class tie는 smallest index.
- 논문 표준 문장: “NCP is the negative Euclidean distance to the nearest raw
  ID-training class mean.”

### 5.3 CTM / Prototype-Cosine `[SUB-3]`

Raw feature를 먼저 class-average한 뒤 mean과 query를 normalize한다.

$$
s_{CTM}(z)=\max_c\frac{z^\top\mu_c}{\|z\|_2\|\mu_c\|_2}.
$$

- 의미/tier: NCP의 angular-only primary control.
- key: `detector/ctm_prototype_cosine`.
- 실패: zero/near-zero query 또는 class-mean norm은 공통 상태 규칙 적용.
- source: `Fsoft-AIC/CTM-OOD@3587259b...`.
- 검증: scaling invariance, direct cosine, class tie.
- 논문 표준 문장: “CTM scores the maximum cosine similarity between the
  query and raw ID-training class means.”

### 5.4 Weight-Cosine `[SUB-4]`

$$s_{WeightCos}(z)=\max_c\frac{z^\top w_c}{\|z\|_2\|w_c\|_2}.$$

- 의미/tier: empirical mean 대신 classifier row를 쓰는 appendix self-duality
  diagnostic.
- key: `detector/weight_cosine_appendix`.
- 검증: CTM과 prototype/weight source가 다름을 fixture로 보장.
- 논문 표준 문장: “Weight-Cosine replaces empirical class prototypes with
  classifier-weight rows and is reported only as a self-duality diagnostic.”

### 5.5 ViM `[SUB-5]`

Classifier origin은

$$u=-W^+b$$

이며 bias가 없으면 `b=0`, `u=0`이다. `id_train`에서

$$X=Z_{train}-u,\qquad \Sigma_u=\frac{1}{N}X^\top X$$

를 계산한다. **`X`의 sample mean을 다시 빼지 않는다.** Float64로
symmetrize/eigh하고, 상위 `DIM` eigenvectors를 principal space, 나머지를
`NS in R^{p x (p-DIM)}`로 둔다. Author rule에서 `p=640`이므로
`DIM=320`, `residual_dim=320`이다.

$$
r(z)=\|(z-u)NS\|_2,
\qquad
\alpha=\frac{\operatorname{mean}_{id\_train}\max_c\ell_c(z)}
{\operatorname{mean}_{id\_train}r(z)},
$$

$$s_{ViM}(z)=\log\sum_c\exp\ell_c(z)-\alpha r(z).$$

- 의미/tier: logit energy와 ID principal-space 밖 residual의 결합, primary.
- key: `detector/vim_author_dim`.
- sensitivity: `DIM in {64,128,256,320}` 전부 appendix에 저장하며 best
  OOD point를 선택하지 않는다.
- 저장: `u`, DIM, residual_dim, full spectrum/rank, `NS` identity, ID residual
  mean/std, alpha, Energy/residual/alpha-residual components, correlation.
- 실패: mean ID residual `<=tol`, non-finite alpha/score, wrong NS shape.
- source: `haoqiwang/vim@dabf9e5b...`.
- 검증: tiny matrix에서 `X^T X/N` hand calculation, additional-centering이
  결과를 바꾸는 fixture, component sum, author DIM.
- 논문 표준 문장: “ViM uses the author assume-centered covariance
  `(Z-u)^T(Z-u)/N` without additional mean centering, with DIM=320 for the
  640-dimensional WRN feature.”

### 5.6 NECO `[SUB-6]`

`StandardScaler`를 `id_train`에 fit한다. Scaler mean과 population variance
(`ddof=0`)로 `z_std`를 만든 뒤 standardized `id_train`에 PCA를 fit한다.
첫 100개 component 좌표를 `P_100(z_std)`라 하면

$$s_{NECO}(z)=\frac{\|P_{100}(z_{std})\|_2}{\|z_{std}\|_2}.$$

- 의미/tier: ID principal subspace가 standardized feature norm을 설명하는
  비율, primary.
- hyperparameter: fixed `neco_dim=100`; WRN/ResNet MaxLogit 미사용.
- key: `detector/neco_author_std_dim100`.
- 금지: raw-feature ID90, explained-variance-selected dimension, OOD tuning.
- 저장: scaler mean/scale, PCA mean/components/variance, input/dim=100, numerator,
  denominator, zero counts.
- 실패: `p<100`, scaler zero scale, zero/near-zero query norm, non-finite PCA.
- source: `drti/neco@6a556406...`.
- 검증: pinned author pipeline parity, scaler/PCA train-only fit, no-MaxLogit,
  fixed dimension assertion.
- 논문 표준 문장: “NECO is the norm fraction captured by the first 100 PCA
  directions after an ID-training StandardScaler; no MaxLogit factor is used
  for the WRN/ResNet backbone.”

## 6. OOD score evaluation metrics

모든 metric은 같은 detector의 `id_test_primary` 10k score와 **한 OOD
dataset의 score**를 먼저 결합해 계산한다. 여러 OOD dataset raw samples를
pooling하지 않는다.

### 6.1 AUROC `[OOD-1]`

ID를 positive (`1`), OOD를 negative (`0`)로 둔다.

$$
\operatorname{AUROC}=P(s(X_{ID})>s(X_{OOD}))
+\tfrac12P(s(X_{ID})=s(X_{OOD})).
$$

- tier/direction/key: primary, 클수록 좋음;
  `ood_metric/auroc_id_positive`.
- backend: `sklearn.metrics.roc_auc_score`.
- 실패: empty group, non-finite score, 한 binary class만 존재.
- 검증: perfect/reversed/tied hand fixture.
- 논문 표준 문장: “OOD AUROC treats ID as positive and uses the declared
  ID-oriented detector score.”

### 6.2 AUPR-In `[OOD-2]`

ID-positive `precision_recall_curve`를 만든다.

- primary trapezoidal PR-AUC:
  `auc(recall, precision)`, key `ood_metric/aupr_in_openood_auc`.
- secondary Average Precision:
  `average_precision_score`, key `ood_metric/ap_in`.
- 방향: 클수록 좋음. 두 scalar를 같은 이름으로 저장하지 않는다.
- 저장: positive prevalence, precision/recall/threshold arrays.
- 검증: direct sklearn parity와 imbalanced fixture.
- 논문 표준 문장: “AUPR-In uses ID as positive; trapezoidal PR-AUC is
  primary and Average Precision is reported separately.”

### 6.3 AUPR-Out `[OOD-3]`

OOD를 positive로 두고 score는 `-s`를 사용한다.

- primary: trapezoidal `ood_metric/aupr_out_openood_auc`.
- secondary: `ood_metric/ap_out`.
- 방향/실패/검증: `[OOD-2]`와 동일; OOD prevalence를 저장한다.
- 논문 표준 문장: “AUPR-Out treats OOD as positive and ranks examples with
  the negated ID-oriented score.”

### 6.4 FPR@95 ID TPR `[OOD-4]`

$$
\tau_{95}=Q_{0.05}(\{s(x):x\in D_{ID}\}),
$$

```python
threshold = np.quantile(id_scores, 0.05, method="linear")
fpr95 = np.mean(ood_scores >= threshold)
achieved_id_tpr = np.mean(id_scores >= threshold)
```

- 의미/tier: ID 약 95%를 accept할 때 OOD가 ID로 통과하는 비율, primary.
- 방향/keys: 작을수록 좋음; `ood_metric/fpr95_id_tpr`,
  `ood_metric/fpr95_threshold`, `ood_metric/fpr95_achieved_id_tpr`.
- tie: inclusive `>=`; achieved TPR이 0.95와 다를 수 있으므로 저장한다.
- source: Mahalanobis++ quantile implementation convention.
- 검증: interpolated quantile fixture, tied threshold fixture, order-statistic
  implementation과 다름을 보이는 regression fixture.
- 논문 표준 문장: “FPR@95 is the fraction of OOD scores above the linear
  5th percentile of ID scores, using an inclusive threshold.”

### 6.5 Dataset aggregation `[OOD-5]`

Near는 CIFAR-100과 TinyImageNet, far는 MNIST, SVHN, Textures, Places365다.
각 dataset scalar를 먼저 계산한 뒤

$$M_{near}=\frac1{|D_{near}|}\sum_{d\in D_{near}}M_d,$$

far와 overall도 같은 arithmetic macro mean을 쓴다.

- tier/key: primary summary; `<metric>/near_macro_mean`,
  `<metric>/far_macro_mean`, `<metric>/overall_macro_mean`.
- 금지: dataset size weighting 또는 pooled raw score metric.
- 저장: per-dataset scalar와 `num_id`, `num_ood`.
- 검증: unequal-size datasets에서 macro와 pooled가 다름을 확인.
- 논문 표준 문장: “Near-, far-, and overall OOD results are arithmetic
  means of per-dataset metrics, not pooled-sample estimates.”

### 6.6 Detector rank concordance `[OOD-6]`

Optimizer pair `(o,o')`의 canonical detector AUROC rank vector에 대해

$$\tau_b(o,o')=\operatorname{KendallTauB}(rank(v_o),rank(v_{o'})).$$

- tier/key: exploratory; `analysis/detector_rank_concordance_tau_b`.
- near/far를 분리하고 모든 optimizer pair를 저장한다.
- oracle/sensitivity/compatibility duplicate는 제외한다.
- 실제 delta-AUROC와 seed uncertainty 없이 단독 해석하지 않는다.
- 실패: detector가 2개 미만이면 degenerate.
- 검증: identical/reversed/tied rank fixture.
- 논문 표준 문장: “Detector-order stability is summarized exploratorily by
  pairwise Kendall tau-b on canonical detector AUROC ranks.”

### 6.7 Restoration and angular-control gaps `[OOD-7]`

$$
\operatorname{RestorationGap}_{Maha}=AUROC(Maha++)-AUROC(Maha),
$$

$$
\operatorname{RestorationGap}_{RMD}=AUROC(RMD++)-AUROC(RMD).
$$

`AUROC(CTM)-AUROC(NCP)`는 동일 detector의 L2 variant가 아니므로
`AngularControlGap`으로 저장한다.

- tier: exploratory mechanism effect size; 양수면 normalized/angular readout이
  raw counterpart보다 높은 AUROC.
- keys: `analysis/restoration_gap_mahalanobis`,
  `analysis/restoration_gap_relative_mahalanobis`,
  `analysis/angular_control_gap_ctm_ncp`.
- 검증: 동일 checkpoint/seed/OOD dataset identity가 아니면 failed.
- 논문 표준 문장: “Restoration gaps are paired within-checkpoint AUROC
  differences between normalized and raw Gaussian readouts.”

## 7. Representation geometry

Primary geometry는 `last.pt`의 `id_train`; `id_validation`은 held-out control이다.

### 7.1 Feature norm distribution `[GEO-1]`

$$r_i=\|z_i\|_2,\qquad CV(r)=\operatorname{std}_{ddof=0}(r)/\operatorname{mean}(r).$$

- tier/direction: primary distribution; 단일 좋음/나쁨 방향을 부여하지 않는다.
- keys: `geometry/feature_norm/global`, `geometry/feature_norm/classwise`,
  `geometry/feature_norm/heldout_validation`.
- 저장: raw norm vector, count, mean, population std, CV, min/max, linear
  quantiles `{.01,.05,.25,.5,.75,.95,.99}`. Histogram은 plot bin edges와
  함께 visualization artifact로만 저장한다.
- 실패: non-finite norm; exact/near-zero count는 상태 규칙 적용.
- 검증: direct Euclidean norm과 class/sample permutation invariance.
- 논문 표준 문장: “We characterize radial feature geometry using the full
  ID-training norm distribution and class-wise summaries.”

### 7.2 CDNV `[GEO-2]`

$$V_c=\frac1{N_c}\sum_{i\in I_c}\|z_i-\mu_c\|_2^2,$$

$$CDNV_{cc'}=\frac{V_c+V_{c'}}{2\|\mu_c-\mu_{c'}\|_2^2},\qquad
CDNV=\frac{2}{K(K-1)}\sum_{c<c'}CDNV_{cc'}.$$

- 의미/tier: within-class compactness 대 between-class separation, primary.
- 방향/key: 작을수록 collapse/separation이 강함; `geometry/cdnv/mean`.
- 저장: `geometry/cdnv/pair_matrix`, class variances, pair denominators.
- 실패: pair distance squared `<=1e-12`이면 affected result failed.
- 검증: translated/scaled feature invariance와 two-class hand fixture.
- 논문 표준 문장: “CDNV averages the pairwise ratio of within-class variance
  to squared class-mean separation.”

### 7.3 Class-pair distance and angle `[GEO-3]`

$$D_{cc'}=\|\mu_c-\mu_{c'}\|_2,\qquad
A_{cc'}=\frac{m_c^\top m_{c'}}{\|m_c\|_2\|m_{c'}\|_2}.$$

- tier: primary matrices; raw-origin angle는 appendix.
- keys: `geometry/class_pair/distance`,
  `geometry/class_pair/centered_cosine`,
  `geometry/class_pair/raw_origin_cosine_appendix`.
- 저장: full symmetric matrix와 off-diagonal mean/population std/CV/min/max.
- 실패: centered mean zero/near-zero norm.
- 검증: symmetry, zero diagonal distance, feature translation invariance of
  centered angle.
- 논문 표준 문장: “Class angles are computed between class means centered by
  the class-balanced global mean.”

### 7.4 NC0 `[GEO-4]`

$$NC0_{raw}=\|W^\top\mathbf1_K\|_2.$$

Appendix audit는 `NC0_Eq12=NC0_raw/p`,
`NC0_theory=NC0_raw^2/K`다.

- 의미/tier: classifier rows의 centered-sum condition; raw가 primary.
- 방향/keys: 작을수록 collapse condition에 가까움;
  `geometry/nc0_row_sum_raw`, `geometry/nc0_eq12_per_dim`,
  `geometry/nc0_theory_squared`.
- split: classifier weight only.
- 검증: direct row sum과 row permutation invariance.
- 논문 표준 문장: “NC0 is the Euclidean norm of the sum of classifier-weight
  rows; scaled expressions are reported only as audits.”

### 7.5 NC1 Moore-Penrose `[GEO-5]`

$$
\Sigma_W=\frac1N\sum_c\sum_{i\in I_c}(z_i-\mu_c)(z_i-\mu_c)^\top,
$$

$$
\Sigma_B=\frac1K\sum_c m_cm_c^\top,
\qquad
NC1=\frac1K\operatorname{Tr}(\Sigma_W\Sigma_B^\dagger).
$$

`Sigma_B`를 float64 symmetrize/eigh하고
`lambda > 1e-15*lambda_max`만 reciprocal한다.

- 의미/tier: between-class subspace에 남은 within-class variability, primary.
- 방향/key: 작을수록 collapse; `geometry/nc1_pinv`.
- diagnostics: `geometry/nc1_svd_diagnostic`,
  `geometry/nc1_trace_quotient_diagnostic`.
- 저장: matrices, eigenvalues, cutoff/rank, trace components, denominators.
- 실패: retained rank 0, materially negative eigenvalue, non-finite trace.
- source: Papyan et al. `arXiv:2008.08186`; audited optimizer code
  `7cab4a59...`.
- 검증: direct Moore-Penrose tiny matrix, rotated-feature invariance, SVD
  diagnostic가 primary key를 덮지 않음.
- 논문 표준 문장: “We report NC1 as
  `Tr(Sigma_W Sigma_B^dagger)/K`, using biased within- and between-class
  covariance estimators computed from raw ID-training features.”

### 7.6 NC2 feature class-mean geometry `[GEO-6]`

Class norm equinormness는

$$NC2n=\frac{\operatorname{std}_{correction=1,c}\|m_c\|_2}
{\operatorname{mean}_c\|m_c\|_2}.$$

Equiangularity는

$$NC2a=\frac1{K(K-1)}\sum_{c\ne c'}
\left|\cos(m_c,m_{c'})+\frac1{K-1}\right|,$$

cosine denominator에는 audited epsilon `1e-9`를 적용하되 공통 zero 상태를
먼저 판정한다. `M=[m_1,...,m_K]`와

$$M^*=\frac1{\sqrt{K-1}}(I_K-J_K/K)$$

에 대해

$$NC2ETF_{raw}=\left\|\frac{M^\top M}{\|M^\top M\|_F}-M^*\right\|_F,
\qquad NC2ETF_{Eq5}=NC2ETF_{raw}/K^2.$$

- tier: NC2n/NC2a/ETF raw primary; Eq5 appendix audit.
- 방향: 모두 작을수록 simplex ETF에 가까움.
- keys: `geometry/nc2_equinorm`, `geometry/nc2_equiangular`,
  `geometry/nc2_etf_raw`, `geometry/nc2_etf_eq5_scaled`.
- 저장: class norms, centered means, cosine/Gram matrices.
- 실패: mean class norm 또는 Gram Frobenius norm이 `<=1e-12`.
- source/검증: `jydzhao/neural_collapse_optimizer@7cab4a59...`;
  regular-simplex fixture, correction=1 regression.
- 논문 표준 문장: “NC2n is the sample coefficient of variation of centered
  class-mean norms, while NC2a and NC2-ETF quantify simplex equiangularity.”

### 7.7 NC2W classifier geometry `[GEO-7]`

$$NC2W_{raw}=\left\|\frac{WW^\top}{\|WW^\top\|_F}-M^*\right\|_F.$$

Classifier-row norm CV (`NC2Wn`)와 equiangular error (`NC2Wa`)도 같은 NC2
규칙으로 계산한다.

- tier: appendix/context.
- keys: `geometry/nc2w_etf_raw`, `geometry/nc2w_equinorm`,
  `geometry/nc2w_equiangular`.
- 실패/검증: zero row/Gram과 regular-simplex weight fixture.
- 논문 표준 문장: “NC2W applies the corresponding ETF geometry diagnostics
  to classifier-weight rows.”

### 7.8 NC3 self-duality `[GEO-8]`

`W:[K,p]`, `M:[p,K]`에서

$$NC3_{raw}=\left\|\frac{W}{\|W\|_F}-
\frac{M^\top}{\|M^\top\|_F}\right\|_F,$$

appendix audit는 `NC3_Eq10=NC3_raw/(Kp)`다.

- tier/direction: raw primary, scaled appendix; 작을수록 self-dual.
- keys: `geometry/nc3_self_duality_raw`, `geometry/nc3_eq10_scaled`.
- 실패: either Frobenius norm `<=1e-12`.
- 검증: `W=M^T` fixture에서 0; shape assertion.
- 논문 표준 문장: “NC3 is the Frobenius distance between globally
  normalized classifier weights and transposed centered class means.”

### 7.9 NC4 behavioral agreement `[GEO-9]`

$$\hat y_{NCC}(z)=\arg\min_c\|z-\mu_c\|_2,$$

$$NC4_{with\_bias}=\frac1N\sum_i
\mathbf1\{\arg\max_c(w_c^\top z_i+b_c)=\hat y_{NCC}(z_i)\}.$$

Bias를 뺀 prediction과의 agreement는 compatibility diagnostic이다.

- tier/split: primary behavior geometry; development `id_validation`, final
  승인 후 `id_test_primary`.
- 방향/keys: 클수록 agreement;
  `geometry/nc4_agreement_with_bias`,
  `geometry/nc4_agreement_without_bias`.
- 저장: classifier/NCC predictions와 각각의 accuracy.
- tie: argmin/argmax 모두 smallest class index.
- 검증: identical classifier/NCC fixture, bias-sensitive fixture.
- 논문 표준 문장: “NC4 is the agreement rate between the affine classifier
  and nearest ID-training class-mean predictions.”

### 7.10 RankMe `[GEO-10]`

Uncentered `Z in R^{N x p}`의 singular value `sigma_i`에 대해

$$q_i=\frac{\sigma_i}{\sum_j\sigma_j}+10^{-7},\qquad
RankMe=\exp\left(-\sum_i q_i\log q_i\right).$$

- 의미/tier: uncentered feature singular-value entropy rank, primary.
- 방향/key: 크면 spectrum이 더 고르게 분산;
  `geometry/rankme_uncentered`.
- 금지: centering, covariance eigenvalue 대체, epsilon 위치 변경.
- 실패: singular-value sum 0/non-finite.
- source: RankMe `arXiv:2210.02885`.
- 검증: equal/rank-one singular spectrum, feature-row permutation.
- 논문 표준 문장: “RankMe is spectral entropy computed from singular values
  of the uncentered ID-training feature matrix.”

### 7.11 Covariance spectra and ranks `[GEO-11]`

Primary/secondary/diagnostic matrices는

$$
\Sigma_W=\frac1N\sum_i(z_i-\mu_{y_i})(z_i-\mu_{y_i})^\top,
$$

$$
\Sigma_{total}=\frac1N\sum_i(z_i-\mu)(z_i-\mu)^\top,
\qquad
\Sigma_B=\frac1K\sum_c m_cm_c^\top.
$$

각 covariance를 float64 symmetrize/eigh한 nonnegative eigenvalue
`lambda_1>=...>=lambda_p`로 저장한다. `q_i=lambda_i/sum_j lambda_j`일 때

$$r_{entropy}=\exp(-\sum_{q_i>0}q_i\log q_i),$$

$$r_{trace/top}=\frac{\sum_i\lambda_i}{\lambda_1},\qquad
r_{PR}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.$$

Canonical metric names are distinct:

```text
covariance_entropy_rank
covariance_trace_top_rank
covariance_participation_ratio
```

- tier: `Sigma_W` primary, total secondary, between diagnostic.
- keys: `geometry/spectrum/{sw|total|between}_entropy_rank`,
  `..._trace_top_rank`, `..._participation_ratio`.
- 이름 경계: 세 값과 RankMe는 서로 대체하거나 alias하지 않는다.
- 저장: full raw/clipped spectrum, negative clip count, trace, tolerance.
- 실패: materially negative 또는 all-zero spectrum.
- 검증: diagonal `[4,1]`에서 trace/top `5/4`, PR `25/17`, entropy formula;
  orthogonal-rotation invariance.
- 논문 표준 문장: “We report covariance entropy rank, trace-to-top rank, and
  participation ratio as three distinct functions of the same nonnegative
  covariance eigenvalue spectrum.”

### 7.12 Numerical rank and condition number `[GEO-12]`

$$r_{num}=|\{i:\lambda_i>tol\}|,
\qquad\kappa_{retained}=\lambda_1/\lambda_{r_{num}}.$$

- tier: `Sigma_W` primary, total secondary.
- keys: `geometry/spectrum/{matrix}_numerical_rank`,
  `..._condition_number_retained`.
- 저장: tolerance, lambda max/min-retained, `log10(kappa)`.
- 실패: retained rank 0; rank 1은 유효하며 kappa=1.
- 검증: explicit diagonal spectrum around tolerance.
- 논문 표준 문장: “Numerical rank retains covariance eigenvalues above
  `lambda_max p eps64`; conditioning uses only that retained spectrum.”

### 7.13 Within-covariance spectral slope `[GEO-13]`

Retained `Sigma_W` eigenvalue 전체에 대해

$$\log\lambda_i=a+bi+\varepsilon_i,\qquad i=1,...,r_{num}$$

를 intercept 포함 ordinary least squares로 fit한다. Natural log를 쓴다.

$$Top20Decay=\frac1m\sum_{i=1}^{m}
(\log\lambda_i-\log\lambda_{i+1}),
\quad m=\min(20,r_{num}-1).$$

- tier/direction: exploratory; slope가 더 음수면 빠른 decay이나 mechanism
  claim은 R-squared와 함께만 한다.
- keys: `geometry/spectrum/sw_log_slope`,
  `geometry/spectrum/sw_top20_mean_log_decay`.
- 저장: slope, abs slope, intercept, R-squared, residuals, retained count.
- 실패: retained count `<2`; SST=0이면 slope는 계산하되 R-squared가
  degenerate/null이다.
- 검증: exact geometric spectrum과 constant spectrum.
- 논문 표준 문장: “The exploratory spectral slope is an OLS fit of natural
  log within-class eigenvalues against one-based rank.”

### 7.14 Local Intrinsic Dimensionality `[GEO-14]`

Raw `id_train`에서 sample index 자신을 제외한 exact float64 Euclidean
거리 `0<r_1<=...<=r_k`에 대해

$$LID_k(z_i)=-\left[\frac1k\sum_{j=1}^{k}
\log\frac{r_j}{r_k}\right]^{-1}.$$

- tier/hyperparameter: `k=50` primary geometry; `{10,25,100}` appendix.
- keys: `geometry/lid_k50`, `geometry/lid_k10`,
  `geometry/lid_k25`, `geometry/lid_k100`.
- deterministic neighbor tie: `(distance, sample_id)`.
- invalid sample: any selected distance `<=1e-12`, non-finite distance, 또는
  absolute mean-log denominator `<=1e-15`.
- status: invalid 0이면 success; 일부면 aggregate degenerate와 valid-subset
  diagnostic; 전부면 failed.
- 저장: per-sample LID/valid mask, invalid reason/count, valid mean/median,
  population std, linear quantiles.
- 검증: line/grid fixture, duplicate fixture, block/batch parity.
- 논문 표준 문장: “LID uses the maximum-likelihood k-neighbor distance-ratio
  estimator on exact raw ID-training neighbors, with k=50 primary.”

### 7.15 TwoNN `[GEO-15]`

각 sample의 first/second non-self neighbor distance에서
`mu_i=r_2(i)/r_1(i)`를 계산한다. `log(mu)`를 오름차순 정렬하고
`n_eff=floor(0.9N)`만 사용한다.

$$x_i=\operatorname{sort}(\log\mu)_i,qquad
y_i=-\log(1-i/N),\quad i=1,...,n_{eff},$$

원점을 지나는 OLS `y=d x`의 slope `d`가 TwoNN intrinsic dimension이다.

- tier/key: appendix cross-check;
  `geometry/twonn_base_mu_fraction_09`.
- settings: raw `id_train`, exact float64 Euclidean, `algorithm=base`,
  `mu_fraction=0.9`, `data_fraction=1`.
- zero/duplicate: any `r_1<=1e-12`이면 metric degenerate, primary scalar null;
  sample 제거로 reference estimator를 바꾸지 않는다.
- 저장: `mu`, sorted log-mu, x/y, n_eff, slope, mean first/second distance.
- source: `sissa-data-science/DADApy@c37e52ca...`.
- 검증: pinned DADApy result parity, exact OLS fixture, duplicate failure.
- 논문 표준 문장: “TwoNN is the DADApy base estimator fitted through the
  origin after retaining the lowest 90% of log second-to-first-neighbor ratios.”

### 7.16 Hypersphere class alignment `[GEO-16]`

공통 zero policy 후 `u_i=z_i/||z_i||_2`로 둔다.

$$Align_c=\frac{1}{N_c(N_c-1)}\sum_{i\ne j\in I_c}
\|u_i-u_j\|_2^2
=\frac{2N_c}{N_c-1}(1-\|\bar u_c\|_2^2),$$

$$ClassAlignment=\frac1K\sum_c Align_c.$$

- 의미/tier: unit sphere에서 same-class angular concentration, exploratory.
- 방향/key: 작을수록 aligned;
  `geometry/hypersphere/class_alignment_exact`.
- 저장: class means/counts/alignment와 macro mean.
- 실패: class count `<2`, zero/near-zero feature.
- 검증: pairwise brute force와 exact moment identity.
- 논문 표준 문장: “Class alignment is the class-macro mean pairwise squared
  distance between L2-normalized same-class features.”

### 7.17 Hypersphere uniformity `[GEO-17]`

$$Uniformity=\log E_{i\ne j}\exp(-2\|u_i-u_j\|_2^2).$$

Empirical sample에서 ordered indices를 iid로 뽑고 same index는 reject한다.
Repeat당 100,000 pairs, seeds `{0,1,2}`, stable `logmeanexp`를 사용한다.

- 의미/tier: unit sphere 전체의 spread, exploratory.
- 방향/key: 더 음수면 더 uniform;
  `geometry/hypersphere/uniformity_t2`.
- 저장: repeat value, pair seed/count, mean과 sample Monte Carlo std.
- 실패: 유효 distinct pair 부족, zero/near-zero feature.
- source: Wang and Isola alignment/uniformity formulation.
- 검증: identical-vector와 antipodal toy set, fixed-seed determinism.
- 논문 표준 문장: “Uniformity is the log mean exponential pair potential at
  t=2 over three prespecified 100,000-pair Monte Carlo repeats.”

## 8. 논문 이름과 artifact key 대응표

### 8.1 Main/control table

| 논문 이름 | formula | artifact key | 방향 | tier |
| --- | --- | --- | --- | --- |
| Accuracy | ID-1 | `id/accuracy` | up | primary/control |
| NLL | ID-2 | `id/nll_raw` | down | primary/control |
| ECE-M15 | ID-3 | `id/ece_raw_m15` | down | primary |
| Temperature-scaled NLL/ECE | ID-4 | `id/nll_ts`, `id/ece_ts_m15` | down | control |
| Misclassification AUROC | ID-5 | `id/misclassification_auroc_msp_raw` | up | primary |
| AUGRC | ID-6 | `id/augrc_msp_raw` | down | primary |
| MSP / MLS / Energy-T1 | LOGIT-1..3 | `ood_score/msp`, `ood_score/max_logit`, `ood_score/energy_t1` | up | primary control |
| Mahalanobis / Mahalanobis++ | GAUSS-1/4 | `detector/mahalanobis_raw`, `detector/mahalanobis_pp` | up | primary |
| Relative Mahalanobis / RMD++ | GAUSS-3/5 | `detector/relative_mahalanobis_raw`, `detector/relative_mahalanobis_pp` | up | primary |
| kNN-L2 | SUB-1 | `detector/knn_l2_k50` | up | primary |
| CTM | SUB-3 | `detector/ctm_prototype_cosine` | up | primary |
| ViM | SUB-5 | `detector/vim_author_dim` | up | primary |
| NECO | SUB-6 | `detector/neco_author_std_dim100` | up | primary |
| AUROC / FPR@95 | OOD-1/4 | `ood_metric/auroc_id_positive`, `ood_metric/fpr95_id_tpr` | up/down | primary |
| NC1 Moore-Penrose | GEO-5 | `geometry/nc1_pinv` | down | primary |
| NC2n / NC2a / NC2-ETF | GEO-6 | `geometry/nc2_equinorm`, `geometry/nc2_equiangular`, `geometry/nc2_etf_raw` | down | primary |
| NC3 / NC4 | GEO-8/9 | `geometry/nc3_self_duality_raw`, `geometry/nc4_agreement_with_bias` | down/up | primary |
| RankMe | GEO-10 | `geometry/rankme_uncentered` | descriptive | primary |
| Covariance ranks | GEO-11 | `geometry/spectrum/*_{entropy_rank,trace_top_rank,participation_ratio}` | descriptive | primary/secondary |

### 8.2 Appendix and diagnostic table

| 논문 이름 | formula | artifact key | tier |
| --- | --- | --- | --- |
| ECE-M10/M30 | ID-3 | `id/ece_raw_m10`, `id/ece_raw_m30` | appendix |
| AURC | ID-7 | `id/aurc_msp_raw` | appendix |
| Marginal Mahalanobis | GAUSS-2 | `detector/marginal_mahalanobis_raw` | diagnostic |
| GDA-ClassDensity | GAUSS-6 | `detector/gda_class_density` | appendix |
| NCP / Weight-Cosine | SUB-2/4 | `detector/ncp_euclidean_raw`, `detector/weight_cosine_appendix` | diagnostic/appendix |
| ViM DIM sensitivity | SUB-5 | `detector/vim_dim{64,128,256,320}` | appendix |
| Detector Kendall tau-b | OOD-6 | `analysis/detector_rank_concordance_tau_b` | exploratory |
| Restoration/Angular gaps | OOD-7 | `analysis/restoration_gap_*`, `analysis/angular_control_gap_ctm_ncp` | exploratory |
| NC0 / NC2W | GEO-4/7 | `geometry/nc0_*`, `geometry/nc2w_*` | context/appendix |
| NC1 SVD/trace quotient | GEO-5 | `geometry/nc1_*_diagnostic` | diagnostic |
| Spectral slope / LID | GEO-13/14 | `geometry/spectrum/sw_*`, `geometry/lid_k*` | exploratory/appendix |
| TwoNN | GEO-15 | `geometry/twonn_base_mu_fraction_09` | appendix |
| Alignment / Uniformity | GEO-16/17 | `geometry/hypersphere/*` | exploratory |

## 9. Metric-to-data matrix

| metric family | checkpoint | fit split | evaluation split | transform |
| --- | --- | --- | --- | --- |
| ID raw performance/ranking | both, separate | none | `id_test_primary` | raw logits |
| Temperature scaling | both, separate | `id_validation` | `id_test_primary` | logits divided by fitted T |
| Logit OOD controls | both, separate | none | `id_test_primary` + each OOD | raw logits |
| Raw Gaussian/RMD | both, separate | `id_train` | `id_test_primary` + each OOD | raw feature |
| Mahalanobis++/RMD++ | both, separate | `id_train` | `id_test_primary` + each OOD | sample L2 normalization before refit |
| GDA-ClassDensity | both, separate | `id_train` | `id_test_primary` + each OOD | raw feature |
| kNN-L2 | both, separate | `id_train` bank | `id_test_primary` + each OOD | L2-normalized feature |
| NCP/CTM/ViM/NECO | both, separate | `id_train` | `id_test_primary` + each OOD | metric-specific fixed transform |
| Primary geometry | `last.pt` | `id_train` | same artifact | raw unless stated |
| Held-out geometry | both, separate | `id_train` when needed | `id_validation` | raw unless stated |
| Final NC4 | both, separate | class means on `id_train` | `id_test_primary` | raw feature/logit |

## 10. Validation oracle

### 10.1 Hand-computable fixtures

1. Perfect/all-wrong/all-equal-confidence classification fixtures validate
   Accuracy, NLL, ECE, misclassification AUROC state, AUGRC, and AURC.
2. A one-dimensional balanced two-class Gaussian validates class means, biased
   tied covariance, unbiased class covariance, Mahalanobis, RMD, and GDA.
3. A regular simplex with `W=M^T` validates NC0-NC4, including NC1 pinv and
   class-tie behavior.
4. Diagonal covariance `diag(4,1)` validates the three spectrum ranks,
   numerical rank, condition number, slope input, and negative-eigen tolerance.
5. A ViM matrix whose column mean is nonzero validates `X^T X/N` and detects an
   erroneous extra-centering implementation.
6. Collinear points and an exact duplicate validate LID, TwoNN, neighbor ties,
   and degenerate states.

### 10.2 Pinned reference parity

- MSP/Energy/Mahalanobis/kNN/CTM: commits already pinned in
  `docs/sources.lock.yaml`.
- Temperature: `gpleiss/temperature_scaling@ce1154ec...`, allowing only the
  documented positive-parameter/fallback project differences.
- AURC: `IML-DKFZ/fd-shifts@c4467aec...`, after removing display scale 1000.
- Mahalanobis++: `mueller-mp/maha-norm@53de550b...`.
- ViM: `haoqiwang/vim@dabf9e5b...`.
- NECO: `drti/neco@6a556406...`.
- NC: `jydzhao/neural_collapse_optimizer@7cab4a59...`, with paper NC1 pinv
  primary and code SVD only diagnostic.
- TwoNN: `sissa-data-science/DADApy@c37e52ca...`.

### 10.3 Invariance and batching

- Sample-order permutation은 모든 aggregate를 바꾸지 않는다.
- Consistent class permutation은 class-indexed arrays만 같은 방식으로
  permute하고 scalar를 바꾸지 않는다.
- Batched/unbatched score와 CPU float64 reference가
  `atol=1e-12`, `rtol=1e-10` 안에서 일치해야 한다. GPU/float32가 이 기준을
  충족하지 못하면 float64 reference를 primary로 유지한다.
- Raw-space distance는 orthogonal rotation에, normalized metric은 positive
  sample scaling에 대해 정의가 요구하는 invariance를 만족해야 한다.

### 10.4 Required failure fixtures

```text
empty split or empty class
non-finite feature, logit, fitted parameter, score
exact-zero and near-zero feature norm
singular or materially indefinite covariance
all-zero covariance spectrum
all-correct or all-wrong misclassification labels
duplicate nearest neighbor
ViM zero residual mean or non-finite alpha
NECO p < 100 or invalid scaler
checkpoint/split/membership identity mismatch
```

실패 fixture에서 scalar를 정상 숫자로 반환하면 acceptance failure다.

## 11. Definition of Done for the later implementation

- 모든 formula ID가 정확히 하나의 canonical artifact key와 구현 entrypoint에
  매핑된다.
- 모든 result에 formula version, source commit, checkpoint/split identity,
  dtype, tolerance, intermediate diagnostics와 status가 있다.
- `last.pt`와 `best_val.pt`, seed, OOD dataset의 fit/result namespace가
  분리된다.
- 10k `id_test_primary`만 v1.2 OOD ID side로 resolve되고 9k compatibility
  artifact가 생성되지 않는다.
- Quantile FPR, AURC tie grouping, temperature fallback, ViM no-extra-centering,
  NECO scaler/dim/no-MaxLogit, NC1 pinv, covariance rank names, LID/TwoNN
  degeneracy가 regression test로 보호된다.
- Per-dataset OOD scalar와 macro means를 raw score artifact에서 재계산할 수
  있다.
- Tiny fixture, pinned-reference parity, invariance, failure tests와 bounded
  authorized checkpoint smoke가 통과한다.
- 이 기준을 통과하기 전에는 metric 값이나 optimizer 차이를 논문 결과로
  해석하지 않는다.

## 12. Revision record

### v1.2 — 2026-08-04

- WRN-28-10/CIFAR-10만 executable scope로 고정했다.
- `last.pt` primary, `best_val.pt` deployment control을 분리했다.
- 모든 OOD metric의 ID side를 project 10k로 통일하고 OpenOOD 9k 실행을
  제외했다.
- FPR@95를 linear 5th-percentile quantile로 고정했다.
- SN-off density를 GDA-ClassDensity로 명명하고 DDU를 future SN ablation에
  남겼다.
- ViM author covariance, NECO author-code pipeline, NC1 Moore-Penrose primary,
  covariance rank 세 공식을 고정했다.
- AURC, temperature fitting/fallback, spectrum slope, LID, TwoNN, tie와
  zero/near-zero failure semantics를 완결했다.
