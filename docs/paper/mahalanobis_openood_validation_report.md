# Mahalanobis 구현 교차 검증 결과와 남은 검증 범위

작성일: 2026-08-18
목적: Raw Mahalanobis와 RMD의 큰 성능 차이가 구현 오류인지 확인하고, 검증된 범위와 아직 검증되지 않은 범위를 교수님께 보고한다.

Repository handoff:

- 최신 분석 generator commit: `d0245f2ad46d2a42c2086b760bb1b6012ce2f3ef`
- 최신 reader pack: `hf://buckets/contra333/ICLR_RUN/aggregate/task_f_frozen_paper_pack_20260819/8e6b8ab9fe52cfd8f2f7fb33681de4e1f10c8d674fba3597dd6427f6e59759dd/`
- 최신 결과 해석 기준: [`task_f_result_analysis.md`](task_f_result_analysis.md)

## 기술 요약

이번 검증의 결론은 다음과 같다.

1. **저장된 feature 이후의 Mahalanobis scoring/evaluation 구현은 pinned OpenOOD와 교차 검증을 통과했다.** 현재 결과를 score sign, RMD formula, covariance fit split, class selection 또는 `assume_centered` 설정 오류로 설명할 근거는 발견되지 않았다.
2. **따라서 다음 질문은 단순히 “Mahalanobis 코드가 틀렸는가”가 아니라, “왜 이 학습으로 얻은 representation에서 Raw MD가 실패하는가”이다.** 다만 이것이 곧바로 “학습 코드가 틀렸다”는 뜻은 아니다. training-induced geometry와 training implementation bug를 구분해야 한다.
3. **Adam seed 0의 raw covariance는 effective rank가 501/640인 반면 AdamW는 640/640이었다.** 이 rank deficiency는 Raw MD의 optimizer·dataset별 민감도를 설명할 수 있는 구체적인 기하·수치 후보다.
4. 이번 검증은 primary cell의 seed 0, epoch 200, penultimate feature, CIFAR-100 및 SVHN에 한정된다. **50-run 전체의 일반성, feature extraction을 포함한 독립 end-to-end 재현, 학습 구현의 정확성은 이번 검증만으로 확정하지 않는다.**

따라서 현 단계에서 metric 코드를 다시 대규모로 작성하거나 모든 실험을 재학습할 필요는 없다. 다음 검증은 저비용 feature-extraction sanity check와, 필요한 경우 한 개의 독립 reference training으로 좁히는 것이 타당하다.

## 1. 어떤 의심 항목을 검증했는가

교수님 피드백에서 제기된 구현 의심을 이번 검증 결과와 대응시키면 다음과 같다.

| 의심 항목 | 이번 검증 결과 | 판정과 경계 |
|---|---|---|
| score sign | pinned OpenOOD confidence score와 동일한 방향으로 AUROC 및 score rank 재현 | **PASS** |
| RMD formula | OpenOOD의 `class score - background score`와 현재 `d0 - dc`가 같은 결과를 냄 | **PASS** |
| covariance fit split | checksum이 검증된 CIFAR-10 `id_train` 45,000장만으로 class/global covariance를 다시 fit | **PASS** |
| OOD leakage | CIFAR-100과 SVHN은 covariance fit이나 checkpoint 선택에 사용하지 않고 query에만 사용 | **PASS**, scorer 입력 단계 기준 |
| class selection | CIFAR-10 열 개 class mean과 가장 큰 ID-like confidence를 선택하는 OpenOOD 방식으로 재현 | **PASS** |
| covariance centering | `float64`에서 `assume_centered=True/False`를 분리 비교했으며 AUROC/FPR95가 동일 | **PASS** |
| inverse covariance 구현 | 동일 `EmpiricalCovariance.precision_` 계열 및 OpenOOD 원형으로 score/metric 재현 | **PASS**, 단 Adam rank deficiency는 별도 robustness 분석 대상 |
| normalization 혼입 | 이번 표는 raw penultimate feature만 사용했으며 L2-normalized fit을 섞지 않음 | **PASS** |
| ID/OOD label 방향 | ID-positive metric과 OpenOOD confidence ordering이 동일하게 재현 | **PASS** |
| feature layer | manifest상 `penultimate`, shape `[N, 640]`, checkpoint SHA binding을 검증 | **provenance PASS**, 독립 재추출은 이번 범위 밖 |
| training correctness | checkpoint identity와 ID 성능은 확인했지만 optimizer update 전체를 외부 코드와 독립 재현하지 않음 | **이번 검증 범위 밖** |

여기서 `PASS`는 “모든 환경에서 코드 전체가 절대적으로 옳다”는 의미가 아니다. **이번에 동결한 checkpoint와 feature에 대해 해당 오류 가설이 관찰 결과를 설명하지 못했다**는 의미다.

## 2. 검증 대상과 입력 provenance

검증에는 새로운 학습이나 새로운 OOD inference를 사용하지 않았다. 기존에 저장된 feature를 읽기 전용으로 사용했다.

| 항목 | Adam | AdamW |
|---|---|---|
| run | `task-f-adam-lr1e-03-wd1e-04-seed0-adam-alpha-1` | `task-f-adam-lr1e-03-wd1e-04-seed0-adamw-alpha-0` |
| cell | `adam_lr1e-3_wd1e-4_anchor` | `adam_lr1e-3_wd1e-4_anchor` |
| seed | 0 | 0 |
| checkpoint | `last`, epoch 200 | `last`, epoch 200 |
| checkpoint SHA256 | `a4156cb9cfa4f3a884938e883ee3fafdde1ead43befa7bc16f8d252f1d91dcc4` | `cf25c55f01c7ab5258f6f702d4c7b0065b208977e2e37584b927a62c502d9030` |
| feature layer | penultimate, 640 dimensions | penultimate, 640 dimensions |
| covariance fit | CIFAR-10 `id_train`, 45,000장 | CIFAR-10 `id_train`, 45,000장 |
| query | 기존 `id_test` 9,000장, CIFAR-100 9,000장, SVHN 26,032장 cache | 기존 `id_test` 9,000장, CIFAR-100 9,000장, SVHN 26,032장 cache |
| protected `id_test` accuracy | 0.93478 | 0.93944 |

train feature, label bridge, geometry fit, protected feature, 기존 score bundle의 `checksums.sha256`를 다시 계산했다. 모든 artifact가 같은 run ID, checkpoint role, epoch, checkpoint SHA 및 penultimate tap에 연결되는 것을 확인했다.

검증 환경은 `curie`의 NVIDIA RTX A5000이며, covariance fit은 CPU의 scikit-learn을 사용하고 대량 quadratic score 계산은 GPU batch multiplication을 사용했다.

## 3. 검증 방법

### 3.1 현재 구현의 재계산

현재 프로젝트가 정의한 tied within-class covariance는 다음과 같다.

\[
r_i=z_i-\mu_{y_i}, \qquad
\Sigma_W=\frac{1}{N}\sum_i r_i r_i^\top.
\]

Raw MD와 RMD의 ID-like score는 다음과 같다.

\[
s_{MD}(z)=-\min_c (z-\mu_c)^\top\Sigma_W^\dagger(z-\mu_c),
\]

\[
s_{RMD}(z)=\max_c\left[d_0(z)-d_c(z)\right].
\]

동일 ID-train feature로 `float64`, `EmpiricalCovariance(assume_centered=True)` fit을 다시 수행했다. class mean과 covariance는 저장된 geometry artifact와 일치했고, 각 split의 첫 512장에 대해 재계산한 score와 기존 score의 Spearman correlation은 1.0이었다.

### 3.2 `assume_centered`만 분리한 검증

dtype와 입력을 그대로 유지하고 `assume_centered=True`만 `False`로 바꿨다.

- `True`: 이미 class mean을 뺀 residual에 대해 \(R^\top R/N\)을 직접 계산한다.
- `False`: residual 전체의 평균을 라이브러리가 다시 추정해 한 번 더 뺀다.

정확한 class residual은 전체 평균이 0이므로 두 방식은 이론적으로 같아야 한다. 실제 residual mean의 최대 절댓값은 Adam `1.75e-16`, AdamW `1.89e-16`이었고, centering option에 따른 covariance 상대 변화는 각각 `1.47e-16`, `1.54e-16`이었다.

결과적으로 `float64`에서 centering option만 바꿨을 때 아래 모든 AUROC와 FPR95가 기존 결과와 동일했다.

### 3.3 pinned OpenOOD full-cache 교차 검증

외부 reference는 OpenOOD commit `3c35632ee91b54b09d1f085d04f94744cece7d0b`로 고정했다.

- [OpenOOD MDSPostprocessor](https://github.com/Jingkang50/OpenOOD/blob/3c35632ee91b54b09d1f085d04f94744cece7d0b/openood/postprocessors/mds_postprocessor.py)
- [OpenOOD RMDSPostprocessor](https://github.com/Jingkang50/OpenOOD/blob/3c35632ee91b54b09d1f085d04f94744cece7d0b/openood/postprocessors/rmds_postprocessor.py)

OpenOOD 원형에 맞추어 다음을 적용했다.

- network feature와 class/global mean: `float32` torch tensor;
- class residual과 global residual: `float32`;
- `EmpiricalCovariance(assume_centered=False)`;
- `precision_`을 다시 `float32` tensor로 변환;
- Raw MD: class별 negative quadratic distance의 maximum;
- RMD: 각 class score에서 background score를 뺀 뒤 maximum.

전체 `id_test`, CIFAR-100, SVHN cache를 동일 ID-positive AUROC/FPR95 정의로 평가했다. OpenOOD 원본은 query batch마다 CPU에서 `N x N` 행렬을 만든 후 diagonal만 사용하므로, full-cache score 계산은 동일 quadratic form을 GPU에서 직접 batch 계산했다.

### 3.4 실제 upstream class smoke test

GPU로 옮긴 식이 원본 코드와 같은지를 확인하기 위해 checked-out OpenOOD에서 실제 `MDSPostprocessor`와 `RMDSPostprocessor` class를 import하여 실행했다. 원본 scoring/fitting 파일은 수정하지 않았다. 서버에 설치되지 않은 `tqdm` 진행률 표시만 input iterable을 그대로 반환하는 stub으로 대체했다.

- MDS source SHA256: `2de0eca4fc43efc3ee68edd115c50a46da0a056e98e1ca450bd2ee4887a7aa79`
- RMDS source SHA256: `34f4276b3634ecee80fa05dfefe7d6e656bb74874ec872a8b89b64329f1f68ce`
- MDS와 RMDS가 독립 fit한 class mean과 within precision은 정확히 동일했다.
- 각 query split의 첫 64장에서 upstream CPU score와 GPU-accelerated equivalent score의 Pearson correlation은 최소 0.99980이었다.

실제 class를 실행할 때 사용한 dummy network는 이미 저장된 feature를 그대로 반환하기 위한 adapter다. 따라서 OpenOOD가 출력한 train accuracy 10%는 dummy logits에 관한 값이며, covariance나 MD/RMD score에는 사용되지 않는다.

## 4. 현재 결과는 OpenOOD에서 재현되었다

아래 표는 `AUROC / FPR95`이며 score가 클수록 ID-like인 현재 metric convention을 동일하게 사용했다. 이 표는 exact lookup과 audit가 목적이므로 별도의 chart는 사용하지 않았다.

| Model | Detector | OOD | 현재 프로젝트 | pinned OpenOOD | AUROC 차이 |
|---|---|---|---:|---:|---:|
| Adam | Raw MD | CIFAR-100 | 0.419843 / 0.967333 | 0.419842 / 0.967333 | -0.00000007 |
| Adam | Raw MD | SVHN | 0.204624 / 0.995813 | 0.204623 / 0.995813 | -0.00000140 |
| Adam | RMD | CIFAR-100 | 0.870032 / 0.608667 | 0.869917 / 0.607889 | -0.00011594 |
| Adam | RMD | SVHN | 0.912613 / 0.573832 | 0.912380 / 0.573294 | -0.00023273 |
| AdamW | Raw MD | CIFAR-100 | 0.605906 / 0.900111 | 0.605906 / 0.900111 | +0.00000009 |
| AdamW | Raw MD | SVHN | 0.861224 / 0.433467 | 0.861224 / 0.433467 | +0.00000003 |
| AdamW | RMD | CIFAR-100 | 0.879251 / 0.583333 | 0.879253 / 0.583333 | +0.00000123 |
| AdamW | RMD | SVHN | 0.927920 / 0.475569 | 0.927921 / 0.475530 | +0.00000113 |

OpenOOD 대비 최대 AUROC 차이는 `0.000233`, 최대 FPR95 차이는 `0.000778`이었다. 전체 score array에서 project/OpenOOD 간 Spearman correlation의 최솟값은 `0.99824`였다.

관측된 RMD minus Raw MD AUROC 차이는 다음과 같다.

| Model | CIFAR-100 | SVHN |
|---|---:|---:|
| Adam | +0.450190 | +0.707988 |
| AdamW | +0.273346 | +0.066696 |

따라서 적어도 이 두 checkpoint에서는 큰 Raw MD→RMD 회복이 현재 코드에만 나타나는 결과가 아니다. **동일 feature를 OpenOOD 방식으로 평가해도 같은 현상이 나타난다.**

## 5. 발견된 수치·기하적 특성

| Model | within covariance effective rank | effective condition number |
|---|---:|---:|
| Adam | 501/640 | 1.56e7 |
| AdamW | 640/640 | 1.64e4 |

Adam covariance는 rank-deficient다. 저장 당시와 현재 runtime의 pseudoinverse는 covariance-null direction에서 매우 다른 값을 가질 수 있었지만, 실제 network feature query에서 재계산 score의 순위는 유지되었고 OpenOOD AUROC/FPR95도 동일하게 재현되었다.

이 관찰의 해석은 다음과 같다.

- 이것은 score sign이나 RMD formula bug의 증거가 아니다.
- 하지만 Adam representation이 Raw MD에 불리한 low-rank/ill-conditioned geometry를 형성했을 가능성을 보여준다.
- pseudoinverse portability와 작은 ridge에 대한 sensitivity는 **코드 신뢰성의 재검증**보다는 **현상 분석 및 robustness 확인**으로 다루는 것이 적절하다.
- 이 결과만으로 rank deficiency가 Raw MD 실패의 인과 원인이라고 결론 내릴 수는 없다. seed와 optimizer/LR/WD cell 전체에서 rank, spectrum, Raw MD, RMD의 관계를 확인해야 한다.

## 6. 문헌 수치와 현재 결과를 어떻게 비교해야 하는가

[RMD 원 논문](https://arxiv.org/abs/2106.09022)은 scratch WRN-28-10에서 다음 near-OOD 결과를 보고한다.

| ID vs OOD | MD | RMD | 차이 |
|---|---:|---:|---:|
| CIFAR-100 vs CIFAR-10 | 74.91% | 81.01% | +6.10%p |
| CIFAR-10 vs CIFAR-100 | 88.49% | 89.71% | +1.22%p |

따라서 현재 gap이 문헌의 대표 scratch 결과보다 훨씬 크다는 지적은 타당하며, 구현을 먼저 의심한 것도 합리적인 검증 순서였다.

다만 이 scratch WRN-28-10 결과는 현재 Adam/AdamW 학습과 같은 optimizer setting이 아니다. 원 논문 Appendix B는 CIFAR 모델 학습 코드로 당시 `google/uncertainty-baselines/baselines/cifar/deterministic.py`를 연결한다. 논문 공개 직전인 2021-06-11 commit `df320d4987deddf2e23a8a7cb45eda87d3c5f210`의 실제 설정은 다음과 같다.

- `tf.keras.optimizers.SGD`;
- momentum `0.9` 및 Nesterov `True`;
- base learning rate `0.1` at total batch size 128;
- 200 epochs;
- learning-rate decay epochs `[60, 120, 160]`, decay ratio `0.2`;
- one-epoch warmup;
- TensorFlow model-loss 방식의 L2 coefficient `2e-4`;
- default `train_proportion=1.0`.

따라서 원 논문의 CIFAR-10 ID vs CIFAR-100 OOD 결과 `MD 88.49% / RMD 89.71%`는 **SGD with Nesterov로 학습한 representation의 결과**다. 이 값은 Adam/AdamW representation에서 기대해야 할 직접적인 reference range가 아니다. 특히 해당 TensorFlow L2는 PyTorch optimizer의 `weight_decay`와 구현 위치가 같지 않으므로 숫자만 대응시켜서도 안 된다.

그러나 위 문헌 수치가 현재 실험의 정답 범위를 정의하지는 않는다.

- 같은 WRN-28-10이라는 이름만으로 weight decay semantics, checkpoint, data split, initialization 및 feature covariance spectrum이 같다고 볼 수 없다. 특히 이번 비교는 SGD/Nesterov 대 Adam/AdamW라는 명시적인 optimizer 차이를 포함한다.
- 현재 검증은 45k ID-train/5k validation 및 protected 9k ID-test query를 따르므로, 원 논문의 표와 split·sample membership까지 동일한 재현 실험은 아니다.
- 원 논문은 RMD가 non-discriminative dimension의 MD contribution을 줄이는 방법이라고 설명한다. 현재 프로젝트에서 분석 중인 global/marginal 및 class-orthogonal component와 문제의식이 연결된다.
- 원 논문 자체도 training iteration에 따라 Raw MD AUROC가 상승했다가 크게 하락하고 RMD가 안정되는 사례를 보고한다. 따라서 좋은 ID accuracy가 Raw MD의 정상 범위를 보장하지 않는다.
- 현재 Adam과 AdamW의 ID accuracy는 각각 약 93.48%, 93.94%로 분류 학습이 실패했다고 보기는 어렵다. 반면 covariance rank와 Raw MD는 크게 다르므로, 분류 정확도와 OOD geometry를 별도로 봐야 한다.

즉 문헌과의 차이는 **버그 검증을 촉발하는 red flag**로는 충분하지만, cross-validation을 통과한 뒤에도 결과가 다르다는 이유만으로 버그라고 단정할 수는 없다.

## 7. 이제 남은 검증은 무엇인가

“이제 학습이 틀렸는지만 보면 되는가?”에 대한 답은 **아니다**. 남은 질문을 다음처럼 분리해야 한다.

### 7.1 저비용으로 남아 있는 코드 검증

1. **feature extraction one-batch parity**
   동일 checkpoint의 작은 고정 batch를 model direct forward와 export path 양쪽으로 통과시켜 logits와 penultimate feature를 `allclose` 비교한다. 이것은 재학습 없이 feature layer 및 serialization 경계를 검증한다.

2. **dataset membership audit**
   현재 manifest와 checksum은 fit/query 분리를 보장하지만, 필요하다면 sample ID를 기준으로 45k train, 5k validation, ID test, CIFAR-100, SVHN membership이 의도한 protocol과 일치하는지 한 번 독립 확인한다.

이 두 항목까지 통과하면 metric 결과로 이어지는 코드 경계는 충분히 닫힌다. 대규모 코드 리뷰나 50개 모델 재학습은 코드 신뢰성 검증을 위해 필요하지 않다.

### 7.2 training-induced representation인가, training bug인가

다음은 코드 검증과 과학적 현상 분석을 구분하는 핵심 질문이다.

- optimizer update와 weight-decay semantics가 PyTorch 공식 정의 및 동결 protocol대로 실행됐는가;
- checkpoint epoch, scheduler boundary, initialization 및 data stream이 sibling 간 의도대로 묶였는가;
- 정상 ID accuracy를 가지면서도 Adam representation만 rank deficient해지는가;
- 이 경향이 seed 0 우연이 아니라 여러 seed와 LR/WD cell에서 반복되는가.

현재 checkpoint provenance와 ID accuracy는 “학습 전체가 망가졌다”는 가설과 맞지 않는다. 그러나 외부 training implementation과의 end-to-end parity는 이번에 실행하지 않았으므로, optimizer semantics가 정확하다는 독립 증명으로 사용해서는 안 된다.

### 7.3 교수님이 외부 repository에 붙인 결과만 신뢰하시는 경우

필요하다면 마지막으로 다음의 한 개 bounded experiment만 수행할 수 있다.

- external/canonical WRN-28-10 training recipe 하나;
- seed 하나와 checkpoint 하나;
- CIFAR-10 ID, CIFAR-100 및 SVHN;
- 실제 pinned OpenOOD MDS/RMDS;
- 현재 project scorer를 같은 feature에 병렬 적용.

이 실험의 목적은 50-run 결과를 다시 만드는 것이 아니라, 완전히 독립된 training pipeline에서도 두 scorer가 같은 값을 내고 canonical Raw MD 범위를 재현하는지 확인하는 것이다. 단, 이것은 “현재 Adam/AdamW representation이 왜 다른가”를 직접 설명하지는 않는다.

## 8. 교수님 GPT 피드백에 대한 답변 초안

아래 내용은 그대로 전달하거나 다른 GPT가 문체만 다듬을 수 있도록 작성한 답변 초안이다.

---

교수님께서 지적하신 것처럼 scratch WRN/CIFAR에서 Raw MD와 RMD의 AUROC 차이가 수십 %p 발생한 것은 문헌의 대표적인 결과보다 매우 큰 값이므로, 먼저 구현 오류를 의심하는 것이 맞다고 판단했습니다. 이에 결과 해석에 들어가기 전에 score sign, RMD formula, covariance fit, label 방향 및 OpenOOD 구현과의 일치 여부를 교차 검증했습니다.

검증에는 새로운 모델 학습이나 새로운 OOD inference를 사용하지 않았습니다. primary setting의 Adam/AdamW seed 0, epoch 200 checkpoint에서 이미 저장된 penultimate feature를 사용했습니다. covariance는 CIFAR-10 ID-train 45,000장만으로 다시 fit했고, CIFAR-100과 SVHN은 query에만 사용했습니다. train feature, label, checkpoint, geometry 및 OOD feature artifact의 checksum과 checkpoint SHA도 다시 확인했습니다.

첫째, 현재 프로젝트의 `float64 + EmpiricalCovariance(assume_centered=True)` 구현을 다시 계산하여 저장된 score와 비교했습니다. class mean과 covariance가 일치했고, split별 첫 512장의 MD/RMD score 순위는 Spearman 1.0이었습니다.

둘째, `assume_centered=True`가 문제인지 확인하기 위해 dtype와 입력을 고정하고 `False`로만 변경했습니다. class residual mean의 최대 절댓값은 약 `1.9e-16` 이하였고, 두 선택의 AUROC와 FPR95는 모든 비교에서 동일했습니다. 따라서 이미 class mean을 뺀 residual에 `assume_centered=True`를 사용하는 현재 구현은 결과 차이의 원인이 아니었습니다.

셋째, pinned OpenOOD commit `3c35632e...`의 MDS/RMDS 방식으로 class mean, within covariance, global covariance 및 score를 다시 계산했습니다. 현재 프로젝트와 OpenOOD 간 최대 AUROC 차이는 `0.000233`, 최대 FPR95 차이는 `0.000778`이었습니다. 전체 score rank의 Spearman correlation도 최소 `0.99824`였습니다. 실제 OpenOOD `MDSPostprocessor`와 `RMDSPostprocessor` class를 직접 import한 sample-level smoke test도 별도로 통과했습니다.

구체적으로 Adam seed 0에서는 현재 프로젝트의 Raw MD/RMD AUROC가 CIFAR-100에서 `0.4198/0.8700`, SVHN에서 `0.2046/0.9126`이었고, OpenOOD 재계산도 각각 `0.4198/0.8699`, `0.2046/0.9124`로 재현됐습니다. AdamW에서도 CIFAR-100 `0.6059/0.8793`, SVHN `0.8612/0.9279`가 OpenOOD와 사실상 동일했습니다.

따라서 교수님 GPT에서 제기한 score sign, RMD formula, covariance fit split, OOD leakage, class selection 및 label 방향 중 frozen feature 이후의 scorer/evaluation 단계에서는 오류가 발견되지 않았습니다. 특히 SVHN에서 Adam Raw MD가 매우 낮은 결과도 OpenOOD 방식에서 그대로 재현되어, 이 결과를 단순 score sign 오류로 설명하기 어렵습니다.

대신 이번 검증에서 Adam의 raw covariance effective rank가 `501/640`이고 AdamW는 `640/640`이라는 차이를 확인했습니다. Adam covariance는 매우 ill-conditioned하지만, 실제 query score는 OpenOOD와 일치했습니다. 따라서 현재로서는 “Mahalanobis 코드가 틀렸다”보다 “학습된 representation의 low-rank/global covariance-sensitive component가 Raw MD를 지배하고, RMD가 이를 상쇄하는가”가 더 직접적인 분석 질문입니다. 다만 rank deficiency가 인과 원인이라는 결론은 아직 아니며, 전체 seed와 LR/WD cell에서 확인해야 합니다.

문헌의 scratch WRN 결과보다 gap이 훨씬 크다는 사실은 그대로 중요한 red flag입니다. 그러나 RMD 원 논문도 training iteration에 따라 Raw MD가 크게 하락하고 RMD가 안정되는 사례를 보고하므로, 분류가 정상적으로 학습됐다는 사실만으로 Raw MD 성능이 보장되지는 않습니다. 현재 두 checkpoint의 ID accuracy도 약 93.5~93.9%로 분류 학습 자체가 실패한 모습은 아닙니다.

또한 원 논문의 scratch WRN-28-10 수치는 Adam/AdamW가 아니라 당시 Uncertainty Baselines의 SGD(momentum 0.9, Nesterov enabled, base LR 0.1) 학습에서 나온 결과입니다. 따라서 `MD 88.49% / RMD 89.71%`를 현재 Adam/AdamW 결과의 직접적인 정상 범위로 사용할 수는 없습니다. 이 차이는 현재 학습이 틀렸다는 증거라기보다 optimizer가 달라 representation geometry도 달라질 수 있다는 점을 별도로 검증해야 한다는 근거입니다.

이번 결과가 검증한 것은 metric/scorer 단계까지입니다. 아직 독립적으로 검증하지 않은 부분은 feature extraction을 동일 checkpoint에서 다시 실행하는 end-to-end parity와, training/optimizer implementation 자체입니다. 다음 단계로는 재학습 없이 작은 고정 batch의 penultimate feature/logit parity를 먼저 확인하고, 그래도 외부 training pipeline 검증이 필요하다면 canonical WRN-28-10 한 모델·한 seed에 실제 OpenOOD scorer를 붙이는 bounded experiment만 수행하는 것이 적절하다고 생각합니다.

따라서 현재 결론은 다음과 같습니다.

> 큰 Raw MD–RMD 차이는 문헌의 일반적인 scratch 결과보다 이례적이므로 의심하고 검증해야 한다는 지적이 맞았습니다. 그 검증을 수행한 결과, primary seed-0 checkpoint의 frozen feature에서는 현재 scorer와 pinned OpenOOD가 같은 현상을 재현했습니다. 따라서 score sign이나 covariance centering 같은 단순 구현 오류 가능성은 크게 낮아졌고, 남은 핵심은 학습 코드 전체가 틀렸다고 전제하는 것이 아니라 training-induced representation geometry와 protocol 차이를 분리해 확인하는 것입니다.

---

## 9. 권장 다음 순서

1. 이 문서와 검증 표를 교수님께 먼저 보고한다.
2. metric/scorer 코드는 당분간 동결한다.
3. 재학습 없이 feature extraction one-batch parity와 dataset membership audit만 수행한다.
4. 50-run 결과를 optimizer/LR/WD/seed와 OOD dataset별로 표로 정리하고 covariance rank·condition·Raw MD·RMD를 함께 본다.
5. Adam rank deficiency와 Raw MD failure의 관계를 seed/cell 전체에서 확인한다.
6. 교수님이 external training repository 기반 검증을 계속 요구하실 때만 canonical WRN 한 모델의 bounded end-to-end reference run을 추가한다.

## 10. 검증 artifact와 재현 정보

전체 parity 검증 runtime은 6.41초, 실제 pinned OpenOOD class smoke test는 2.60초였다.

- 상세 diagnostic report: `artifacts/diagnostics/mahalanobis_openood_parity_seed0/report.md`
- full result: `artifacts/diagnostics/mahalanobis_openood_parity_seed0/result.json`
- actual upstream smoke result: `artifacts/diagnostics/mahalanobis_openood_parity_seed0/openood_actual_smoke.json`
- `result.json` SHA256: `5322f902eee60ee270d3472b47b0945f953e609bc1a1c4e47f633631693beaf3`
- `openood_actual_smoke.json` SHA256: `2323c686a699d5f4ab46808b2c045257826e1ac63fe39b4556cdd527f81756b4`

관련 specification과 외부 reference:

- `docs/reference_cards/11_metric_contract_v1_2.md`, Sections 4.1–4.4
- `docs/reference_cards/13_active_paper_protocol.md`
- [A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection](https://arxiv.org/abs/2106.09022)
- [2021-06-11 Uncertainty Baselines CIFAR deterministic training source](https://github.com/google/uncertainty-baselines/blob/df320d4987deddf2e23a8a7cb45eda87d3c5f210/baselines/cifar/deterministic.py)
- [2021-06-11 Uncertainty Baselines CIFAR default hyperparameters](https://github.com/google/uncertainty-baselines/blob/df320d4987deddf2e23a8a7cb45eda87d3c5f210/baselines/cifar/utils.py)
- [pinned OpenOOD MDS source](https://github.com/Jingkang50/OpenOOD/blob/3c35632ee91b54b09d1f085d04f94744cece7d0b/openood/postprocessors/mds_postprocessor.py)
- [pinned OpenOOD RMDS source](https://github.com/Jingkang50/OpenOOD/blob/3c35632ee91b54b09d1f085d04f94744cece7d0b/openood/postprocessors/rmds_postprocessor.py)

## 11. 이 문서가 주장하지 않는 것

- seed 0 결과가 50-run 전체에 일반화된다고 주장하지 않는다.
- Adam rank deficiency가 Raw MD 실패의 인과 원인이라고 확정하지 않는다.
- 현재 training implementation 전체가 외부 reference와 동일하다고 주장하지 않는다.
- RMD가 모든 OOD dataset과 optimizer에서 Raw MD보다 우월하다고 주장하지 않는다.
- 현재 관찰만으로 논문의 최종 주제를 변경해야 한다고 결론 내리지 않는다.
- 문헌보다 큰 gap 자체만으로 novelty를 주장하지 않는다.
