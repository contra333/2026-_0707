# 서버별 Hugging Face CLI 모델 업로드 통합 안내

## 1. 목적과 적용 범위

이 문서는 학습 서버에서 생성한 체크포인트와 모델 파일을 Hugging Face
bucket에 업로드할 때 사용하는 공통 절차와, 2026-07-28에 세 서버에서
확인한 HF CLI 환경을 통합해 기록한다.

실제 업로드는 이 작업에서 수행하지 않았다. 아래의 업로드 경로는 서버별
파일 충돌을 막기 위한 고정 규칙이며, 실제 업로드 성공을 의미하지 않는다.

- 학습용 `candidate-venv`는 변경하지 않는다.
- Hugging Face CLI는 학습 환경과 분리된 virtual environment에서 실행한다.
- 인증 토큰, 비밀번호, device code는 문서나 Git에 기록하지 않는다.
- 공개 bucket에는 공개해도 되는 파일만 업로드한다.
- 업로드 전에는 반드시 `--dry-run` 결과와 파일 용량을 확인한다.

## 2. 서버별 확인 결과

모든 서버에서 확인된 공통 결과는 HF 계정 `contra333`, bucket
`contra333/ICLR_RUN`, 공개 상태(`private: false`), 확인 당시 파일 0개,
모델·체크포인트 업로드 `NOT_RUN`이다.

### 2.1 175 서버 (`lise`)

확인일: 2026-07-28

| 항목 | 확인된 값 |
|---|---|
| 실제 hostname | `lise` |
| HF 전용 virtual environment | `/home/ghjin/0707_exp/hf-cli-venv` |
| HF CLI 경로 | `/home/ghjin/0707_exp/hf-cli-venv/bin/hf` |
| HF CLI / `huggingface_hub` | `1.25.1` |
| `httpx` | `0.28.1` |
| 인증 계정 | `contra333` |
| bucket | `contra333/ICLR_RUN` |
| bucket 상태 | 공개, 파일 0개 |
| `candidate-venv` 변경 여부 | 변경하지 않음 |
| 모델·체크포인트 업로드 | `NOT_RUN` |

### 2.2 193 서버 (`curie`)

확인일: 2026-07-28

| 항목 | 확인된 값 |
|---|---|
| 서버 식별자 | `193` |
| 실제 hostname | `curie` |
| `BASE_PY` | `/home/ghjin/miniconda3/bin/python` |
| `BASE_PY` Python 버전 | `3.13.11` |
| HF 전용 virtual environment | `/home/ghjin/0707_exp/hf-cli-venv` |
| HF CLI 경로 | `/home/ghjin/0707_exp/hf-cli-venv/bin/hf` |
| HF CLI / `huggingface_hub` | `1.25.1` |
| `httpx` | `0.28.1` |
| 인증 계정 | `contra333` |
| bucket | `contra333/ICLR_RUN` |
| bucket 상태 | 공개, 파일 0개, recursive list 비어 있음 |
| `candidate-venv` 변경 여부 | 변경하지 않음 |
| 모델·체크포인트 업로드 | `NOT_RUN` |

193 서버에서는 기존 token이 유효하지 않아 `hf auth login --force`를 실행한
뒤 브라우저 device 인증으로 `contra333` 계정을 확인했다. token과 device
code 자체는 기록하지 않는다.

### 2.3 맞춤의학 서버 (`precision_medicine`)

확인일: 2026-07-28

| 항목 | 확인된 값 |
|---|---|
| 서버 식별자 | `precision_medicine` |
| 실제 hostname | `math-SYS-740GP-TNRT` |
| `BASE_PY` | `/home/lab1/anaconda3/bin/python` |
| `BASE_PY` Python 버전 | `3.11.5` |
| HF 전용 virtual environment | `/mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/hf-cli-venv` |
| HF CLI 경로 | `/mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/hf-cli-venv/bin/hf` |
| HF CLI / `huggingface_hub` | `1.25.1` |
| `httpx` | `0.28.1` |
| 인증 계정 | `contra333` |
| bucket | `contra333/ICLR_RUN` |
| bucket 상태 | 공개, 파일 0개, recursive list 비어 있음 |
| `candidate-venv` 변경 여부 | 변경하지 않음 |
| 모델·체크포인트 업로드 | `NOT_RUN` |

## 3. 서버별 bucket 경로 규칙

실제 업로드 시 서버 식별자와 실행 이름을 함께 사용한다.

| 서버 | 원격 경로 규칙 |
|---|---|
| `lise` | `hf://buckets/contra333/ICLR_RUN/servers/lise/<run-id>` |
| `193` (`curie`) | `hf://buckets/contra333/ICLR_RUN/servers/193/<run-id>` |
| `precision_medicine` | `hf://buckets/contra333/ICLR_RUN/servers/precision_medicine/<run-id>` |

`<run-id>`는 체크포인트가 생성된 학습 실행을 식별하는 값으로 대체한다.
공개하면 안 되는 원본 데이터, 개인정보, 비공개 실험자료는 업로드하지
않는다.

## 4. CLI 확인과 인증

각 서버의 HF 전용 CLI 절대경로를 사용한다. 학습용 환경을 활성화했는지와
관계없이 실행할 수 있다.

```bash
export HF_VENV=/absolute/path/to/hf-cli-venv
export HF="$HF_VENV/bin/hf"

"$HF" version
"$HF" auth whoami
```

인증이 필요하면 해당 서버 작업자가 직접 입력한다.

```bash
"$HF" auth login
"$HF" auth whoami
```

저장된 token이 유효하지 않을 때만 다음 명령으로 재인증한다.

```bash
"$HF" auth login --force
```

계정을 바꾸는 경우 token 이름을 확인한 뒤 로그아웃한다.

```bash
"$HF" auth list
TOKEN_NAME=stored-token-name
"$HF" auth logout --token-name "$TOKEN_NAME"
"$HF" auth login
"$HF" auth whoami
```

## 5. Bucket 확인과 체크포인트 업로드

먼저 bucket 상태를 확인한다.

```bash
HF_BUCKET=contra333/ICLR_RUN

"$HF" buckets info "$HF_BUCKET"
"$HF" buckets list "$HF_BUCKET" --recursive
```

체크포인트 폴더가 확정된 뒤 실제 업로드 대상만 확인한다.

```bash
CHECKPOINT_DIR=/absolute/path/to/checkpoint
REMOTE_DIR=hf://buckets/contra333/ICLR_RUN/servers/<server-id>/<run-id>

"$HF" buckets sync "$CHECKPOINT_DIR" "$REMOTE_DIR" --dry-run
```

출력된 파일 목록과 용량이 예상과 일치할 때만 실제 업로드를 실행한다.

```bash
"$HF" buckets sync "$CHECKPOINT_DIR" "$REMOTE_DIR"
```

기본 절차에서는 `--delete`를 사용하지 않는다. 마지막으로 원격 목록에서
서버 경로와 실행 이름을 확인한다.

```bash
"$HF" buckets list "$HF_BUCKET" --recursive
```

## 6. 확인 근거와 제한사항

- CLI 명령의 의미는 기존 `lise` 환경에서 확인한 `hf ... --help` 결과와
  193·맞춤의학 서버의 실제 실행 기록을 근거로 한다.
- 세 서버에서 HF CLI 설치, 인증, bucket 정보 확인은 완료되었다.
- 이 문서 통합 작업에서는 서버 명령을 재실행하지 않았고, 실제 모델·체크포인트
  업로드도 수행하지 않았다.
- 따라서 업로드 결과는 모두 `NOT_RUN`이며, bucket 경로 규칙은 향후 실행을
  위한 문서상 규칙이다.
