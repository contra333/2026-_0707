# 서버별 Hugging Face CLI 모델 업로드 통합 안내

## 1. 이 문서의 목적

이 문서는 학습 서버에서 생성한 체크포인트와 모델 파일을 Hugging Face에
업로드할 때 따라야 할 공통 절차와 서버별 확인 결과를 기록한다.

현재 서버에서 실제로 확인한 내용을 먼저 기록하고, 193 서버와 맞춤의학
서버의 설정이 끝나면 같은 문서에 서버별 정보를 이어서 추가한다. 토큰,
비밀번호, 일회용 인증 코드와 같은 비밀 정보는 이 문서나 Git에 기록하지
않는다.

이 작업의 기본 원칙은 다음과 같다.

- 학습용 `candidate-venv`는 건드리지 않는다.
- Hugging Face CLI는 별도 virtual environment에서 실행한다.
- 새 터미널에서는 HF CLI 경로를 다시 지정하거나 절대경로를 사용한다.
- 업로드 전에는 `--dry-run`으로 대상 파일을 확인한다.
- 업로드 결과와 서버별 환경 정보는 실제로 확인한 내용만 기록한다.

## 2. 현재 서버에서 확인된 설정

확인일: 2026-07-28

| 항목 | 확인된 값 |
|---|---|
| HF 전용 virtual environment | `/home/ghjin/0707_exp/hf-cli-venv` |
| virtual environment 생성에 사용한 Python | `/home/ghjin/miniconda3/bin/python` |
| `huggingface_hub` | `1.25.1` |
| `httpx` | `0.28.1` |
| 인증 계정 확인 결과 | `contra333` |
| 버킷 | `contra333/ICLR_RUN` |
| 버킷 공개 여부 | 공개(`private: false`) |
| 확인 당시 버킷 파일 수 | 0개 |
| `candidate-venv` 변경 여부 | 변경하지 않음 |

현재 서버에서 로그인은 브라우저 device 인증으로 완료되었다. 인증 토큰의
값 자체와 인증 코드는 기록하지 않는다. 각 서버는 각자의 환경에서 별도로
로그인해야 한다.

## 3. 현재 서버에서 HF CLI 확인하기

가상환경을 활성화하지 않아도 된다. 절대경로를 사용하면 현재 어떤 학습용
환경이 활성화되어 있는지와 관계없이 HF 전용 CLI를 실행할 수 있다.

```bash
/home/ghjin/0707_exp/hf-cli-venv/bin/hf version
/home/ghjin/0707_exp/hf-cli-venv/bin/hf auth whoami
```

`auth whoami`가 다음과 같이 나오면 인증이 정상이다.

```text
user=contra333
```

현재 터미널에서 여러 번 사용할 때는 다음처럼 경로를 기억시킬 수 있다.

```bash
export HF_VENV=/home/ghjin/0707_exp/hf-cli-venv

"$HF_VENV/bin/hf" version
"$HF_VENV/bin/hf" auth whoami
```

`export`는 현재 터미널과 그 터미널에서 실행한 하위 프로세스에만 적용된다.
새 터미널을 열거나 별도 작업 스크립트를 실행하면 다시 설정해야 한다.
헷갈릴 때는 위의 절대경로 명령을 사용한다.

## 4. 인증하기

현재 서버 또는 다른 서버에서 처음 인증할 때 다음을 실행한다.

```bash
/home/ghjin/0707_exp/hf-cli-venv/bin/hf auth login
```

이미 로그인되어 있으면 이미 로그인되어 있다는 안내가 나온다. 저장된
토큰이 유효하지 않으면 다음 명령으로 재로그인한다.

```bash
/home/ghjin/0707_exp/hf-cli-venv/bin/hf auth login --force
```

브라우저 인증 또는 토큰 입력 화면이 나타나면 해당 서버의 작업자가 직접
입력한다. 토큰을 Chat, Issue, 문서, Git commit, 셸 명령 기록에 남기지 않는다.

인증 후 다음 명령으로 계정을 확인한다.

```bash
/home/ghjin/0707_exp/hf-cli-venv/bin/hf auth whoami
```

## 5. 현재 버킷 확인하기

```bash
/home/ghjin/0707_exp/hf-cli-venv/bin/hf buckets info contra333/ICLR_RUN
/home/ghjin/0707_exp/hf-cli-venv/bin/hf buckets list contra333/ICLR_RUN --recursive
```

현재 버킷은 공개 상태이므로 공개하면 안 되는 파일을 업로드하지 않는다.
서버별 파일이 서로 덮어쓰지 않도록 버킷 안에 서버와 실행 이름을 포함한
경로를 사용한다.

예시는 다음과 같다.

```text
hf://buckets/contra333/ICLR_RUN/servers/current/run001
hf://buckets/contra333/ICLR_RUN/servers/193/run001
hf://buckets/contra333/ICLR_RUN/servers/precision_medicine/run001
```

`current`, `193`, `precision_medicine`은 실제 작업에서 사용할 서버 식별자와
일치시킨다. 실행 이름은 체크포인트가 어느 학습 실행에서 만들어졌는지
알 수 있도록 정한다.

## 6. 체크포인트 업로드 절차

### 6.1 업로드 전 확인

`CHECKPOINT_DIR`에는 실제 체크포인트 폴더의 절대경로를 넣는다.

```bash
HF=/home/ghjin/0707_exp/hf-cli-venv/bin/hf
CHECKPOINT_DIR=/absolute/path/to/checkpoint
REMOTE_DIR=hf://buckets/contra333/ICLR_RUN/servers/current/run001

"$HF" buckets sync "$CHECKPOINT_DIR" "$REMOTE_DIR" --dry-run
```

출력된 파일 목록과 용량이 맞는지 확인한다. 예상하지 않은 파일이 있으면
업로드를 중단하고 먼저 폴더를 정리한다.

### 6.2 실제 업로드

`--dry-run` 결과를 확인한 후에만 실행한다.

```bash
"$HF" buckets sync "$CHECKPOINT_DIR" "$REMOTE_DIR"
```

기본 안내에서는 `--delete`를 사용하지 않는다. 원격에 이미 있는 파일을
삭제할 위험이 있기 때문이다.

### 6.3 업로드 결과 확인

```bash
"$HF" buckets list contra333/ICLR_RUN --recursive
```

업로드한 서버 경로와 실행 이름이 보이는지 확인하고, 작업 완료 보고에
업로드 대상과 확인 결과를 기록한다.

## 7. 서버별 통합 기록

아래 표는 설정이 실제로 확인된 서버만 채운다. 경로와 버전은 추측해서
작성하지 않는다.

| 서버 식별자 | 실제 호스트명 | HF CLI 경로 | HF 계정 | 버킷 경로 규칙 | 확인일 | 상태 |
|---|---|---|---|---|---|---|
| 현재 서버 | `lise` | `/home/ghjin/0707_exp/hf-cli-venv/bin/hf` | `contra333` | `servers/current/<run-id>` | 2026-07-28 | 인증·버킷 확인 완료, 업로드 미실행 |
| 193 서버 | 미기록 | 미기록 | 미기록 | `servers/193/<run-id>` | 미기록 | 설정 필요 |
| 맞춤의학 서버 | 미기록 | 미기록 | 미기록 | `servers/precision_medicine/<run-id>` | 미기록 | 설정 필요 |

193 서버와 맞춤의학 서버의 설정이 끝나면 작업지시서의 보고 양식에서
확인된 값만 가져와 이 표와 서버별 기록에 추가한다. 토큰 값은 추가하지
않는다.

## 8. 원격 저장소를 통한 문서 전달

다른 서버는 문서를 별도로 복사하지 않고 원격 저장소에서 받는다.

```bash
git status --short --branch
git pull --ff-only origin main
```

작업 브랜치에서 문서를 시험해야 할 때는 담당자가 지정한 작업 브랜치를
받는다. 로컬에 커밋하지 않은 변경이 있으면 `pull` 전에 중단하고 담당자에게
알린다. `git reset --hard`로 변경을 지우지 않는다.

다른 서버의 설정 결과는 작업 완료 보고 양식으로 현재 서버 담당자에게
전달한다. 현재 서버에서 이 통합 문서에 결과를 추가한 뒤, 같은 저장소에
커밋하고 Pull Request를 통해 반영한다.

## 9. 보안 및 범위 주의사항

- HF 토큰, 비밀번호, device code를 문서나 Git에 저장하지 않는다.
- 현재 버킷은 공개 상태이므로 데이터셋 원본, 개인정보, 비공개 실험자료를
  업로드하지 않는다.
- 학습용 `candidate-venv`에 `pip install`하지 않는다.
- HF CLI 설정을 위해 학습 코드나 프로젝트 의존성을 변경하지 않는다.
- 업로드 전 `--dry-run` 결과를 확인한다.
- 서버별 접두사와 실행 이름을 사용해 파일 충돌을 피한다.
