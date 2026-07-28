# 다른 서버용 Hugging Face CLI 연결 설정 작업지시서

## 1. 작업 대상

이 문서는 193 서버와 맞춤의학 서버에서 Hugging Face CLI를 설정할 때
그대로 따라 하는 작업지시서다.

목표는 학습용 `candidate-venv`를 건드리지 않고, 서버별 HF 전용 virtual
environment에서 로그인·버킷 확인·모델 업로드를 할 수 있게 만드는 것이다.

이 문서의 명령은 다른 서버에 맞게 아래 변수만 먼저 바꾸어 사용한다.

```bash
SERVER_LABEL=193
BASE_PY=/absolute/path/to/base/python
HF_VENV=/absolute/path/to/hf-cli-venv
HF_BUCKET=contra333/ICLR_RUN
```

맞춤의학 서버에서는 다음처럼 서버 식별자를 바꾼다.

```bash
SERVER_LABEL=precision_medicine
```

`BASE_PY`는 학습용 `candidate-venv`의 Python이 아닌, 해당 서버의 base
Python 또는 별도 Conda Python을 지정한다. 실제 경로를 모르면 먼저 다음을
실행해 확인하고, 후보 경로를 임의로 사용하지 않는다.

```bash
command -v python
command -v python3
```

## 2. 원격 저장소에서 최신 문서 받기

먼저 저장소 상태를 확인한다.

```bash
REPO_DIR=/absolute/path/to/2026-_0707
git -C "$REPO_DIR" status --short --branch
```

커밋하지 않은 변경이 있으면 `pull`하지 말고 담당자에게 알린다. 변경이
없을 때 문서가 있는 브랜치에서 최신 내용을 받는다.

Pull Request가 이미 merge된 뒤라면:

```bash
git -C "$REPO_DIR" pull --ff-only origin main
```

작업 브랜치에서 먼저 시험하라는 안내를 받은 경우에는 `main` 대신 지정된
작업 브랜치를 사용한다. `git reset --hard`로 기존 변경을 삭제하지 않는다.

## 3. HF 전용 virtual environment 만들기

아래 명령은 학습용 환경과 분리된 HF 전용 환경을 만든다.

```bash
set -e

SERVER_LABEL=193
BASE_PY=/absolute/path/to/base/python
HF_VENV=/absolute/path/to/hf-cli-venv

if [ ! -x "$BASE_PY" ]; then
  echo "BASE_PY 경로를 확인하세요: $BASE_PY"
  exit 1
fi

if [ ! -x "$HF_VENV/bin/python" ]; then
  "$BASE_PY" -m venv "$HF_VENV"
fi

"$HF_VENV/bin/python" -m pip install --upgrade huggingface_hub httpx
"$HF_VENV/bin/python" - <<'PY'
import sys
import httpx
import huggingface_hub

print("python:", sys.executable)
print("httpx:", httpx.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
PY

"$HF_VENV/bin/hf" version
```

현재 서버에서 사용한 경로는 다음과 같다.

```text
BASE_PY=/home/ghjin/miniconda3/bin/python
HF_VENV=/home/ghjin/0707_exp/hf-cli-venv
```

다른 서버에서는 위 경로를 그대로 복사하지 말고 그 서버의 실제 경로로
바꾼다. `candidate-venv` 안에서 `pip install`하지 않는다.

## 4. 인증하기

현재 터미널에서 HF CLI 경로를 변수로 지정한다.

```bash
export HF_VENV=/absolute/path/to/hf-cli-venv
HF="$HF_VENV/bin/hf"
"$HF" version
```

처음 로그인할 때:

```bash
"$HF" auth login
```

이미 로그인되어 있으면 이미 로그인되어 있다는 안내가 나온다. 다음
명령으로 계정을 확인한다.

```bash
"$HF" auth whoami
```

저장된 토큰이 유효하지 않다는 오류가 나오면 다음 명령으로 재로그인한다.

```bash
"$HF" auth login --force
```

브라우저 device 인증 또는 토큰 입력 화면이 나오면 해당 서버를 사용하는
사람이 직접 입력한다. 토큰과 device code를 이 문서, Chat, Issue, Git,
셸 스크립트에 기록하지 않는다. 인증 화면이 나타나면 자동 작업을 멈추고
사용자 입력을 기다린다.

## 5. 버킷 접근 확인

공통 버킷과 서버별 경로를 확인한다.

```bash
HF_BUCKET=contra333/ICLR_RUN

"$HF" buckets info "$HF_BUCKET"
"$HF" buckets list "$HF_BUCKET" --recursive
```

모델을 업로드할 때는 서버별 접두사를 사용한다.

```text
193 서버:
hf://buckets/contra333/ICLR_RUN/servers/193/<run-id>

맞춤의학 서버:
hf://buckets/contra333/ICLR_RUN/servers/precision_medicine/<run-id>
```

현재 `contra333/ICLR_RUN` 버킷은 공개 상태다. 공개하면 안 되는 파일이나
개인정보가 포함된 파일은 업로드하지 않는다.

## 6. 학습이 끝난 뒤 모델 업로드하기

학습이 끝나고 체크포인트 폴더가 확정된 뒤에만 실행한다.

```bash
CHECKPOINT_DIR=/absolute/path/to/checkpoint
REMOTE_DIR=hf://buckets/contra333/ICLR_RUN/servers/${SERVER_LABEL}/run001
```

먼저 업로드 대상만 확인한다.

```bash
"$HF" buckets sync "$CHECKPOINT_DIR" "$REMOTE_DIR" --dry-run
```

파일 목록과 용량이 맞는지 확인한 뒤 실제 업로드한다.

```bash
"$HF" buckets sync "$CHECKPOINT_DIR" "$REMOTE_DIR"
```

기본 절차에서는 `--delete`를 사용하지 않는다. 마지막으로 업로드 결과를
확인한다.

```bash
"$HF" buckets list "$HF_BUCKET" --recursive
```

## 7. 작업 완료 보고

설정이 끝나면 아래 양식을 채워 현재 서버 담당자에게 전달한다. 토큰,
비밀번호, device code는 절대 포함하지 않는다.

```text
서버 식별자:
실제 hostname:
확인일:
저장소 commit 또는 branch:
BASE_PY 경로:
HF_VENV 경로:
HF CLI 버전:
huggingface_hub 버전:
httpx 버전:
hf auth whoami 결과의 계정명:
HF 버킷:
hf buckets info 결과 요약:
hf buckets list 결과 요약:
업로드를 수행했는가: 예 / 아니오
업로드한 경우의 원격 경로:
dry-run 확인 결과:
실제 업로드 결과:
실패 또는 미실행 항목:
```

현재 서버 담당자는 이 보고에서 확인된 값만
`docs/HF_MODEL_UPLOAD_GUIDE.md`의 서버별 통합 기록에 추가한다.

## 8. 문제 발생 시 보고할 내용

- 전체 토큰이나 비밀번호를 복사하지 않는다.
- 오류 문장에서 비밀 값은 가린다.
- 실행한 명령의 구조, 종료 코드, 오류의 마지막 부분을 전달한다.
- `candidate-venv`를 변경하지 않는다.
- 인증 입력 화면이 나오면 입력을 멈추고 사용자에게 넘긴다.
- 버킷이 공개 상태이므로 파일을 임의로 시험 업로드하지 않는다.
