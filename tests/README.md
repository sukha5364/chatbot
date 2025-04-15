# 챗봇 테스트 가이드

이 문서는 `tests/` 디렉토리에 포함된 도구들을 사용하여 데카트론 AI 챗봇의 기능과 성능을 테스트하는 방법을 안내합니다.

## 1. 테스트 개요

본 테스트 프레임워크는 `pytest`와 같은 표준 라이브러리 대신, 프로젝트의 특정 요구사항에 맞춰 개발된 자체 스크립트(`test_runner.py`, `test_generator.py`, `visualize_results.py`)를 사용합니다.

**주요 목적:**

* 챗봇의 핵심 기능(RAG, 모델 라우팅, CoT, 슬롯 추출 등) 검증
* 다양한 프롬프트 구성 방식(Mode 1, 2, 3)에 따른 성능(응답 속도, 응답 내용) 비교
* LLM(GPT-4o)을 이용한 자동 응답 만족도 평가
* 테스트 결과의 정량적 분석 및 시각화

## 2. 테스트 구조

* **`test_cases/`**: `test_generator.py`에 의해 자동 생성된 테스트 질문들이 저장되는 디렉토리입니다 (`.jsonl` 형식).
* **`test_generator.py`**: 키워드와 템플릿을 기반으로 다양한 테스트 질문을 생성하는 스크립트입니다.
* **`test_runner.py`**: `test_cases/`의 질문들을 사용하여 실제 챗봇 API를 호출하고, 응답, 성능 지표, 만족도 평가 결과를 기록하는 스크립트입니다.
* **`visualize_results.py`**: `test_runner.py`가 생성한 결과 파일을 분석하여 그래프와 요약 리포트를 생성하는 스크립트입니다.
* **`test_results/`**: (자동 생성됨) `test_runner.py` 실행 결과 파일이 저장되는 디렉토리입니다.
* **`test_plots/`**: (자동 생성됨) `visualize_results.py` 실행 결과 그래프 및 리포트가 저장되는 디렉토리입니다.

## 3. 테스트 절차

### 3.1. 테스트 케이스 생성

테스트를 실행하기 전에, `test_generator.py`를 사용하여 테스트 질문들을 생성합니다.

```bash
# 프로젝트 루트 디렉토리에서 실행
python tests/test_generator.py
```

* 이 스크립트는 `test_generator.py` 내의 `TEST_SET_CONFIG` 설정 (또는 `config.yaml`의 관련 설정)에 따라 기능 테스트(`functional_tests.jsonl`) 및 여러 세트의 전체 테스트(`overall_tests_setX.jsonl`) 파일을 `tests/test_cases/` 디렉토리에 생성합니다.
* 생성되는 질문 수는 설정에 따라 조절할 수 있습니다.

### 3.2. 테스트 실행 (`test_runner.py`)

**중요:** 테스트를 실행하기 전에 **챗봇 서버(`chatbot/app.py`)가 반드시 실행 중**이어야 합니다 (`http://127.0.0.1:8000`).

프로젝트 루트 디렉토리에서 아래 형식의 명령어를 사용하여 테스트를 실행합니다. `-m` 옵션을 사용하여 모듈로 실행하는 것이 좋습니다.

```bash
# 기능 테스트 실행 예시
python -m tests.test_runner --test-type function

# 전체 테스트 세트 1번 실행 예시
python -m tests.test_runner --test-type overall --set 1

# 전체 테스트 세트 2번을 동시 요청 10개로 실행 예시
python -m tests.test_runner --test-type overall --set 2 --concurrency 10
```

**명령줄 인자:**

* `--test-type` (필수): 실행할 테스트 유형을 지정합니다 (`function` 또는 `overall`).
* `--set` ( `--test-type`이 `overall`일 때 필수): 실행할 전체 테스트 세트 번호를 지정합니다 (예: 1, 2, ...). `test_generator.py` 설정의 `num_sets` 범위 내에서 지정해야 합니다.
* `--concurrency` (선택): 동시에 챗봇 API에 요청을 보낼 최대 개수를 지정합니다. API Rate Limit이나 로컬 서버 부하를 고려하여 조절합니다 (기본값: 5).

**테스트 모드 설명:**

`test_runner.py`는 각 질문에 대해 아래 3가지 모드로 API를 호출하고 결과를 비교합니다.

1.  **Mode 1 (User Prompt Only):** 사용자 질문만 GPT 모델(Baseline 모델, 예: `gpt-3.5-turbo`)에 직접 전달합니다.
2.  **Mode 2 (User + System Prompt):** 기본 시스템 프롬프트와 사용자 질문을 함께 Baseline 모델에 전달합니다.
3.  **Mode 3 (Full Pipeline):** 챗봇 서버의 `/chat` 엔드포인트를 호출합니다. 이 때 `X-Test-Mode: true` 헤더를 전송하여 RAG, 모델 라우팅, CoT, 슬롯 추출, 요약 등 모든 파이프라인이 동작하도록 하고, 상세한 디버그 정보(사용된 모델, 라우팅 결과 등)를 함께 반환받습니다.

**자동 만족도 평가:**

* Mode 3 (Full Pipeline)의 응답에 대해서는 자동으로 **GPT-4o 모델**을 호출하여 응답 만족도(관련성, 정확성, 완결성, 간결성, 어조, 종합 만족도)를 평가하고 그 결과를 함께 기록합니다.
* **주의:** 이 평가는 추가적인 OpenAI API 호출 비용을 발생시킵니다.

**결과 파일:**

* 테스트 실행 결과는 `tests/test_results/` 디렉토리에 `test_run_<유형>_<세트번호?>_<타임스탬프>.jsonl` 형식의 파일로 저장됩니다.
* 각 줄은 하나의 테스트 케이스에 대한 결과를 나타내는 JSON 객체이며, 원본 질문, 각 모드별 응답 내용, 응답 지연 시간(latency), 성공 여부, Mode 3의 경우 디버그 정보, 만족도 평가 결과 등을 포함합니다.

### 3.3. 결과 분석 및 시각화 (`visualize_results.py`)

`test_runner.py`로 생성된 결과 파일을 분석하여 성능 지표를 시각화하고 요약 리포트를 생성할 수 있습니다.

```bash
# 특정 결과 파일 하나를 분석하는 예시
python tests/visualize_results.py --input tests/test_results/test_run_overall_set1_20250407_153000.jsonl

# 특정 패턴(예: 모든 overall 테스트 결과)에 맞는 파일들을 함께 분석하는 예시
python tests/visualize_results.py --input "tests/test_results/test_run_overall_*.jsonl"

# 결과를 'report_images' 디렉토리에 저장하는 예시
python tests/visualize_results.py --input "tests/test_results/*.jsonl" --output-dir report_images
```

**명령줄 인자:**

* `--input` (필수): 분석할 `.jsonl` 결과 파일의 경로 또는 여러 파일을 지정하는 glob 패턴 (예: `"tests/test_results/*.jsonl"`)을 입력합니다.
* `--output-dir` (선택): 생성된 그래프 이미지 파일(`.png`)과 요약 리포트 파일(`summary_report.txt`)을 저장할 디렉토리를 지정합니다 (기본값: `test_plots`).

**생성되는 결과물:**

지정된 `--output-dir`에 아래와 같은 파일들이 생성됩니다.

* **그래프 이미지 파일 (`.png`):**
    * `latency_comparison_boxplot.png`: Mode 1/2/3 응답 속도 비교
    * `response_length_comparison_boxplot.png`: Mode 1/2/3 응답 길이 비교
    * `mode3_satisfaction_distribution.png`: Mode 3 만족도 점수 분포
    * `mode3_model_routing_piechart.png`: Mode 3에서 사용된 모델 비율
    * `mode3_latency_by_difficulty.png`: 사전 정의된 질문 난이도별 Mode 3 응답 속도
    * `mode3_satisfaction_by_difficulty.png`: 사전 정의된 질문 난이도별 Mode 3 만족도
* **요약 리포트 파일 (`summary_report.txt`):**
    * 평균 응답 속도, 성공률, 평균 응답 길이 (Mode별 비교)
    * Mode 3 평균 만족도 점수
    * Mode 3 모델 사용률
    * Mode 3과 Mode 2 간의 성능 개선율 요약 등

이러한 결과물들을 통해 챗봇의 성능을 다각도로 평가하고, 개선 방향을 모색하는 데 활용할 수 있습니다.
