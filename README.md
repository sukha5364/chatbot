# 데카트론 코리아 GPT 기반 AI 챗봇

## 1. 프로젝트 개요

데카트론 코리아 자체 플랫폼(앱, 웹사이트)에 탑재될 GPT 기반의 지능형 AI 챗봇 개발 프로젝트입니다. 본 프로젝트는 고객 만족도 향상, 운영 효율 개선 및 관련 연구(논문) 수행을 목표로 합니다.

**주요 기능:**

* 타 브랜드 사이즈 정보를 이용한 데카트론 제품 사이즈 추천
* RAG(Retrieval-Augmented Generation) 기반의 신뢰도 높은 정보 제공 (공개된 제품 정보, FAQ 등 활용)
* 자연스러운 멀티턴 대화 및 맥락 유지
* 질문 복잡도에 따른 모델 라우팅 (GPT-3.5-Turbo, GPT-4, GPT-4o)
* CoT(Chain-of-Thought)를 활용한 복잡한 질문 처리
* 비용 효율화를 위한 토큰 절약 전략 적용 (요약, Slot 관리 등)

**기술 스택:**

* Python 3.11
* FastAPI (웹 프레임워크)
* OpenAI API (GPT-3.5/4/4o, Text Embedding)
* FAISS (벡터 검색 라이브러리)
* Langchain (Text Splitter)
* asyncio, aiohttp (비동기 처리)
* Uvicorn (ASGI 서버)
* YAML (설정 관리)
* dotenv (환경 변수 관리)

## 2. 아키텍처

본 챗봇은 FastAPI 기반의 웹 서버로 구동되며, 비동기 API 호출 스케줄러를 통해 Slot 추출, 모델 라우팅, 임베딩 생성 등의 작업을 효율적으로 처리합니다. RAG는 FAISS 인덱스를 사용하여 관련 문서를 검색하고, 모델 라우팅과 CoT는 질문의 복잡도에 따라 동적으로 적용됩니다. 상세 설정은 `config.yaml` 파일을 통해 관리됩니다.

*(선택 사항: 여기에 간단한 아키텍처 다이어그램 이미지나 설명을 추가할 수 있습니다.)*

## 3. 디렉토리 구조

```
chatbot/
├── chatbot/            # 핵심 챗봇 로직 (FastAPI 앱, 스케줄러, 모듈 등)
├── data/               # RAG용 데이터 (원본, 인덱스, 메타데이터)
├── documents/          # 프로젝트 관련 문서
├── logs/               # 로그 파일 저장 위치
├── pipeline/           # RAG 인덱스 생성 파이프라인 스크립트
├── static/             # 웹 UI 파일 (HTML, CSS, JS)
├── tests/              # 테스트 관련 파일 (생성기, 실행기, 결과 등)
├── .env                # (생성 필요) 환경 변수 파일 (API 키 등)
├── .env.example        # 환경 변수 예시 파일
├── config.yaml         # 주요 설정 파일
├── requirements.txt    # Python 의존성 목록
├── README.md           # 프로젝트 개요 (이 파일)
└── 실행 가이드.md        # 상세 실행 방법 안내
```

## 4. 사전 요구 사항

* Python 3.11
* `pip` (Python 패키지 관리자)
* OpenAI API 키
* (선택) FAISS 설치를 위한 시스템 의존성 (C++ 컴파일러 등)

## 5. 설치 및 설정

1.  **저장소 클론:**
    ```bash
    git clone <저장소_URL>
    cd chatbot
    ```
2.  **가상 환경 생성 및 활성화 (권장):**
    * Linux/Mac:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
3.  **필수 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **환경 변수 설정:**
    * `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.
        ```bash
        # Linux/Mac
        cp .env.example .env
        # Windows
        copy .env.example .env
        ```
    * 생성된 `.env` 파일을 열어 `OPENAI_API_KEY=` 뒷부분에 자신의 OpenAI API 키를 입력합니다. (따옴표 없이 키만 입력)
        ```dotenv
        OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ```

## 6. RAG 파이프라인 실행 (최초 1회 또는 데이터 변경 시)

1.  **원본 데이터 준비:** (선택 사항) `data/original/` 디렉토리에 RAG에 사용할 자체 `.txt` 파일(UTF-8 인코딩)을 추가할 수 있습니다. (예시 파일 포함됨)
2.  **파이프라인 스크립트 실행:**
    ```bash
    python pipeline/rag_generator.py
    ```
    * *주의:* 이 과정은 OpenAI Embedding API를 호출하므로 비용이 발생하며, 데이터 양에 따라 시간이 소요될 수 있습니다.
3.  **결과 확인:** 실행 완료 후 `data/` 디렉토리에 `index.faiss` (벡터 인덱스)와 `doc_meta.jsonl` (문서 조각 및 메타데이터) 파일이 생성되었는지 확인합니다.

## 7. 챗봇 서버 실행

프로젝트 루트 디렉토리(`chatbot/`)에서 아래 명령어를 실행합니다.

```bash
uvicorn chatbot.app:app --host 127.0.0.1 --port 8000 --reload
```

* `--reload` 옵션은 개발 중 코드 변경 시 서버 자동 재시작을 위해 사용되며, 배포 시에는 제거합니다.
* 서버 중지는 `Ctrl+C`를 누릅니다.

## 8. 웹 UI 접속

챗봇 서버가 실행 중인 상태에서 웹 브라우저를 열고 아래 주소로 접속합니다.

`http://127.0.0.1:8000`

채팅 인터페이스를 통해 챗봇과 대화할 수 있습니다.

## 9. 테스트 실행

자동화된 테스트를 통해 챗봇의 기능과 성능을 검증할 수 있습니다. 상세한 내용은 `tests/README.md` 파일을 참고하세요.

* **테스트 케이스 생성:**
    ```bash
    python tests/test_generator.py
    ```
* **테스트 실행 (예: overall 세트 1):**
    ```bash
    # 챗봇 서버가 실행 중이어야 합니다!
    python -m tests.test_runner --test-type overall --set 1
    ```
* **결과 시각화 (예: 모든 결과 파일 대상):**
    ```bash
    python tests/visualize_results.py --input "tests/test_results/test_run_*.jsonl"
    ```

## 10. 설정 파일 (`config.yaml`)

챗봇의 동작 방식(사용 모델, 프롬프트, RAG 설정, 로깅 레벨 등)은 `config.yaml` 파일을 통해 상세하게 제어할 수 있습니다. 필요에 따라 이 파일을 수정하여 챗봇의 동작을 변경할 수 있습니다.

## 11. 로깅

챗봇 서버 실행 중 발생하는 주요 이벤트 및 OpenAI API 호출/응답 내역은 `logs/` 디렉토리에 `api_history.txt` (또는 날짜별 파일 `api_history.txt.YYYYMMDD`) 형식으로 기록됩니다. 로그 상세 수준은 `config.yaml` 파일의 `logging.log_level` 설정으로 조절할 수 있습니다.
