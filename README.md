# 데카트론 코리아 GPT 기반 AI 챗봇

## 1. 프로젝트 개요

데카트론 코리아 자체 플랫폼(앱, 웹사이트)에 탑재될 GPT 기반의 지능형 AI 챗봇 개발 프로젝트입니다. 본 프로젝트는 OpenAI GPT API (GPT-3.5-Turbo, GPT-4, GPT-4o 모델)와 RAG(Retrieval-Augmented Generation) 기술을 활용하여 고객 만족도 향상, 운영 효율 개선 및 관련 연구(논문) 수행을 목표로 합니다.

**주요 기능:**

* 타 브랜드 사이즈 정보를 이용한 데카트론 제품 사이즈 추천 (프롬프트 엔지니어링 및 RAG 정보 활용)
* RAG(Retrieval-Augmented Generation) 기반의 신뢰도 높은 정보 제공 (자체 파싱 제품 정보 블록 활용)
* **Tool Use (Function Calling) 기반의 동적 정보 검색 및 응답 생성**: `scheduler.py`를 중심으로 사용자의 복잡한 질문을 이해하고, 필요한 정보를 얻기 위해 `product_search`와 같은 Tool(함수)을 여러 차례 호출하며 그 결과를 종합하여 최종 답변을 생성합니다.
* 자연스러운 멀티턴 대화 및 맥락 유지 (대화 요약, 최근 K턴 대화 활용)
* 비용 효율화를 위한 토큰 절약 전략 적용 (Slot 추출, 주기적 요약, RAG 결과 수 제한 등)
* 상세 로깅을 통한 연구 데이터 확보 및 디버깅 지원 (`gpt_interface.py`)
* 자동화된 테스트 프레임워크 (`tests/` 디렉토리)를 통한 기능 및 성능 검증

**기술 스택:**

* Python 3.12 (코드 내 `.pyc` 파일 및 최신 경향 반영)
* FastAPI (웹 프레임워크)
* Uvicorn (ASGI 서버)
* OpenAI API (`openai` 라이브러리: GPT-3.5-Turbo, GPT-4, GPT-4o, text-embedding-3-large)
* FAISS (벡터 검색 라이브러리 - `IndexFlatIP` 사용)
* NumPy, Pandas (데이터 처리 및 분석)
* aiohttp (비동기 HTTP 요청)
* PyYAML (설정 관리 - `config.yaml`)
* python-dotenv (환경 변수 관리 - `.env`)
* Matplotlib, Seaborn (테스트 결과 시각화)
* *참고: Langchain은 `rag_generator.py` 주석에 언급되어 있으나, 현재 제품 블록의 주 분할 로직에는 직접 사용되지 않습니다.*

## 2. 아키텍처

본 챗봇은 FastAPI 기반의 웹 서버로 구동되며, `chatbot/scheduler.py` 모듈이 전체 대화 흐름을 오케스트레이션합니다. 핵심 로직은 **Tool Use (Function Calling)** 패러다임을 따릅니다. 사용자 입력이 들어오면, `scheduler.py`는 대화 맥락(요약, 이전 대화, Slot 정보 등)을 구성하고, 정의된 Tool(예: `product_search`)과 함께 LLM을 호출합니다. LLM은 상황에 따라 직접 답변하거나, 필요한 정보를 얻기 위해 Tool 호출을 요청할 수 있습니다. Tool 호출이 발생하면, `scheduler.py`는 해당 Tool을 실행(예: RAG 검색 수행)하고 그 결과를 다시 LLM에게 전달하여 최종 답변을 생성하도록 합니다. 이 과정은 필요에 따라 여러 번 반복될 수 있습니다.

RAG는 `pipeline/rag_generator.py`를 통해 오프라인으로 구축된 FAISS 인덱스(`data/index.faiss`)와 메타데이터(`data/doc_meta.jsonl`)를 사용합니다. 제품 정보는 원본 텍스트 파일의 "제품 블록" 단위로 청크화되어 저장 및 검색됩니다.

상세 설정은 `config.yaml` 파일을 통해 관리됩니다.

## 3. 디렉토리 구조

.
├── chatbot/                  # 핵심 챗봇 애플리케이션 로직
│   ├── app.py                # FastAPI 애플리케이션 (메인 실행 파일)
│   ├── scheduler.py          # Tool Use 기반 대화 흐름 및 RAG 오케스트레이션
│   ├── gpt_interface.py      # OpenAI API 연동 및 상세 로깅
│   ├── conversation_state.py # 대화 상태(히스토리, 슬롯, 요약) 관리
│   ├── searcher.py           # FAISS 인덱스 로드 및 벡터 검색
│   ├── slot_extractor.py     # 사용자 입력에서 Slot 정보 추출
│   ├── summarizer.py         # 대화 요약 생성
│   └── config_loader.py      # config.yaml 설정 로드
├── data/                     # RAG 데이터 및 원본 문서
│   ├── original/             # 원본 제품 정보 텍스트 파일 (브랜드별)
│   ├── index.faiss           # FAISS 벡터 인덱스 파일
│   └── doc_meta.jsonl        # 문서 메타데이터 파일 (raw_block_text 포함)
├── document/                 # 프로젝트 관련 문서 (예: 수행계획서)
├── logs/                     # API 호출 및 인터랙션 로그
├── pipeline/                 # RAG 데이터 생성 파이프라인
│   └── rag_generator.py      # 원본 문서에서 FAISS 인덱스 및 메타데이터 생성
├── static/                   # 웹 UI 정적 파일
│   └── index.html            # 기본 채팅 UI
├── tests/                    # 테스트 관련 스크립트 및 데이터
│   ├── test_cases/           # 생성된 테스트 케이스 (.jsonl)
│   ├── test_results/         # 테스트 실행 결과 (.jsonl)
│   ├── test_generator.py     # 테스트 케이스 생성기
│   ├── test_runner.py        # 테스트 실행기 (3모드 비교, 자동 평가)
│   └── visualize_results.py  # 테스트 결과 시각화 및 리포트 생성
├── .env                      # (생성 필요) 환경 변수 파일 (OPENAI_API_KEY 등)
├── config.yaml               # 프로젝트 전체 설정 파일
├── requirements.txt          # Python 패키지 의존성
├── README.md                 # 프로젝트 개요 (이 파일)
└── 실행 가이드.md              # 상세 실행 방법 안내 (프로젝트 루트에 위치)


## 4. 사전 요구 사항

* Python 3.12 (또는 호환 가능한 3.x 버전)
* `pip` (Python 패키지 관리자)
* OpenAI API 키
* (선택) FAISS 설치를 위한 시스템 의존성 (C++ 컴파일러 등). `faiss-cpu`는 일반적으로 추가 의존성 없이 설치 가능합니다.

## 5. 설치 및 설정

1.  **저장소 클론:**
```bash
    git clone <저장소_URL>
    cd <repository_name> # 예: cd decathlon-chatbot
```
2.  **가상 환경 생성 및 활성화 (권장):**
```bash
    * Linux/Mac:
        python3 -m venv venv
        source venv/bin/activate
    * Windows:
        python -m venv venv
        venv\Scripts\activate
```
3.  **필수 패키지 설치:**
```bash
    pip install -r requirements.txt
```
4.  **환경 변수 설정:**
```bash
    * `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.
        # Linux/Mac
        cp .env.example .env
        # Windows
        copy .env.example .env

    * 생성된 `.env` 파일을 열어 `OPENAI_API_KEY=` 뒷부분에 자신의 OpenAI API 키를 입력합니다. (따옴표 없이 키만 입력)
        OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 6. RAG 파이프라인 실행 (최초 1회 또는 데이터 변경 시)

1.  **원본 데이터 준비:** (선택 사항) `data/original/` 디렉토리에 RAG에 사용할 자체 `.txt` 파일(UTF-8 인코딩)을 추가할 수 있습니다. (프로젝트 내 예시 파일 참고)
2.  **파이프라인 스크립트 실행:**
```bash
    python pipeline/rag_generator.py

    * *주의:* 이 과정은 OpenAI Embedding API를 호출하므로 비용이 발생하며, 데이터 양에 따라 시간이 소요될 수 있습니다.
    * 스크립트는 `data/original/` 내의 텍스트 파일을 읽어 제품 블록 단위로 청크를 나누고, 각 청크의 원본 텍스트(`raw_block_text`)에 대한 임베딩을 생성합니다.
```
3.  **결과 확인:** 실행 완료 후 `data/` 디렉토리에 `index.faiss` (벡터 인덱스)와 `doc_meta.jsonl` (문서 청크 메타데이터 및 `raw_block_text` 포함) 파일이 생성되었는지 확인합니다.

## 7. 챗봇 서버 실행

프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다. (예: `decathlon-chatbot/`)

```bash
python chatbot/app.py
```

또는 config.yaml의 server 설정을 활용하여 Uvicorn으로 직접 실행:

```bash
uvicorn chatbot.app:app --host 127.0.0.1 --port 8000 --reload
```

--reload 옵션은 개발 중 코드 변경 시 서버 자동 재시작을 위해 사용되며, 배포 시에는 제거합니다.
서버 중지는 Ctrl+C를 누릅니다.

## 8. 웹 UI 접속
챗봇 서버가 실행 중인 상태에서 웹 브라우저를 열고 아래 주소로 접속합니다.

http://127.0.0.1:8000 (또는 config.yaml에 설정된 주소)

채팅 인터페이스를 통해 챗봇과 대화할 수 있습니다.

## 9. 설정 파일 (config.yaml)
챗봇의 동작 방식(사용 모델, 프롬프트, RAG 설정, 로깅 레벨 등)은 config.yaml 파일을 통해 상세하게 제어할 수 있습니다. 주요 설정 항목은 다음과 같습니다:

tasks: Slot 추출, 요약, 만족도 평가, Tool Use 등 각 작업별 모델 및 파라미터.
rag: RAG 검색 관련 설정 (검색할 K개, 임베딩 모델 등).
tools: Tool Use (Function Calling)에 사용될 함수(Tool) 정의 (product_search 등).
prompts: 다양한 작업에 사용될 프롬프트 템플릿.
logging: 로그 레벨, 파일명 형식 등.
testing: 테스트 케이스 생성 및 실행 관련 설정.

## 10. 로깅
챗봇 서버 실행 중 발생하는 주요 이벤트 및 OpenAI API 호출/응답 내역은 logs/ 디렉토리에 동적 파일명 ({설정된 기본 이름}_{YYYYMMDD_HHMM}.txt 형식, 예: api_history_20250520_1030.txt)으로 기록됩니다. 로그 상세 수준은 config.yaml 파일의 logging.log_level 설정으로 조절할 수 있으며, 현재 연구 데이터 확보를 위해 DEBUG 레벨로 고정되어 운영될 수 있습니다.