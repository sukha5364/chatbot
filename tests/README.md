# tests/README.md

## 데카트론 챗봇 테스트 가이드

이 디렉토리에는 챗봇의 성능을 테스트하기 위한 스크립트와 데이터가 포함되어 있습니다.

### 구성 요소

* **`test_cases/`**: 테스트 질문 세트가 저장되는 디렉토리 (`.jsonl` 형식).
    * `test_generator.py`를 실행하여 생성합니다.
* **`test_results/`**: 테스트 실행 결과가 저장되는 디렉토리 (`.jsonl` 형식).
    * `test_runner.py` 실행 시 자동으로 생성됩니다.
* **`test_generator.py`**: 템플릿과 키워드를 조합하여 `test_cases/`에 테스트 질문 세트를 자동으로 생성하는 스크립트입니다.
* **`test_runner.py`**: `test_cases/`의 질문들을 읽어 실행 중인 챗봇 서버 API (`http://127.0.0.1:8000/chat`)에 요청을 보내고, 응답과 처리 시간 등의 결과를 `test_results/`에 저장하는 스크립트입니다.

### 사용 방법

1.  **테스트셋 생성 (필요시):**
    * 새로운 테스트 질문 세트를 생성하려면 프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.
        ```bash
        python tests/test_generator.py
        ```
    * 실행 결과로 `tests/test_cases/` 디렉토리에 `test_set_1.jsonl`, `test_set_2.jsonl` 등의 파일이 생성됩니다.

2.  **챗봇 서버 실행:**
    * 테스트를 실행하기 전에, 메인 챗봇 서버가 실행 중이어야 합니다.
        ```bash
        uvicorn chatbot.app:app --reload
        ```

3.  **테스트 실행:**
    * 챗봇 서버가 실행된 상태에서, 프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 테스트를 시작합니다.
        ```bash
        python tests/test_runner.py
        ```
    * 스크립트는 `tests/test_cases/`의 모든 `.jsonl` 파일을 읽어 각 질문을 챗봇 API로 보냅니다.
    * 테스트 완료 후, 결과는 `tests/test_results/` 디렉토리에 타임스탬프가 포함된 `.jsonl` 파일로 저장됩니다.
    * 콘솔에 테스트 요약 정보(성공/실패 수, 평균 응답 시간 등)가 출력됩니다.

### 참고

* `test_runner.py`는 현재 응답 내용과 처리 시간만 기록합니다. 응답의 정확성이나 품질에 대한 자동 평가는 포함되어 있지 않으므로, 저장된 결과 파일(`test_results/*.jsonl`)을 검토하여 수동으로 평가하거나 별도의 평가 스크립트를 구현해야 합니다.
* 테스트 실행 전 `requirements.txt`에 명시된 라이브러리 (`requests` 등)가 설치되어 있는지 확인하세요.