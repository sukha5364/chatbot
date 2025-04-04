# chatbot/gpt_interface.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트에 .env 파일이 있다고 가정)
load_dotenv()

# --- 환경 변수 로드 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")

# --- 로깅 설정 ---
# 로그 파일 경로 설정 (project/logs/)
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'api_call_history.txt')
# logs 디렉토리가 없으면 생성
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# 로거 설정
# 기본 로거 대신 파일 핸들러 사용
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

api_logger = logging.getLogger('api_logger')
api_logger.setLevel(logging.INFO)
# 콘솔 핸들러 추가 (선택 사항: 콘솔에도 로그 출력)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(formatter)
# api_logger.addHandler(console_handler)
if not api_logger.handlers: # 핸들러 중복 추가 방지
    api_logger.addHandler(file_handler)


# --- 비동기 OpenAI API 호출 함수 ---
async def call_gpt_async(
    messages: list[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    OpenAI ChatCompletion API를 비동기적으로 호출하고 결과를 로깅합니다.

    Args:
        messages (list): OpenAI API 형식의 메시지 리스트 [{'role': 'user', 'content': '...'}, ...]
        model (str): 사용할 GPT 모델 이름 (기본값: "gpt-3.5-turbo")
        temperature (float): 샘플링 온도 (기본값: 0.7)
        max_tokens (int): 최대 생성 토큰 수 (기본값: 1000)
        session (aiohttp.ClientSession, optional): 재사용할 aiohttp 세션. 없으면 새로 생성.

    Returns:
        Optional[Dict[str, Any]]: OpenAI API 응답 JSON 객체. 실패 시 None.
    """
    openai_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    request_timestamp = datetime.now().isoformat()
    log_entry_request = {
        "timestamp": request_timestamp,
        "direction": "request",
        "model": model,
        "payload": payload,
    }
    api_logger.info(json.dumps(log_entry_request, ensure_ascii=False))

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_data = await response.json()
            response_timestamp = datetime.now().isoformat()

            log_entry_response = {
                "timestamp": response_timestamp,
                "direction": "response",
                "status_code": response.status,
                "data": response_data,
                "request_timestamp": request_timestamp, # 요청 타임스탬프 연결
            }
            api_logger.info(json.dumps(log_entry_response, ensure_ascii=False))

            if response.status == 200:
                return response_data
            else:
                api_logger.error(f"API Error: Status Code {response.status}, Response: {response_data}")
                return None
    except aiohttp.ClientError as e:
        error_timestamp = datetime.now().isoformat()
        api_logger.error(f"Network Error at {error_timestamp}: {e}")
        return None
    except Exception as e:
        error_timestamp = datetime.now().isoformat()
        api_logger.error(f"Unexpected Error at {error_timestamp}: {e}")
        return None
    finally:
        if close_session and session:
            await session.close()


# --- 동기 OpenAI API 호출 함수 (필요시 사용) ---
# import requests # requests 라이브러리 필요

# def call_gpt_sync(
#     messages: list[Dict[str, str]],
#     model: str = "gpt-3.5-turbo",
#     temperature: float = 0.7,
#     max_tokens: int = 1000
# ) -> Optional[Dict[str, Any]]:
#     """
#     OpenAI ChatCompletion API를 동기적으로 호출하고 결과를 로깅합니다. (requests 사용)
#     """
#     # ... (requests 라이브러리를 사용한 동기 호출 로직 구현) ...
#     # 로깅 로직은 비동기 함수와 유사하게 적용
#     pass


# --- 예시 사용법 (테스트용) ---
if __name__ == "__main__":
    import asyncio

    async def test_api_call():
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What's the weather like today?"}
        ]
        print("Calling OpenAI API asynchronously...")
        # aiohttp 세션 관리 예시
        async with aiohttp.ClientSession() as session:
            response = await call_gpt_async(test_messages, session=session)
            if response:
                print("API Call Successful:")
                print(json.dumps(response, indent=2, ensure_ascii=False))
                # 실제 응답 내용 추출
                if response.get("choices") and len(response["choices"]) > 0:
                     print("\nAssistant's Message:")
                     print(response["choices"][0].get("message", {}).get("content"))
            else:
                print("API Call Failed.")

    asyncio.run(test_api_call())

    # 로그 파일 확인: project/logs/api_call_history.txt
    print(f"\nCheck log file at: {log_file_path}")