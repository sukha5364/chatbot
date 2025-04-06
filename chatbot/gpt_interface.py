# chatbot/gpt_interface.py

import os
import json
import logging
import logging.handlers # TimedRotatingFileHandler 사용 위해 추가
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp
from dotenv import load_dotenv

# 설정 로더 임포트
from .config_loader import get_config

# 설정 로드
config = get_config()

# .env 파일 로드 (프로젝트 루트에 .env 파일이 있다고 가정)
load_dotenv()

# --- 환경 변수 로드 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # API 키가 없을 경우 로깅 후 예외 발생
    logging.critical("OPENAI_API_KEY environment variable not set.") # 로거 생성 전이므로 기본 로거 사용될 수 있음
    raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")

# --- 로깅 설정 수정 ---
logger_config = config.get('logging', {})
log_level_str = logger_config.get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
log_date_format = logger_config.get('log_date_format', '%Y%m%d') # config에서 포맷 읽기
log_file_base_name = logger_config.get('log_file_base_name', 'api_history.txt')
log_backup_count = logger_config.get('log_backup_count', 30)

log_file_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', log_file_base_name)
os.makedirs(os.path.dirname(log_file_base), exist_ok=True)

# 로거 가져오기
api_logger = logging.getLogger('api_logger')
api_logger.setLevel(log_level)

# 포매터 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 핸들러 중복 추가 방지
if not api_logger.handlers:
    try:
        # 1. 날짜별 파일 로테이션 핸들러 설정 (suffix 생성자 인자 제거, 속성으로 설정)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file_base,
            when='midnight',
            interval=1,
            backupCount=log_backup_count,
            encoding='utf-8'
            # suffix 인자 제거됨
        )
        # 생성 후 suffix 속성 설정
        file_handler.suffix = log_date_format # 예: "%Y%m%d" -> .20250406 형태로 붙음

        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        api_logger.addHandler(file_handler)

        # 2. 콘솔 출력 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        api_logger.addHandler(console_handler)

        api_logger.info("API Logger initialized with File and Console handlers.")
    except Exception as e:
        # 핸들러 설정 중 오류 발생 시 기본 로깅으로 대체 또는 오류 출력
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        api_logger = logging.getLogger('api_logger_fallback')
        api_logger.error(f"Failed to initialize custom logging handlers: {e}. Using basicConfig.", exc_info=True)


# --- 비동기 OpenAI ChatCompletion API 호출 함수 ---
async def call_gpt_async(
    messages: list[Dict[str, str]],
    model: str = None, # 기본값 None으로 변경
    temperature: float = None, # 기본값 None으로 변경
    max_tokens: int = None, # 기본값 None으로 변경
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    OpenAI ChatCompletion API를 비동기적으로 호출하고 결과를 로깅합니다.
    model, temperature, max_tokens 기본값은 config에서 가져옵니다.
    """
    # config에서 기본값 가져오기
    gen_config = config.get('generation', {})
    router_map = config.get('model_router', {}).get('routing_map', {})
    # 모델 기본값은 easy 모델로 설정
    effective_model = model if model is not None else router_map.get('easy', 'gpt-3.5-turbo')
    effective_temperature = temperature if temperature is not None else gen_config.get('default_temperature', 0.7)
    effective_max_tokens = max_tokens if max_tokens is not None else gen_config.get('default_max_tokens', 500)

    openai_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": effective_model,
        "messages": messages,
        "temperature": effective_temperature,
        "max_tokens": effective_max_tokens,
    }

    request_timestamp = datetime.now().isoformat()
    log_entry_request = {
        "timestamp": request_timestamp, "direction": "request", "type": "chat_completion",
        "model": effective_model,
        "payload_preview": {
             "messages_summary": f"{len(messages)} messages (first: {messages[0]['role']}, last: {messages[-1]['role']})",
             "temperature": payload['temperature'], "max_tokens": payload['max_tokens']
        }
    }
    api_logger.info(f"Sending ChatCompletion request for model {effective_model}")
    api_logger.debug(json.dumps(log_entry_request, ensure_ascii=False, indent=2))

    close_session = False
    if session is None:
        api_logger.warning("aiohttp session not provided, creating a new one for this call.")
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            api_logger.info(f"Received ChatCompletion response status: {response.status} for model {effective_model}")
            response_data = {}
            try:
                 response_data = await response.json()
            except (json.JSONDecodeError, aiohttp.ContentTypeError):
                 response_text = await response.text()
                 api_logger.error(f"Failed to decode JSON or unexpected content type. Status: {response.status}, Response text: {response_text[:500]}...")
                 # 실패 시 None 반환 전에 세션 정리
                 if close_session and session and not session.closed: await session.close()
                 return None

            response_timestamp = datetime.now().isoformat()
            log_entry_response = {
                "timestamp": response_timestamp, "direction": "response", "type": "chat_completion",
                "status_code": response.status, "model": effective_model,
                "data_preview": {
                    "choices_count": len(response_data.get("choices", [])),
                    "usage": response_data.get("usage")
                },
                "request_timestamp": request_timestamp,
            }
            api_logger.debug(json.dumps(log_entry_response, ensure_ascii=False, indent=2))

            if response.status == 200:
                 usage = response_data.get("usage")
                 if usage:
                     api_logger.info(f"ChatCompletion success. Model: {effective_model}, Tokens: {usage.get('total_tokens')} (Prompt: {usage.get('prompt_tokens')}, Completion: {usage.get('completion_tokens')})")
                 else:
                     api_logger.info(f"ChatCompletion success. Model: {effective_model}, Usage info not available.")
                 return response_data
            else:
                 api_logger.error(f"ChatCompletion API Error. Status: {response.status}, Model: {effective_model}, Response: {response_data}")
                 return None
    except aiohttp.ClientError as e:
        api_logger.error(f"Network Error during ChatCompletion call: {e}")
        return None
    except Exception as e:
        api_logger.error(f"Unexpected Error during ChatCompletion call: {e}", exc_info=True)
        return None
    finally:
        if close_session and session and not session.closed:
            await session.close()


# --- [신규] 비동기 OpenAI Embedding API 호출 함수 ---
async def get_openai_embedding_async(
    text: str,
    session: Optional[aiohttp.ClientSession] = None,
    model: str = None
) -> Optional[List[float]]:
    """OpenAI Embedding API를 비동기적으로 호출하고 결과를 로깅합니다."""
    # config에서 임베딩 모델 가져오기
    embedding_model = model if model else config.get('rag', {}).get('embedding_model')
    if not embedding_model:
         api_logger.error("Embedding model name not configured or provided.")
         return None

    # ... (이하 Embedding 함수 로직은 이전 답변과 동일) ...
    openai_url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": text, "model": embedding_model}

    request_timestamp = datetime.now().isoformat()
    log_entry_request = {
        "timestamp": request_timestamp, "direction": "request", "type": "embedding",
        "model": embedding_model, "payload_preview": {"input_length": len(text)}
    }
    api_logger.info(f"Sending Embedding request for model {embedding_model}")
    api_logger.debug(json.dumps(log_entry_request, ensure_ascii=False, indent=2))

    close_session = False
    if session is None:
        api_logger.warning("aiohttp session not provided for embedding, creating a new one.")
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            api_logger.info(f"Received Embedding response status: {response.status} for model {embedding_model}")
            response_data = {}
            try:
                response_data = await response.json()
            except (json.JSONDecodeError, aiohttp.ContentTypeError):
                 response_text = await response.text()
                 api_logger.error(f"Failed to decode JSON or unexpected content type for embedding. Status: {response.status}, Response text: {response_text[:500]}...")
                 if close_session and session and not session.closed: await session.close()
                 return None

            response_timestamp = datetime.now().isoformat()
            log_entry_response = {
                "timestamp": response_timestamp, "direction": "response", "type": "embedding",
                "status_code": response.status, "model": embedding_model,
                "data_preview": {
                    "embeddings_count": len(response_data.get("data", [])),
                    "usage": response_data.get("usage")
                },
                "request_timestamp": request_timestamp,
            }
            api_logger.debug(json.dumps(log_entry_response, ensure_ascii=False, indent=2))

            if response.status == 200 and response_data.get("data"):
                 usage = response_data.get("usage")
                 if usage: api_logger.info(f"Embedding success. Model: {embedding_model}, Tokens: {usage.get('total_tokens')} (Prompt: {usage.get('prompt_tokens')})")
                 else: api_logger.info(f"Embedding success. Model: {embedding_model}, Usage info not available.")

                 embedding_vector = response_data["data"][0].get("embedding")
                 if embedding_vector and isinstance(embedding_vector, list):
                     return embedding_vector
                 else:
                     api_logger.error("Embedding vector not found or invalid format in response.")
                     return None
            else:
                 api_logger.error(f"Embedding API Error. Status: {response.status}, Model: {embedding_model}, Response: {response_data}")
                 return None
    except aiohttp.ClientError as e:
        api_logger.error(f"Network Error during Embedding call: {e}")
        return None
    except Exception as e:
        api_logger.error(f"Unexpected Error during Embedding call: {e}", exc_info=True)
        return None
    finally:
        if close_session and session and not session.closed:
            await session.close()


# --- 예시 사용법 (테스트용) ---
if __name__ == "__main__":
    import asyncio

    async def test_apis():
        # 로깅 기본 설정 (테스트용)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Running gpt_interface tests...")
        # config 로드 확인
        try:
            get_config()
        except Exception as e:
            logger.error(f"Config load failed in test: {e}")
            return

        # Chat Completion 테스트
        # ... (기존 테스트 코드 유지, config 값 사용) ...
        test_messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}]
        print("\n--- Testing Chat Completion API ---")
        api_logger.info("Starting Chat Completion Test")
        async with aiohttp.ClientSession() as session:
             response = await call_gpt_async(test_messages, session=session) # 모델 등은 config 기본값 사용
             if response: print("Chat API Call Successful.")
             else: print("Chat API Call Failed.")
        api_logger.info("Finished Chat Completion Test")

        # Embedding API 테스트
        # ... (기존 테스트 코드 유지) ...
        test_text = "데카트론 러닝화 추천"
        print("\n--- Testing Embedding API ---")
        api_logger.info("Starting Embedding Test")
        async with aiohttp.ClientSession() as session:
            embedding = await get_openai_embedding_async(test_text, session=session)
            if embedding: print(f"Embedding API Call Successful. Dim: {len(embedding)}")
            else: print("Embedding API Call Failed.")
        api_logger.info("Finished Embedding Test")

    # 비동기 함수 실행
    try:
        asyncio.run(test_apis())
    except FileNotFoundError:
         print("\nError: config.yaml not found.")
    except ValueError as e: # API 키 오류 등
         print(f"\nConfiguration Error: {e}")
    except Exception as e:
         print(f"\nAn error occurred: {e}")

    print(f"\nCheck log file at: {log_file_base} (and its rotated versions like {log_file_base}.{datetime.now().strftime(log_date_format)})")