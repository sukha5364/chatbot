# chatbot/gpt_interface.py (env 경로 수정 및 time 임포트 추가됨)
# COMPLETE FILE - Using Custom Text Formatter for human-readable file logs.

import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp
from dotenv import load_dotenv
import traceback
import time # time 모듈 임포트

# 설정 로더 임포트
try:
    from .config_loader import get_config
    config = get_config()
except ImportError as ie:
    print(f"CRITICAL ERROR (gpt_interface): Failed to import config_loader: {ie}.")
    config = {}
except Exception as config_e:
    print(f"CRITICAL ERROR (gpt_interface): Failed to load configuration: {config_e}")
    config = {}

# --- .env 파일 로드 [수정됨: 경로 계산 방식 수정] ---
try:
    # 현재 파일(gpt_interface.py)의 디렉토리 -> chatbot/chatbot/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # chatbot/chatbot/의 부모 디렉토리 -> chatbot/ (이것이 프로젝트 루트)
    project_root = os.path.dirname(current_dir) # *** 여기가 수정된 부분 ***

    # 프로젝트 루트 디렉토리 안에 있는 .env 파일 경로 계산
    dotenv_path = os.path.join(project_root, '.env')
    print(f"INFO (gpt_interface): Attempting to load .env from project root: {dotenv_path}") # 계산된 경로 로깅

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"INFO (gpt_interface): Successfully loaded .env file from: {dotenv_path}")
    else:
        print(f"INFO (gpt_interface): .env file not found at project root ({dotenv_path}), relying on environment variables.")
except Exception as env_e:
    print(f"ERROR (gpt_interface): Error loading .env: {env_e}")


# --- 환경 변수 로드 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("CRITICAL ERROR (gpt_interface): OPENAI_API_KEY was not found after checking .env and environment variables.")
else:
    # 키 값 자체는 로깅하지 않음
    print("INFO (gpt_interface): OPENAI_API_KEY seems to be loaded (found a value).")


# --- Custom Text Formatter Definition ---
class ReadableTextFormatter(logging.Formatter):
    _basic_fmt_str = "[%(asctime)s] [%(levelname)s] - %(message)s"

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None):
        # ValueError 수정: datefmt 유효성 검사 또는 기본값 사용 강화
        safe_datefmt = datefmt if datefmt else logging.Formatter.default_time_format # None이면 기본 포맷 사용
        super().__init__(fmt or self._basic_fmt_str, safe_datefmt, style, validate, defaults=defaults)
        # self.default_msec_format = logging.Formatter.default_msec_format # msec 포맷은 formatTime 내에서 처리
        self.converter = time.localtime # 로컬 시간대 사용 (time.gmtime 대신)

    def formatTime(self, record, datefmt=None):
        """record.created를 사용하여 시간을 포맷팅 (ValueError 방지)"""
        ct = self.converter(record.created)
        _datefmt = datefmt or self.default_time_format # None이면 기본 포맷 사용
        if _datefmt:
            try:
                s = time.strftime(_datefmt, ct)
            except ValueError as e:
                # 잘못된 포맷 문자열일 경우 경고 로깅 및 기본 포맷 사용
                print(f"WARNING (ReadableTextFormatter): Invalid date format string '{_datefmt}'. Falling back to default. Error: {e}")
                s = time.strftime(logging.Formatter.default_time_format, ct) # 기본 ISO 포맷 사용
        else:
            # datefmt가 명시적으로 None 또는 빈 문자열인 경우 (가능성은 낮음)
            t = time.strftime(logging.Formatter.default_time_format, ct) # 안전하게 기본 포맷 사용
            s = t # 밀리초 제외

        # 밀리초 추가 (선택 사항, 필요 시)
        # if self.default_msec_format:
        #     try:
        #         s = self.default_msec_format % (s, record.msecs)
        #     except TypeError: # 포맷 문자열에 %가 없는 경우 등
        #         pass # 밀리초 추가 실패 시 그냥 넘어감

        return s

    def format(self, record: logging.LogRecord) -> str:
        # formatTime 호출 시 self.datefmt 전달 (수정된 formatTime에서 처리)
        record.asctime = self.formatTime(record, self.datefmt)
        log_string = f"[{record.asctime}] [{record.levelname}] - {record.getMessage()}"
        if hasattr(record, 'event_type') and record.event_type in ["api_call", "api_response", "api_call_error", "api_response_error"]:
            lines = ["=" * 70, log_string, "-" * 70]
            lines.append(f"Event Type : {getattr(record, 'event_type', 'N/A')}")
            lines.append(f"Direction  : {getattr(record, 'direction', 'N/A')}")
            lines.append(f"API Type   : {getattr(record, 'api_type', 'N/A')}")
            lines.append(f"Model      : {getattr(record, 'model', 'N/A')}")
            lines.append(f"Request TS : {getattr(record, 'request_timestamp', 'N/A')}")
            if hasattr(record, 'response_timestamp'): lines.append(f"Response TS: {record.response_timestamp}")
            if hasattr(record, 'status_code'): lines.append(f"Status Code: {record.status_code}")
            if hasattr(record, 'payload_info') and isinstance(record.payload_info, dict):
                lines.append("\n--- Payload Info ---")
                lines.append(f"  Messages Count: {record.payload_info.get('messages_count', 'N/A')}")
                if 'temperature' in record.payload_info: lines.append(f"  Temperature   : {record.payload_info['temperature']}")
                if 'max_tokens' in record.payload_info: lines.append(f"  Max Tokens    : {record.payload_info['max_tokens']}")
                if 'input_length' in record.payload_info: lines.append(f"  Input Length  : {record.payload_info['input_length']}")
                if 'messages_preview' in record.payload_info: lines.append(f"  Messages Preview: {record.payload_info['messages_preview']}")
                if 'additional_params' in record.payload_info and record.payload_info['additional_params']:
                    lines.append(f"  Additional Params: {record.payload_info['additional_params']}")
                if 'messages_formatted' in record.payload_info:
                    lines.append("\n--- Formatted Prompt (DEBUG Level Only) ---")
                    lines.extend(record.payload_info['messages_formatted'].splitlines())
            if hasattr(record, 'response_info') and isinstance(record.response_info, dict):
                lines.append("\n--- Response Info ---")
                if 'id' in record.response_info: lines.append(f"  ID                : {record.response_info['id']}")
                if 'choices_count' in record.response_info: lines.append(f"  Choices Count     : {record.response_info['choices_count']}")
                if 'embeddings_count' in record.response_info: lines.append(f"  Embeddings Count  : {record.response_info['embeddings_count']}")
                usage = record.response_info.get('usage')
                if isinstance(usage, dict):
                    lines.append(f"  Usage (P/C/T)     : {usage.get('prompt_tokens','?')} / {usage.get('completion_tokens','?')} / {usage.get('total_tokens','?')}")
                if 'content_preview' in record.response_info: lines.append(f"  Content Preview   : {record.response_info['content_preview']}")
            if hasattr(record, 'error_details') and record.error_details:
                lines.append("\n--- Error Details ---")
                error_data = record.error_details
                if isinstance(error_data, dict):
                    lines.append(f"  Type   : {error_data.get('type')}")
                    lines.append(f"  Code   : {error_data.get('code')}")
                    lines.append(f"  Param  : {error_data.get('param')}")
                    lines.append(f"  Message: {error_data.get('message')}")
                else: lines.append(f"  Details: {error_data}")
            if hasattr(record, 'error_message') and not hasattr(record, 'error_details'):
                lines.append("\n--- Error Message ---")
                lines.append(f"  {record.error_message}")
            failed_prompt_data = getattr(record, 'failed_prompt_messages', None)
            failed_input_text = getattr(record, 'failed_input_text', None)
            if failed_prompt_data:
                lines.append("\n--- Failed Prompt ---")
                try:
                    formatted_fail_lines = []
                    for msg in failed_prompt_data:
                        role = msg.get('role', 'unknown').upper(); content = msg.get('content', '')
                        indented_content = "\n".join(["  " + line for line in content.split('\n')])
                        formatted_fail_lines.append(f"[{role}]:\n{indented_content}")
                    lines.append("\n".join(formatted_fail_lines))
                except Exception: lines.append(f"  (Error formatting failed prompt, raw: {failed_prompt_data})")
            elif failed_input_text:
                lines.append("\n--- Failed Input Text ---"); lines.append(f"  {failed_input_text}")
            lines.append("=" * 70 + "\n"); log_string = "\n".join(lines)
        else:
            log_string = f"[{record.asctime}] [{record.levelname}] [{record.name}] - {record.getMessage()}"
            if record.exc_info: log_string += "\n" + self.formatException(record.exc_info)
        return log_string

# --- 로깅 설정 ---
# 로거 설정은 이전과 동일하게 유지 (config.yaml 값 사용)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
bootstrap_logger = logging.getLogger('gpt_interface_setup')
try:
    if not config or not isinstance(config, dict): raise ValueError("Configuration is not loaded or invalid.")
    logger_config = config.get('logging', {});
    if not logger_config: raise ValueError("'logging' section not found in configuration.")
    config_log_level_str = logger_config.get('log_level', 'INFO').upper()
    config_log_level_int = getattr(logging, config_log_level_str, logging.INFO)
    log_file_base_name = logger_config.get('log_file_base_name', 'api_history.txt')
    log_backup_count = logger_config.get('log_backup_count', 30)
    log_date_format_suffix = logger_config.get('log_date_format', '%Y%m%d') # 파일명 날짜 형식
    log_timestamp_format = logger_config.get('log_timestamp_format', '%Y-%m-%d %H:%M:%S,%f') # 로그 내 시간 형식

    bootstrap_logger.info(f"Read logging config: Level='{config_log_level_str}'({config_log_level_int}), File='{log_file_base_name}', BackupCount={log_backup_count}")
    # 로그 디렉토리 경로 수정 (gpt_interface.py 기준 상위 -> 상위)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_base = os.path.join(log_dir, log_file_base_name)
    bootstrap_logger.info(f"Log directory set to: {log_dir}")
    api_logger = logging.getLogger('api_logger')
    api_logger.setLevel(config_log_level_int); api_logger.propagate = False
    if not api_logger.handlers:
        bootstrap_logger.info(f"Configuring 'api_logger' (Name: {api_logger.name}) with level {logging.getLevelName(api_logger.level)} ({api_logger.level}). Adding handlers.")
        # 수정된 포맷터 사용 (로그 내 타임스탬프 형식 전달)
        text_formatter = ReadableTextFormatter(datefmt=log_timestamp_format) # log_timestamp_format 사용
        try:
            file_handler = logging.handlers.TimedRotatingFileHandler(filename=log_file_base, when='midnight', interval=1,backupCount=log_backup_count, encoding='utf-8')
            file_handler.suffix = log_date_format_suffix; file_handler.setFormatter(text_formatter); file_handler.setLevel(config_log_level_int)
            api_logger.addHandler(file_handler)
            bootstrap_logger.info(f"Added Custom Text File Handler: Path='{log_file_base}', Level={logging.getLevelName(file_handler.level)}")
        except Exception as fh_e: bootstrap_logger.error(f"Failed to create or add file handler: {fh_e}", exc_info=True)

        # 콘솔 핸들러는 기본 포맷터 사용 가능 (또는 동일 포맷터 사용)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
        try:
            console_handler = logging.StreamHandler(); console_handler.setFormatter(console_formatter); console_handler.setLevel(config_log_level_int)
            api_logger.addHandler(console_handler)
            bootstrap_logger.info(f"Added Standard Console Handler: Level={logging.getLevelName(console_handler.level)}")
        except Exception as ch_e: bootstrap_logger.error(f"Failed to create or add console handler: {ch_e}", exc_info=True)

        if api_logger.hasHandlers(): api_logger.info(f"Logger '{api_logger.name}' initialized. Effective Level: {logging.getLevelName(api_logger.getEffectiveLevel())}.")
        else: bootstrap_logger.error("No handlers could be added to api_logger.")
    else: bootstrap_logger.info(f"'api_logger' already has handlers. Skipping setup.")
except Exception as setup_e:
    bootstrap_logger.critical(f"CRITICAL ERROR during logger setup: {setup_e}", exc_info=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    api_logger = logging.getLogger('api_logger_fallback')
    api_logger.error("Using basicConfig fallback logger due to setup error.")


# --- 비동기 OpenAI ChatCompletion API 호출 함수 ---
async def call_gpt_async(
    messages: list[Dict[str, str]], model: str, temperature: float, max_tokens: int,
    session: Optional[aiohttp.ClientSession] = None, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    # 함수 내용은 이전 답변과 동일 (API 키 확인 로직 포함)
    if not model or not isinstance(temperature, (int, float)) or not isinstance(max_tokens, int) or max_tokens <= 0:
        api_logger.error(f"Invalid API parameters provided: model='{model}', temp={temperature}, max_tokens={max_tokens}")
        return None
    if not OPENAI_API_KEY:
        # API 키가 없을 때 즉시 에러 로깅 및 None 반환 (중요)
        api_logger.error("OpenAI API Key is missing, cannot make API call.")
        return None

    try:
        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}","Content-Type": "application/json",}
        payload = {"model": model,"messages": messages,"temperature": temperature,"max_tokens": max_tokens,**kwargs}
    except Exception as param_e:
        api_logger.error(f"Error setting up API parameters: {param_e}", exc_info=True, extra={"event_type":"param_error"}); return None

    request_timestamp = datetime.now().isoformat()
    log_data_request = {"event_type": "api_call", "direction": "request", "api_type": "chat_completion","model": model,
        "payload_info": {"messages_count": len(payload['messages']),"temperature": temperature,"max_tokens": max_tokens,
             "messages_preview": f"System: {payload['messages'][0].get('content', '')[:50]}... User: {payload['messages'][-1].get('content', '')[:50]}..." if len(payload['messages']) > 0 else "No messages",
             "additional_params": kwargs},"request_timestamp": request_timestamp}

    if api_logger.getEffectiveLevel() <= logging.DEBUG:
        try:
            formatted_prompt_lines = [f"--- Prompt Messages ({len(payload['messages'])}) ---"]
            for msg in payload['messages']:
                role = msg.get('role', 'unknown').upper(); content = msg.get('content', '')
                indented_content = "\n".join(["  " + line for line in content.split('\n')])
                formatted_prompt_lines.append(f"[{role}]:\n{indented_content}")
            formatted_prompt_lines.append("--- End Prompt ---"); formatted_prompt_string = "\n".join(formatted_prompt_lines)
            log_data_request["payload_info"]["messages_formatted"] = formatted_prompt_string
        except Exception as fmt_e:
            api_logger.warning(f"Error formatting prompt messages: {fmt_e}", extra={"event_type":"logging_format_error"})
            log_data_request["payload_info"]["messages_formatted"] = "Error formatting prompt"
        log_message = f"Sending ChatCompletion request to {model} (DEBUG level)"
    else: log_message = f"Sending ChatCompletion request to {model}"

    api_logger.info(log_message, extra=log_data_request)

    close_session = False
    if session is None:
        api_logger.warning("aiohttp session not provided, creating a new one.", extra={"event_type": "session_warning"})
        try: session = aiohttp.ClientSession(); close_session = True
        except Exception as session_e: api_logger.error(f"Failed to create new aiohttp session: {session_e}", exc_info=True, extra={"event_type":"session_error"}); return None
    elif session.closed: api_logger.error("Provided aiohttp session is closed."); return None

    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(); response_status = response.status; response_data = {}; response_text_content = await response.text()
            try: response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + '...'
                log_data_error = {"event_type": "api_response_error", "direction": "response", "api_type": "chat_completion", "error_type": "decode_error", "model": model, "status_code": response_status, "error_message": str(decode_e),"response_text_preview": response_text_preview, "request_timestamp": request_timestamp, "response_timestamp": response_timestamp,"failed_prompt_messages": payload['messages']}
                api_logger.error("Failed to decode JSON or unexpected content type", extra=log_data_error)
                if close_session and session and not session.closed: await session.close()
                return None

            log_data_response = {"event_type": "api_response", "direction": "response", "api_type": "chat_completion", "model": model, "status_code": response_status,
                "response_info": {"id": response_data.get("id"), "choices_count": len(response_data.get("choices", [])),"usage": response_data.get("usage"),"content_preview": response_data.get("choices", [{}])[0].get("message", {}).get("content", "")[:80] + "..." if response_data.get("choices") else None},
                "request_timestamp": request_timestamp, "response_timestamp": response_timestamp}

            if response.status == 200:
                usage = response_data.get("usage", {}); total_tokens = usage.get('total_tokens', 'N/A')
                log_message_resp = f"ChatCompletion success. Tokens: {total_tokens}"
                api_logger.info(log_message_resp, extra=log_data_response); return response_data
            else:
                if response.status == 401: log_message_resp = f"ChatCompletion API Error (HTTP 401 Unauthorized). Check your API key."
                elif response.status == 429: log_message_resp = f"ChatCompletion API Error (HTTP 429 Rate Limit Exceeded)."
                else: log_message_resp = f"ChatCompletion API Error (HTTP {response.status})"
                log_data_response["error_details"] = response_data.get("error"); log_data_response["failed_prompt_messages"] = payload['messages']
                api_logger.error(log_message_resp, extra=log_data_response); return None # 실패 시 None 반환

    except aiohttp.ClientError as e:
        log_data_exception = {"event_type": "api_call_error", "api_type": "chat_completion", "error_type": "network_error","model": model, "error_message": str(e), "request_timestamp": request_timestamp,"failed_prompt_messages": payload['messages']}
        api_logger.error(f"Network Error: {e}", extra=log_data_exception); return None
    except Exception as e:
        log_data_exception = {"event_type": "api_call_error", "api_type": "chat_completion", "error_type": "unexpected_error", "model": model, "error_message": str(e), "request_timestamp": request_timestamp,"failed_prompt_messages": payload['messages']}
        api_logger.error(f"Unexpected Error: {e}", extra=log_data_exception, exc_info=True); return None
    finally:
        if close_session and session and not session.closed: await session.close()


# --- 비동기 OpenAI Embedding API 호출 함수 ---
async def get_openai_embedding_async(
    text: str, session: Optional[aiohttp.ClientSession] = None, model: str = None
) -> Optional[List[float]]:
    # 함수 내용은 이전 답변과 동일 (API 키 확인 로직 포함)
    if not OPENAI_API_KEY:
        api_logger.error("OpenAI API Key is missing, cannot make Embedding API call.")
        return None

    try:
        if model is None:
            local_config = get_config()
            embedding_model = local_config.get('rag', {}).get('embedding_model')
        else: embedding_model = model

        if not embedding_model:
            log_data_error = { "event_type": "config_error", "error_message": "Embedding model name not configured or provided." }
            api_logger.error("Embedding model name missing", extra=log_data_error); return None

        openai_url = "https://api.openai.com/v1/embeddings"; headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}; payload = {"input": text, "model": embedding_model}
    except Exception as param_e: api_logger.error(f"Error setting up Embedding API parameters: {param_e}", exc_info=True, extra={"event_type":"param_error"}); return None

    request_timestamp = datetime.now().isoformat()
    log_data_request = {"event_type": "api_call", "direction": "request", "api_type": "embedding","model": embedding_model,"payload_info": { "input_length": len(payload['input']) },"request_timestamp": request_timestamp}
    api_logger.info(f"Sending Embedding request to {embedding_model}", extra=log_data_request)

    close_session = False
    if session is None:
        api_logger.warning("aiohttp session not provided for embedding, creating a new one.", extra={"event_type": "session_warning"})
        try: session = aiohttp.ClientSession(); close_session = True
        except Exception as session_e: api_logger.error(f"Failed to create new aiohttp session for embedding: {session_e}", exc_info=True, extra={"event_type":"session_error"}); return None
    elif session.closed: api_logger.error("Provided aiohttp session for embedding is closed."); return None

    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(); response_status = response.status; response_data = {}; response_text_content = await response.text()
            try: response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + '...'; log_data_error = {"event_type": "api_response_error", "direction": "response", "api_type": "embedding", "error_type": "decode_error", "model": embedding_model, "status_code": response_status, "error_message": str(decode_e), "response_text_preview": response_text_preview, "request_timestamp": request_timestamp, "response_timestamp": response_timestamp, "failed_input_text": payload['input']}
                api_logger.error("Failed to decode JSON for embedding response", extra=log_data_error);
                if close_session and session and not session.closed: await session.close()
                return None

            log_data_response = {"event_type": "api_response", "direction": "response", "api_type": "embedding", "model": embedding_model, "status_code": response_status,"response_info": {"embeddings_count": len(response_data.get("data", [])), "usage": response_data.get("usage")},"request_timestamp": request_timestamp, "response_timestamp": response_timestamp}

            if response.status == 200 and response_data.get("data"):
                usage = response_data.get("usage", {}); total_tokens = usage.get('total_tokens', 'N/A'); log_message_emb = f"Embedding success. Tokens: {total_tokens}"; api_logger.info(log_message_emb, extra=log_data_response)
                embedding_vector = response_data["data"][0].get("embedding")
                if embedding_vector and isinstance(embedding_vector, list): return embedding_vector
                else: log_data_response["error_details"] = "Embedding vector not found or invalid format"; log_data_response["failed_input_text"] = payload['input']; api_logger.error("Embedding vector error", extra=log_data_response); return None
            else:
                if response.status == 401: log_message_emb = f"Embedding API Error (HTTP 401 Unauthorized). Check your API key."
                elif response.status == 429: log_message_emb = f"Embedding API Error (HTTP 429 Rate Limit Exceeded)."
                else: log_message_emb = f"Embedding API Error (HTTP {response.status})"
                log_data_response["error_details"] = response_data.get("error"); log_data_response["failed_input_text"] = payload['input']; api_logger.error(log_message_emb, extra=log_data_response); return None

    except aiohttp.ClientError as e:
        log_data_exception = {"event_type": "api_call_error", "api_type": "embedding", "error_type": "network_error", "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp, "failed_input_text": payload['input']}
        api_logger.error(f"Network Error during Embedding call: {e}", extra=log_data_exception); return None
    except Exception as e:
        log_data_exception = {"event_type": "api_call_error", "api_type": "embedding", "error_type": "unexpected_error", "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp, "failed_input_text": payload['input']}
        api_logger.error(f"Unexpected Error during Embedding call: {e}", extra=log_data_exception, exc_info=True); return None
    finally:
        if close_session and session and not session.closed: await session.close()


# --- 예시 사용법 (테스트용) ---
if __name__ == "__main__":
    import asyncio
    # import time # 맨 위에 이미 임포트됨

    print("--- Running gpt_interface.py as main script for testing ---")
    # 로거 레벨 강제 DEBUG 설정
    if 'api_logger' in locals() and isinstance(api_logger, logging.Logger):
        api_logger.setLevel(logging.DEBUG);
        for handler in api_logger.handlers: handler.setLevel(logging.DEBUG)
        print(f"api_logger effective level forced to: {logging.getLevelName(api_logger.getEffectiveLevel())}")
    else: print("WARNING: api_logger not properly configured."); logging.basicConfig(level=logging.DEBUG)

    async def test_apis():
        print("Running test_apis() function...")
        try: test_config = get_config(); print("Config loaded successfully within test_apis.")
        except Exception as e: print(f"Config load failed in test_apis: {e}"); return

        # API 키 존재 여부 확인 후 테스트 진행
        if not OPENAI_API_KEY:
            print("CRITICAL ERROR: OPENAI_API_KEY is missing. Cannot run API call tests.")
            return

        # Chat Completion 테스트
        test_messages = [{"role": "user", "content": "Say 'Hello Test!'"}]
        print("\n--- Testing Chat Completion API ---")
        try:
            test_model = test_config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
            test_temp = test_config.get('generation', {}).get('final_response_temperature', 0.7)
            test_max_tokens = test_config.get('generation', {}).get('final_response_max_tokens', 50)
            if not all([test_model, isinstance(test_temp, (int,float)), isinstance(test_max_tokens, int)]): raise ValueError("Test parameters missing in config.")
        except Exception as e: print(f"Error reading test parameters from config: {e}"); return

        async with aiohttp.ClientSession() as session:
            print(f"Calling chat API with model={test_model}, temp={test_temp}, max_tokens={test_max_tokens}")
            response = await call_gpt_async(messages=test_messages,model=test_model,temperature=test_temp,max_tokens=test_max_tokens,session=session)
            print(f"Chat API Call Result: {'Success' if response else 'Failed'}")
            if response: print(f"Response Content Preview: {response.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")

        # Embedding API 테스트
        test_text = "Test embedding input text."; print("\n--- Testing Embedding API ---")
        async with aiohttp.ClientSession() as session:
            emb_model_from_config = test_config.get('rag', {}).get('embedding_model')
            print(f"Calling embedding API with model={emb_model_from_config or 'Default from config'}")
            embedding = await get_openai_embedding_async(test_text, session=session, model=emb_model_from_config)
            print(f"Embedding API Call Result: {'Success (Dim: ' + str(len(embedding)) + ')' if embedding else 'Failed'}")

    try: asyncio.run(test_apis())
    except Exception as e: print(f"\nAn error occurred during test execution: {e}", exc_info=True)
    try: # 로그 파일 경로 확인
        final_config = get_config()
        log_file_final_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', final_config.get('logging', {}).get('log_file_base_name', 'api_history.txt'))
        print(f"\nCheck log file at: {log_file_final_path}")
    except Exception as e: print(f"Error getting log file path from config: {e}")
    print("Log formatter issues should be resolved.")