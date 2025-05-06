# chatbot/gpt_interface.py (evaluate_satisfaction_async 함수 추가 최종 버전)

import os
import json # json 임포트 확인
import logging # logging 임포트 확인
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import aiohttp # aiohttp 임포트 확인
from dotenv import load_dotenv
import traceback
import time # Formatter 및 테스트에서 사용
import re # re 임포트 추가 (만족도 평가 파싱용)
import asyncio # 테스트용 asyncio 임포트

# --- 설정 로더 임포트 ---
try:
    # gpt_interface.py는 chatbot/chatbot/ 안에 있으므로, 프로젝트 루트는 상위의 상위
    current_dir_gpt = os.path.dirname(os.path.abspath(__file__))
    project_root_gpt = os.path.dirname(current_dir_gpt) # 프로젝트 루트 계산
    # chatbot 모듈 경로 추가 (get_config 접근 위함)
    import sys
    if project_root_gpt not in sys.path: # 중복 추가 방지
        sys.path.insert(0, project_root_gpt)
    from chatbot.config_loader import get_config
    config = get_config() # 설정 로드 시도
    logging.info("Configuration loaded successfully in gpt_interface.")
except ImportError as ie:
    logging.error(f"CRITICAL ERROR (gpt_interface): Failed to import config_loader: {ie}. Ensure correct path and file existence.", exc_info=True)
    config = {} # 빈 dict로 설정하여 이후 .get() 사용 시 오류 방지
except Exception as config_e:
    logging.error(f"CRITICAL ERROR (gpt_interface): Failed to load configuration: {config_e}", exc_info=True)
    config = {}

# --- .env 파일 로드 (프로젝트 루트 기준) ---
try:
    dotenv_path = os.path.join(project_root_gpt, '.env')
    logging.debug(f"Attempting to load .env file from project root: {dotenv_path}")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Successfully loaded environment variables from: {dotenv_path}")
    else:
        logging.info(f".env file not found at ({dotenv_path}), relying solely on system environment variables.")
except Exception as env_e:
    logging.error(f"Error loading .env file: {env_e}", exc_info=True)

# --- 환경 변수 로드 (OpenAI API 키) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("CRITICAL ERROR (gpt_interface): OPENAI_API_KEY was not found in .env file or environment variables. API calls will fail.")
else:
    # 키 값 자체는 로깅하지 않음
    logging.info("OpenAI API Key found in environment.")

# --- Custom Text Formatter (Tool Use 로깅 추가) ---
class ReadableTextFormatter(logging.Formatter):
    """
    API 호출 및 응답 로그를 사람이 읽기 쉬운 텍스트 형식으로 포맷합니다.
    DEBUG 레벨에서는 상세한 프롬프트/페이로드/Tool 정보 등을 포함합니다.
    """
    def __init__(self, fmt="[%(asctime)s] [%(levelname)s] - %(message)s", datefmt=None, style='%', validate=True, *, defaults=None):
        """
        Formatter 초기화. config에서 타임스탬프 형식 읽기.
        """
        safe_datefmt = datefmt or config.get('logging', {}).get('log_timestamp_format', '%Y-%m-%d %H:%M:%S')
        super().__init__(fmt, safe_datefmt, style, validate, defaults=defaults)
        self.converter = time.localtime # 로컬 시간대 사용

    def formatTime(self, record, datefmt=None):
        """로그 레코드의 생성 시간을 지정된 포맷으로 변환합니다."""
        ct = self.converter(record.created)
        _datefmt = datefmt or self.default_time_format
        if _datefmt:
            try:
                s = time.strftime(_datefmt, ct)
            except ValueError as e:
                logging.warning(f"WARNING (ReadableTextFormatter): Invalid date format string '{_datefmt}'. Falling back to default. Error: {e}")
                s = time.strftime(logging.Formatter.default_time_format, ct)
        else:
            s = time.strftime(logging.Formatter.default_time_format, ct)
        return s

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 포맷된 문자열로 변환합니다."""
        record.asctime = self.formatTime(record, self.datefmt) # 시간 포맷팅
        log_string = super().format(record) # 기본 Formatter 활용

        # API 관련 이벤트 상세 정보 추가
        if hasattr(record, 'event_type') and record.event_type.startswith("api_"):
            lines = ["=" * 80, log_string, "-" * 80]

            # 공통 정보
            lines.append(f"Event Type : {getattr(record, 'event_type', 'N/A')}")
            lines.append(f"Direction  : {getattr(record, 'direction', 'N/A')}")
            lines.append(f"API Type   : {getattr(record, 'api_type', 'N/A')}")
            lines.append(f"Model      : {getattr(record, 'model', 'N/A')}")
            lines.append(f"Request TS : {getattr(record, 'request_timestamp', 'N/A')}")
            if hasattr(record, 'response_timestamp'): lines.append(f"Response TS: {record.response_timestamp}")
            if hasattr(record, 'status_code'): lines.append(f"Status Code: {record.status_code}")

            # --- Request Payload / Input ---
            if hasattr(record, 'payload_info') and isinstance(record.payload_info, dict):
                lines.append("\n--- Request Payload Info ---")
                lines.append(f"  Messages Count: {record.payload_info.get('messages_count', 'N/A')}")
                if 'temperature' in record.payload_info: lines.append(f"  Temperature   : {record.payload_info['temperature']}")
                if 'max_tokens' in record.payload_info: lines.append(f"  Max Tokens    : {record.payload_info['max_tokens']}")
                if 'input_length' in record.payload_info: lines.append(f"  Input Length  : {record.payload_info['input_length']}") # Embedding
                # Tool 관련 정보
                if 'tools_count' in record.payload_info: lines.append(f"  Tools Count   : {record.payload_info['tools_count']}")
                if 'tool_choice' in record.payload_info: lines.append(f"  Tool Choice   : {record.payload_info['tool_choice']}")

                if 'messages_preview' in record.payload_info: lines.append(f"  Messages Preview: {record.payload_info['messages_preview']}")
                if 'additional_params' in record.payload_info and record.payload_info['additional_params']:
                    lines.append(f"  Additional Params: {record.payload_info['additional_params']}")

                # DEBUG 레벨: 상세 프롬프트 및 Tools 정보
                if record.levelno <= logging.DEBUG:
                    if 'messages_formatted' in record.payload_info:
                        lines.append("\n--- Formatted Prompt (DEBUG Level Only) ---")
                        lines.extend(record.payload_info['messages_formatted'].splitlines())
                    if 'tools_formatted' in record.payload_info:
                        lines.append("\n--- Tools Definition (DEBUG Level Only) ---")
                        lines.extend(record.payload_info['tools_formatted'].splitlines())

            # --- Response / Output ---
            if hasattr(record, 'response_info') and isinstance(record.response_info, dict):
                lines.append("\n--- Response Info ---")
                if 'id' in record.response_info: lines.append(f"  ID               : {record.response_info['id']}")
                if 'choices_count' in record.response_info: lines.append(f"  Choices Count    : {record.response_info['choices_count']}")
                if 'embeddings_count' in record.response_info: lines.append(f"  Embeddings Count: {record.response_info['embeddings_count']}")
                usage = record.response_info.get('usage')
                if isinstance(usage, dict):
                    lines.append(f"  Usage (P/C/T)    : {usage.get('prompt_tokens','?')} / {usage.get('completion_tokens','?')} / {usage.get('total_tokens','?')}")
                if 'content_preview' in record.response_info: lines.append(f"  Content Preview  : {record.response_info['content_preview']}")
                # Tool Calls 정보
                if 'tool_calls_count' in record.response_info and record.response_info['tool_calls_count'] > 0:
                    lines.append(f"  Tool Calls Count: {record.response_info['tool_calls_count']}")
                    if 'tool_calls_summary' in record.response_info: lines.append(f"  Tool Calls Summ  : {record.response_info['tool_calls_summary']}")
                    if record.levelno <= logging.DEBUG and 'tool_calls_details' in record.response_info:
                        lines.append("\n--- Tool Calls Details (DEBUG Level Only) ---")
                        try:
                            lines.append(json.dumps(record.response_info['tool_calls_details'], indent=2, ensure_ascii=False))
                        except Exception: lines.append("  (Error formatting tool_calls details)")

            # --- Tool Result (Sent to LLM) ---
            # event_type 이 "api_tool_call_response" 인 경우 (scheduler.py에서 로깅 시 사용)
            if record.event_type == "api_tool_call_response":
                if hasattr(record, 'tool_result_info') and isinstance(record.tool_result_info, dict):
                    lines.append("\n--- Tool Result Sent to LLM ---")
                    lines.append(f"  Tool Call ID: {record.tool_result_info.get('tool_call_id')}")
                    lines.append(f"  Function Name: {record.tool_result_info.get('function_name')}")
                    lines.append(f"  Result Preview: {record.tool_result_info.get('result_preview')}")
                    if record.levelno <= logging.DEBUG and 'result_content' in record.tool_result_info:
                        lines.append("\n--- Full Tool Result Content (DEBUG Level Only) ---")
                        lines.append(record.tool_result_info['result_content'])

            # --- Error Info ---
            if hasattr(record, 'error_details') and record.error_details:
                lines.append("\n--- Error Details ---")
                error_data = record.error_details
                if isinstance(error_data, dict):
                    lines.append(f"  Type    : {error_data.get('type')}")
                    lines.append(f"  Code    : {error_data.get('code')}")
                    lines.append(f"  Param   : {error_data.get('param')}")
                    lines.append(f"  Message : {error_data.get('message')}")
                else: lines.append(f"  Details : {error_data}")
            elif hasattr(record, 'error_message') and not hasattr(record, 'error_details'):
                lines.append("\n--- Error Message ---")
                lines.append(f"  {record.error_message}")

            # --- Failed Input (Error 발생 시) ---
            failed_prompt_data = getattr(record, 'failed_prompt_messages', None)
            failed_input_text = getattr(record, 'failed_input_text', None)
            if record.levelno <= logging.DEBUG: # DEBUG 레벨일 때만 실패 프롬프트/입력 기록
                if failed_prompt_data:
                    lines.append("\n--- Failed Prompt (DEBUG Level Only) ---")
                    try:
                        formatted_fail_lines = []
                        for msg in failed_prompt_data:
                            role = msg.get('role', 'unknown').upper(); content = msg.get('content', '')
                            content_str = str(content)[:1000] + '...' if len(str(content)) > 1000 else str(content) # 너무 길면 자르기
                            indented_content = "\n".join(["    " + line for line in content_str.split('\n')])
                            formatted_fail_lines.append(f"  [{role}]:\n{indented_content}")
                        lines.append("\n".join(formatted_fail_lines))
                    except Exception: lines.append("  (Error formatting failed prompt)")
                elif failed_input_text:
                    lines.append("\n--- Failed Input Text (DEBUG Level Only) ---"); lines.append(f"  {failed_input_text}")
            elif failed_prompt_data or failed_input_text: # DEBUG 아니어도 실패 입력 있었음은 표시
                lines.append("\n--- Failed Input Exists (Set DEBUG level for details) ---")


            lines.append("=" * 80 + "\n")
            log_string = "\n".join(lines)

        # 일반 로그 (API 관련 아닐 때) + 예외 정보
        elif record.exc_info:
            log_string += "\n" + self.formatException(record.exc_info)

        return log_string


# --- 로깅 설정 (DEBUG 고정 및 동적 파일명) ---
api_logger: Optional[logging.Logger] = None
try:
    logger_config = config.get('logging', {})
    if not isinstance(logger_config, dict):
        raise ValueError("Invalid 'logging' configuration section.")

    log_file_base_name = logger_config.get('log_file_base_name', 'api_history')
    current_timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
    dynamic_log_filename = f"{log_file_base_name}_{current_timestamp_str}.txt"
    log_dir = os.path.join(project_root_gpt, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, dynamic_log_filename)
    logging.info(f"API log file path set to: {log_file_path}")

    api_logger = logging.getLogger('api_logger')
    api_logger.setLevel(logging.DEBUG) # DEBUG 고정
    api_logger.propagate = False

    if not api_logger.handlers:
        logging.info(f"Configuring 'api_logger' (Name: {api_logger.name}) with DEBUG level...")
        log_timestamp_format = logger_config.get('log_timestamp_format', '%Y-%m-%d %H:%M:%S')
        text_formatter = ReadableTextFormatter(datefmt=log_timestamp_format)

        try: # 파일 핸들러
            file_handler = logging.FileHandler(filename=log_file_path, encoding='utf-8')
            file_handler.setFormatter(text_formatter)
            file_handler.setLevel(logging.DEBUG)
            api_logger.addHandler(file_handler)
            logging.info(f"Added File Handler: Path='{log_file_path}', Level=DEBUG")
        except Exception as fh_e:
            logging.error(f"Failed to create File Handler: {fh_e}", exc_info=True)

        try: # 콘솔 핸들러
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%H:%M:%S')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG) # 콘솔도 DEBUG 레벨
            api_logger.addHandler(console_handler)
            logging.info(f"Added Stream Handler (Console): Level=DEBUG")
        except Exception as ch_e:
            logging.error(f"Failed to create Stream Handler: {ch_e}", exc_info=True)

        if api_logger.hasHandlers():
            api_logger.info(f"Logger '{api_logger.name}' initialized with DEBUG level. Check logs at: {log_file_path}")
        else:
            logging.error("CRITICAL: No handlers added to api_logger. Logging might not work.")
    else:
        logging.info(f"'api_logger' already configured. Skipping setup.")

except Exception as setup_e:
    logging.error(f"CRITICAL ERROR during logger setup: {setup_e}", exc_info=True)
    logging.basicConfig(level=logging.DEBUG) # Fallback
    api_logger = logging.getLogger('api_logger_fallback')
    api_logger.error("Using basicConfig fallback logger due to setup error.")


# --- 비동기 OpenAI ChatCompletion API 호출 함수 (Tool Use 지원) ---
async def call_gpt_async(
    messages: list[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    session: Optional[aiohttp.ClientSession] = None,
    tools: Optional[List[Dict]] = None, # Tool 정의
    tool_choice: Optional[Union[str, Dict]] = None, # Tool 선택 방식
    **kwargs: Any # response_format 등 추가 파라미터
) -> Optional[Dict[str, Any]]:
    """
    비동기적으로 OpenAI Chat Completion API를 호출하고 결과를 반환합니다.
    Tool Use 파라미터를 지원하고 상세한 로그를 api_logger를 통해 기록합니다.
    """
    # 필수 요소 검증
    if not api_logger:
        logging.error("API Logger not configured.") # api_logger는 위에서 fallback 처리됨
    if not OPENAI_API_KEY:
        # api_logger는 존재 보장됨
        api_logger.error("OpenAI API Key missing.", extra={"event_type": "config_error"})
        return None
    if not model or not isinstance(temperature, (int, float)) or not isinstance(max_tokens, int) or max_tokens <= 0:
        api_logger.error(f"Invalid API params: model='{model}', temp={temperature}, max_tokens={max_tokens}", extra={"event_type":"param_error"})
        return None
    if not messages:
        api_logger.error("Messages list cannot be empty.", extra={"event_type": "param_error"})
        return None

    # API 요청 준비
    try:
        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if tools: payload["tools"] = tools
        if tool_choice: payload["tool_choice"] = tool_choice

    except Exception as param_e:
        api_logger.error(f"Error setting up API parameters: {param_e}", exc_info=True, extra={"event_type":"param_error"})
        return None

    # 요청 로그 데이터 준비
    request_timestamp = datetime.now().isoformat(timespec='milliseconds')
    log_data_request = {
        "event_type": "api_call", "direction": "request", "api_type": "chat_completion", "model": model,
        "payload_info": {
            "messages_count": len(payload.get('messages', [])),
            "temperature": temperature, "max_tokens": max_tokens,
            "tools_count": len(payload.get('tools', [])),
            "tool_choice": payload.get('tool_choice', 'Not Specified'),
            "messages_preview": f"System: {payload.get('messages', [{}])[0].get('content', '')[:50]}... User: {payload.get('messages', [{}, {}])[-1].get('content', '')[:50]}..." if len(payload.get('messages', [])) > 0 else "No messages",
            "additional_params": kwargs
        },
        "request_timestamp": request_timestamp
    }

    # DEBUG 레벨 상세 로그 준비
    log_message = f"Sending ChatCompletion request to {model}"
    if api_logger and api_logger.getEffectiveLevel() <= logging.DEBUG:
        try:
            # Formatted Prompt
            formatted_prompt_lines = [f"--- Prompt Messages ({len(payload.get('messages', []))}) ---"]
            for msg in payload.get('messages', []):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', ''); tool_call_id = msg.get('tool_call_id') # tool 결과 로깅 추가
                content_str = str(content)[:1000] + ('...' if len(str(content)) > 1000 else '')
                indented_content = "\n".join(["  " + line for line in content_str.split('\n')])
                prefix = f"[{role}]"
                if tool_call_id: prefix += f" (ID: {tool_call_id})" # Tool Call ID 표시
                formatted_prompt_lines.append(f"{prefix}:\n{indented_content}")
            formatted_prompt_lines.append("--- End Prompt ---")
            log_data_request["payload_info"]["messages_formatted"] = "\n".join(formatted_prompt_lines)

            # Tools Definition
            if tools:
                formatted_tools_string = json.dumps(tools, indent=2, ensure_ascii=False)
                log_data_request["payload_info"]["tools_formatted"] = formatted_tools_string
            log_message += " (DEBUG: full prompt/tools logged)"
        except Exception as fmt_e:
            if api_logger: api_logger.warning(f"Error formatting prompt/tools messages for logging: {fmt_e}", extra={"event_type":"logging_format_error"})
            log_data_request["payload_info"]["messages_formatted"] = "Error formatting"
            if tools: log_data_request["payload_info"]["tools_formatted"] = "Error formatting"
            log_message += " (Error formatting details)"

    # API 요청 로그 기록
    if api_logger: api_logger.info(log_message, extra=log_data_request)

    # aiohttp 세션 관리
    close_session = False
    if session is None:
        if api_logger: api_logger.warning("aiohttp session not provided, creating new.", extra={"event_type": "session_warning"})
        try:
            session = aiohttp.ClientSession()
            close_session = True
        except Exception as session_e:
            if api_logger: api_logger.error(f"Failed to create new aiohttp session: {session_e}", exc_info=True, extra={"event_type":"session_error"})
            return None
    elif session.closed:
        if api_logger: api_logger.error("Provided aiohttp session is closed.", extra={"event_type":"session_error"})
        return None

    # API 호출 및 응답 처리
    response_data: Optional[Dict[str, Any]] = None
    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(timespec='milliseconds')
            response_status = response.status
            response_text_content = await response.text()

            try: # JSON 파싱
                response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + ('...' if len(response_text_content) > 500 else '')
                log_data_error = {
                    "event_type": "api_response_error", "direction": "response", "api_type": "chat_completion",
                    "error_type": "decode_error", "model": model, "status_code": response_status,
                    "error_message": str(decode_e), "response_text_preview": response_text_preview,
                    "request_timestamp": request_timestamp, "response_timestamp": response_timestamp,
                    "failed_prompt_messages": payload.get('messages') # 실패 시 프롬프트 저장
                }
                if api_logger: api_logger.error("Failed to decode JSON API response", extra=log_data_error)
                if close_session and session and not session.closed: await session.close()
                return None

            # 응답 로그 데이터 준비
            response_choices = response_data.get("choices", [])
            first_choice_message = response_choices[0].get("message", {}) if response_choices else {}
            tool_calls = first_choice_message.get("tool_calls")

            log_data_response = {
                "event_type": "api_response", "direction": "response", "api_type": "chat_completion",
                "model": model, "status_code": response_status,
                "response_info": {
                    "id": response_data.get("id"),
                    "choices_count": len(response_choices),
                    "usage": response_data.get("usage"),
                    "content_preview": first_choice_message.get("content", "")[:80] + "..." if first_choice_message.get("content") else "N/A (Check tool_calls)",
                    "tool_calls_count": len(tool_calls) if tool_calls else 0,
                    "tool_calls_summary": [{"id": tc.get("id"), "function": tc.get("function", {}).get("name")} for tc in tool_calls] if tool_calls else "None",
                    "tool_calls_details": tool_calls # DEBUG 레벨용
                },
                "request_timestamp": request_timestamp, "response_timestamp": response_timestamp
            }

            if response.status == 200:
                usage = response_data.get("usage", {})
                total_tokens = usage.get('total_tokens', 'N/A')
                log_message_resp = f"ChatCompletion success from {model}. Tokens: {total_tokens}"
                if tool_calls: log_message_resp += f". Received {len(tool_calls)} tool call(s)."
                if api_logger: api_logger.info(log_message_resp, extra=log_data_response)
                return response_data # 성공 시 응답 데이터 반환
            else: # API 레벨 에러
                if response.status == 401: log_message_resp = f"API Error (401 Unauthorized)"
                elif response.status == 429: log_message_resp = f"API Error (429 Rate Limit)"
                elif response.status == 400: log_message_resp = f"API Error (400 Bad Request)"
                else: log_message_resp = f"API Error (HTTP {response.status})"

                log_data_response["event_type"] = "api_response_error"
                log_data_response["error_details"] = response_data.get("error")
                log_data_response["failed_prompt_messages"] = payload.get('messages') # 실패 시 프롬프트 저장

                if api_logger: api_logger.error(log_message_resp, extra=log_data_response)
                return None # API 오류 시 None 반환

    except aiohttp.ClientError as e: # 네트워크 오류
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "network_error",
            "model": model, "error_message": str(e), "request_timestamp": request_timestamp,
            "failed_prompt_messages": payload.get('messages')
        }
        if api_logger: api_logger.error(f"Network Error during ChatCompletion: {e}", extra=log_data_exception)
        return None
    except asyncio.TimeoutError: # 타임아웃
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "timeout_error",
            "model": model, "error_message": "Request timed out", "request_timestamp": request_timestamp,
            "failed_prompt_messages": payload.get('messages')
        }
        if api_logger: api_logger.error("ChatCompletion request timed out", extra=log_data_exception)
        return None
    except Exception as e: # 기타 오류
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "unexpected_error",
            "model": model, "error_message": str(e), "request_timestamp": request_timestamp,
            "failed_prompt_messages": payload.get('messages')
        }
        if api_logger: api_logger.error(f"Unexpected Error during ChatCompletion: {e}", extra=log_data_exception, exc_info=True)
        return None
    finally:
        if close_session and session and not session.closed:
            await session.close()


# --- 비동기 OpenAI Embedding API 호출 함수 (변경 없음) ---
async def get_openai_embedding_async(
    text: str,
    session: Optional[aiohttp.ClientSession] = None,
    model: Optional[str] = None # 모델명 직접 지정 옵션
) -> Optional[List[float]]:
    """
    비동기적으로 OpenAI Embedding API를 호출하여 텍스트의 임베딩 벡터를 반환합니다.
    """
    global config, api_logger, OPENAI_API_KEY # 전역 설정 사용

    if not api_logger: logging.error("API Logger not configured for Embedding.")
    if not OPENAI_API_KEY:
        if api_logger: api_logger.error("OpenAI API Key missing for Embedding.", extra={"event_type": "config_error", "api_type": "embedding"})
        else: logging.error("OpenAI API Key missing.")
        return None
    if not text or not isinstance(text, str):
        if api_logger: api_logger.error("Invalid text for embedding.", extra={"event_type":"param_error", "api_type": "embedding"})
        else: logging.error("Invalid text for embedding.")
        return None

    try: # 임베딩 모델 결정
        embedding_model = model or config.get('rag', {}).get('embedding_model')
        if not embedding_model: raise ValueError("Embedding model name missing.")
    except Exception as e:
        if api_logger: api_logger.error(f"Error determining embedding model: {e}", extra={"event_type":"config_error", "api_type": "embedding"})
        else: logging.error(f"Error determining embedding model: {e}")
        return None

    try: # API 요청 준비
        openai_url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"input": text.replace("\n", " "), "model": embedding_model, "encoding_format": "float"}
    except Exception as param_e:
        if api_logger: api_logger.error(f"Error setting up Embedding params: {param_e}", exc_info=True, extra={"event_type":"param_error", "api_type": "embedding"})
        else: logging.error(f"Error setting up Embedding params: {param_e}")
        return None

    request_timestamp = datetime.now().isoformat(timespec='milliseconds')
    log_data_request = {
        "event_type": "api_call", "direction": "request", "api_type": "embedding", "model": embedding_model,
        "payload_info": {"input_length": len(payload.get('input', '')) },
        "request_timestamp": request_timestamp
    }
    if api_logger: api_logger.info(f"Sending Embedding request to {embedding_model}", extra=log_data_request)
    else: logging.info(f"Sending Embedding request to {embedding_model}")

    # 세션 관리
    close_session = False
    if session is None:
        if api_logger: api_logger.warning("aiohttp session not provided for embedding, creating new.", extra={"event_type": "session_warning", "api_type": "embedding"})
        else: logging.warning("No session for embedding, creating new.")
        try: session = aiohttp.ClientSession(); close_session = True
        except Exception as session_e:
            if api_logger: api_logger.error(f"Failed to create session for embedding: {session_e}", exc_info=True, extra={"event_type":"session_error", "api_type": "embedding"})
            else: logging.error(f"Failed to create session for embedding: {session_e}")
            return None
    elif session.closed:
        if api_logger: api_logger.error("Provided session for embedding is closed.", extra={"event_type":"session_error", "api_type": "embedding"})
        else: logging.error("Provided session for embedding is closed.")
        return None

    # API 호출 및 응답 처리
    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(timespec='milliseconds')
            response_status = response.status
            response_text_content = await response.text()

            try: response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                # ... (오류 로깅 - ChatCompletion과 유사하게 처리) ...
                response_text_preview = response_text_content[:500] + ('...' if len(response_text_content) > 500 else '')
                log_data_error = {
                    "event_type": "api_response_error", "direction": "response", "api_type": "embedding",
                    "error_type": "decode_error", "model": embedding_model, "status_code": response_status,
                    "error_message": str(decode_e), "response_text_preview": response_text_preview,
                    "request_timestamp": request_timestamp, "response_timestamp": response_timestamp,
                    "failed_input_text": payload.get('input')
                }
                if api_logger: api_logger.error("Failed to decode JSON embedding response", extra=log_data_error)
                else: logging.error("Failed to decode embedding JSON response.")
                if close_session and session and not session.closed: await session.close()
                return None

            log_data_response = { # 응답 로그 데이터
                "event_type": "api_response", "direction": "response", "api_type": "embedding",
                "model": embedding_model, "status_code": response_status,
                "response_info": {"embeddings_count": len(response_data.get("data", [])), "usage": response_data.get("usage")},
                "request_timestamp": request_timestamp, "response_timestamp": response_timestamp
            }

            if response.status == 200 and response_data.get("data"):
                usage = response_data.get("usage", {}); total_tokens = usage.get('total_tokens', 'N/A')
                log_message_emb = f"Embedding success from {embedding_model}. Tokens: {total_tokens}"
                if api_logger: api_logger.info(log_message_emb, extra=log_data_response)
                else: logging.info(log_message_emb)
                embedding_vector = response_data["data"][0].get("embedding")
                if embedding_vector and isinstance(embedding_vector, list): return embedding_vector
                else: # 성공 응답이나 벡터 없음
                    log_data_response["event_type"] = "api_response_error"
                    log_data_response["error_details"] = "Embedding vector error in success response"
                    if api_logger: api_logger.error("Embedding vector error", extra=log_data_response)
                    else: logging.error("Embedding vector error.")
                    return None
            else: # API 레벨 에러
                if response.status == 401: log_message_emb = f"Embedding API Error (401)"
                elif response.status == 429: log_message_emb = f"Embedding API Error (429)"
                else: log_message_emb = f"Embedding API Error ({response.status})"
                log_data_response["event_type"] = "api_response_error"
                log_data_response["error_details"] = response_data.get("error")
                log_data_response["failed_input_text"] = payload.get('input')
                if api_logger: api_logger.error(log_message_emb, extra=log_data_response)
                else: logging.error(log_message_emb)
                return None

    except aiohttp.ClientError as e: # 네트워크 오류
        # ... (오류 로깅 - ChatCompletion과 유사하게 처리) ...
        log_data_exception = { "event_type": "api_call_error", "api_type": "embedding", "error_type": "network_error", "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp, "failed_input_text": payload.get('input') }
        if api_logger: api_logger.error(f"Network Error (Embedding): {e}", extra=log_data_exception)
        else: logging.error(f"Network Error (Embedding): {e}")
        return None
    except asyncio.TimeoutError: # 타임아웃
        # ... (오류 로깅 - ChatCompletion과 유사하게 처리) ...
        log_data_exception = { "event_type": "api_call_error", "api_type": "embedding", "error_type": "timeout_error", "model": embedding_model, "error_message": "Request timed out", "request_timestamp": request_timestamp, "failed_input_text": payload.get('input') }
        if api_logger: api_logger.error("Embedding request timed out", extra=log_data_exception)
        else: logging.error("Embedding request timed out")
        return None
    except Exception as e: # 기타 오류
        # ... (오류 로깅 - ChatCompletion과 유사하게 처리) ...
        log_data_exception = { "event_type": "api_call_error", "api_type": "embedding", "error_type": "unexpected_error", "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp, "failed_input_text": payload.get('input') }
        if api_logger: api_logger.error(f"Unexpected Error (Embedding): {e}", extra=log_data_exception, exc_info=True)
        else: logging.error(f"Unexpected Error (Embedding): {e}")
        return None
    finally:
        if close_session and session and not session.closed:
            await session.close()


# ##### 만족도 평가 함수 추가된 부분 시작 #####

logger = logging.getLogger(__name__) # 필요 시 재정의

async def evaluate_satisfaction_async(
    question: str,
    response: str,
    session: aiohttp.ClientSession
) -> Optional[Dict[str, Any]]:
    """
    GPT (config에서 지정한 모델)를 사용하여 응답 만족도를 평가합니다.
    gpt_interface 모듈 내에서 사용하기 위해 구현되었습니다.

    Args:
        question (str): 사용자 질문.
        response (str): 챗봇의 응답.
        session (aiohttp.ClientSession): API 호출에 사용할 aiohttp 세션.

    Returns:
        Optional[Dict[str, Any]]: 평가 결과 딕셔너리 (점수, 이유, 상태 포함). 오류 시 None 또는 에러 정보 포함 딕셔너리.
    """
    global config # 전역 config 사용

    # 만족도 평가 모델 및 프롬프트 로드
    try:
        if config is None: raise ValueError("Config not loaded")
        satisfaction_config = config.get('tasks', {}).get('satisfaction_evaluation', {})
        SATISFACTION_MODEL = satisfaction_config.get('model')
        if not SATISFACTION_MODEL: raise ValueError("Satisfaction model not found in config (tasks.satisfaction_evaluation.model)")

        evaluation_prompt_template = config.get('prompts', {}).get('satisfaction_evaluation_prompt_template')
        if not evaluation_prompt_template: raise ValueError("Satisfaction prompt template not found in config (prompts.satisfaction_evaluation_prompt_template)")
    except Exception as e:
        logger.error(f"Configuration error for satisfaction evaluation: {e}", exc_info=True)
        return {"error": f"Configuration error: {e}", "status": "evaluation_failed_config"}

    # 입력 검증
    if not response or not isinstance(response, str) or len(response.strip()) == 0:
        logger.warning("Empty or invalid response received for satisfaction evaluation.")
        return {"error": "Empty or invalid response provided", "status": "evaluation_skipped"}

    # 평가 프롬프트 포맷팅
    try:
        evaluation_prompt = evaluation_prompt_template.format(question=question, response=response)
    except KeyError as e:
        logger.error(f"Error formatting satisfaction prompt template. Missing key: {e}")
        return {"error": f"Prompt format error: Missing key {e}", "status": "evaluation_failed_prompt_format"}
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return {"error": f"Unexpected prompt format error: {e}", "status": "evaluation_failed_prompt_format"}

    # API 호출 메시지 준비
    messages = [{"role": "user", "content": evaluation_prompt}]
    logger.debug(f"Requesting satisfaction evaluation from {SATISFACTION_MODEL}...")
    try:
        # call_gpt_async 함수 재사용 (이 파일 내에 정의되어 있음)
        eval_response_data = await call_gpt_async(
            messages=messages,
            model=SATISFACTION_MODEL,
            temperature=0.1, # 일관된 평가를 위해 낮은 온도
            max_tokens=300,  # JSON 결과 받기에 충분한 토큰
            session=session,
            response_format={"type": "json_object"} # JSON 모드 요청
        )

        # API 응답 처리
        if eval_response_data and eval_response_data.get("choices"):
            eval_content = eval_response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw satisfaction evaluation response: {eval_content[:150]}...")

            # JSON 파싱 강화
            clean_eval_content = eval_content.strip()
            if clean_eval_content.startswith("```json"): clean_eval_content = clean_eval_content[7:-3].strip()
            elif clean_eval_content.startswith("```"): clean_eval_content = clean_eval_content[3:-3].strip()

            try:
                json_start = clean_eval_content.find('{'); json_end = clean_eval_content.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_string = clean_eval_content[json_start:json_end+1]
                else:
                    json_string = clean_eval_content # 중괄호 없으면 그대로 시도

                # 빈 문자열 방지
                if not json_string.strip():
                     logger.warning("Evaluation response content became empty after cleaning.")
                     return {"error": "Empty content after cleaning", "raw_content": eval_content, "status": "evaluation_failed_parsing"}

                evaluation_result = json.loads(json_string)

                # 결과 검증 및 정제
                required_scores = ["relevance_score", "accuracy_score", "completeness_score", "conciseness_score", "tone_score", "overall_satisfaction_score"]
                valid_result = {"status": "success"}
                missing_scores = []
                conversion_errors = []

                for key in required_scores:
                    if key not in evaluation_result:
                        missing_scores.append(key)
                        valid_result[key] = None
                        continue
                    try:
                        # None 값 처리 추가
                        score_value = evaluation_result[key]
                        if score_value is None:
                             logger.warning(f"Evaluation score '{key}' is None. Setting to None.")
                             valid_result[key] = None
                             conversion_errors.append(f"{key}: received_none")
                             continue

                        score = int(score_value)
                        if 1 <= score <= 5:
                            valid_result[key] = score
                        else:
                            logger.warning(f"Evaluation score '{key}' ({score}) out of range (1-5). Setting to None.")
                            valid_result[key] = None
                            conversion_errors.append(f"{key}: out_of_range({score})")
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert score '{key}' to int: {evaluation_result[key]}. Setting to None.")
                        valid_result[key] = None
                        conversion_errors.append(f"{key}: conversion_error({evaluation_result[key]})")

                valid_result["evaluation_reason"] = evaluation_result.get("evaluation_reason", "N/A") # 이유 텍스트

                # 상태 업데이트 (경고 처리)
                if missing_scores:
                    logger.warning(f"Evaluation result missing scores: {missing_scores}")
                    valid_result["status"] = "parsing_warning_missing_scores"
                if conversion_errors:
                    logger.warning(f"Evaluation score conversion issues: {conversion_errors}")
                    if valid_result["status"] == "success": # 이전 경고 없을 때만 업데이트
                         valid_result["status"] = "parsing_warning_conversion_error"

                logger.info("Satisfaction evaluation successful.")
                return valid_result # 정제된 결과 반환

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from satisfaction evaluation: {e}. Cleaned Content: '{clean_eval_content}'")
                return {"error": f"JSON Decode Error: {e}", "raw_content": eval_content, "status": "evaluation_failed_parsing"}
        else:
            logger.warning(f"Failed to get valid response/choices from evaluation model {SATISFACTION_MODEL}.")
            # call_gpt_async 내부에서 API 실패 로그 기록됨
            return {"error": "No valid choices from evaluation model", "status": "evaluation_failed_api_no_choice"}

    except Exception as e:
        logger.error(f"An unexpected error occurred during satisfaction evaluation API call: {e}", exc_info=True)
        return {"error": f"Exception during evaluation: {e}", "status": "evaluation_failed_exception"}

# ###########################################
# ##### 만족도 평가 함수 추가된 부분 끝 #####
# ###########################################


# --- 예시 사용법 (__main__ 블록 수정) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__) # 로거 재정의 필요 없음 (파일 상단에서 정의됨)
    logger.info("--- Running gpt_interface.py as main script for testing ---")

    async def test_apis():
        """기존 API 테스트 함수 (변경 없음)"""
        logging.info("Running test_apis() function...")
        if not config: logging.error("Config not loaded, cannot run tests."); return
        if not OPENAI_API_KEY: logging.error("API Key missing."); return

        # Chat Completion Test (Basic)
        test_messages = [{"role": "user", "content": "Say 'Hello GPT Interface Test!'"}]
        logging.info("\n--- Testing Basic Chat Completion API ---")
        try:
            test_model = config.get('testing', {}).get('default_baseline_model', 'gpt-3.5-turbo')
            # 기본값 설정을 위해 task 설정 접근 방식 수정
            tool_use_config = config.get('tasks', {}).get('tool_use', {})
            test_temp = tool_use_config.get('generation_temperature', 0.7)
            test_max_tokens = tool_use_config.get('generation_max_tokens', 100)

            async with aiohttp.ClientSession() as session:
                response = await call_gpt_async(messages=test_messages, model=test_model, temperature=test_temp, max_tokens=test_max_tokens, session=session)
                logging.info(f"Basic Chat Result: {'Success' if response else 'Failed'}")
                if response: logging.info(f"Content Preview: {response.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
        except Exception as e: logging.error(f"Error in Basic Chat test: {e}", exc_info=True)

        # Chat Completion Test (With Tools Definition - but no forced call)
        logging.info("\n--- Testing Chat Completion API (With Tools Definition) ---")
        test_messages_tool = [{"role": "user", "content": "What is the weather in Seoul today?"}] # Tool 사용 안 할 질문
        dummy_tool = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
        try:
            # task 설정 접근 방식 수정
            tool_use_config = config.get('tasks', {}).get('tool_use', {})
            test_model_tool = tool_use_config.get('model', 'gpt-4o')
            test_temp_tool = tool_use_config.get('decision_temperature', 0.2)
            test_max_tokens_tool = tool_use_config.get('decision_max_tokens', 500)

            async with aiohttp.ClientSession() as session:
                response_tool = await call_gpt_async(messages=test_messages_tool, model=test_model_tool, temperature=test_temp_tool, max_tokens=test_max_tokens_tool, session=session, tools=dummy_tool, tool_choice="auto")
                logging.info(f"Chat API Call (With Tools Def) Result: {'Success' if response_tool else 'Failed'}")
                if response_tool:
                    logging.info(f"Content Preview: {response_tool.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
                    logging.info(f"Tool Calls: {response_tool.get('choices', [{}])[0].get('message', {}).get('tool_calls')}") # Tool call 확인
        except Exception as e: logging.error(f"Error during Chat (With Tools Def) test: {e}", exc_info=True)

        # Embedding API Test
        test_text = "Test embedding for gpt_interface.py"; logging.info("\n--- Testing Embedding API ---")
        try:
            emb_model = config.get('rag', {}).get('embedding_model')
            async with aiohttp.ClientSession() as session:
                embedding = await get_openai_embedding_async(test_text, session=session, model=emb_model)
                logging.info(f"Embedding API Result: {'Success (Dim: ' + str(len(embedding)) + ')' if embedding else 'Failed'}")
                if embedding: logging.debug(f"Embedding vector preview: {embedding[:5]}...")
        except Exception as e: logging.error(f"Error during Embedding test: {e}", exc_info=True)

    # --- [추가] 만족도 평가 테스트 함수 ---
    async def test_satisfaction():
        """만족도 평가 함수 테스트"""
        print("\n--- Testing Satisfaction Evaluation ---")
        if not os.getenv("OPENAI_API_KEY"): print("API Key missing. Cannot run satisfaction test."); return
        try:
            # 설정 로드 재확인 (필요시)
            if config is None: config = get_config(); assert config
            print("Config loaded for satisfaction test.")
            async with aiohttp.ClientSession() as session:
                test_q = "가벼운 런닝화 추천해줘"
                test_r = "네, 고객님. 킵런 KD500 모델은 가볍고 쿠션도 적당하여 가볍게 뛰기에 좋습니다. 가격은 89,000원입니다."
                print(f"Q: {test_q}")
                print(f"R: {test_r}")
                eval_result = await evaluate_satisfaction_async(test_q, test_r, session) # 이 파일 내 함수 호출
                print("Evaluation Result:")
                # 결과가 None일 수 있으므로 확인
                if eval_result:
                    print(json.dumps(eval_result, indent=2, ensure_ascii=False))
                else:
                    print("Satisfaction evaluation returned None (check logs for errors).")
        except Exception as e:
             print(f"Error during satisfaction test: {e}")

    # --- [수정] 모든 테스트 실행 함수 ---
    async def run_all_tests():
        """모든 테스트를 순차적으로 실행"""
        await test_apis() # 기존 API 테스트
        await test_satisfaction() # 추가된 만족도 테스트

    # --- [수정] 테스트 실행 ---
    try:
        asyncio.run(run_all_tests()) # 수정된 함수 호출
    except Exception as e:
        logging.error(f"Test execution error: {e}", exc_info=True)

    # --- 로그 파일 경로 안내 (기존 유지) ---
    if 'log_file_path' in locals(): print(f"\nCheck logs in: {log_file_path}")