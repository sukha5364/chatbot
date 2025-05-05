# chatbot/gpt_interface.py (Tool Use 파라미터 및 로깅 추가)

import os
import json
import logging
import logging.handlers # 직접 사용 안 함
from datetime import datetime
from typing import Dict, Any, Optional, List, Union # Union 추가
import aiohttp
from dotenv import load_dotenv
import traceback
import time # Formatter에서 사용

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
        Formatter 초기화.
        """
        # config에서 타임스탬프 형식 읽기 (없으면 기본값 사용)
        safe_datefmt = datefmt or config.get('logging', {}).get('log_timestamp_format', '%Y-%m-%d %H:%M:%S')
        super().__init__(fmt, safe_datefmt, style, validate, defaults=defaults)
        self.converter = time.localtime # 로컬 시간대 사용

    def formatTime(self, record, datefmt=None):
        """로그 레코드의 생성 시간을 지정된 포맷으로 변환합니다."""
        ct = self.converter(record.created)
        _datefmt = datefmt or self.default_time_format # None이면 기본 포맷 사용
        if _datefmt:
            try:
                s = time.strftime(_datefmt, ct)
            except ValueError as e:
                # 잘못된 포맷 문자열일 경우 경고 로깅 및 기본 포맷 사용
                logging.warning(f"WARNING (ReadableTextFormatter): Invalid date format string '{_datefmt}'. Falling back to default. Error: {e}")
                s = time.strftime(logging.Formatter.default_time_format, ct) # 기본 ISO 포맷 사용
        else:
            s = time.strftime(logging.Formatter.default_time_format, ct) # 안전하게 기본 포맷 사용
        return s

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 포맷된 문자열로 변환합니다."""
        # 기본 로그 메시지 생성
        record.asctime = self.formatTime(record, self.datefmt) # 시간 포맷팅 먼저 수행
        log_string = super().format(record) # 기본 Formatter의 format 활용

        # API 관련 특정 이벤트 타입에 대한 상세 정보 추가
        # [수정] event_type 에 tool 관련 타입 추가
        if hasattr(record, 'event_type') and record.event_type.startswith("api_"):
            lines = ["=" * 80, log_string, "-" * 80] # 구분선 추가

            # 공통 정보
            lines.append(f"Event Type : {getattr(record, 'event_type', 'N/A')}")
            lines.append(f"Direction  : {getattr(record, 'direction', 'N/A')}")
            lines.append(f"API Type   : {getattr(record, 'api_type', 'N/A')}")
            lines.append(f"Model      : {getattr(record, 'model', 'N/A')}")
            lines.append(f"Request TS : {getattr(record, 'request_timestamp', 'N/A')}")
            if hasattr(record, 'response_timestamp'): lines.append(f"Response TS: {record.response_timestamp}")
            if hasattr(record, 'status_code'): lines.append(f"Status Code: {record.status_code}")

            # --- Payload / Input ---
            if hasattr(record, 'payload_info') and isinstance(record.payload_info, dict):
                lines.append("\n--- Request Payload Info ---")
                lines.append(f"  Messages Count: {record.payload_info.get('messages_count', 'N/A')}")
                if 'temperature' in record.payload_info: lines.append(f"  Temperature   : {record.payload_info['temperature']}")
                if 'max_tokens' in record.payload_info: lines.append(f"  Max Tokens    : {record.payload_info['max_tokens']}")
                if 'input_length' in record.payload_info: lines.append(f"  Input Length  : {record.payload_info['input_length']}") # Embedding
                # [신규] Tool 관련 정보 로깅
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
                    # [신규] Tools 정의 로깅
                    if 'tools_formatted' in record.payload_info:
                        lines.append("\n--- Tools Definition (DEBUG Level Only) ---")
                        lines.extend(record.payload_info['tools_formatted'].splitlines())


            # --- Response / Output ---
            if hasattr(record, 'response_info') and isinstance(record.response_info, dict):
                lines.append("\n--- Response Info ---")
                if 'id' in record.response_info: lines.append(f"  ID             : {record.response_info['id']}")
                if 'choices_count' in record.response_info: lines.append(f"  Choices Count  : {record.response_info['choices_count']}")
                if 'embeddings_count' in record.response_info: lines.append(f"  Embeddings Count: {record.response_info['embeddings_count']}")
                usage = record.response_info.get('usage')
                if isinstance(usage, dict):
                    lines.append(f"  Usage (P/C/T)  : {usage.get('prompt_tokens','?')} / {usage.get('completion_tokens','?')} / {usage.get('total_tokens','?')}")
                if 'content_preview' in record.response_info: lines.append(f"  Content Preview: {record.response_info['content_preview']}")
                # [신규] Tool Calls 로깅
                if 'tool_calls_count' in record.response_info and record.response_info['tool_calls_count'] > 0:
                    lines.append(f"  Tool Calls Count: {record.response_info['tool_calls_count']}")
                    if 'tool_calls_summary' in record.response_info: lines.append(f"  Tool Calls Summ : {record.response_info['tool_calls_summary']}") # 함수 이름, ID 등 요약
                    # DEBUG 레벨: 상세 Tool Calls 정보
                    if record.levelno <= logging.DEBUG and 'tool_calls_details' in record.response_info:
                         lines.append("\n--- Tool Calls Details (DEBUG Level Only) ---")
                         try:
                             lines.append(json.dumps(record.response_info['tool_calls_details'], indent=2, ensure_ascii=False))
                         except Exception:
                             lines.append("  (Error formatting tool_calls details)")

            # --- [신규] Tool Result (애플리케이션 -> LLM 전송 시) ---
            # event_type 이 "api_tool_call_response" 인 경우 (scheduler.py에서 이 타입으로 로깅 필요)
            if record.event_type == "api_tool_call_response":
                if hasattr(record, 'tool_result_info') and isinstance(record.tool_result_info, dict):
                    lines.append("\n--- Tool Result Sent to LLM ---")
                    lines.append(f"  Tool Call ID: {record.tool_result_info.get('tool_call_id')}")
                    lines.append(f"  Function Name: {record.tool_result_info.get('function_name')}")
                    lines.append(f"  Result Preview: {record.tool_result_info.get('result_preview')}")
                    # DEBUG 레벨: 상세 Tool Result 내용
                    if record.levelno <= logging.DEBUG and 'result_content' in record.tool_result_info:
                        lines.append("\n--- Full Tool Result Content (DEBUG Level Only) ---")
                        lines.append(record.tool_result_info['result_content'])


            # --- Error Info ---
            if hasattr(record, 'error_details') and record.error_details:
                lines.append("\n--- Error Details ---")
                error_data = record.error_details
                if isinstance(error_data, dict):
                    lines.append(f"  Type   : {error_data.get('type')}")
                    lines.append(f"  Code   : {error_data.get('code')}")
                    lines.append(f"  Param  : {error_data.get('param')}")
                    lines.append(f"  Message: {error_data.get('message')}")
                else: lines.append(f"  Details: {error_data}")
            elif hasattr(record, 'error_message') and not hasattr(record, 'error_details'):
                 lines.append("\n--- Error Message ---")
                 lines.append(f"  {record.error_message}")

            # --- Failed Input (Error 발생 시) ---
            failed_prompt_data = getattr(record, 'failed_prompt_messages', None)
            failed_input_text = getattr(record, 'failed_input_text', None)
            if failed_prompt_data:
                lines.append("\n--- Failed Prompt (DEBUG Level Only) ---")
                if record.levelno <= logging.DEBUG:
                    try:
                        formatted_fail_lines = []
                        for msg in failed_prompt_data:
                            role = msg.get('role', 'unknown').upper(); content = msg.get('content', '')
                            indented_content = "\n".join(["    " + line for line in str(content).split('\n')])
                            formatted_fail_lines.append(f"  [{role}]:\n{indented_content}")
                        lines.append("\n".join(formatted_fail_lines))
                    except Exception: lines.append("  (Error formatting failed prompt)")
                else: lines.append("  (Set log level to DEBUG to view full failed prompt)")
            elif failed_input_text:
                lines.append("\n--- Failed Input Text ---"); lines.append(f"  {failed_input_text}")

            lines.append("=" * 80 + "\n")
            log_string = "\n".join(lines)

        # 일반 로그 (API 관련 아닐 때)
        elif record.exc_info:
             log_string += "\n" + self.formatException(record.exc_info)

        return log_string


# --- 로깅 설정 (DEBUG 고정 및 동적 파일명) ---
api_logger: Optional[logging.Logger] = None
try:
    # 로깅 설정 읽기
    logger_config = config.get('logging', {})
    if not isinstance(logger_config, dict):
        raise ValueError("Invalid 'logging' configuration section in config.yaml")

    # 로그 파일 기본 이름 가져오기
    log_file_base_name = logger_config.get('log_file_base_name', 'api_history')

    # 동적 로그 파일명 생성 ({base_name}_{YYYYMMDD_HHMM}.txt)
    current_timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
    dynamic_log_filename = f"{log_file_base_name}_{current_timestamp_str}.txt"

    # 로그 디렉토리 경로 설정 (프로젝트 루트/logs)
    log_dir = os.path.join(project_root_gpt, 'logs')
    os.makedirs(log_dir, exist_ok=True) # 디렉토리 없으면 생성
    log_file_path = os.path.join(log_dir, dynamic_log_filename)
    logging.info(f"Dynamic log file path set to: {log_file_path}")

    # api_logger 인스턴스 생성 및 설정
    api_logger = logging.getLogger('api_logger')
    api_logger.setLevel(logging.DEBUG) # 로거 레벨 DEBUG로 고정
    api_logger.propagate = False # 루트 로거로 전파 방지

    # 핸들러 중복 추가 방지
    if not api_logger.handlers:
        logging.info(f"Configuring 'api_logger' (Name: {api_logger.name}) with DEBUG level. Adding handlers...")

        # 파일 핸들러 설정 (FileHandler 사용, 로테이션 없음)
        try:
            # Text Formatter 인스턴스 생성 (Timestamp 형식 전달)
            log_timestamp_format = logger_config.get('log_timestamp_format', '%Y-%m-%d %H:%M:%S')
            text_formatter = ReadableTextFormatter(datefmt=log_timestamp_format) # [수정] 새 포매터 사용

            file_handler = logging.FileHandler(filename=log_file_path, encoding='utf-8')
            file_handler.setFormatter(text_formatter)
            file_handler.setLevel(logging.DEBUG) # 핸들러 레벨 DEBUG로 고정
            api_logger.addHandler(file_handler)
            logging.info(f"Added File Handler: Path='{log_file_path}', Level=DEBUG")
        except Exception as fh_e:
            logging.error(f"Failed to create or add File Handler: {fh_e}", exc_info=True)

        # 콘솔 핸들러 설정
        try:
            # 콘솔 핸들러는 간단한 포맷 사용 가능
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%H:%M:%S')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG) # 핸들러 레벨 DEBUG로 고정
            api_logger.addHandler(console_handler)
            logging.info(f"Added Stream Handler (Console): Level=DEBUG")
        except Exception as ch_e:
            logging.error(f"Failed to create or add Stream Handler: {ch_e}", exc_info=True)

        if api_logger.hasHandlers():
            api_logger.info(f"Logger '{api_logger.name}' initialized with DEBUG level. Check log file at: {log_file_path}")
        else:
            logging.error("CRITICAL: No handlers could be added to api_logger. Logging might not work correctly.")
    else:
        logging.info(f"'api_logger' already has handlers configured. Skipping setup.")

except Exception as setup_e:
    # 로거 설정 중 심각한 오류 발생 시 폴백 로거 사용
    logging.error(f"CRITICAL ERROR during logger setup: {setup_e}", exc_info=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') # 기본 설정으로 대체
    api_logger = logging.getLogger('api_logger_fallback')
    api_logger.error("Using basicConfig fallback logger due to setup error.")


# --- 비동기 OpenAI ChatCompletion API 호출 함수 ([수정됨]) ---
async def call_gpt_async(
    messages: list[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    session: Optional[aiohttp.ClientSession] = None,
    tools: Optional[List[Dict]] = None, # [신규] Tools 파라미터
    tool_choice: Optional[Union[str, Dict]] = None, # [신규] Tool Choice 파라미터
    **kwargs: Any # response_format 등 추가 파라미터
) -> Optional[Dict[str, Any]]:
    """
    비동기적으로 OpenAI Chat Completion API를 호출하고 결과를 반환합니다.
    Tool Use 파라미터를 지원하고 상세한 로그를 api_logger를 통해 기록합니다.

    Args:
        messages (list[Dict[str, str]]): OpenAI API 형식의 메시지 리스트.
        model (str): 사용할 GPT 모델 이름.
        temperature (float): 샘플링 온도 (0.0 ~ 2.0).
        max_tokens (int): 생성할 최대 토큰 수.
        session (Optional[aiohttp.ClientSession], optional): 사용할 aiohttp 세션. 없으면 새로 생성. Defaults to None.
        tools (Optional[List[Dict]], optional): API에 제공할 Tool 정의 리스트. Defaults to None.
        tool_choice (Optional[Union[str, Dict]], optional): 사용할 Tool 선택 방식 ('auto', 'none', 또는 특정 함수 지정). Defaults to None.
        **kwargs (Any): OpenAI API에 전달할 추가 파라미터 (예: response_format).

    Returns:
        Optional[Dict[str, Any]]: API 응답 딕셔너리 (tool_calls 포함 가능). 오류 발생 시 None 반환.
    """
    # 필수 요소 확인
    if not api_logger:
        logging.error("API Logger is not configured, cannot log API call details.")
        # 로깅 없이 진행하거나 None 반환
        # return None

    if not OPENAI_API_KEY:
        if api_logger: api_logger.error("OpenAI API Key is missing, cannot make API call.", extra={"event_type": "config_error"})
        else: logging.error("OpenAI API Key is missing.")
        return None
    if not model or not isinstance(temperature, (int, float)) or not isinstance(max_tokens, int) or max_tokens <= 0:
        if api_logger: api_logger.error(f"Invalid API parameters provided: model='{model}', temp={temperature}, max_tokens={max_tokens}", extra={"event_type":"param_error"})
        else: logging.error("Invalid API parameters.")
        return None
    if not messages: # messages 리스트가 비어있는 경우 추가
        if api_logger: api_logger.error("Messages list cannot be empty.", extra={"event_type": "param_error"})
        else: logging.error("Messages list cannot be empty.")
        return None

    # API 요청 준비
    try:
        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        # 기본 페이로드 구성
        payload: Dict[str, Any] = { # 타입 명시
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs # response_format 등 추가 인자 포함
        }
        # [신규] tools 와 tool_choice 추가 (None이 아닐 경우)
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    except Exception as param_e:
        if api_logger: api_logger.error(f"Error setting up API parameters: {param_e}", exc_info=True, extra={"event_type":"param_error"})
        else: logging.error(f"Error setting up API parameters: {param_e}")
        return None

    request_timestamp = datetime.now().isoformat(timespec='milliseconds')
    log_data_request = {
        "event_type": "api_call", "direction": "request", "api_type": "chat_completion", "model": model,
        "payload_info": {
            "messages_count": len(payload.get('messages', [])),
            "temperature": temperature,
            "max_tokens": max_tokens,
            # [신규] Tool 관련 정보 추가
            "tools_count": len(payload.get('tools', [])),
            "tool_choice": payload.get('tool_choice', 'Not Specified'),
            "messages_preview": f"System: {payload.get('messages', [{}])[0].get('content', '')[:50]}... User: {payload.get('messages', [{}, {}])[-1].get('content', '')[:50]}..." if len(payload.get('messages', [])) > 0 else "No messages",
            "additional_params": kwargs
        },
        "request_timestamp": request_timestamp
    }

    # DEBUG 레벨일 때만 상세 프롬프트 및 Tools 포맷팅
    log_message = f"Sending ChatCompletion request to {model}"
    if api_logger and api_logger.getEffectiveLevel() <= logging.DEBUG:
        try:
            # 프롬프트 포맷팅
            formatted_prompt_lines = [f"--- Prompt Messages ({len(payload.get('messages', []))}) ---"]
            for msg in payload.get('messages', []):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                # role: tool 인 경우 content 는 JSON 문자열일 수 있음
                if role == 'TOOL':
                    content_str = str(content)[:500] + '...' if len(str(content)) > 500 else str(content)
                else:
                    content_str = str(content)
                indented_content = "\n".join(["  " + line for line in content_str.split('\n')])
                formatted_prompt_lines.append(f"[{role}]:\n{indented_content}")
            formatted_prompt_lines.append("--- End Prompt ---")
            log_data_request["payload_info"]["messages_formatted"] = "\n".join(formatted_prompt_lines)

            # [신규] Tools 포맷팅
            if tools:
                formatted_tools_string = json.dumps(tools, indent=2, ensure_ascii=False)
                log_data_request["payload_info"]["tools_formatted"] = formatted_tools_string

            log_message += " (DEBUG level - full prompt/tools logged)"
        except Exception as fmt_e:
            if api_logger: api_logger.warning(f"Error formatting prompt/tools messages for logging: {fmt_e}", extra={"event_type":"logging_format_error"})
            log_data_request["payload_info"]["messages_formatted"] = "Error formatting messages"
            if tools: log_data_request["payload_info"]["tools_formatted"] = "Error formatting tools"
            log_message += " (Error formatting details for log)"

    # API 요청 로그 기록
    if api_logger: api_logger.info(log_message, extra=log_data_request)
    else: logging.info(log_message)

    # aiohttp 세션 관리
    close_session = False
    if session is None:
        if api_logger: api_logger.warning("aiohttp session not provided, creating a new one.", extra={"event_type": "session_warning"})
        else: logging.warning("aiohttp session not provided, creating new.")
        try:
            session = aiohttp.ClientSession()
            close_session = True
        except Exception as session_e:
            if api_logger: api_logger.error(f"Failed to create new aiohttp session: {session_e}", exc_info=True, extra={"event_type":"session_error"})
            else: logging.error(f"Failed to create new aiohttp session: {session_e}")
            return None
    elif session.closed:
        if api_logger: api_logger.error("Provided aiohttp session is closed, cannot make API call.", extra={"event_type":"session_error"})
        else: logging.error("Provided aiohttp session is closed.")
        return None

    # API 호출 및 응답 처리
    response_data: Optional[Dict[str, Any]] = None # 응답 데이터 초기화
    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(timespec='milliseconds')
            response_status = response.status
            response_text_content = await response.text()

            # JSON 파싱 시도
            try:
                response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + ('...' if len(response_text_content) > 500 else '')
                log_data_error = {
                    "event_type": "api_response_error", "direction": "response", "api_type": "chat_completion",
                    "error_type": "decode_error", "model": model, "status_code": response_status,
                    "error_message": str(decode_e), "response_text_preview": response_text_preview,
                    "request_timestamp": request_timestamp, "response_timestamp": response_timestamp,
                    "failed_prompt_messages": payload.get('messages')
                }
                if api_logger: api_logger.error("Failed to decode JSON or unexpected content type from API response", extra=log_data_error)
                else: logging.error("Failed to decode API JSON response.")
                if close_session and session and not session.closed: await session.close()
                return None # 파싱 실패 시 None 반환

            # 응답 로그 데이터 준비
            # [수정] tool_calls 정보 포함
            response_choices = response_data.get("choices", [])
            first_choice_message = response_choices[0].get("message", {}) if response_choices else {}
            tool_calls = first_choice_message.get("tool_calls") # tool_calls 추출

            log_data_response = {
                "event_type": "api_response", "direction": "response", "api_type": "chat_completion",
                "model": model, "status_code": response_status,
                "response_info": {
                    "id": response_data.get("id"),
                    "choices_count": len(response_choices),
                    "usage": response_data.get("usage"),
                    "content_preview": first_choice_message.get("content", "")[:80] + "..." if first_choice_message.get("content") else "N/A (Check tool_calls)",
                    # [신규] Tool Calls 정보 로깅
                    "tool_calls_count": len(tool_calls) if tool_calls else 0,
                    "tool_calls_summary": [{"id": tc.get("id"), "function": tc.get("function", {}).get("name")} for tc in tool_calls] if tool_calls else "None",
                    "tool_calls_details": tool_calls # DEBUG 레벨에서만 포매터가 사용
                },
                "request_timestamp": request_timestamp, "response_timestamp": response_timestamp
            }

            # 응답 상태 코드에 따른 처리
            if response.status == 200:
                usage = response_data.get("usage", {})
                total_tokens = usage.get('total_tokens', 'N/A')
                log_message_resp = f"ChatCompletion success from {model}. Tokens: {total_tokens}"
                if tool_calls: log_message_resp += f". Received {len(tool_calls)} tool call(s)."

                if api_logger: api_logger.info(log_message_resp, extra=log_data_response)
                else: logging.info(log_message_resp)
                # [수정] 응답 데이터에 tool_calls가 포함될 수 있으므로 그대로 반환
                return response_data
            else:
                # API 레벨 에러 처리
                if response.status == 401: log_message_resp = f"ChatCompletion API Error (HTTP 401 Unauthorized). Check API key."
                elif response.status == 429: log_message_resp = f"ChatCompletion API Error (HTTP 429 Rate Limit Exceeded)."
                elif response.status == 400: log_message_resp = f"ChatCompletion API Error (HTTP 400 Bad Request). Check payload/parameters/tools."
                else: log_message_resp = f"ChatCompletion API Error (HTTP {response.status})"

                log_data_response["event_type"] = "api_response_error" # 에러 타입 명시
                log_data_response["error_details"] = response_data.get("error")
                if api_logger and api_logger.getEffectiveLevel() <= logging.DEBUG:
                    log_data_response["failed_prompt_messages"] = payload.get('messages')

                if api_logger: api_logger.error(log_message_resp, extra=log_data_response)
                else: logging.error(log_message_resp)
                return None # API 오류 시 None 반환

    except aiohttp.ClientError as e:
        # 네트워크 관련 오류
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "network_error",
            "model": model, "error_message": str(e), "request_timestamp": request_timestamp,
        }
        if api_logger and api_logger.getEffectiveLevel() <= logging.DEBUG:
             log_data_exception["failed_prompt_messages"] = payload.get('messages')
        if api_logger: api_logger.error(f"Network Error during ChatCompletion call: {e}", extra=log_data_exception)
        else: logging.error(f"Network Error: {e}")
        return None
    except asyncio.TimeoutError:
         # 타임아웃 오류
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "timeout_error",
            "model": model, "error_message": "Request timed out", "request_timestamp": request_timestamp,
        }
        if api_logger and api_logger.getEffectiveLevel() <= logging.DEBUG:
             log_data_exception["failed_prompt_messages"] = payload.get('messages')
        if api_logger: api_logger.error("ChatCompletion request timed out", extra=log_data_exception)
        else: logging.error("Request timed out")
        return None
    except Exception as e:
        # 기타 예상치 못한 오류
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "unexpected_error",
            "model": model, "error_message": str(e), "request_timestamp": request_timestamp,
        }
        if api_logger and api_logger.getEffectiveLevel() <= logging.DEBUG:
            log_data_exception["failed_prompt_messages"] = payload.get('messages')
        if api_logger: api_logger.error(f"Unexpected Error during ChatCompletion call: {e}", extra=log_data_exception, exc_info=True)
        else: logging.error(f"Unexpected Error: {e}")
        return None
    finally:
        # 세션을 직접 생성한 경우 닫기
        if close_session and session and not session.closed:
            await session.close()


# --- 비동기 OpenAI Embedding API 호출 함수 (변경 없음) ---
async def get_openai_embedding_async(
    text: str,
    session: Optional[aiohttp.ClientSession] = None,
    model: Optional[str] = None # 모델명 직접 지정 옵션 추가
) -> Optional[List[float]]:
    """
    비동기적으로 OpenAI Embedding API를 호출하여 텍스트의 임베딩 벡터를 반환합니다.
    (이 함수는 이전 코드와 동일하게 유지됩니다.)
    """
    global config # 전역 설정 사용

    if not api_logger: logging.error("API Logger not configured, cannot log Embedding call.")
    if not OPENAI_API_KEY:
        if api_logger: api_logger.error("OpenAI API Key missing for Embedding call.", extra={"event_type": "config_error", "api_type": "embedding"})
        else: logging.error("OpenAI API Key missing for Embedding.")
        return None
    if not text or not isinstance(text, str):
        if api_logger: api_logger.error("Invalid or empty text provided for embedding.", extra={"event_type":"param_error", "api_type": "embedding"})
        else: logging.error("Invalid text for embedding.")
        return None

    # 임베딩 모델 결정
    try:
        embedding_model = model # 인자로 받은 모델 우선 사용
        if not embedding_model:
            embedding_model = config.get('rag', {}).get('embedding_model') # config에서 읽기
        if not embedding_model:
            raise ValueError("Embedding model name not provided via argument or configuration.")
    except Exception as e:
        if api_logger: api_logger.error(f"Error determining embedding model: {e}", extra={"event_type":"config_error", "api_type": "embedding"})
        else: logging.error(f"Error determining embedding model: {e}")
        return None

    # API 요청 준비
    try:
        openai_url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"input": text, "model": embedding_model, "encoding_format": "float"} # encoding_format 명시
    except Exception as param_e:
        if api_logger: api_logger.error(f"Error setting up Embedding API parameters: {param_e}", exc_info=True, extra={"event_type":"param_error", "api_type": "embedding"})
        else: logging.error(f"Error setting up Embedding API params: {param_e}")
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
        try:
            session = aiohttp.ClientSession()
            close_session = True
        except Exception as session_e:
            if api_logger: api_logger.error(f"Failed to create new session for embedding: {session_e}", exc_info=True, extra={"event_type":"session_error", "api_type": "embedding"})
            else: logging.error(f"Failed to create session for embedding: {session_e}")
            return None
    elif session.closed:
        if api_logger: api_logger.error("Provided aiohttp session for embedding is closed.", extra={"event_type":"session_error", "api_type": "embedding"})
        else: logging.error("Provided session for embedding is closed.")
        return None

    # API 호출 및 응답 처리
    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(timespec='milliseconds')
            response_status = response.status
            response_data = {}
            response_text_content = await response.text()

            try:
                response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + ('...' if len(response_text_content) > 500 else '')
                log_data_error = {
                    "event_type": "api_response_error", "direction": "response", "api_type": "embedding",
                    "error_type": "decode_error", "model": embedding_model, "status_code": response_status,
                    "error_message": str(decode_e), "response_text_preview": response_text_preview,
                    "request_timestamp": request_timestamp, "response_timestamp": response_timestamp,
                    "failed_input_text": payload.get('input')
                }
                if api_logger: api_logger.error("Failed to decode JSON from embedding response", extra=log_data_error)
                else: logging.error("Failed to decode embedding JSON response.")
                if close_session and session and not session.closed: await session.close()
                return None

            log_data_response = {
                "event_type": "api_response", "direction": "response", "api_type": "embedding",
                "model": embedding_model, "status_code": response_status,
                "response_info": {
                    "embeddings_count": len(response_data.get("data", [])),
                    "usage": response_data.get("usage")
                },
                "request_timestamp": request_timestamp, "response_timestamp": response_timestamp
            }

            if response.status == 200 and response_data.get("data"):
                usage = response_data.get("usage", {})
                total_tokens = usage.get('total_tokens', 'N/A')
                log_message_emb = f"Embedding success from {embedding_model}. Tokens: {total_tokens}"
                if api_logger: api_logger.info(log_message_emb, extra=log_data_response)
                else: logging.info(log_message_emb)

                embedding_vector = response_data["data"][0].get("embedding")
                if embedding_vector and isinstance(embedding_vector, list):
                     # 차원 수 검증 (선택 사항)
                     # expected_dim = config.get('rag', {}).get('embedding_dimension')
                     # if expected_dim and len(embedding_vector) != expected_dim:
                     #     logger.warning(f"Embedding dimension mismatch! Expected {expected_dim}, Got {len(embedding_vector)}")
                    return embedding_vector
                else:
                    log_data_response["event_type"] = "api_response_error" # 에러 타입 명시
                    log_data_response["error_details"] = "Embedding vector not found or invalid format in response data"
                    if api_logger: api_logger.error("Embedding vector error in successful response", extra=log_data_response)
                    else: logging.error("Embedding vector error in response.")
                    return None
            else:
                # API 레벨 에러
                if response.status == 401: log_message_emb = f"Embedding API Error (HTTP 401 Unauthorized)."
                elif response.status == 429: log_message_emb = f"Embedding API Error (HTTP 429 Rate Limit)."
                else: log_message_emb = f"Embedding API Error (HTTP {response.status})"
                log_data_response["event_type"] = "api_response_error" # 에러 타입 명시
                log_data_response["error_details"] = response_data.get("error")
                log_data_response["failed_input_text"] = payload.get('input')
                if api_logger: api_logger.error(log_message_emb, extra=log_data_response)
                else: logging.error(log_message_emb)
                return None

    except aiohttp.ClientError as e:
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "embedding", "error_type": "network_error",
            "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp,
            "failed_input_text": payload.get('input')
        }
        if api_logger: api_logger.error(f"Network Error during Embedding call: {e}", extra=log_data_exception)
        else: logging.error(f"Network Error (Embedding): {e}")
        return None
    except asyncio.TimeoutError:
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "embedding", "error_type": "timeout_error",
            "model": embedding_model, "error_message": "Request timed out", "request_timestamp": request_timestamp,
            "failed_input_text": payload.get('input')
        }
        if api_logger: api_logger.error("Embedding request timed out", extra=log_data_exception)
        else: logging.error("Embedding request timed out")
        return None
    except Exception as e:
        log_data_exception = {
            "event_type": "api_call_error", "api_type": "embedding", "error_type": "unexpected_error",
            "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp,
            "failed_input_text": payload.get('input')
        }
        if api_logger: api_logger.error(f"Unexpected Error during Embedding call: {e}", extra=log_data_exception, exc_info=True)
        else: logging.error(f"Unexpected Error (Embedding): {e}")
        return None
    finally:
        if close_session and session and not session.closed:
            await session.close()


# --- 예시 사용법 (테스트용 - 변경 없음) ---
if __name__ == "__main__":
    # 로깅 레벨 강제 DEBUG (이미 위에서 설정됨)
    logging.info("--- Running gpt_interface.py as main script for testing ---")
    if not api_logger or api_logger.level > logging.DEBUG:
        logging.warning("API logger might not be set to DEBUG level for detailed test output.")

    async def test_apis():
        """테스트 API 호출 실행"""
        logging.info("Running test_apis() function...")
        if not config: # 설정 로드 실패 시 테스트 불가
            logging.error("Configuration not loaded, cannot run API tests.")
            return
        if not OPENAI_API_KEY:
            logging.error("CRITICAL ERROR: OPENAI_API_KEY is missing. Cannot run API call tests.")
            return

        # Chat Completion 테스트 (기본)
        test_messages = [{"role": "user", "content": "Say 'Hello GPT Interface Test!'"}]
        logging.info("\n--- Testing Basic Chat Completion API ---")
        try:
            test_model = config.get('testing', {}).get('default_baseline_model', 'gpt-3.5-turbo')
            test_temp = config.get('tasks', {}).get('tool_use', {}).get('generation_temperature', 0.7)
            test_max_tokens = config.get('tasks', {}).get('tool_use', {}).get('generation_max_tokens', 100)

            async with aiohttp.ClientSession() as session:
                logging.info(f"Calling chat API with model={test_model}, temp={test_temp}, max_tokens={test_max_tokens}")
                response = await call_gpt_async(
                    messages=test_messages, model=test_model,
                    temperature=test_temp, max_tokens=test_max_tokens, session=session
                )
                logging.info(f"Basic Chat API Call Result: {'Success' if response else 'Failed'}")
                if response: logging.info(f"Response Content Preview: {response.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
        except Exception as e:
            logging.error(f"Error during Basic Chat Completion test: {e}", exc_info=True)

        # [신규] Chat Completion 테스트 (Tool Use - 도구 미정의)
        logging.info("\n--- Testing Chat Completion API (No Tools) ---")
        test_messages_tool = [{"role": "user", "content": "나이키 러닝화 찾아줘."}]
        try:
            test_model_tool = config.get('tasks', {}).get('tool_use', {}).get('model', 'gpt-4o')
            test_temp_tool = config.get('tasks', {}).get('tool_use', {}).get('decision_temperature', 0.2)
            test_max_tokens_tool = config.get('tasks', {}).get('tool_use', {}).get('decision_max_tokens', 500)
            async with aiohttp.ClientSession() as session:
                logging.info(f"Calling chat API (no tools defined) with model={test_model_tool}")
                response_tool = await call_gpt_async(
                    messages=test_messages_tool, model=test_model_tool,
                    temperature=test_temp_tool, max_tokens=test_max_tokens_tool, session=session
                    # tools=None, tool_choice=None
                )
                logging.info(f"Chat API Call (No Tools) Result: {'Success' if response_tool else 'Failed'}")
                if response_tool:
                    logging.info(f"Response Content Preview: {response_tool.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
                    logging.info(f"Tool Calls: {response_tool.get('choices', [{}])[0].get('message', {}).get('tool_calls')}") # tool_calls 확인
        except Exception as e:
            logging.error(f"Error during Chat Completion (No Tools) test: {e}", exc_info=True)

        # Embedding API 테스트 (변경 없음)
        test_text = "Test embedding for gpt_interface.py"; logging.info("\n--- Testing Embedding API ---")
        try:
            emb_model = config.get('rag', {}).get('embedding_model')
            async with aiohttp.ClientSession() as session:
                logging.info(f"Calling embedding API with model={emb_model or 'Not specified'}")
                embedding = await get_openai_embedding_async(test_text, session=session, model=emb_model)
                logging.info(f"Embedding API Call Result: {'Success (Dim: ' + str(len(embedding)) + ')' if embedding else 'Failed'}")
                if embedding: logging.debug(f"Embedding vector preview: {embedding[:5]}...")
        except Exception as e:
            logging.error(f"Error during Embedding test: {e}", exc_info=True)

    # 비동기 테스트 실행
    try:
        import asyncio
        asyncio.run(test_apis())
    except Exception as e:
        logging.error(f"\nAn error occurred during test execution: {e}", exc_info=True)

    # 최종 로그 파일 경로 안내
    if 'log_file_path' in locals(): # 로거 설정 성공 시 변수 존재
        print(f"\nCheck detailed logs (including potential errors) in: {log_file_path}")
    else:
        print("\nCheck console output for logs (logger setup might have failed).")