# chatbot/gpt_interface.py (NameError 수정 및 o3 모델 토큰 파라미터 처리, max_output_tokens 인자명 통일)

import os
import json
import logging
import logging.handlers # 직접 사용 안 함
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import aiohttp
from dotenv import load_dotenv
import traceback
import time # Formatter에서 사용

# --- 설정 로더 임포트 ---
try:
    current_dir_gpt = os.path.dirname(os.path.abspath(__file__))
    project_root_gpt = os.path.dirname(current_dir_gpt)
    import sys
    if project_root_gpt not in sys.path:
        sys.path.insert(0, project_root_gpt)
    from chatbot.config_loader import get_config
    config = get_config()
    logging.info("Configuration loaded successfully in gpt_interface.")
except ImportError as ie:
    logging.error(f"CRITICAL ERROR (gpt_interface): Failed to import config_loader: {ie}.", exc_info=True)
    config = {}
except Exception as config_e:
    logging.error(f"CRITICAL ERROR (gpt_interface): Failed to load configuration: {config_e}", exc_info=True)
    config = {}

# --- .env 파일 로드 ---
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("CRITICAL ERROR (gpt_interface): OPENAI_API_KEY was not found in .env file or environment variables. API calls will fail.")
else:
    logging.info("OpenAI API Key found in environment.")

class ReadableTextFormatter(logging.Formatter):
    def __init__(self, fmt="[%(asctime)s] [%(levelname)s] - %(message)s", datefmt=None, style='%', validate=True, *, defaults=None):
        safe_datefmt = datefmt or config.get('logging', {}).get('log_timestamp_format', '%Y-%m-%d %H:%M:%S')
        super().__init__(fmt, safe_datefmt, style, validate, defaults=defaults)
        self.converter = time.localtime

    def formatTime(self, record, datefmt=None):
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
        record.asctime = self.formatTime(record, self.datefmt)
        log_string = super().format(record)

        if hasattr(record, 'event_type') and record.event_type.startswith("api_"):
            lines = ["=" * 80, log_string, "-" * 80]
            lines.append(f"Event Type : {getattr(record, 'event_type', 'N/A')}")
            lines.append(f"Direction  : {getattr(record, 'direction', 'N/A')}")
            lines.append(f"API Type   : {getattr(record, 'api_type', 'N/A')}")
            lines.append(f"Model      : {getattr(record, 'model', 'N/A')}")
            lines.append(f"Request TS : {getattr(record, 'request_timestamp', 'N/A')}")
            if hasattr(record, 'response_timestamp'): lines.append(f"Response TS: {record.response_timestamp}")
            if hasattr(record, 'status_code'): lines.append(f"Status Code: {record.status_code}")

            if hasattr(record, 'payload_info') and isinstance(record.payload_info, dict):
                lines.append("\n--- Request Payload Info ---")
                lines.append(f"  Messages Count: {record.payload_info.get('messages_count', 'N/A')}")
                if 'temperature' in record.payload_info: lines.append(f"  Temperature        : {record.payload_info['temperature']}")
                # [수정] max_output_tokens 관련 로깅 수정
                if 'actual_max_tokens_param_name' in record.payload_info and 'max_output_tokens_value' in record.payload_info:
                     lines.append(f"  {record.payload_info['actual_max_tokens_param_name'].ljust(20)} : {record.payload_info['max_output_tokens_value']}")
                elif 'max_output_tokens' in record.payload_info: # fallback, should not happen if actual_max_tokens_param_name is set
                     lines.append(f"  Max Output Tokens    : {record.payload_info['max_output_tokens']}")

                if 'input_length' in record.payload_info: lines.append(f"  Input Length         : {record.payload_info['input_length']}")
                if 'tools_count' in record.payload_info: lines.append(f"  Tools Count          : {record.payload_info['tools_count']}")
                if 'tool_choice' in record.payload_info: lines.append(f"  Tool Choice          : {record.payload_info['tool_choice']}")
                messages_full = getattr(record, 'messages_full', None)
                if messages_full:
                    lines.append("\n--- Full Prompt Messages ---")
                    try: lines.append(json.dumps(messages_full, indent=2, ensure_ascii=False))
                    except Exception: lines.append("  (Error formatting messages)")
                elif 'messages_preview' in record.payload_info:
                    lines.append(f"  Messages Preview: {record.payload_info['messages_preview']}")
                if 'additional_params' in record.payload_info and record.payload_info['additional_params']:
                    lines.append(f"  Additional Params    : {record.payload_info['additional_params']}")
                tools_full = getattr(record, 'tools_full', None)
                if tools_full:
                    lines.append("\n--- Full Tools Definition ---")
                    try: lines.append(json.dumps(tools_full, indent=2, ensure_ascii=False))
                    except Exception: lines.append("  (Error formatting tools definition)")

            if hasattr(record, 'response_info') and isinstance(record.response_info, dict):
                lines.append("\n--- Response Info ---")
                if 'id' in record.response_info: lines.append(f"  ID                 : {record.response_info['id']}")
                if 'choices_count' in record.response_info: lines.append(f"  Choices Count        : {record.response_info['choices_count']}")
                if 'embeddings_count' in record.response_info: lines.append(f"  Embeddings Count     : {record.response_info['embeddings_count']}")
                usage = record.response_info.get('usage')
                if isinstance(usage, dict):
                    lines.append(f"  Usage (P/C/T)        : {usage.get('prompt_tokens','?')} / {usage.get('completion_tokens','?')} / {usage.get('total_tokens','?')}")
                content_full = getattr(record, 'content_full', None)
                if content_full is not None:
                    lines.append("\n--- Full Response Content ---")
                    lines.append(str(content_full))
                elif 'content_preview' in record.response_info:
                    lines.append(f"  Content Preview      : {record.response_info['content_preview']}")
                if 'tool_calls_count' in record.response_info and record.response_info['tool_calls_count'] > 0:
                    lines.append(f"  Tool Calls Count     : {record.response_info['tool_calls_count']}")
                    if 'tool_calls_summary' in record.response_info: lines.append(f"  Tool Calls Summ      : {record.response_info['tool_calls_summary']}")
                    tool_calls_details = getattr(record, 'tool_calls_details', None)
                    if tool_calls_details:
                        lines.append("\n--- Full Tool Calls Details ---")
                        try: lines.append(json.dumps(tool_calls_details, indent=2, ensure_ascii=False))
                        except Exception: lines.append("  (Error formatting tool_calls details)")

            if record.event_type == "api_tool_call_response":
                if hasattr(record, 'tool_result_info') and isinstance(record.tool_result_info, dict):
                    lines.append("\n--- Tool Result Sent to LLM ---")
                    lines.append(f"  Tool Call ID: {record.tool_result_info.get('tool_call_id')}")
                    lines.append(f"  Function Name: {record.tool_result_info.get('function_name')}")
                    result_content_full = getattr(record, 'result_content_full', None)
                    if result_content_full is not None:
                        lines.append("\n--- Full Tool Result Content ---")
                        try:
                            parsed_content = json.loads(result_content_full)
                            lines.append(json.dumps(parsed_content, indent=2, ensure_ascii=False))
                        except (json.JSONDecodeError, TypeError): lines.append(str(result_content_full))
                        except Exception: lines.append("  (Error formatting tool result content)")
                    elif 'result_preview' in record.tool_result_info:
                        lines.append(f"  Result Preview: {record.tool_result_info.get('result_preview')}")

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
                lines.append("\n--- Error Message ---"); lines.append(f"  {record.error_message}")

            failed_prompt_data = getattr(record, 'failed_prompt_messages_full', None)
            failed_input_text = getattr(record, 'failed_input_text_full', None)
            if failed_prompt_data:
                lines.append("\n--- Failed Prompt ---")
                try: lines.append(json.dumps(failed_prompt_data, indent=2, ensure_ascii=False))
                except Exception: lines.append("  (Error formatting failed prompt)")
            elif failed_input_text:
                lines.append("\n--- Failed Input Text ---"); lines.append(f"  {failed_input_text}")
            lines.append("=" * 80 + "\n"); log_string = "\n".join(lines)
        elif record.exc_info:
            log_string += "\n" + self.formatException(record.exc_info)
        return log_string

api_logger: Optional[logging.Logger] = None
EXPECTED_EMBEDDING_DIM = config.get('rag', {}).get('embedding_dimension', 3072) if config else 3072
try:
    logger_config = config.get('logging', {})
    if not isinstance(logger_config, dict): raise ValueError("Invalid 'logging' configuration section.")
    log_file_base_name = logger_config.get('log_file_base_name', 'api_history')
    current_timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
    dynamic_log_filename = f"{log_file_base_name}_{current_timestamp_str}.txt"
    log_dir = os.path.join(project_root_gpt, 'logs'); os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, dynamic_log_filename)
    logging.info(f"API log file path set to: {log_file_path}")
    api_logger = logging.getLogger('api_logger'); api_logger.setLevel(logging.DEBUG); api_logger.propagate = False
    if not api_logger.handlers:
        logging.info(f"Configuring 'api_logger' (Name: {api_logger.name}) with DEBUG level...")
        log_timestamp_format = logger_config.get('log_timestamp_format', '%Y-%m-%d %H:%M:%S')
        text_formatter = ReadableTextFormatter(datefmt=log_timestamp_format)
        try:
            file_handler = logging.FileHandler(filename=log_file_path, encoding='utf-8')
            file_handler.setFormatter(text_formatter); file_handler.setLevel(logging.DEBUG)
            api_logger.addHandler(file_handler)
            logging.info(f"Added File Handler: Path='{log_file_path}', Level=DEBUG")
        except Exception as fh_e: logging.error(f"Failed to create File Handler: {fh_e}", exc_info=True)
        try:
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%H:%M:%S')
            console_handler = logging.StreamHandler(); console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG); api_logger.addHandler(console_handler)
            logging.info(f"Added Stream Handler (Console): Level=DEBUG")
        except Exception as ch_e: logging.error(f"Failed to create Stream Handler: {ch_e}", exc_info=True)
        if api_logger.hasHandlers(): api_logger.info(f"Logger '{api_logger.name}' initialized with DEBUG level. Check logs at: {log_file_path}")
        else: logging.error("CRITICAL: No handlers added to api_logger. Logging might not work.")
    else: logging.info(f"'api_logger' already configured. Skipping setup.")
except Exception as setup_e:
    logging.error(f"CRITICAL ERROR during logger setup: {setup_e}", exc_info=True)
    logging.basicConfig(level=logging.DEBUG)
    api_logger = logging.getLogger('api_logger_fallback')
    api_logger.error("Using basicConfig fallback logger due to setup error.")

async def call_gpt_async(
    messages: list[Dict[str, Any]], model: str, temperature: float,
    max_output_tokens: int, # [수정] 인자명 변경: max_tokens -> max_output_tokens
    session: Optional[aiohttp.ClientSession] = None, tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = None, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    if not api_logger: logging.error("API Logger not configured.")
    if not OPENAI_API_KEY:
        if api_logger: api_logger.error("OpenAI API Key missing.", extra={"event_type": "config_error"})
        else: logging.error("OpenAI API Key missing.")
        return None
    # [수정] max_output_tokens 유효성 검사
    if not model or not isinstance(temperature, (int, float)) or not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        if api_logger: api_logger.error(f"Invalid API params: model='{model}', temp={temperature}, max_output_tokens_arg={max_output_tokens}", extra={"event_type":"param_error"})
        else: logging.error("Invalid API params.")
        return None
    if not messages:
        if api_logger: api_logger.error("Messages list cannot be empty.", extra={"event_type": "param_error"})
        else: logging.error("Messages list cannot be empty.")
        return None
    try:
        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, **kwargs}

        actual_max_tokens_param_name = ""
        if model.lower().startswith("o3"):
            # [수정] o3 모델은 max_completion_tokens 사용
            payload["max_completion_tokens"] = max_output_tokens
            actual_max_tokens_param_name = "max_completion_tokens"
            if api_logger: api_logger.debug(f"Using '{actual_max_tokens_param_name}: {max_output_tokens}' for o3 model family.")
        else:
            # [수정] 그 외 모델은 max_tokens 사용
            payload["max_tokens"] = max_output_tokens
            actual_max_tokens_param_name = "max_tokens"
            if api_logger: api_logger.debug(f"Using '{actual_max_tokens_param_name}: {max_output_tokens}' for non-o3 model family.")

        if tools: payload["tools"] = tools
        if tool_choice: payload["tool_choice"] = tool_choice
    except Exception as param_e:
        if api_logger: api_logger.error(f"Error setting up API parameters: {param_e}", exc_info=True, extra={"event_type":"param_error"})
        else: logging.error(f"Error setting up API parameters: {param_e}")
        return None

    request_timestamp = datetime.now().isoformat(timespec='milliseconds')
    log_data_request = {
        "event_type": "api_call", "direction": "request", "api_type": "chat_completion", "model": model,
        "payload_info": {
            "messages_count": len(payload.get('messages', [])), "temperature": temperature,
            "actual_max_tokens_param_name": actual_max_tokens_param_name, # [수정] 로깅용 파라미터명
            "max_output_tokens_value": max_output_tokens, # [수정] 로깅용 파라미터 값
            "tools_count": len(payload.get('tools', [])) if tools else 0,
            "tool_choice": payload.get('tool_choice', 'Not Specified'),
            "messages_preview": f"System: {str(payload.get('messages', [{}])[0].get('content', ''))[:50]}... User: {str(payload.get('messages', [{}, {}])[-1].get('content', ''))[:50]}..." if len(payload.get('messages', [])) > 0 else "No messages",
            "additional_params": {k: v for k, v in kwargs.items()}
        },
        "request_timestamp": request_timestamp
    }
    log_data_request["messages_full"] = messages
    if tools: log_data_request["tools_full"] = tools
    # [수정] 로깅 메시지 수정
    log_message = f"Sending ChatCompletion request to {model} (Full details logged, using '{actual_max_tokens_param_name}' with value {max_output_tokens})"
    if api_logger: api_logger.info(log_message, extra=log_data_request)
    else: logging.info(log_message)

    close_session = False
    if session is None:
        if api_logger: api_logger.warning("aiohttp session not provided, creating new.", extra={"event_type": "session_warning"})
        else: logging.warning("aiohttp session not provided, creating new.")
        try: session = aiohttp.ClientSession(); close_session = True
        except Exception as session_e:
            if api_logger: api_logger.error(f"Failed to create new aiohttp session: {session_e}", exc_info=True, extra={"event_type":"session_error"})
            else: logging.error(f"Failed to create session: {session_e}")
            return None
    elif session.closed:
        if api_logger: api_logger.error("Provided aiohttp session is closed.", extra={"event_type":"session_error"})
        else: logging.error("Provided session is closed.")
        return None

    response_data: Optional[Dict[str, Any]] = None
    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(timespec='milliseconds')
            response_status = response.status
            response_text_content = await response.text()
            try: response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + ('...' if len(response_text_content) > 500 else '')
                log_data_error = {"event_type": "api_response_error", "direction": "response", "api_type": "chat_completion", "error_type": "decode_error", "model": model, "status_code": response_status, "error_message": str(decode_e), "response_text_preview": response_text_preview, "request_timestamp": request_timestamp, "response_timestamp": response_timestamp}
                log_data_error["failed_prompt_messages_full"] = messages
                if api_logger: api_logger.error("Failed to decode JSON API response", extra=log_data_error)
                else: logging.error("Failed to decode JSON API response")
                if close_session and session and not session.closed: await session.close()
                return None
            response_choices = response_data.get("choices", []); first_choice_message = response_choices[0].get("message", {}) if response_choices else {}; tool_calls = first_choice_message.get("tool_calls"); content = first_choice_message.get("content")
            log_data_response = {"event_type": "api_response", "direction": "response", "api_type": "chat_completion", "model": model, "status_code": response_status, "response_info": {"id": response_data.get("id"), "choices_count": len(response_choices), "usage": response_data.get("usage"), "content_preview": str(content)[:80] + "..." if content else "N/A (Check tool_calls)", "tool_calls_count": len(tool_calls) if tool_calls else 0, "tool_calls_summary": [{"id": tc.get("id"), "function": tc.get("function", {}).get("name")} for tc in tool_calls] if tool_calls else "None"}, "request_timestamp": request_timestamp, "response_timestamp": response_timestamp}
            log_data_response["content_full"] = content
            if tool_calls: log_data_response["tool_calls_details"] = tool_calls
            if response.status == 200:
                usage = response_data.get("usage", {}); total_tokens = usage.get('total_tokens', 'N/A')
                log_message_resp = f"ChatCompletion success from {model}. Tokens: {total_tokens}"
                if tool_calls: log_message_resp += f". Received {len(tool_calls)} tool call(s)."
                if api_logger: api_logger.info(log_message_resp, extra=log_data_response)
                else: logging.info(log_message_resp)
                return response_data
            else:
                if response.status == 401: log_message_resp = f"API Error (401 Unauthorized)"
                elif response.status == 429: log_message_resp = f"API Error (429 Rate Limit Exceeded)"
                elif response.status >= 500: log_message_resp = f"API Error (>=500 Server Error)"
                else: log_message_resp = f"API Error (HTTP {response.status})"
                log_data_response["event_type"] = "api_response_error"; log_data_response["error_details"] = response_data.get("error"); log_data_response["failed_prompt_messages_full"] = messages
                if api_logger: api_logger.error(log_message_resp, extra=log_data_response)
                else: logging.error(log_message_resp)
                return None
    except aiohttp.ClientError as e:
        log_data_exception = { "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "network_error", "model": model, "error_message": str(e), "request_timestamp": request_timestamp}; log_data_exception["failed_prompt_messages_full"] = messages
        if api_logger: api_logger.error(f"Network Error during ChatCompletion: {e}", extra=log_data_exception)
        else: logging.error(f"Network Error: {e}")
        return None
    except asyncio.TimeoutError:
        log_data_exception = { "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "timeout_error", "model": model, "error_message": "Request timed out", "request_timestamp": request_timestamp}; log_data_exception["failed_prompt_messages_full"] = messages
        if api_logger: api_logger.error("ChatCompletion request timed out", extra=log_data_exception)
        else: logging.error("Request timed out")
        return None
    except Exception as e:
        log_data_exception = { "event_type": "api_call_error", "api_type": "chat_completion", "error_type": "unexpected_error", "model": model, "error_message": str(e), "request_timestamp": request_timestamp}; log_data_exception["failed_prompt_messages_full"] = messages
        if api_logger: api_logger.error(f"Unexpected Error during ChatCompletion: {e}", extra=log_data_exception, exc_info=True)
        else: logging.error(f"Unexpected Error: {e}")
        return None
    finally:
        if close_session and session and not session.closed: await session.close()

async def get_openai_embedding_async(text: str, session: Optional[aiohttp.ClientSession] = None, model: Optional[str] = None) -> Optional[List[float]]:
    global config, api_logger, OPENAI_API_KEY, EXPECTED_EMBEDDING_DIM
    if not api_logger: logging.error("API Logger not configured for Embedding.")
    if not OPENAI_API_KEY:
        if api_logger: api_logger.error("OpenAI API Key missing for Embedding.", extra={"event_type": "config_error", "api_type": "embedding"})
        else: logging.error("OpenAI API Key missing.")
        return None
    if not text or not isinstance(text, str):
        if api_logger: api_logger.error("Invalid text for embedding.", extra={"event_type":"param_error", "api_type": "embedding"})
        else: logging.error("Invalid text for embedding.")
        return None
    try:
        rag_cfg = config.get('rag', {}) if config else {}
        embedding_model = model or rag_cfg.get('embedding_model', 'text-embedding-3-large')
        if not embedding_model: raise ValueError("Embedding model name missing.")
    except Exception as e:
        if api_logger: api_logger.error(f"Error determining embedding model: {e}", extra={"event_type":"config_error", "api_type": "embedding"})
        else: logging.error(f"Error determining embedding model: {e}")
        return None
    try:
        openai_url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        input_text_processed = text.replace("\n", " ")
        payload = {"input": input_text_processed, "model": embedding_model, "encoding_format": "float"}
    except Exception as param_e:
        if api_logger: api_logger.error(f"Error setting up Embedding params: {param_e}", exc_info=True, extra={"event_type":"param_error", "api_type": "embedding"})
        else: logging.error(f"Error setting up Embedding params: {param_e}")
        return None
    request_timestamp = datetime.now().isoformat(timespec='milliseconds')
    log_data_request = {"event_type": "api_call", "direction": "request", "api_type": "embedding", "model": embedding_model, "payload_info": {"input_length": len(payload.get('input', '')) }, "request_timestamp": request_timestamp}
    log_data_request["input_text_full"] = payload.get('input')
    log_message_emb = f"Sending Embedding request to {embedding_model} (Full details logged)"
    if api_logger: api_logger.info(log_message_emb, extra=log_data_request)
    else: logging.info(log_message_emb)
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
    try:
        async with session.post(openai_url, headers=headers, json=payload) as response:
            response_timestamp = datetime.now().isoformat(timespec='milliseconds'); response_status = response.status; response_text_content = await response.text()
            try: response_data = json.loads(response_text_content)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as decode_e:
                response_text_preview = response_text_content[:500] + ('...' if len(response_text_content) > 500 else '')
                log_data_error = {"event_type": "api_response_error", "direction": "response", "api_type": "embedding", "error_type": "decode_error", "model": embedding_model, "status_code": response_status, "error_message": str(decode_e), "response_text_preview": response_text_preview, "request_timestamp": request_timestamp, "response_timestamp": response_timestamp}
                log_data_error["failed_input_text_full"] = payload.get('input')
                if api_logger: api_logger.error("Failed to decode JSON embedding response", extra=log_data_error)
                else: logging.error("Failed to decode embedding JSON response.")
                if close_session and session and not session.closed: await session.close()
                return None
            log_data_response = {"event_type": "api_response", "direction": "response", "api_type": "embedding", "model": embedding_model, "status_code": response_status, "response_info": {"embeddings_count": len(response_data.get("data", [])), "usage": response_data.get("usage")}, "request_timestamp": request_timestamp, "response_timestamp": response_timestamp}
            if response.status == 200 and response_data.get("data"):
                usage = response_data.get("usage", {}); total_tokens = usage.get('total_tokens', 'N/A')
                log_message_emb_resp = f"Embedding success from {embedding_model}. Tokens: {total_tokens}"
                if api_logger: api_logger.info(log_message_emb_resp, extra=log_data_response)
                else: logging.info(log_message_emb_resp)
                embedding_vector = response_data["data"][0].get("embedding")
                if embedding_vector and isinstance(embedding_vector, list):
                    if EXPECTED_EMBEDDING_DIM is not None and len(embedding_vector) != EXPECTED_EMBEDDING_DIM:
                        logger.error(f"Embedding dimension mismatch! Expected {EXPECTED_EMBEDDING_DIM}, got {len(embedding_vector)}")
                        return None
                    return embedding_vector
                else:
                    log_data_response["event_type"] = "api_response_error"; log_data_response["error_details"] = "Embedding vector error in success response"
                    if api_logger: api_logger.error("Embedding vector error", extra=log_data_response)
                    else: logging.error("Embedding vector error.")
                    return None
            else:
                if response.status == 401: log_message_emb_resp = f"Embedding API Error (401)"
                elif response.status == 429: log_message_emb_resp = f"Embedding API Error (429 Rate Limit)"
                elif response.status >= 500: log_message_emb_resp = f"Embedding API Error (>=500 Server Error)"
                else: log_message_emb_resp = f"Embedding API Error ({response.status})"
                log_data_response["event_type"] = "api_response_error"; log_data_response["error_details"] = response_data.get("error"); log_data_response["failed_input_text_full"] = payload.get('input')
                if api_logger: api_logger.error(log_message_emb_resp, extra=log_data_response)
                else: logging.error(log_message_emb_resp)
                return None
    except aiohttp.ClientError as e:
        log_data_exception = { "event_type": "api_call_error", "api_type": "embedding", "error_type": "network_error", "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp}; log_data_exception["failed_input_text_full"] = payload.get('input')
        if api_logger: api_logger.error(f"Network Error (Embedding): {e}", extra=log_data_exception)
        else: logging.error(f"Network Error (Embedding): {e}")
        return None
    except asyncio.TimeoutError:
        log_data_exception = { "event_type": "api_call_error", "api_type": "embedding", "error_type": "timeout_error", "model": embedding_model, "error_message": "Request timed out", "request_timestamp": request_timestamp}; log_data_exception["failed_input_text_full"] = payload.get('input')
        if api_logger: api_logger.error("Embedding request timed out", extra=log_data_exception)
        else: logging.error("Embedding request timed out")
        return None
    except Exception as e:
        log_data_exception = { "event_type": "api_call_error", "api_type": "embedding", "error_type": "unexpected_error", "model": embedding_model, "error_message": str(e), "request_timestamp": request_timestamp}; log_data_exception["failed_input_text_full"] = payload.get('input')
        if api_logger: api_logger.error(f"Unexpected Error (Embedding): {e}", extra=log_data_exception, exc_info=True)
        else: logging.error(f"Unexpected Error (Embedding): {e}")
        return None
    finally:
        if close_session and session and not session.closed: await session.close()

if __name__ == "__main__":
    if api_logger:
        api_logger.setLevel(logging.DEBUG)
        for handler in api_logger.handlers: handler.setLevel(logging.DEBUG)
        logging.info("--- Set api_logger level to DEBUG for testing ---")
    else:
        logging.basicConfig(level=logging.DEBUG)
        logging.info("--- Using basicConfig (DEBUG) for testing ---")

    logging.info("--- Running gpt_interface.py as main script for testing (max_output_tokens unified) ---")

    async def test_apis():
        logging.info("Running test_apis() function...")
        global config, OPENAI_API_KEY
        if not config: logging.error("Config not loaded, cannot run tests."); return
        if not OPENAI_API_KEY: logging.error("API Key missing."); return

        test_messages_basic = [{"role": "system", "content": "You are a test bot."}, {"role": "user", "content": "Say 'Hello GPT Interface Test!'"}]
        logging.info("\n--- Testing Basic Chat Completion API (non-o3 model) ---")
        try:
            test_model_basic = config.get('testing', {}).get('default_baseline_model', 'gpt-3.5-turbo')
            # [수정] max_output_tokens 사용
            max_tokens_for_basic = 150 # 예시 값, 실제로는 config 등에서 관리
            async with aiohttp.ClientSession() as session:
                response = await call_gpt_async(
                    messages=test_messages_basic,
                    model=test_model_basic,
                    temperature=0.7,
                    max_output_tokens=max_tokens_for_basic, # [수정] 인자명 변경
                    session=session
                )
                logging.info(f"Basic Chat (non-o3) Result: {'Success' if response else 'Failed'}")
                if response: logging.info(f"Content: {response.get('choices', [{}])[0].get('message', {}).get('content')}")
        except Exception as e: logging.error(f"Error in Basic Chat (non-o3) test: {e}", exc_info=True)

        test_messages_o3_mini = [{"role": "system", "content": "You are o3-mini."}, {"role": "user", "content": "Tell me a short joke using o3-mini."}]
        logging.info("\n--- Testing Chat Completion API (o3-mini model) ---")
        try:
            slot_config = config.get('tasks', {}).get('slot_extraction', {})
            model_o3_mini = slot_config.get('model', 'o3-mini')
            temp_o3_mini = slot_config.get('temperature', 0.5)
            # [수정] max_output_tokens 사용 (config의 max_completion_tokens 값을 읽어옴)
            max_output_o3_mini = slot_config.get('max_completion_tokens', 200)
            async with aiohttp.ClientSession() as session:
                response_o3_mini = await call_gpt_async(
                    messages=test_messages_o3_mini,
                    model=model_o3_mini,
                    temperature=temp_o3_mini,
                    max_output_tokens=max_output_o3_mini, # [수정] 인자명 변경
                    session=session
                )
                logging.info(f"Chat (o3-mini) Result: {'Success' if response_o3_mini else 'Failed'}")
                if response_o3_mini: logging.info(f"Content: {response_o3_mini.get('choices', [{}])[0].get('message', {}).get('content')}")
        except Exception as e: logging.error(f"Error in Chat (o3-mini) test: {e}", exc_info=True)

        logging.info("\n--- Testing Chat Completion API (With Tools Definition, o3 model) ---")
        test_messages_tool_o3 = [{"role": "system", "content": "You can use tools with o3."}, {"role": "user", "content": "What is the weather in Seoul today? Use a tool if you have one."}]
        dummy_tool = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
        try:
            tool_use_cfg = config.get('tasks', {}).get('tool_use', {})
            model_o3_tool = tool_use_cfg.get('model', 'o3')
            temp_o3_tool = tool_use_cfg.get('decision_temperature', 0.2)
            # [수정] max_output_tokens 사용 (config의 decision_max_completion_tokens 값을 읽어옴)
            max_output_o3_tool = tool_use_cfg.get('decision_max_completion_tokens', 500)
            async with aiohttp.ClientSession() as session:
                response_tool_o3 = await call_gpt_async(
                    messages=test_messages_tool_o3,
                    model=model_o3_tool,
                    temperature=temp_o3_tool,
                    max_output_tokens=max_output_o3_tool, # [수정] 인자명 변경
                    session=session,
                    tools=dummy_tool,
                    tool_choice="auto"
                )
                logging.info(f"Chat API Call (With Tools Def, o3) Result: {'Success' if response_tool_o3 else 'Failed'}")
                if response_tool_o3:
                    logging.info(f"Content: {response_tool_o3.get('choices', [{}])[0].get('message', {}).get('content')}")
                    logging.info(f"Tool Calls: {response_tool_o3.get('choices', [{}])[0].get('message', {}).get('tool_calls')}")
        except Exception as e: logging.error(f"Error during Chat (With Tools Def, o3) test: {e}", exc_info=True)

        test_text = "Test embedding for gpt_interface.py detailed logging"; logging.info("\n--- Testing Embedding API ---")
        try:
            emb_model = config.get('rag', {}).get('embedding_model')
            async with aiohttp.ClientSession() as session:
                embedding = await get_openai_embedding_async(test_text, session=session, model=emb_model)
                logging.info(f"Embedding API Result: {'Success (Dim: ' + str(len(embedding)) + ')' if embedding else 'Failed'}")
                if embedding: logging.debug(f"Embedding vector preview: {embedding[:5]}...")
        except Exception as e: logging.error(f"Error during Embedding test: {e}", exc_info=True)

    try: asyncio.run(test_apis())
    except Exception as e: logging.error(f"Test execution error: {e}", exc_info=True)

    if 'log_file_path' in locals() and log_file_path:
        print(f"\nCheck detailed logs in: {log_file_path}")
    else: print("\nLog file path not determined. Check logger setup.")