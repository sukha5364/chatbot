# chatbot/summarizer.py (o3-mini 모델 및 토큰 파라미터 처리, "brief" 모드 강화, gpt_interface.py 호출부 수정 반영)

import json
import logging
from typing import List, Dict, Optional, Any
import aiohttp

# --- 필요한 모듈 임포트 ---
try:
    # summarizer.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    logging.info("gpt_interface and config_loader imported successfully in summarizer.")
except ImportError as ie:
    logging.error(f"ERROR (summarizer): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 기능 사용 불가 처리
    call_gpt_async = None
    get_config = None

# --- 로거 설정 (기본 설정 상속) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

# --- [기존] 히스토리 포맷팅 함수 (변경 없음) ---
def format_history_for_prompt(history_list: List[Dict[str, Any]]) -> str:
    """주어진 대화 기록 리스트를 LLM 프롬프트에 적합한 문자열로 포맷합니다."""
    lines = []
    for turn in history_list:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content")
        # Tool 관련 정보는 요약 프롬프트에서 제외 (별도 처리 예정)
        if role == 'Tool': continue # Tool 결과는 여기서 제외
        if role == 'Assistant' and turn.get('tool_calls'):
            # Tool 호출한 Assistant 메시지는 간단히 표시하거나 내용만 표시
            content_str = "(Tool call requested)" if not content else str(content)
        else:
            content_str = str(content) if content is not None else ""

        # 너무 길거나 불필요한 content 필터링 가능 (선택 사항)
        if content_str: # 내용이 있을 때만 추가
            lines.append(f"{role}: {content_str}")
    return "\n".join(lines).strip()

# --- [수정됨] 대화 요약 함수 (gpt_interface.py 호출부 수정) ---
async def summarize_conversation_async(
    history: List[Dict[str, Any]], # history 타입 Any 허용
    previous_summary: Optional[str] = None, # 이전 요약본
    session: Optional[aiohttp.ClientSession] = None,
    # --- 추가된 파라미터 ---
    current_slots: Optional[Dict[str, Any]] = None,
    include_slots: bool = False, # config에서 읽은 값 전달받음
    include_tool_results: str = "none" # config에서 읽은 값 전달받음 ("none", "brief", "full")
) -> Optional[str]:
    """
    주어진 대화 기록, 이전 요약, 현재 슬롯, Tool 결과를 바탕으로 업데이트된 요약을 생성합니다.
    config.yaml 설정을 참조하여 모델, 파라미터, 프롬프트, 증분 업데이트 여부 등을 결정합니다.
    "brief" 모드 시 Tool 결과 요약에 첫 번째 제품명을 포함하도록 수정되었습니다.

    Args:
        history (List[Dict[str, Any]]): 요약 대상이 될 전체 대화 기록 리스트.
        previous_summary (Optional[str], optional): 이전 턴에서 생성된 요약.
        session (Optional[aiohttp.ClientSession], optional): API 호출에 사용할 aiohttp 세션.
        current_slots (Optional[Dict[str, Any]], optional): 현재 추출된 슬롯 정보.
        include_slots (bool): 요약 프롬프트에 슬롯 정보를 포함할지 여부.
        include_tool_results (str): Tool 결과를 요약 프롬프트에 포함할 방식 ("none", "brief", "full").

    Returns:
        Optional[str]: 생성된 요약 텍스트. 오류 발생 시 None.
    """
    # --- 필수 모듈 및 설정 로드 확인 (기존과 동일) ---
    if not call_gpt_async or not get_config:
        logger.error("Required modules (gpt_interface, config_loader) not available in summarize_conversation_async.")
        return None
    try:
        config_data = get_config() # 변수명 변경 config -> config_data
        if not config_data: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
        logger.error(f"Failed to get configuration in summarizer: {conf_e}", exc_info=True)
        return None

    # --- 요약 설정 로드 및 확인 ---
    try:
        summarization_config = config_data.get('tasks', {}).get('summarization', {})
        prompt_template = config_data.get('prompts', {}).get('summarization_prompt_template')

        enabled = summarization_config.get('enabled', False)
        if not enabled:
            logger.debug("Summarization is disabled in config. Skipping summarization.")
            return None

        model = summarization_config.get('model')
        temperature = summarization_config.get('temperature')
        # [수정] config.yaml에서 max_completion_tokens 읽기 (o3 모델용)
        max_output_tokens_val = summarization_config.get('max_completion_tokens') # 변수명 변경 및 값 할당
        target_summary_tokens = summarization_config.get('target_summary_tokens', 500)
        update_incrementally = summarization_config.get('update_summary_incrementally', True)
        summarize_every_n = summarization_config.get('summarize_every_n_turns', 1)
        if not isinstance(summarize_every_n, int) or summarize_every_n <= 0:
            logger.warning(f"Invalid 'summarize_every_n_turns' ({summarize_every_n}). Using 1.")
            summarize_every_n = 1

        # [수정] 설정값 누락 검사 시 max_output_tokens_val 확인
        if not all([model, isinstance(temperature, (int, float)), isinstance(max_output_tokens_val, int), prompt_template]):
            logger.error(f"Summarization configuration missing or incomplete in config.yaml. Needed: model, temperature, max_completion_tokens, prompt_template. Found: model={model}, temp={temperature}, max_output_tokens(config:max_completion_tokens)={max_output_tokens_val}")
            return None

        if not history:
            logger.debug("Conversation history is empty, cannot summarize.")
            return None

    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing summarization configuration: {e}", exc_info=True)
        return None

    # --- 입력 데이터 준비 (슬롯, Tool 결과 포함) ---
    logger.info(f"Attempting to summarize conversation (Include Slots: {include_slots}, Include Tools: {include_tool_results})...")

    history_for_prompt_str = ""
    summary_for_prompt = previous_summary if previous_summary else "N/A"
    history_to_process_for_tools = history

    slot_info_str = ""
    if include_slots and current_slots:
        try:
            filtered_slots = {k: v for k, v in current_slots.items() if v is not None}
            if filtered_slots:
                slot_info_str = f"[현재 슬롯 정보]:\n{json.dumps(filtered_slots, indent=2, ensure_ascii=False)}\n"
                logger.debug("Formatted slot information for summary prompt.")
        except Exception as e:
            logger.warning(f"Failed to format slots for summary prompt: {e}")

    if update_incrementally and previous_summary:
        logger.debug("Incremental update mode: Using previous summary and recent history for context.")
        num_recent_messages = summarize_every_n * 2 
        recent_history = history[-num_recent_messages:]
        history_for_prompt_str = format_history_for_prompt(recent_history)
        history_to_process_for_tools = recent_history
        logger.debug(f"Using last {len(recent_history)} messages (approximating {summarize_every_n} turns) for incremental summary.")
    else:
        logger.debug("Full history mode (or no previous summary): Using entire history for context.")
        history_for_prompt_str = format_history_for_prompt(history)

    tool_info_str = ""
    if include_tool_results in ["brief", "full"] and history_to_process_for_tools:
        tool_summaries = []
        processed_tool_call_ids = set()
        max_tool_summaries = 3 
        logger.debug(f"Extracting up to {max_tool_summaries} recent tool results (Mode: {include_tool_results})...")

        for msg in reversed(history_to_process_for_tools):
            if len(tool_summaries) >= max_tool_summaries: break

            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id or tool_call_id in processed_tool_call_ids: continue

                tool_name = msg.get("name", "?")
                tool_content_str = msg.get("content", "{}")
                summary_line = ""
                try:
                    tool_content_obj = json.loads(tool_content_str)
                    results_found = tool_content_obj.get("results_found", False)
                    error_msg = tool_content_obj.get("error")
                    results = tool_content_obj.get("results", [])
                    results_count = len(results)

                    if error_msg:
                        summary_line = f"- Tool '{tool_name}' 실패: {error_msg}"
                    elif include_tool_results == "brief":
                        if results_found and results_count > 0:
                            first_result_name = results[0].get('product_name', '알 수 없음')
                            summary_line = f"- Tool '{tool_name}' 실행: {results_count}개 찾음 (예: {first_result_name})"
                        elif results_found and results_count == 0:
                            summary_line = f"- Tool '{tool_name}' 실행: 결과 없음 (성공: True)"
                        else: 
                            summary_line = f"- Tool '{tool_name}' 실행: 결과 없음 (성공: False)"
                    elif include_tool_results == "full":
                        if results_found and results_count > 0:
                            names = [r.get('product_name', '?') for r in results[:2]]
                            summary_line = f"- Tool '{tool_name}' 실행: {results_count}개 찾음 (결과: {', '.join(names)}{'...' if results_count > 2 else ''})"
                        else:
                            summary_line = f"- Tool '{tool_name}' 실행: 결과 없음 (성공: {results_found})"
                    else:
                        continue

                    if summary_line:
                        tool_summaries.append(summary_line)
                        processed_tool_call_ids.add(tool_call_id)

                except json.JSONDecodeError:
                    logger.warning(f"Could not parse tool content for {tool_call_id} in summary: {tool_content_str[:100]}...")
                except Exception as e:
                    logger.warning(f"Error processing tool message for summary (ID: {tool_call_id}): {e}")

        if tool_summaries:
            tool_info_str = "[최근 Tool 실행 요약]:\n" + "\n".join(reversed(tool_summaries)) + "\n"
            logger.debug("Formatted tool result summary for prompt (names included in brief).")


    if not history_for_prompt_str.strip() and not slot_info_str and not tool_info_str:
        logger.warning("Formatted conversation history and additional context (slots/tools) are empty. Cannot summarize.")
        return None

    # --- 프롬프트 생성 (기존과 동일) ---
    try:
        prompt = prompt_template.format(
            slot_information=slot_info_str,
            tool_result_summary=tool_info_str,
            conversation_history=history_for_prompt_str,
            previous_summary=summary_for_prompt,
            target_summary_tokens=target_summary_tokens
        )
        logger.debug(f"Summarization prompt created (Context Enhanced). Length: {len(prompt)} chars.")
    except KeyError as e:
        logger.error(f"Error formatting summarization prompt template. Missing key: {e}. Check placeholders in config.yaml and code. Template: {prompt_template[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return None

    messages = [{"role": "user", "content": prompt}]

    # --- GPT 호출하여 요약 생성 ---
    # [수정] 로깅 메시지에 max_output_tokens_val 사용
    logger.debug(f"Calling GPT for summarization (model: {model}, temp: {temperature}, max_output_tokens: {max_output_tokens_val})")
    try:
        # [수정] call_gpt_async 호출 시 max_output_tokens 인자 사용
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens_val, # [수정] 인자명 변경
            session=session
        )

        if response_data and response_data.get("choices"):
            summary_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            if summary_text:
                usage = response_data.get("usage", {})
                total_tokens = usage.get("total_tokens", "?")
                logger.info(f"Summarization successful. Generated summary length: {len(summary_text)} chars. Tokens: {total_tokens}")
                logger.debug(f"Generated summary preview: {summary_text[:150]}...")
                return summary_text
            else:
                logger.warning("Summarization API call successful but resulted in empty content.")
                return None
        else:
            logger.warning("Failed to get valid choices from GPT for summarization.")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during summarization API call: {e}", exc_info=True)
        return None

# --- 예시 사용법 (테스트 코드 변경 없음 - 이미 수정된 시그니처 사용) ---
if __name__ == "__main__":
    import asyncio
    import os # os 임포트 추가
    import time # time 임포트 추가 (테스트용)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running summarizer.py as main script for testing (gpt_interface call updated, brief with names) ---")

    async def test_summarization_enhanced():
        """테스트 요약 실행 함수 (수정된 시그니처 반영)"""
        try:
            config_data_test = get_config(); assert config_data_test # 변수명 변경 및 사용
            assert os.getenv("OPENAI_API_KEY") # os.getenv 사용
            logger.info("Config and API Key ready for enhanced summarizer test.")
        except Exception as e:
            logger.error(f"Prerequisites missing for test: {e}"); return

        test_history = [
            {"role": "user", "content": "발볼 넓은 남성용 런닝화 추천해주세요. 가격은 10만원대로요."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "product_search", "arguments": "{\"search_keywords\": \"발볼 넓은 남성용 런닝화 10만원대\", \"filters\": {\"logic\": \"AND\", \"conditions\": [{\"field\": \"target_audience\", \"operator\": \"==\", \"value\": \"남성\"}, {\"field\": \"price_numeric\", \"operator\": \"<=\", \"value\": 199999}]}}"}}]},
            {"role": "tool", "tool_call_id": "call_abc", "name": "product_search", "content": json.dumps({"results": [{"product_name": "킵런 KS500 2", "brand": "Kiprun", "price": "99000원", "features": ["fit:넓은 발볼", "review_good:편안함"]}], "results_found": True})},
            {"role": "assistant", "content": "발볼이 넓으시다면 킵런 KS500 2 모델을 추천합니다. 99,000원이고 사용자 리뷰에서도 편안하다는 평이 많습니다."},
            {"role": "user", "content": "그거 말고 다른 건 없나요? 킵런 브랜드 말고 다른 걸로요."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_def", "type": "function", "function": {"name": "product_search", "arguments": "{\"search_keywords\": \"발볼 넓은 남성용 런닝화 10만원대 (킵런 제외)\", \"filters\": {\"logic\": \"AND\", \"conditions\": [{\"field\": \"target_audience\", \"operator\": \"==\", \"value\": \"남성\"}, {\"field\": \"price_numeric\", \"operator\": \"<=\", \"value\": 199999}, {\"field\": \"brand\", \"operator\": \"!=\", \"value\": \"Kiprun\"}]}}"}}]},
            {"role": "tool", "tool_call_id": "call_def", "name": "product_search", "content": json.dumps({"results_found": False})}
        ]
        test_previous_summary = "사용자가 런닝화를 찾기 시작함."
        test_slots = {"product_category": "런닝화", "user_preference": ["발볼 넓음", "10만원대"], "target_audience": "남성"}

        summarization_cfg_test = config_data_test.get('tasks', {}).get('summarization', {}) # config_data_test 사용
        inc_slots_test = summarization_cfg_test.get('include_slots_in_summary_prompt', False) # 변수명 _test 추가
        inc_tools_brief_test = "brief"
        inc_tools_full_test = "full"

        async with aiohttp.ClientSession() as session:
            print(f"\n--- Testing Summarization (Slots: {inc_slots_test}, Tools: {inc_tools_brief_test}) ---")
            start_t_brief = time.time() # time.time() 사용
            summary_brief = await summarize_conversation_async(
                history=test_history,
                previous_summary=test_previous_summary,
                session=session,
                current_slots=test_slots,
                include_slots=inc_slots_test,
                include_tool_results=inc_tools_brief_test
            )
            dur_t_brief = time.time() - start_t_brief # time.time() 사용
            print(f"(Took {dur_t_brief:.3f}s)")
            if summary_brief: print(f"Generated Summary (Brief Tools):\n{summary_brief}")
            else: print("Summarization (Brief Tools) failed.")
            print("-" * 30)

            print(f"\n--- Testing Summarization (Slots: {inc_slots_test}, Tools: {inc_tools_full_test}) ---")
            start_t_full = time.time() # time.time() 사용
            summary_full = await summarize_conversation_async(
                history=test_history, previous_summary=test_previous_summary, session=session,
                current_slots=test_slots, include_slots=inc_slots_test, include_tool_results=inc_tools_full_test
            )
            dur_t_full = time.time() - start_t_full # time.time() 사용
            print(f"(Took {dur_t_full:.3f}s)")
            if summary_full: print(f"Generated Summary (Full Tools):\n{summary_full}")
            else: print("Summarization (Full Tools) failed.")
            print("-" * 30)

    try:
        asyncio.run(test_summarization_enhanced())
    except Exception as e:
        logging.error(f"\nSummarizer test error: {e}", exc_info=True)