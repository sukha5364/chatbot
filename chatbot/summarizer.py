# chatbot/summarizer.py (최종 수정 계획 확인 - 로직 변경 없음)

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

# --- 대화 요약 함수 ---
async def summarize_conversation_async(
    history: List[Dict[str, Any]], # history 타입 Any 허용 (tool_calls 등 포함 가능성)
    previous_summary: Optional[str] = None, # 이전 요약본
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    주어진 대화 기록과 이전 요약(선택 사항)을 바탕으로 업데이트된 요약을 생성합니다.
    config.yaml 설정을 참조하여 모델, 파라미터, 프롬프트, 증분 업데이트 여부 등을 결정합니다.
    증분 업데이트 시에는 이전 요약과 최근 N턴의 대화 기록만 사용합니다.

    Args:
        history (List[Dict[str, Any]]): 요약 대상이 될 전체 대화 기록 리스트.
        previous_summary (Optional[str], optional): 이전 턴에서 생성된 요약.
        session (Optional[aiohttp.ClientSession], optional): API 호출에 사용할 aiohttp 세션.

    Returns:
        Optional[str]: 생성된 요약 텍스트. 오류 발생 시 None.
    """
    # 필수 모듈 및 설정 로드 확인
    if not call_gpt_async or not get_config:
        logger.error("Required modules (gpt_interface, config_loader) not available in summarize_conversation_async.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
        logger.error(f"Failed to get configuration in summarizer: {conf_e}", exc_info=True)
        return None

    # 요약 설정 로드 및 확인
    try:
        summarization_config = config.get('tasks', {}).get('summarization', {})
        prompt_template = config.get('prompts', {}).get('summarization_prompt_template')

        # 요약 기능 활성화 여부 확인 (app.py에서도 체크하지만 여기서도 확인 가능)
        enabled = summarization_config.get('enabled', False)
        if not enabled:
             logger.debug("Summarization is disabled in config. Skipping summarization.")
             return None # 이전 요약 유지 또는 None 반환 (호출 측에서 처리)

        model = summarization_config.get('model')
        temperature = summarization_config.get('temperature')
        max_tokens = summarization_config.get('max_tokens')
        target_summary_tokens = summarization_config.get('target_summary_tokens', 500)
        update_incrementally = summarization_config.get('update_summary_incrementally', True)
        summarize_every_n = summarization_config.get('summarize_every_n_turns', 1)
        if not isinstance(summarize_every_n, int) or summarize_every_n <= 0:
            logger.warning(f"Invalid 'summarize_every_n_turns' ({summarize_every_n}). Using 1.")
            summarize_every_n = 1

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Summarization configuration missing or incomplete in config.yaml.")
            return None

        if not history:
            logger.debug("Conversation history is empty, cannot summarize.")
            return None # 빈 히스토리면 요약 불가

    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing summarization configuration: {e}", exc_info=True)
        return None

    logger.info(f"Attempting to summarize conversation (Incremental: {update_incrementally})...")

    # --- 히스토리 및 이전 요약 준비 ---
    history_for_prompt_str = ""
    summary_for_prompt = previous_summary if previous_summary else "N/A" # 이전 요약 없으면 "N/A"

    # 히스토리 포맷팅 함수 (가독성 위해 분리)
    def format_history_for_prompt(history_list: List[Dict[str, Any]]) -> str:
        lines = []
        for turn in history_list:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content")
            # Tool 관련 정보는 요약 프롬프트에서 제외 (필요시 포함 가능)
            if role == 'Tool': continue # Tool 결과는 제외
            if role == 'Assistant' and turn.get('tool_calls'):
                 # Tool 호출한 Assistant 메시지는 간단히 표시하거나 내용만 표시
                 content_str = "(Tool call requested)" if not content else str(content)
            else:
                 content_str = str(content) if content is not None else ""

            lines.append(f"{role}: {content_str}")
        return "\n".join(lines).strip()

    if update_incrementally and previous_summary:
        logger.debug("Incremental update mode: Using previous summary and recent history.")
        # 최근 N 턴(user+assistant = 2*N 메시지) 추출
        num_recent_messages = summarize_every_n * 2
        recent_history = history[-num_recent_messages:]
        history_for_prompt_str = format_history_for_prompt(recent_history)
        logger.debug(f"Using last {len(recent_history)} messages ({summarize_every_n} turns) for incremental summary.")
    else: # 증분 업데이트 아니거나 이전 요약 없음
        logger.debug("Full history mode (or no previous summary): Using entire history.")
        history_for_prompt_str = format_history_for_prompt(history)

    if not history_for_prompt_str.strip():
        logger.warning("Formatted conversation history for prompt is empty. Cannot summarize.")
        return None

    # --- 프롬프트 생성 ---
    try:
        prompt = prompt_template.format(
            conversation_history=history_for_prompt_str,
            previous_summary=summary_for_prompt,
            target_summary_tokens=target_summary_tokens
        )
        logger.debug(f"Summarization prompt created. Length: {len(prompt)} chars.")
    except KeyError as e:
        logger.error(f"Error formatting summarization prompt template. Missing key: {e}. Template: {prompt_template[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return None

    messages = [{"role": "user", "content": prompt}]

    # --- GPT 호출하여 요약 생성 ---
    logger.debug(f"Calling GPT for summarization (model: {model}, temp: {temperature}, max_tokens: {max_tokens})")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
            # 요약 시에는 JSON 모드 불필요
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
                return None # 빈 내용이면 실패 처리
        else:
            logger.warning("Failed to get valid choices from GPT for summarization.")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during summarization API call: {e}", exc_info=True)
        return None

# --- 예시 사용법 (변경 없음) ---
if __name__ == "__main__":
    import asyncio
    import os # os 임포트 추가
    import time # time 임포트 추가

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running summarizer.py as main script for testing ---")

    async def test_summarization():
        """테스트 요약 실행 함수"""
        try:
            config = get_config(); assert config
            assert os.getenv("OPENAI_API_KEY")
            logger.info("Config and API Key ready for summarizer test.")
        except Exception as e:
            logger.error(f"Prerequisites missing for test: {e}"); return

        test_history_short = [ # 이전 예시 사용
             {"role": "user", "content": "러닝화 추천해주세요."},
             {"role": "assistant", "content": "네, 어떤 종류의 러닝을 주로 하시나요?"},
             {"role": "user", "content": "주로 공원에서 가볍게 뛰어요."},
             {"role": "assistant", "content": "공원에서 가볍게 뛰신다면 쿠션이 좋은 데일리 러닝화를 추천드립니다. 킵런 KD500 모델은 어떠신가요?"},
         ]
        test_history_long = [ # 이전 예시 사용
             {"role": "user", "content": "안녕하세요, 데카트론 킵런 KS900 신발 사이즈 문의합니다."},
             {"role": "assistant", "content": "네, 고객님. 어떤 사이즈를 찾으시나요? 평소 신으시는 운동화 사이즈를 알려주시겠어요?"},
             {"role": "user", "content": "나이키 270mm 신는데, 발볼이 좀 넓은 편이에요."},
             {"role": "assistant", "content": "나이키 270mm 신으시고 발볼이 넓으시다면, 킵런 KS900은 270mm 또는 275mm를 고려해보실 수 있습니다."},
             # Tool 호출/결과 추가 (예시)
             {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type":"function", "function": {"name": "product_search", "arguments": "{...}"}}]},
             {"role": "tool", "tool_call_id": "t1", "name": "product_search", "content": json.dumps({"results_found": False})},
             {"role": "assistant", "content": "죄송하지만 현재 KS900 관련 상세 정보 조회가 어렵습니다. 온라인몰 확인 부탁드립니다."},
             {"role": "user", "content": "알겠습니다."}
         ]

        summarize_n = config.get('tasks', {}).get('summarization', {}).get('summarize_every_n_turns', 1)
        update_inc = config.get('tasks', {}).get('summarization', {}).get('update_summary_incrementally', True)

        async with aiohttp.ClientSession() as session:
             print(f"\n--- Testing Summarization (Short History, Incremental={update_inc}, N={summarize_n}) ---")
             prev_summary1 = "사용자가 러닝화 추천을 요청함."
             start_t1 = time.time()
             summary1 = await summarize_conversation_async(test_history_short, previous_summary=prev_summary1, session=session)
             dur_t1 = time.time() - start_t1
             print(f"(Took {dur_t1:.3f}s)")
             if summary1: print(f"Generated Summary 1:\n{summary1}")
             else: print("Summarization 1 failed.")
             print("-" * 30)

             print(f"\n--- Testing Summarization (Long History, Incremental={update_inc}, N={summarize_n}) ---")
             prev_summary2 = "고객은 킵런 KS900 사이즈(나이키 270mm, 발볼 넓음)를 문의했고, 270/275mm 추천받음."
             start_t2 = time.time()
             summary2 = await summarize_conversation_async(test_history_long, previous_summary=prev_summary2, session=session)
             dur_t2 = time.time() - start_t2
             print(f"(Took {dur_t2:.3f}s)")
             if summary2: print(f"Generated Summary 2 (Incremental):\n{summary2}")
             else: print("Summarization 2 failed.")
             print("-" * 30)

             print(f"\n--- Testing Summarization (Long History, From Scratch) ---")
             start_t3 = time.time()
             summary3 = await summarize_conversation_async(test_history_long, previous_summary=None, session=session)
             dur_t3 = time.time() - start_t3
             print(f"(Took {dur_t3:.3f}s)")
             if summary3: print(f"Generated Summary 3 (From Scratch):\n{summary3}")
             else: print("Summarization 3 failed.")
             print("-" * 30)

    try:
        asyncio.run(test_summarization())
    except Exception as e:
        logging.error(f"\nSummarizer test error: {e}", exc_info=True)