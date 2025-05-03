# chatbot/summarizer.py (요구사항 반영 최종본: 증분 요약 로직 개선)

import json
import logging
from typing import List, Dict, Optional
import aiohttp

# --- 필요한 모듈 임포트 ---
try:
    # summarizer.py는 chatbot/chatbot/ 안에 있으므로, 상위 경로 추가 불필요
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    logging.info("gpt_interface and config_loader imported successfully in summarizer.")
except ImportError as ie:
    logging.error(f"ERROR (summarizer): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    call_gpt_async = None
    get_config = None

# --- 로거 설정 (기본 설정 상속 또는 명시적 설정) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 로거 레벨 직접 설정

# --- 대화 요약 함수 ---
async def summarize_conversation_async(
    history: List[Dict[str, str]],
    previous_summary: Optional[str] = None, # 이전 요약본 추가
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    주어진 대화 기록과 이전 요약(선택 사항)을 바탕으로 업데이트된 요약을 생성합니다.
    config.yaml 설정을 참조하여 모델, 파라미터, 프롬프트, 증분 업데이트 여부 등을 결정합니다.
    증분 업데이트 시에는 이전 요약과 최근 N턴의 대화 기록만 사용합니다.

    Args:
        history (List[Dict[str, str]]): 요약 대상이 될 전체 대화 기록 리스트.
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

        model = summarization_config.get('model')
        temperature = summarization_config.get('temperature')
        max_tokens = summarization_config.get('max_tokens')
        target_summary_tokens = summarization_config.get('target_summary_tokens', 100)
        update_incrementally = summarization_config.get('update_summary_incrementally', True)
        # [수정] 증분 업데이트 시 사용할 턴 수 읽기 (기본값 1 -> 최근 1턴(user+assist))
        summarize_every_n = summarization_config.get('summarize_every_n_turns', 1)
        # N값이 0 이하거나 너무 크면 기본값 사용 (예: 1)
        if not isinstance(summarize_every_n, int) or summarize_every_n <= 0:
            logger.warning(f"Invalid 'summarize_every_n_turns' value ({summarize_every_n}). Using default 1.")
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

    logger.info(f"Attempting to summarize conversation history (Update Incrementally: {update_incrementally})...")

    # --- [수정] 히스토리 및 이전 요약 준비 ---
    history_for_prompt_str = ""
    summary_for_prompt = "N/A" # 기본값

    if update_incrementally and previous_summary:
        logger.debug("Incremental update enabled and previous summary exists.")
        summary_for_prompt = previous_summary
        # 최근 N 턴에 해당하는 메시지 추출 (N * 2 개 메시지)
        num_recent_messages = summarize_every_n * 2
        recent_history = history[-num_recent_messages:]
        logger.debug(f"Extracting last {len(recent_history)} messages ({summarize_every_n} turns) for incremental summary.")

        formatted_recent_history_lines = []
        for turn in recent_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            formatted_recent_history_lines.append(f"{role}: {content}")
        history_for_prompt_str = "\n".join(formatted_recent_history_lines)

    else: # 증분 업데이트 아니거나 이전 요약 없을 경우
        if not update_incrementally: logger.debug("Incremental update disabled.")
        if not previous_summary: logger.debug("No previous summary provided.")
        logger.debug("Using full history for summarization.")

        formatted_full_history_lines = []
        for turn in history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            formatted_full_history_lines.append(f"{role}: {content}")
        history_for_prompt_str = "\n".join(formatted_full_history_lines)
        # summary_for_prompt는 "N/A" 유지

    if not history_for_prompt_str.strip():
        logger.warning("Formatted conversation history for prompt is empty.")
        # 이 경우 요약을 시도하는 의미가 없을 수 있으므로 None 반환 또는 다른 처리
        return None

    # --- 프롬프트 생성 ---
    try:
        # .format()으로 변수 주입
        prompt = prompt_template.format(
            conversation_history=history_for_prompt_str.strip(), # 포맷팅된 히스토리 전달
            previous_summary=summary_for_prompt,           # 이전 요약 또는 "N/A" 전달
            target_summary_tokens=target_summary_tokens
        )
        logger.debug(f"Summarization prompt created. Length: {len(prompt)} chars.")

    except KeyError as e:
        logger.error(f"Error formatting summarization prompt template. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return None

    messages = [{"role": "user", "content": prompt}]

    # --- GPT 호출하여 요약 생성 ---
    logger.debug(f"Calling GPT for summarization using model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
        )

        if response_data and response_data.get("choices"):
            summary_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            if summary_text:
                usage = response_data.get("usage", {})
                total_tokens = usage.get("total_tokens", "?")
                logger.info(f"Summarization successful. Generated summary length: {len(summary_text)} chars. Tokens used: {total_tokens}")
                logger.debug(f"Generated summary: {summary_text}")
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

# --- 예시 사용법 (기존과 동일, 변경된 로직 테스트 가능) ---
if __name__ == "__main__":
    import asyncio
    import os # os 임포트 추가 (getenv 사용)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running summarizer.py as main script for testing ---")

    async def test_summarization():
        """테스트 요약 실행 함수"""
        # 설정 로드 및 API 키 확인
        try:
            config = get_config()
            if not config: raise ValueError("Config not loaded")
            if not os.getenv("OPENAI_API_KEY"): raise ValueError("API Key not found")
            logger.info("Config and API Key seem available for summarizer test.")
        except Exception as e:
             logger.error(f"Prerequisites missing for test: {e}")
             return

        # 테스트용 대화 기록
        test_history_short = [
            {"role": "user", "content": "러닝화 추천해주세요."},
            {"role": "assistant", "content": "네, 어떤 종류의 러닝을 주로 하시나요?"},
            {"role": "user", "content": "주로 공원에서 가볍게 뛰어요."},
            {"role": "assistant", "content": "공원에서 가볍게 뛰신다면 쿠션이 좋은 데일리 러닝화를 추천드립니다. 킵런 KD500 모델은 어떠신가요?"}, # 2턴 (4개 메시지)
        ]
        test_history_long = [
            {"role": "user", "content": "안녕하세요, 데카트론 킵런 KS900 신발 사이즈 문의합니다."}, # 1
            {"role": "assistant", "content": "네, 고객님. 어떤 사이즈를 찾으시나요? 평소 신으시는 운동화 사이즈를 알려주시겠어요?"}, # 2
            {"role": "user", "content": "나이키 270mm 신는데, 발볼이 좀 넓은 편이에요."}, # 3
            {"role": "assistant", "content": "나이키 270mm 신으시고 발볼이 넓으시다면, 킵런 KS900은 270mm 또는 275mm를 고려해보실 수 있습니다."}, # 4
             {"role": "user", "content": "혹시 270 사이즈 강남점 재고 있나요?"}, # 5 (3번째 턴 시작)
             {"role": "assistant", "content": "실시간 재고 확인은 어렵습니다. 온라인 스토어나 매장 연락을 통해 확인 가능합니다."}, # 6
             {"role": "user", "content": "알겠습니다. 온라인으로 볼게요."}, # 7
             {"role": "assistant", "content": "네, 감사합니다."} # 8 (4번째 턴 끝)
        ]

        # config에서 summarize_every_n_turns 읽기 (없으면 기본값 1 사용)
        summarize_n = config.get('tasks', {}).get('summarization', {}).get('summarize_every_n_turns', 1)
        if not isinstance(summarize_n, int) or summarize_n <= 0: summarize_n = 1

        async with aiohttp.ClientSession() as session:
            print(f"\n--- Testing Summarization (Short History, Incremental=True, N={summarize_n}) ---")
            prev_summary1 = "사용자가 러닝화 추천을 요청함."
            summary1 = await summarize_conversation_async(test_history_short, previous_summary=prev_summary1, session=session)
            if summary1: print(f"Generated Summary 1:\n{summary1}")
            else: print("Summarization 1 failed.")
            print("-" * 30)

            print(f"\n--- Testing Summarization (Long History, Incremental=True, N={summarize_n}) ---")
            # 긴 히스토리의 마지막 N턴만 사용하게 됨
            # 예: N=1이면 마지막 2개 메시지, N=2면 마지막 4개 메시지 사용
            prev_summary2 = "고객은 킵런 KS900 사이즈(나이키 270mm, 발볼 넓음)를 문의했고, 270/275mm 추천받음."
            summary2 = await summarize_conversation_async(test_history_long, previous_summary=prev_summary2, session=session)
            if summary2: print(f"Generated Summary 2 (Incremental from N turns):\n{summary2}")
            else: print("Summarization 2 failed.")
            print("-" * 30)

            print(f"\n--- Testing Summarization (Long History, Incremental=False) ---")
            # 전체 히스토리 사용
            # update_summary_incrementally 값을 False로 임시 변경하여 테스트 필요 -> 여기서는 None 전달로 테스트
            summary3 = await summarize_conversation_async(test_history_long, previous_summary=None, session=session)
            if summary3: print(f"Generated Summary 3 (From Scratch):\n{summary3}")
            else: print("Summarization 3 failed.")
            print("-" * 30)

    # 비동기 테스트 실행
    try:
        asyncio.run(test_summarization())
    except Exception as e:
        logging.error(f"\nAn error occurred during summarizer testing: {e}", exc_info=True)