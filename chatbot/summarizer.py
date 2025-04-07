# chatbot/summarizer.py [신규]

import json
import logging
from typing import List, Dict, Optional
import aiohttp

# 필요한 모듈 임포트
try:
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
except ImportError as ie:
    print(f"ERROR (summarizer): Failed to import modules: {ie}. Check relative paths.")
    call_gpt_async = None
    get_config = None

# 로거 설정
logger = logging.getLogger(__name__)

# --- 대화 요약 함수 [신규] ---
async def summarize_conversation_async(
    history: List[Dict[str, str]],
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    주어진 대화 기록을 바탕으로 요약을 생성합니다.
    config.yaml의 설정을 참조하여 실행 여부, 모델, 파라미터, 프롬프트를 결정합니다.
    """
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in summarize_conversation_async.")
        return None

    # 설정 로드 및 확인
    try:
        config = get_config()
        summarization_config = config.get('tasks', {}).get('summarization', {})
        prompt_template = config.get('prompts', {}).get('summarization_prompt_template')

        is_enabled = summarization_config.get('enabled', False)
        model = summarization_config.get('model')
        temperature = summarization_config.get('temperature')
        max_tokens = summarization_config.get('max_tokens')
        # max_history_turns 설정 값 읽기 (기본값 설정)
        max_history_pairs = summarization_config.get('max_history_turns_for_summary', 4) # user+assistant 쌍 기준

        if not is_enabled:
            logger.debug("Summarization is disabled in config.")
            return None

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Summarization configuration missing or incomplete in config.yaml (tasks.summarization and prompts.summarization_prompt_template)")
            return None

        if not history:
            logger.debug("Conversation history is empty, cannot summarize.")
            return None

    except Exception as e:
        logger.error(f"Error loading summarization configuration: {e}", exc_info=True)
        return None

    # 실제 요약에 사용할 history 턴 수 계산 (user+assistant 쌍 기준이므로 *2)
    actual_history_length = len(history)
    history_turns_to_use = min(actual_history_length, max_history_pairs * 2)

    logger.info(f"Attempting to summarize conversation history (using last {history_turns_to_use} turns / {max_history_pairs} pairs).")

    # 요약 대상 히스토리 추출
    turns_to_summarize = history[-history_turns_to_use:]

    # 히스토리 포맷팅
    formatted_history = ""
    for turn in turns_to_summarize:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        # 간단한 줄바꿈으로 구분
        formatted_history += f"{role}: {content}\n"

    if not formatted_history.strip():
        logger.warning("Formatted history for summarization is empty.")
        return None

    # 프롬프트 생성
    try:
        prompt = prompt_template.format(conversation_history=formatted_history.strip())
    except KeyError as e:
        logger.error(f"Error formatting summarization prompt template. Missing key: {e}. Template: {prompt_template[:200]}...")
        return None

    messages = [{"role": "user", "content": prompt}] # 요약 요청은 보통 user 롤

    # GPT 호출하여 요약 생성
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
                logger.info("Summarization successful.")
                logger.debug(f"Generated summary: {summary_text}")
                return summary_text
            else:
                logger.warning("Summarization resulted in empty content.")
                return None
        else:
            logger.warning("Failed to get a valid response from GPT for summarization.")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during summarization: {e}", exc_info=True)
        return None

# --- 예시 사용법 ---
if __name__ == "__main__":
    import asyncio
    import os # os 모듈 임포트 추가
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_summarization():
        try:
            get_config()
            logger.info("Config loaded for summarizer test.")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return

        test_history = [
            {"role": "user", "content": "안녕하세요, 데카트론 킵런 KS900 신발 사이즈 문의합니다."},
            {"role": "assistant", "content": "네, 고객님. 어떤 사이즈를 찾으시나요? 평소 신으시는 운동화 사이즈를 알려주시겠어요?"},
            {"role": "user", "content": "나이키 270mm 신는데, 발볼이 좀 넓은 편이에요."},
            {"role": "assistant", "content": "나이키 270mm 신으시고 발볼이 넓으시다면, 킵런 KS900은 270mm 또는 275mm를 고려해보실 수 있습니다. 킵런 시리즈가 발볼이 여유있게 나온 편이지만, 개인차가 있을 수 있어 매장에서 직접 신어보시는 것을 가장 추천드립니다."},
            {"role": "user", "content": "알겠습니다. 혹시 오늘 강남점 재고 있을까요?"},
            {"role": "assistant", "content": "죄송하지만 실시간 재고 확인은 어렵습니다. 데카트론 온라인 스토어에서 매장별 재고 확인이 가능하시거나, 해당 매장으로 직접 연락해보시는 것이 가장 정확합니다."},
            {"role": "user", "content": "네, 온라인으로 확인해볼게요. 감사합니다."},
            {"role": "assistant", "content": "네, 감사합니다. 다른 문의사항 있으시면 언제든지 말씀해주세요."}
        ]

        print("\n--- Testing Summarization ---")
        logger.info("Running summarization test...")
        async with aiohttp.ClientSession() as session:
            summary = await summarize_conversation_async(test_history, session=session)

            if summary:
                print("\nGenerated Summary:")
                print(summary)
            else:
                print("\nSummarization failed or disabled.")
        print("-" * 30)

    try:
         # .env 파일 로드 확인 (API 키 필요)
        if not os.getenv("OPENAI_API_KEY"):
             print("Error: OPENAI_API_KEY not found. Cannot run tests that call the API.")
        else:
            asyncio.run(test_summarization())
    except FileNotFoundError:
        print("\nError: config.yaml not found.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")