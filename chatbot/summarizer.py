# chatbot/summarizer.py (요구사항 반영 최종본: 증분 요약, 토큰 목표 반영)

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
# logging.basicConfig(level=logging.DEBUG) # 필요 시 명시적 설정 가능
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

    Args:
        history (List[Dict[str, str]]): 요약 대상이 될 전체 대화 기록 리스트.
        previous_summary (Optional[str], optional): 이전 턴에서 생성된 요약.
                                                   None이거나 비어있으면 처음부터 요약합니다.
                                                   증분 업데이트 설정이 False면 무시됩니다.
                                                   Defaults to None.
        session (Optional[aiohttp.ClientSession], optional): API 호출에 사용할 aiohttp 세션.
                                                             없으면 새로 생성됩니다. Defaults to None.

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

        # is_enabled 체크는 호출하는 쪽(app.py)에서 하므로 여기서는 불필요
        model = summarization_config.get('model')
        temperature = summarization_config.get('temperature')
        max_tokens = summarization_config.get('max_tokens') # GPT 호출 시 사용
        target_summary_tokens = summarization_config.get('target_summary_tokens', 100) # 프롬프트 지침용
        update_incrementally = summarization_config.get('update_summary_incrementally', True) # 증분 업데이트 여부

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

    # --- 히스토리 포맷팅 ---
    # 항상 전체 히스토리를 사용 (필터링 로직 제거)
    formatted_history = ""
    for turn in history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        # 간단한 줄바꿈으로 구분 (템플릿에서 처리 방식에 따라 변경 가능)
        formatted_history += f"{role}: {content}\n"

    if not formatted_history.strip():
        logger.warning("Formatted conversation history is empty after processing.")
        return None

    # --- 프롬프트 생성 ---
    # 증분 업데이트 여부 및 이전 요약 존재 여부 확인
    current_previous_summary = previous_summary if update_incrementally and previous_summary else "N/A" # 이전 요약 없으면 "N/A" 전달

    try:
        # .format()으로 변수 주입
        prompt = prompt_template.format(
            conversation_history=formatted_history.strip(),
            previous_summary=current_previous_summary,
            target_summary_tokens=target_summary_tokens
        )
        logger.debug(f"Summarization prompt created. Length: {len(prompt)} chars.")
        # DEBUG 레벨일 때만 전체 프롬프트 로깅 (매우 길 수 있음 주의)
        # logger.debug(f"Full Summarization Prompt:\n------\n{prompt}\n------")

    except KeyError as e:
        logger.error(f"Error formatting summarization prompt template. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return None

    # 메시지 형식 구성 (보통 user 롤로 요약 요청)
    messages = [{"role": "user", "content": prompt}]

    # --- GPT 호출하여 요약 생성 ---
    logger.debug(f"Calling GPT for summarization using model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens, # config에서 읽은 max_tokens 사용
            session=session
            # response_format은 요약 결과가 자유 형식이므로 보통 설정 안 함
        )

        if response_data and response_data.get("choices"):
            summary_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            if summary_text:
                # 성공 로그에 토큰 사용량 포함 (있다면)
                usage = response_data.get("usage", {})
                total_tokens = usage.get("total_tokens", "?")
                logger.info(f"Summarization successful. Generated summary length: {len(summary_text)} chars. Tokens used: {total_tokens}")
                logger.debug(f"Generated summary: {summary_text}")
                return summary_text
            else:
                logger.warning("Summarization API call successful but resulted in empty content.")
                return None # 빈 내용이면 실패 처리
        else:
            # API 호출은 성공했으나 유효한 응답(choices)이 없는 경우
            logger.warning("Failed to get valid choices from GPT for summarization.")
            # call_gpt_async 내부에서 이미 에러 로그 기록되었을 것임
            return None

    except Exception as e:
        # API 호출 중 발생한 예외 (네트워크 오류, 타임아웃 등 포함)
        logger.error(f"An unexpected error occurred during summarization API call: {e}", exc_info=True)
        # call_gpt_async 내부에서 이미 에러 로그 기록되었을 것임
        return None

# --- 예시 사용법 ---
if __name__ == "__main__":
    # 로깅 레벨 강제 DEBUG (이미 위에서 설정됨)
    logging.info("--- Running summarizer.py as main script for testing ---")

    async def test_summarization():
        """테스트 요약 실행 함수"""
        if not config or not call_gpt_async:
            logging.error("Config or call_gpt_async not available. Cannot run test.")
            return
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY missing. Cannot run API test.")
            return

        # 테스트용 대화 기록
        test_history_short = [
            {"role": "user", "content": "러닝화 추천해주세요."},
            {"role": "assistant", "content": "네, 어떤 종류의 러닝을 주로 하시나요?"},
            {"role": "user", "content": "주로 공원에서 가볍게 뛰어요."},
        ]
        test_history_long = [
            {"role": "user", "content": "안녕하세요, 데카트론 킵런 KS900 신발 사이즈 문의합니다."},
            {"role": "assistant", "content": "네, 고객님. 어떤 사이즈를 찾으시나요? 평소 신으시는 운동화 사이즈를 알려주시겠어요?"},
            {"role": "user", "content": "나이키 270mm 신는데, 발볼이 좀 넓은 편이에요."},
            {"role": "assistant", "content": "나이키 270mm 신으시고 발볼이 넓으시다면, 킵런 KS900은 270mm 또는 275mm를 고려해보실 수 있습니다. 킵런 시리즈가 발볼이 여유있게 나온 편이지만, 개인차가 있을 수 있어 매장에서 직접 신어보시는 것을 가장 추천드립니다."},
            {"role": "user", "content": "알겠습니다. 혹시 오늘 강남점 재고 있을까요?"},
            {"role": "assistant", "content": "죄송하지만 실시간 재고 확인은 어렵습니다. 데카트론 온라인 스토어에서 매장별 재고 확인이 가능하시거나, 해당 매장으로 직접 연락해보시는 것이 가장 정확합니다."},
            {"role": "user", "content": "네, 온라인으로 확인해볼게요. 감사합니다."},
            {"role": "assistant", "content": "네, 감사합니다. 다른 문의사항 있으시면 언제든지 말씀해주세요."}
        ]

        async with aiohttp.ClientSession() as session:
            print("\n--- Testing Summarization (Short History, No Previous Summary) ---")
            summary1 = await summarize_conversation_async(test_history_short, session=session)
            if summary1: print(f"Generated Summary 1:\n{summary1}")
            else: print("Summarization 1 failed.")
            print("-" * 30)

            print("\n--- Testing Summarization (Long History, With Previous Summary - assuming update_incrementally=true) ---")
            # 이전 요약이 있다고 가정
            previous = "고객은 킵런 KS900 사이즈를 문의했으며, 나이키 270mm를 신고 발볼이 넓다고 함."
            summary2 = await summarize_conversation_async(test_history_long, previous_summary=previous, session=session)
            if summary2: print(f"Generated Summary 2 (Incremental):\n{summary2}")
            else: print("Summarization 2 failed.")
            print("-" * 30)

            print("\n--- Testing Summarization (Long History, No Previous Summary) ---")
            summary3 = await summarize_conversation_async(test_history_long, session=session)
            if summary3: print(f"Generated Summary 3 (From Scratch):\n{summary3}")
            else: print("Summarization 3 failed.")
            print("-" * 30)

    # 비동기 테스트 실행
    try:
        asyncio.run(test_summarization())
    except Exception as e:
        logging.error(f"\nAn error occurred during summarizer testing: {e}", exc_info=True)