# chatbot/slot_extractor.py (최종 수정 계획 확인 - 로직 변경 없음)

import json
import logging
from typing import Dict, Any, Optional
import aiohttp
import re # JSON 파싱 개선 위해 re 임포트

# --- 필요한 모듈 임포트 ---
try:
    # slot_extractor.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    logging.info("gpt_interface and config_loader imported successfully in slot_extractor.")
except ImportError as ie:
    logging.error(f"ERROR (slot_extractor): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 기능 사용 불가 처리
    call_gpt_async = None
    get_config = None

# --- 로거 설정 (기본 설정 상속) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

# --- Slot 추출 함수 ---
async def extract_slots_with_gpt(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    GPT를 사용하여 주어진 사용자 입력 텍스트에서 미리 정의된 Slot 정보를 추출합니다.
    config.yaml 파일에서 Slot 추출에 사용할 모델, 파라미터, 프롬프트 템플릿 설정을 로드합니다.
    GPT 모델에는 JSON 형식의 응답을 요청합니다.

    Args:
        user_input (str): Slot을 추출할 사용자 입력 문자열.
        session (Optional[aiohttp.ClientSession], optional): API 호출에 사용할 aiohttp 세션.

    Returns:
        Optional[Dict[str, Any]]: 추출된 Slot 정보를 담은 딕셔너리. 추출 실패 시 None.
    """
    # 필수 모듈 및 설정 로드 확인
    if not call_gpt_async or not get_config:
        logger.error("Required modules (gpt_interface, config_loader) not available in extract_slots_with_gpt.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
        logger.error(f"Failed to get configuration in extract_slots_with_gpt: {conf_e}", exc_info=True)
        return None

    # Slot 추출 관련 설정 읽기
    try:
        task_config = config.get('tasks', {}).get('slot_extraction', {})
        prompt_template = config.get('prompts', {}).get('slot_extraction_prompt_template')

        model = task_config.get('model')
        temperature = task_config.get('temperature')
        max_tokens = task_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Slot extraction configuration missing or incomplete in config.yaml.")
            return None
    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing slot extraction configuration: {e}", exc_info=True)
        return None

    # 입력이 너무 짧으면 슬롯 추출 시도하지 않음 (선택적 최적화)
    if not user_input or len(user_input.strip()) < 5:
         logger.debug(f"Input too short ('{user_input}'), skipping slot extraction.")
         return {} # 빈 딕셔너리 반환 (실패가 아니라 추출할 슬롯 없음)

    logger.info(f"Attempting to extract slots from input: '{user_input[:70]}...'")

    # 프롬프트 포맷팅
    try:
        prompt = prompt_template.format(user_input=user_input)
        logger.debug("Slot extraction prompt formatted successfully.")
    except KeyError as e:
        logger.error(f"Error formatting slot extraction prompt template. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return None

    messages = [{"role": "user", "content": prompt}]

    logger.debug(f"Calling GPT for slot extraction using model: {model}, temp: {temperature}, max_tokens: {max_tokens}, requesting JSON object.")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session,
            response_format={"type": "json_object"} # JSON 모드 요청
        )

        # --- 응답 처리 및 JSON 파싱 ---
        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            raw_response_preview = response_content[:300] + ('...' if len(response_content) > 300 else '')
            logger.debug(f"Raw response content from slot extractor GPT: {raw_response_preview}")

            # 1차: 직접 JSON 파싱 시도
            try:
                # 추가: 응답이 비어있는 경우 처리
                if not response_content.strip():
                     logger.info("Slot extractor GPT returned empty content. Assuming no slots extracted.")
                     return {} # 빈 딕셔너리 반환

                extracted_slots = json.loads(response_content)
                # 반환값이 dict인지 확인
                if not isinstance(extracted_slots, dict):
                     logger.warning(f"Slot extractor returned non-dict JSON: {type(extracted_slots)}. Treating as empty.")
                     return {}
                logger.info(f"Successfully extracted slots (direct JSON parsing): {list(extracted_slots.keys())}")
                logger.debug(f"Extracted slot values: {extracted_slots}")
                return extracted_slots
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {e}. Trying fallback parsing...")

                # 2차: Fallback 파싱 (코드 블록 제거 등 시도)
                try:
                    # 코드 블록 제거 (```json ... ``` 또는 ``` ... ```)
                    clean_response_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_content.strip(), flags=re.MULTILINE)
                    # 가장 바깥쪽 중괄호 찾기 (정규식 개선)
                    json_match = re.search(r'^\s*(\{.*?\})\s*$', clean_response_content, re.DOTALL)
                    if json_match:
                        json_string = json_match.group(1)
                    else: # 중괄호 없거나 형식이 다르면 그냥 시도
                        json_string = clean_response_content

                    # 파싱 전 빈 문자열 체크
                    if not json_string.strip():
                         logger.info("Fallback parsing: content became empty after cleaning. Assuming no slots.")
                         return {}

                    extracted_slots = json.loads(json_string)
                    if not isinstance(extracted_slots, dict):
                         logger.warning(f"Slot extractor (fallback) returned non-dict JSON: {type(extracted_slots)}. Treating as empty.")
                         return {}
                    logger.info(f"Successfully extracted slots (fallback parsing): {list(extracted_slots.keys())}")
                    logger.debug(f"Extracted slot values: {extracted_slots}")
                    return extracted_slots
                except json.JSONDecodeError as fallback_e:
                    logger.error(f"Fallback JSON parsing also failed: {fallback_e}. Giving up on slot extraction. Cleaned content preview: '{clean_response_content[:200]}...'")
                    return None # 파싱 완전 실패 시 None 반환
                except Exception as fallback_parse_e:
                    logger.error(f"Unexpected error during fallback JSON parsing: {fallback_parse_e}", exc_info=True)
                    return None
        else:
            logger.warning("Failed to get valid response/choices from GPT for slot extraction.")
            return None # API 호출 자체가 실패했거나 choices 없는 경우

    except Exception as e:
        logger.error(f"An unexpected error occurred during slot extraction API call: {e}", exc_info=True)
        return None # 예외 발생 시 None 반환

# --- 예시 사용법 (기존 유지) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # logger 재정의 필요 없음
    logger.info("--- Running slot_extractor.py as main script for testing ---")

    # asyncio 및 aiohttp 임포트 (테스트 실행용)
    import asyncio
    import aiohttp
    import os # getenv 사용 위해
    import time # time 임포트 추가

    async def test_slot_extraction():
        """Slot 추출 기능 테스트 실행"""
        try:
            get_config() # 설정 로드 가능한지 확인
            logger.info("Configuration loaded successfully for slot extraction test.")
        except Exception as e:
            logger.error(f"Failed to load configuration for test: {e}. Cannot run test.", exc_info=True)
            return
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY missing. Cannot run API dependent tests.")
            return

        test_inputs = [
            "30대 남자인데, 주말에 가볍게 등산할 때 신을 발볼 넓은 트레킹화 15만원 이하로 추천해주세요. 초보입니다.",
            "캠핑 가서 쓸 2인용 텐트 보고 있는데, 퀘차 제품 방수 잘 되나요?",
            "지난번에 산 킵런 운동화 왼쪽 발 뒤꿈치가 아픈데, 사이즈 문제일까요? 사이즈는 275mm 신어요.",
            "여자친구 선물로 러닝할 때 입을 M사이즈 기능성 티셔츠 보고 있어요.",
            "그냥 구경왔어요.", # Slot 없는 경우 테스트
            "안녕" # 매우 짧은 입력 테스트
        ]
        async with aiohttp.ClientSession() as session:
            for i, test_input in enumerate(test_inputs):
                print(f"\n--- Testing Slot Extraction for Input #{i+1} --- \n'{test_input}'")
                logger.info(f"Running test extraction for: '{test_input}'")
                try:
                    start_t = time.time()
                    slots = await extract_slots_with_gpt(test_input, session=session)
                    duration_t = time.time() - start_t
                    print(f"(Took {duration_t:.3f}s)")
                    if slots is not None: # None이 아닌 경우 (성공 또는 빈 dict)
                        print("\nExtraction Result:")
                        # ensure_ascii=False 로 한국어 깨짐 방지
                        print(json.dumps(slots, indent=2, ensure_ascii=False))
                    else: # None인 경우 (실패)
                        print("\nExtraction Failed (Returned None). Check logs for details.")
                except Exception as test_e:
                    logger.error(f"Error during test execution for input '{test_input}': {test_e}", exc_info=True)
                    print(f"\nERROR during test: {test_e}")
                print("-" * 40)

    # 비동기 테스트 실행
    try:
        asyncio.run(test_slot_extraction())
    except Exception as e:
        logger.critical(f"\nAn critical error occurred during testing: {e}", exc_info=True)