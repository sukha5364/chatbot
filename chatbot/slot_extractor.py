# chatbot/slot_extractor.py (요구사항 반영 최종본: 주석 보강 및 JSON 파싱 개선)

import json
import logging
from typing import Dict, Any, Optional
import aiohttp

# --- 필요한 모듈 임포트 ---
try:
    # slot_extractor.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    logging.info("gpt_interface and config_loader imported successfully in slot_extractor.")
except ImportError as ie:
    logging.error(f"ERROR (slot_extractor): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 기능 사용 불가
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
                                                             없으면 새로 생성됩니다. Defaults to None.

    Returns:
        Optional[Dict[str, Any]]: 추출된 Slot 정보를 담은 딕셔너리. Slot 이름이 키, 추출된 값이 값.
                                  추출 실패 또는 오류 발생 시 None을 반환합니다.
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

        # 필수 설정값 확인
        model = task_config.get('model')
        temperature = task_config.get('temperature')
        max_tokens = task_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Slot extraction configuration missing or incomplete in config.yaml. Check tasks.slot_extraction and prompts.slot_extraction_prompt_template")
            return None # 설정 오류 시 None 반환
    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing slot extraction configuration: {e}", exc_info=True)
        return None # 설정 오류 시 None 반환

    # 입력값 로깅
    logger.info(f"Attempting to extract slots from input: '{user_input[:70]}...'")

    # 프롬프트 포맷팅
    try:
        # 템플릿에 사용자 입력 주입
        prompt = prompt_template.format(user_input=user_input)
        logger.debug("Slot extraction prompt formatted successfully.")
    except KeyError as e:
        logger.error(f"Error formatting slot extraction prompt template. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None # 프롬프트 포맷팅 오류
    except Exception as e:
         logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
         return None

    # API 요청 메시지 생성
    messages = [{"role": "user", "content": prompt}]

    # GPT 호출하여 Slot 추출 시도
    logger.debug(f"Calling GPT for slot extraction using model: {model}, temp: {temperature}, max_tokens: {max_tokens}, requesting JSON object.")
    try:
        # call_gpt_async 호출 시 response_format 명시하여 JSON 모드 요청
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
            # 응답 내용 추출
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            raw_response_preview = response_content[:300] + ('...' if len(response_content) > 300 else '')
            logger.debug(f"Raw response content from slot extractor GPT: {raw_response_preview}")

            # 1차: 직접 JSON 파싱 시도 (JSON 모드가 성공했을 경우)
            try:
                extracted_slots = json.loads(response_content)
                logger.info(f"Successfully extracted slots (direct JSON parsing): {list(extracted_slots.keys())}")
                logger.debug(f"Extracted slot values: {extracted_slots}")
                return extracted_slots # 성공 시 결과 반환
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {e}. Trying fallback parsing...")

                # 2차: Fallback 파싱 (코드 블록 제거 등 시도)
                clean_response_content = response_content.strip()
                # 마크다운 코드 블록 제거
                if clean_response_content.startswith("```json"):
                    clean_response_content = clean_response_content[7:-3].strip()
                elif clean_response_content.startswith("```"):
                     clean_response_content = clean_response_content[3:-3].strip()

                # 중괄호 기준으로 문자열 찾기 (가장 바깥쪽)
                try:
                    json_start = clean_response_content.find('{')
                    json_end = clean_response_content.rfind('}')
                    if json_start != -1 and json_end != -1 and json_start < json_end:
                        json_string = clean_response_content[json_start:json_end+1]
                    else:
                        # 중괄호를 못 찾거나 순서가 이상하면 원본 사용 (파싱 실패 예상)
                        json_string = clean_response_content

                    extracted_slots = json.loads(json_string)
                    logger.info(f"Successfully extracted slots (fallback parsing): {list(extracted_slots.keys())}")
                    logger.debug(f"Extracted slot values: {extracted_slots}")
                    return extracted_slots # Fallback 성공 시 결과 반환
                except json.JSONDecodeError as fallback_e:
                    # Fallback 파싱도 실패
                    logger.error(f"Fallback JSON parsing also failed: {fallback_e}. Giving up on slot extraction for this input. Cleaned content preview: '{clean_response_content[:200]}...'")
                    return None # 최종 실패 시 None 반환
                except Exception as fallback_parse_e: # 예상치 못한 파싱 오류
                    logger.error(f"Unexpected error during fallback JSON parsing: {fallback_parse_e}", exc_info=True)
                    return None
        else:
            # API 호출은 성공했으나 유효한 응답(choices)이 없는 경우
            logger.warning("Failed to get valid response/choices from GPT for slot extraction.")
            # call_gpt_async 내부에서 관련 에러 로그 기록됨
            return None # 실패 시 None 반환

    except Exception as e:
        # call_gpt_async 호출 자체 또는 네트워크 오류 등
        logger.error(f"An unexpected error occurred during slot extraction API call: {e}", exc_info=True)
        # call_gpt_async 내부에서 관련 에러 로그 기록됨
        return None # 실패 시 None 반환

# --- 예시 사용법 (기존 유지) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # __name__으로 로거 다시 가져오기
    logger.info("--- Running slot_extractor.py as main script for testing ---")

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
            "그냥 구경왔어요." # Slot 없는 경우 테스트
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