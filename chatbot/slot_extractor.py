# chatbot/slot_extractor.py (오류 수정 및 리팩토링 최종본)

import json
from typing import Dict, Any, Optional
import aiohttp
import logging

# 필요한 모듈 임포트
try:
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
except ImportError as ie:
    print(f"ERROR (slot_extractor): Failed to import modules: {ie}. Check relative paths.")
    # 에러 발생 시 모듈 사용 불가, 함수 호출 시 에러 발생 예상
    call_gpt_async = None
    get_config = None

# 로거 설정
logger = logging.getLogger(__name__)

# --- Slot 추출 함수 ---
async def extract_slots_with_gpt(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    GPT를 사용하여 사용자 입력에서 Slot 정보를 추출합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.
    """
    # 모듈 임포트 실패 시 함수 실행 중단
    if not call_gpt_async or not get_config:
         logger.error("Required modules (gpt_interface, config_loader) not imported correctly.")
         return None

    # config 로드 및 설정값 읽기
    try:
        config = get_config()
        task_config = config.get('tasks', {}).get('slot_extraction', {})
        prompt_template = config.get('prompts', {}).get('slot_extraction_prompt_template')

        # 필수 설정값 확인
        model = task_config.get('model')
        temperature = task_config.get('temperature')
        max_tokens = task_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Slot extraction configuration missing or incomplete in config.yaml (tasks.slot_extraction and prompts.slot_extraction_prompt_template)")
            return None
    except Exception as e:
        logger.error(f"Error loading slot extraction configuration: {e}", exc_info=True)
        return None

    logger.info(f"Attempting to extract slots from input: '{user_input[:50]}...'")

    # config에서 읽은 프롬프트 템플릿 사용
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting slot extraction prompt template. Missing key: {e}. Template: {prompt_template[:200]}...")
        return None

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출 (config에서 읽은 파라미터 명시적 전달)
    logger.debug(f"Calling GPT for slot extraction using model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session,
            # JSON 모드 사용 시도 (모델 지원 여부에 따라 동작)
            response_format={"type": "json_object"}
        )

        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw response content from slot extractor GPT: {response_content[:300]}...")

            # JSON 파싱 시도
            try:
                extracted_slots = json.loads(response_content)
                logger.info(f"Successfully extracted slots: {list(extracted_slots.keys())}")
                logger.debug(f"Extracted slot values: {extracted_slots}")
                return extracted_slots
            except json.JSONDecodeError as e:
                # JSON 모드 실패 또는 미사용 시, 기존 방식(코드 블록 제거 등) 재시도
                logger.warning(f"Initial JSON parsing failed: {e}. Trying fallback parsing...")
                clean_response_content = response_content.strip()
                if clean_response_content.startswith("```json"):
                    clean_response_content = clean_response_content[7:-3].strip()
                elif clean_response_content.startswith("```"):
                    clean_response_content = clean_response_content[3:-3].strip()

                try:
                    json_start = clean_response_content.find('{')
                    json_end = clean_response_content.rfind('}')
                    if json_start != -1 and json_end != -1:
                        json_string = clean_response_content[json_start:json_end+1]
                    else:
                        json_string = clean_response_content

                    extracted_slots = json.loads(json_string)
                    logger.info(f"Successfully extracted slots (fallback parsing): {list(extracted_slots.keys())}")
                    logger.debug(f"Extracted slot values: {extracted_slots}")
                    return extracted_slots
                except json.JSONDecodeError as fallback_e:
                    logger.error(f"Fallback JSON parsing also failed: {fallback_e}. Cleaned content: '{clean_response_content}'")
                    return None
        else:
            logger.warning("Failed to get a valid response from GPT for slot extraction.")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during slot extraction: {e}", exc_info=True)
        return None

# --- 예시 사용법 ---
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_slot_extraction():
        try:
            get_config() # 설정 로드 확인
            logger.info("Configuration loaded successfully for test.")
        except Exception as e:
            logger.error(f"Failed to load configuration for test: {e}")
            return

        test_inputs = [
            "30대 남자인데, 주말에 가볍게 등산할 때 신을 발볼 넓은 트레킹화 15만원 이하로 추천해주세요. 초보입니다.",
            "캠핑 가서 쓸 2인용 텐트 보고 있는데, 퀘차 제품 방수 잘 되나요?",
            "지난번에 산 킵런 운동화 왼쪽 발 뒤꿈치가 아픈데, 사이즈 문제일까요? 사이즈는 275mm 신어요.",
            "여자친구 선물로 러닝할 때 입을 M사이즈 기능성 티셔츠 보고 있어요."
        ]
        async with aiohttp.ClientSession() as session:
            for test_input in test_inputs:
                print(f"\n--- Testing slot extraction for input: --- \n'{test_input}'")
                logger.info(f"Running test extraction for: '{test_input}'")
                slots = await extract_slots_with_gpt(test_input, session=session)
                if slots:
                    print("\nExtraction Successful:")
                    # ensure_ascii=False 로 한국어 깨짐 방지
                    print(json.dumps(slots, indent=2, ensure_ascii=False))
                else:
                    print("\nExtraction Failed.")
                print("-" * 30)

    try:
        # .env 파일 로드 확인 (API 키 필요)
        if not os.getenv("OPENAI_API_KEY"):
             print("Error: OPENAI_API_KEY not found. Cannot run tests that call the API.")
        else:
             asyncio.run(test_slot_extraction())
    except FileNotFoundError:
        print("\nError: config.yaml not found. Please ensure it exists in the project root.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")