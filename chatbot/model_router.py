# chatbot/model_router.py (요구사항 반영 최종본: 단일 범용 CoT 생성, 컨텍스트 활용)

import json
import logging
from typing import Optional, Dict, Any, List, Union # List, Union 추가
import aiohttp
import time # if __name__ 블록 테스트용

# --- 필요한 모듈 임포트 ---
try:
    # model_router.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    logging.info("gpt_interface and config_loader imported successfully in model_router.")
except ImportError as ie:
    logging.error(f"ERROR (model_router): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 기능 사용 불가
    call_gpt_async = None
    get_config = None

# --- 로거 설정 (기본 설정 상속 또는 명시적 설정) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 로거 레벨 직접 설정

# --- 함수 구현 ---

async def classify_complexity_level(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    GPT(설정된 모델)를 사용하여 사용자 질문의 복잡도를 분석하고,
    "easy", "medium", "hard" 중 하나로 분류합니다. (이 함수는 변경 없음)
    """
    # 필수 모듈 및 설정 로드 확인
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in classify_complexity_level.")
        return "easy" # 비상시 기본값
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
        logger.error(f"Failed to get configuration in classify_complexity_level: {conf_e}")
        return "easy"

    # 복잡도 분류 설정 읽기
    try:
        router_config = config.get('model_router', {}).get('complexity_classification', {})
        prompt_template = config.get('prompts', {}).get('complexity_classification_prompt_template')

        # 필수 설정값 확인
        model = router_config.get('model')
        temperature = router_config.get('temperature')
        max_tokens = router_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Complexity classification configuration missing or incomplete in config.yaml. Check model_router.complexity_classification and prompts.complexity_classification_prompt_template")
            return "easy" # 설정 오류 시 기본값
    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing complexity classification configuration: {e}", exc_info=True)
        return "easy" # 설정 오류 시 기본값

    logger.info(f"Classifying complexity for input: '{user_input[:50]}...'")

    # 프롬프트 포맷팅
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting complexity classification prompt. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return "easy" # 프롬프트 포맷팅 오류 시 기본값
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return "easy"

    messages = [{"role": "user", "content": prompt}]

    logger.debug(f"Calling complexity classification model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session,
            response_format={"type": "json_object"}
        )

        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw complexity classification response: {response_content}")

            try:
                clean_response = response_content.strip()
                if clean_response.startswith("```json"): clean_response = clean_response[7:-3].strip()
                elif clean_response.startswith("```"): clean_response = clean_response[3:-3].strip()
                json_start = clean_response.find('{'); json_end = clean_response.rfind('}')
                json_string = clean_response[json_start:json_end+1] if json_start != -1 and json_end != -1 else clean_response

                classification_result = json.loads(json_string)
                level = classification_result.get("complexity_level", "easy").lower()

                if level in ["easy", "medium", "hard"]:
                    logger.info(f"Question complexity classified as: {level}")
                    return level
                else:
                    logger.warning(f"Unexpected classification level value received: '{level}'. Defaulting to 'easy'.")
                    return "easy"
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from complexity classification response: {e}. Raw content: '{response_content}'")
                return "easy"
            except Exception as e:
                logger.error(f"Error processing classification response content: {e}. Raw content: '{response_content}'")
                return "easy"
        else:
            logger.warning("No valid response/choices received from complexity classification model.")
            return "easy"

    except Exception as e:
        logger.error(f"An error occurred during complexity classification API call: {e}", exc_info=True)
        return "easy"

# --- [삭제됨] generate_cot_steps_async 함수 ---

# --- [삭제됨] generate_hard_cot_instructions_async 함수 ---

# --- [신규 추가] 범용 CoT 생성 함수 ---
async def generate_general_cot_async(
    user_input: str,
    previous_summary: Optional[str],
    previous_slots: Dict[str, Any],
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    주어진 사용자 입력과 이전 대화 컨텍스트(요약, 슬롯)를 바탕으로
    범용적인 단계별 사고 과정(Chain-of-Thought) 계획을 생성합니다.
    이 함수는 병렬로 실행되며, 결과는 medium/hard 난이도에서 사용됩니다.

    Args:
        user_input (str): 현재 사용자의 입력 문자열.
        previous_summary (Optional[str]): 이전 턴까지의 대화 요약.
        previous_slots (Dict[str, Any]): 이전 턴까지 추출된 슬롯 정보.
        session (Optional[aiohttp.ClientSession], optional): API 호출 세션. Defaults to None.

    Returns:
        Optional[str]: 생성된 CoT 계획 문자열. 오류 발생 시 None.
    """
    # 필수 모듈 및 설정 로드 확인
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in generate_general_cot_async.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
        logger.error(f"Failed to get configuration in generate_general_cot_async: {conf_e}")
        return None

    # 범용 CoT 생성 설정 읽기
    try:
        # [수정] general_cot_generation 섹션 사용
        cot_config = config.get('model_router', {}).get('general_cot_generation', {})
        prompt_template = config.get('prompts', {}).get('general_cot_generation_prompt_template')

        model = cot_config.get('model')
        temperature = cot_config.get('temperature')
        max_tokens = cot_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("General CoT generation configuration missing or incomplete in config.yaml.")
            return None
    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing general CoT generation configuration: {e}", exc_info=True)
        return None

    logger.info("Generating general CoT plan using provided context...")

    # 이전 슬롯/요약이 없거나 비어있으면 'N/A' 처리
    summary_for_prompt = previous_summary if previous_summary else "N/A"
    # 슬롯 정보는 보기 좋게 문자열로 변환 (예: JSON 또는 key-value 리스트)
    try:
        slots_for_prompt = json.dumps(previous_slots, indent=2, ensure_ascii=False) if previous_slots else "N/A"
    except Exception:
        slots_for_prompt = str(previous_slots) # JSON 변환 실패 시 문자열로

    # 프롬프트 포맷팅 (새로운 템플릿과 플레이스홀더 사용)
    try:
        prompt = prompt_template.format(
            user_input=user_input,
            previous_summary=summary_for_prompt,
            previous_slots=slots_for_prompt
        )
    except KeyError as e:
        logger.error(f"Error formatting general CoT prompt. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return None

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출하여 CoT 생성
    logger.debug(f"Calling general CoT generation model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
        )
        if response_data and response_data.get("choices"):
            cot_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            if cot_text:
                logger.info("General CoT plan generated successfully.")
                logger.debug(f"Generated CoT plan:\n------\n{cot_text}\n------")
                return cot_text
            else:
                logger.warning("General CoT generation API call successful but returned empty content.")
                return None # 빈 내용이면 실패
        else:
            logger.warning("Failed to get valid response/choices from general CoT generation model.")
            return None

    except Exception as e:
        logger.error(f"Error occurred during general CoT generation API call: {e}", exc_info=True)
        return None

# --- [수정됨] 라우팅 결정 함수 (CoT 생성 로직 제거) ---
async def determine_routing_and_reasoning(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Dict[str, Any]:
    """
    사용자 질문에 대해 복잡도를 분류하고, 해당 복잡도 레벨에 맞는
    최종 응답 모델 이름을 결정하여 반환합니다.
    (CoT 생성은 별도의 병렬 태스크로 분리됨)

    Args:
        user_input (str): 처리할 사용자 입력 문자열.
        session (Optional[aiohttp.ClientSession], optional): API 호출 세션. Defaults to None.

    Returns:
        Dict[str, Any]: 라우팅 결정 결과 딕셔너리.
                       {'level': str, 'model': str} 형식.
                       오류 발생 시 기본값 {'level': 'easy', 'model': <default_model>} 반환.
    """
    # 설정 로드 및 기본 모델/맵 정의
    default_model = 'gpt-3.5-turbo' # 최후의 기본값
    model_map = {'easy': default_model, 'medium': default_model, 'hard': default_model} # 기본 맵

    if not get_config:
        logger.error("Config loader not available in determine_routing_and_reasoning. Using fallback defaults.")
    else:
        try:
            config = get_config()
            if not config: raise ValueError("Configuration could not be loaded.")
            loaded_map = config.get('model_router', {}).get('routing_map', {})
            if isinstance(loaded_map, dict) and all(k in loaded_map and isinstance(loaded_map[k], str) for k in ['easy', 'medium', 'hard']):
                model_map = loaded_map
                default_model = model_map.get('easy', default_model)
                logger.debug(f"Loaded routing map from config: {model_map}")
            else:
                logger.warning("Routing map in config is incomplete or invalid. Using default models.")
                default_model = model_map.get('easy', default_model)
        except Exception as e:
            logger.error(f"Error loading config for routing map: {e}. Using default models.", exc_info=True)

    logger.info(f"Determining routing model for: '{user_input[:50]}...'")

    # 1. 복잡도 분류 (비동기 호출)
    complexity_level = await classify_complexity_level(user_input, session)
    logger.debug(f"Complexity classification result: {complexity_level}")

    # 2. 최종 응답 모델 선택 (분류 결과와 맵 기반)
    chosen_model = model_map.get(complexity_level, default_model)
    logger.debug(f"Chosen final response model based on complexity '{complexity_level}': {chosen_model}")

    # --- CoT 생성 로직은 여기서 제거됨 ---

    # 3. 최종 결과 조합 (level과 model만 포함)
    result = {
        "level": complexity_level,
        "model": chosen_model,
        # "cot_data": None # CoT 데이터는 더 이상 이 함수에서 관리하지 않음
    }
    logger.info(f"Routing determination complete: Level='{result['level']}', Model='{result['model']}'")
    return result

# --- [수정됨] 예시 사용법 ---
if __name__ == "__main__":
    import asyncio
    import os

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- Running model_router.py as main script for testing ---")

    async def test_model_router_components():
        """라우터 컴포넌트 테스트 실행"""
        try:
            get_config() # 설정 로드 확인
            logger.info("Config loaded successfully for model_router test.")
        except Exception as e:
            logger.error(f"Failed to load configuration for test: {e}. Cannot run test.")
            return
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found. Cannot run API dependent tests.")
            return

        test_inputs = {
            "easy": "매장 전화번호 뭐에요?",
            "medium": "나이키 270mm 신는데 데카트론 러닝화 추천해줘.",
            "hard": "작년에 산 퀘차 등산화 신고 겨울 설산 갔는데 발 시려웠어요. 평발이고 발볼 넓은 편인데, 보온/방수 되면서 발 편한 다른 데카트론 등산화 추천과 비교 설명해주세요."
        }

        # 테스트용 컨텍스트
        test_summary = "사용자는 이전에 퀘차 등산화에 대해 문의했었음."
        test_slots = {"brand_mentioned": "퀘차", "user_preference": ["발볼 넓음", "평발"], "activity": "겨울 등산"}

        async with aiohttp.ClientSession() as session:
            # 1. 복잡도 분류 및 모델 선택 테스트
            print("\n--- Testing Complexity Classification & Model Routing ---")
            for level_key, input_text in test_inputs.items():
                print(f"\nInput ({level_key}): '{input_text[:60]}...'")
                routing_result = await determine_routing_and_reasoning(input_text, session=session)
                print(f"Result: {routing_result}")

            # 2. 범용 CoT 생성 테스트
            print("\n--- Testing General CoT Generation (using 'hard' input) ---")
            hard_input = test_inputs['hard']
            start_cot_time = time.time()
            general_cot = await generate_general_cot_async(
                user_input=hard_input,
                previous_summary=test_summary,
                previous_slots=test_slots,
                session=session
            )
            cot_duration = time.time() - start_cot_time
            print(f"(Took {cot_duration:.3f}s)")
            if general_cot:
                print("Generated General CoT:")
                print("-" * 10)
                print(general_cot)
                print("-" * 10)
            else:
                print("General CoT generation failed.")

    # 비동기 테스트 실행
    try:
        asyncio.run(test_model_router_components())
    except Exception as e:
        logger.critical(f"\nAn critical error occurred during testing: {e}", exc_info=True)