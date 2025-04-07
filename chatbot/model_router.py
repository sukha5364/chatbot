# chatbot/model_router.py (오류 수정 및 리팩토링 최종본)

import json
from typing import Optional, Dict, Any, List, Union
import aiohttp
import logging

# 필요한 모듈 임포트
try:
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
except ImportError as ie:
    print(f"ERROR (model_router): Failed to import modules: {ie}. Check relative paths.")
    call_gpt_async = None
    get_config = None

# 로거 설정
logger = logging.getLogger(__name__)

# --- 함수 구현 ---

async def classify_complexity_level(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    GPT(설정된 모델)를 활용하여 질문 복잡도를 "easy", "medium", "hard"로 분류합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.
    """
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in classify_complexity_level.")
        return "easy" # 비상시 기본값

    # config 로드 및 설정값 읽기
    try:
        config = get_config()
        router_config = config.get('model_router', {}).get('complexity_classification', {})
        prompt_template = config.get('prompts', {}).get('complexity_classification_prompt_template')

        model = router_config.get('model')
        temperature = router_config.get('temperature')
        max_tokens = router_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Complexity classification configuration missing or incomplete in config.yaml")
            return "easy"
    except Exception as e:
        logger.error(f"Error loading complexity classification configuration: {e}", exc_info=True)
        return "easy"

    logger.info(f"Classifying complexity for input: '{user_input[:50]}...'")

    # config에서 읽은 프롬프트 템플릿 사용
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting complexity classification prompt. Missing key: {e}. Template: {prompt_template[:200]}...")
        return "easy"

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출 (config에서 읽은 파라미터 명시적 전달)
    logger.debug(f"Using model for complexity classification: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session,
            response_format={"type": "json_object"} # JSON 모드 사용
        )

        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw complexity classification response: {response_content}")

            # JSON 파싱 시도 (response_format 사용 전제)
            try:
                classification_result = json.loads(response_content)
                level = classification_result.get("complexity_level", "easy").lower()

                if level in ["easy", "medium", "hard"]:
                    logger.info(f"Question complexity classified as: {level}")
                    return level
                else:
                    logger.warning(f"Unexpected classification level value: '{level}'. Defaulting to 'easy'.")
                    return "easy"
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from complexity classification response: {e}. Content: '{response_content}'")
                return "easy"
        else:
            logger.warning("No valid response from complexity classification.")
            return "easy"

    except Exception as e:
        logger.error(f"An error occurred during complexity classification: {e}", exc_info=True)
        return "easy"

async def generate_cot_steps_async(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    'medium' 난이도 질문에 대한 CoT 스텝을 생성합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.
    """
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in generate_cot_steps_async.")
        return None

    # config 로드 및 설정값 읽기
    try:
        config = get_config()
        cot_config = config.get('model_router', {}).get('medium_cot_generation', {})
        prompt_template = config.get('prompts', {}).get('medium_cot_generation_prompt_template')

        model = cot_config.get('model')
        temperature = cot_config.get('temperature')
        max_tokens = cot_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Medium CoT generation configuration missing or incomplete in config.yaml")
            return None
    except Exception as e:
        logger.error(f"Error loading medium CoT generation configuration: {e}", exc_info=True)
        return None

    logger.info("Generating CoT steps for medium complexity question...")

    # config에서 읽은 프롬프트 템플릿 사용
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting medium CoT prompt. Missing key: {e}. Template: {prompt_template[:200]}...")
        return None

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출 (config에서 읽은 파라미터 명시적 전달)
    logger.debug(f"Using model for medium CoT step generation: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
        )
        if response_data and response_data.get("choices"):
            steps_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            logger.info("CoT steps generated successfully.")
            logger.debug(f"Generated CoT steps:\n{steps_text}")
            return steps_text
        else:
            logger.warning("Failed to generate CoT steps.")
            return None
    except Exception as e:
        logger.error(f"Error generating CoT steps: {e}", exc_info=True)
        return None

async def generate_hard_cot_instructions_async(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    'hard' 난이도 질문에 대한 상세 CoT 지침을 생성합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.
    """
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in generate_hard_cot_instructions_async.")
        return None

    # config 로드 및 설정값 읽기
    try:
        config = get_config()
        cot_config = config.get('model_router', {}).get('hard_cot_generation', {})
        prompt_template = config.get('prompts', {}).get('hard_cot_generation_prompt_template')

        model = cot_config.get('model')
        temperature = cot_config.get('temperature')
        max_tokens = cot_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Hard CoT generation configuration missing or incomplete in config.yaml")
            return None
    except Exception as e:
        logger.error(f"Error loading hard CoT generation configuration: {e}", exc_info=True)
        return None

    logger.info("Generating detailed CoT instructions for hard complexity question...")

    # config에서 읽은 프롬프트 템플릿 사용
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting hard CoT prompt. Missing key: {e}. Template: {prompt_template[:200]}...")
        return None

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출 (config에서 읽은 파라미터 명시적 전달)
    logger.debug(f"Using model for hard CoT instruction generation: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
        )
        if response_data and response_data.get("choices"):
            instructions_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            logger.info("Detailed CoT instructions generated successfully.")
            logger.debug(f"Generated CoT instructions:\n{instructions_text}")
            return instructions_text
        else:
            logger.warning("Failed to generate detailed CoT instructions.")
            return None
    except Exception as e:
        logger.error(f"Error generating detailed CoT instructions: {e}", exc_info=True)
        return None

async def determine_routing_and_reasoning(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Dict[str, Any]:
    """
    질문 복잡도 분류, 최종 모델 선택, 필요한 CoT 데이터 생성을 수행하고 결과를 딕셔너리로 반환합니다.
    """
    if not get_config:
         logger.error("Config loader not available in determine_routing_and_reasoning.")
         # 비상용 기본값 반환
         return {"level": "easy", "model": "gpt-3.5-turbo", "cot_data": None}

    # config에서 라우팅 맵 로드
    try:
        config = get_config()
        model_map = config.get('model_router', {}).get('routing_map', {})
        if not model_map or not all(k in model_map for k in ['easy', 'medium', 'hard']):
            logger.error("Routing map is incomplete or missing in model_router configuration.")
            default_model = 'gpt-3.5-turbo' # 비상용
            model_map = {'easy': default_model, 'medium': default_model, 'hard': default_model}
    except Exception as e:
        logger.error(f"Error loading config for routing decision: {e}", exc_info=True)
        default_model = 'gpt-3.5-turbo'
        model_map = {'easy': default_model, 'medium': default_model, 'hard': default_model}

    logger.info("Determining routing and reasoning strategy...")
    # 1. 복잡도 분류
    complexity_level = await classify_complexity_level(user_input, session)

    # 2. 최종 응답 모델 선택
    default_model_for_level = model_map.get('easy', 'gpt-3.5-turbo') # fallback 기본값
    chosen_model = model_map.get(complexity_level, default_model_for_level)

    # 3. CoT 데이터 생성 (필요시)
    cot_data: Optional[str] = None
    if complexity_level == "medium":
        cot_data = await generate_cot_steps_async(user_input, session)
    elif complexity_level == "hard":
        cot_data = await generate_hard_cot_instructions_async(user_input, session)

    result = {
        "level": complexity_level,
        "model": chosen_model,
        "cot_data": cot_data
    }
    logger.info(f"Routing decision: Level='{result['level']}', Model='{result['model']}', CoT data generated={'yes' if result['cot_data'] else 'no'}")
    return result

# --- 예시 사용법 ---
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_routing_and_reasoning():
        try:
            get_config()
            logger.info("Config loaded for routing test.")
        except Exception as e:
            logger.error(f"Failed to load config for routing test: {e}")
            return

        test_inputs = {
            "easy": "매장 전화번호 뭐에요?",
            "medium": "나이키 270mm랑 아디다스 275mm 신발 특징 비교해주고, 발볼 넓은 사람한테 뭐가 더 나을지 알려줘.",
            "hard": "작년에 구매한 퀘차 등산화(MH500)를 신고 겨울 설산 트레킹을 갔는데 발이 너무 시려웠어요. 제 발은 평발에 발볼도 넓은 편인데, 다음 겨울 산행을 위해 보온성과 방수성이 뛰어나면서 제 발에도 편한 다른 등산화 모델이 있다면 추천해주시고, MH500과 비교해서 어떤 점이 더 나은지 구체적으로 설명해주세요."
        }
        async with aiohttp.ClientSession() as session:
            for level_tag, input_text in test_inputs.items():
                print(f"\n--- Testing routing for ({level_tag}): '{input_text[:50]}...' ---")
                logger.info(f"Running full routing and reasoning test for: '{input_text[:50]}...'")
                routing_result = await determine_routing_and_reasoning(input_text, session=session)
                print("\nRouting Result:")
                result_display = routing_result.copy()
                if result_display.get('cot_data'):
                    result_display['cot_data_preview'] = result_display['cot_data'][:100] + '...'
                    del result_display['cot_data']
                # ensure_ascii=False 추가
                print(json.dumps(result_display, indent=2, ensure_ascii=False, default=str))
                print("-" * 30)

    try:
        # .env 파일 로드 확인
        if not os.getenv("OPENAI_API_KEY"):
             print("Error: OPENAI_API_KEY not found. Cannot run tests that call the API.")
        else:
            asyncio.run(test_routing_and_reasoning())
    except FileNotFoundError:
        print("\nError: config.yaml not found. Please ensure it exists.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")