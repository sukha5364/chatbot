# chatbot/model_router.py (요구사항 반영 최종본: 주석 보강 및 품질 개선)

import json
import logging
from typing import Optional, Dict, Any, List, Union # List, Union 추가
import aiohttp

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
# logging.basicConfig(level=logging.DEBUG) # 필요 시 명시적 설정 가능
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 로거 레벨 직접 설정

# --- 함수 구현 ---

async def classify_complexity_level(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    GPT(설정된 모델)를 사용하여 사용자 질문의 복잡도를 분석하고,
    "easy", "medium", "hard" 중 하나로 분류합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.

    Args:
        user_input (str): 분류할 사용자 입력 문자열.
        session (Optional[aiohttp.ClientSession], optional): API 호출에 사용할 aiohttp 세션.
                                                             없으면 새로 생성됩니다. Defaults to None.

    Returns:
        str: 분류된 복잡도 레벨 문자열 ('easy', 'medium', 'hard').
             오류 발생 또는 설정 누락 시 'easy'를 기본값으로 반환.
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
        # 프롬프트 템플릿에 user_input 변수 주입
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting complexity classification prompt. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return "easy" # 프롬프트 포맷팅 오류 시 기본값
    except Exception as e:
         logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
         return "easy"

    # API 요청 메시지 생성
    messages = [{"role": "user", "content": prompt}]

    # GPT 호출하여 복잡도 분류 (JSON 모드 시도)
    logger.debug(f"Calling complexity classification model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session,
            response_format={"type": "json_object"} # JSON 응답 형식 요청
        )

        # 응답 처리
        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw complexity classification response: {response_content}")

            # JSON 파싱 시도
            try:
                # 코드 블록 등 불필요 문자 제거 시도 (안전 장치)
                clean_response = response_content.strip()
                if clean_response.startswith("```json"): clean_response = clean_response[7:-3].strip()
                elif clean_response.startswith("```"): clean_response = clean_response[3:-3].strip()
                json_start = clean_response.find('{'); json_end = clean_response.rfind('}')
                json_string = clean_response[json_start:json_end+1] if json_start != -1 and json_end != -1 else clean_response

                classification_result = json.loads(json_string)
                level = classification_result.get("complexity_level", "easy").lower() # 키 부재 시 'easy' 기본값

                # 유효한 레벨 값인지 확인
                if level in ["easy", "medium", "hard"]:
                    logger.info(f"Question complexity classified as: {level}")
                    return level
                else:
                    logger.warning(f"Unexpected classification level value received: '{level}'. Defaulting to 'easy'.")
                    return "easy" # 예상 외 값일 경우 기본값
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from complexity classification response: {e}. Raw content: '{response_content}'")
                return "easy" # JSON 파싱 실패 시 기본값
            except Exception as e: # 기타 파싱 관련 오류
                 logger.error(f"Error processing classification response content: {e}. Raw content: '{response_content}'")
                 return "easy"
        else:
            # API 호출 성공했으나 choices 없는 경우 등
            logger.warning("No valid response/choices received from complexity classification model.")
            # call_gpt_async 내부에서 에러 로그 기록되었을 것임
            return "easy" # API 응답 문제 시 기본값

    except Exception as e:
        # call_gpt_async 호출 자체 또는 네트워크 오류 등
        logger.error(f"An error occurred during complexity classification API call: {e}", exc_info=True)
        # call_gpt_async 내부에서 에러 로그 기록되었을 것임
        return "easy" # API 호출 실패 시 기본값

async def generate_cot_steps_async(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    'medium' 난이도로 분류된 질문에 대해, 답변 생성을 위한 간결한
    단계별 사고(Chain-of-Thought) 스텝을 생성합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.

    Args:
        user_input (str): CoT 스텝을 생성할 사용자 입력 문자열.
        session (Optional[aiohttp.ClientSession], optional): API 호출 세션. Defaults to None.

    Returns:
        Optional[str]: 생성된 CoT 스텝 문자열. 오류 발생 시 None.
    """
    # 필수 모듈 및 설정 로드 확인
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in generate_cot_steps_async.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
         logger.error(f"Failed to get configuration in generate_cot_steps_async: {conf_e}")
         return None

    # Medium CoT 생성 설정 읽기
    try:
        cot_config = config.get('model_router', {}).get('medium_cot_generation', {})
        prompt_template = config.get('prompts', {}).get('medium_cot_generation_prompt_template')

        model = cot_config.get('model')
        temperature = cot_config.get('temperature')
        max_tokens = cot_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Medium CoT generation configuration missing or incomplete in config.yaml.")
            return None
    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing medium CoT generation configuration: {e}", exc_info=True)
        return None

    logger.info("Generating CoT steps for medium complexity question...")

    # 프롬프트 포맷팅
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting medium CoT prompt. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None
    except Exception as e:
         logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
         return None

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출하여 CoT 스텝 생성
    logger.debug(f"Calling medium CoT step generation model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
            # JSON 모드는 CoT 스텝 생성에는 부적합할 수 있음
        )
        if response_data and response_data.get("choices"):
            # 응답 텍스트 추출 및 앞뒤 공백 제거
            steps_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            if steps_text: # 내용이 있을 경우에만 성공 처리
                logger.info("CoT steps generated successfully.")
                logger.debug(f"Generated CoT steps:\n------\n{steps_text}\n------")
                return steps_text
            else:
                 logger.warning("Medium CoT generation API call successful but returned empty content.")
                 return None # 빈 내용이면 실패
        else:
            logger.warning("Failed to get valid response/choices from medium CoT generation model.")
            return None

    except Exception as e:
        logger.error(f"Error occurred during medium CoT step generation API call: {e}", exc_info=True)
        return None

async def generate_hard_cot_instructions_async(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    'hard' 난이도로 분류된 질문에 대해, 답변 생성을 위한 상세한
    단계별 사고 지침(Instruction)을 생성합니다.
    모델, 파라미터, 프롬프트는 config.yaml에서 로드합니다.

    Args:
        user_input (str): CoT 지침을 생성할 사용자 입력 문자열.
        session (Optional[aiohttp.ClientSession], optional): API 호출 세션. Defaults to None.

    Returns:
        Optional[str]: 생성된 상세 CoT 지침 문자열. 오류 발생 시 None.
    """
    # 필수 모듈 및 설정 로드 확인
    if not call_gpt_async or not get_config:
        logger.error("Required modules not imported correctly in generate_hard_cot_instructions_async.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
    except Exception as conf_e:
         logger.error(f"Failed to get configuration in generate_hard_cot_instructions_async: {conf_e}")
         return None

    # Hard CoT 생성 설정 읽기
    try:
        cot_config = config.get('model_router', {}).get('hard_cot_generation', {})
        prompt_template = config.get('prompts', {}).get('hard_cot_generation_prompt_template')

        model = cot_config.get('model')
        temperature = cot_config.get('temperature')
        max_tokens = cot_config.get('max_tokens')

        if not all([model, isinstance(temperature, (int, float)), isinstance(max_tokens, int), prompt_template]):
            logger.error("Hard CoT generation configuration missing or incomplete in config.yaml.")
            return None
    except (KeyError, TypeError, Exception) as e:
        logger.error(f"Error accessing hard CoT generation configuration: {e}", exc_info=True)
        return None

    logger.info("Generating detailed CoT instructions for hard complexity question...")

    # 프롬프트 포맷팅
    try:
        prompt = prompt_template.format(user_input=user_input)
    except KeyError as e:
        logger.error(f"Error formatting hard CoT prompt. Missing key: {e}. Template preview: {prompt_template[:200]}...")
        return None
    except Exception as e:
         logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
         return None

    messages = [{"role": "user", "content": prompt}]

    # GPT 호출하여 CoT 지침 생성
    logger.debug(f"Calling hard CoT instruction generation model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session=session
        )
        if response_data and response_data.get("choices"):
            # 응답 텍스트 추출 및 앞뒤 공백 제거
            instructions_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            if instructions_text: # 내용이 있을 경우에만 성공 처리
                logger.info("Detailed CoT instructions generated successfully.")
                logger.debug(f"Generated CoT instructions:\n------\n{instructions_text}\n------")
                return instructions_text
            else:
                 logger.warning("Hard CoT instruction generation API call successful but returned empty content.")
                 return None # 빈 내용이면 실패
        else:
            logger.warning("Failed to get valid response/choices from hard CoT instruction generation model.")
            return None

    except Exception as e:
        logger.error(f"Error occurred during hard CoT instruction generation API call: {e}", exc_info=True)
        return None

async def determine_routing_and_reasoning(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Dict[str, Any]:
    """
    사용자 질문에 대해 복잡도 분류, 최종 응답 모델 선택,
    그리고 필요한 경우 CoT 데이터(스텝 또는 지침) 생성을 수행하고,
    그 결과를 딕셔너리로 반환합니다.

    Args:
        user_input (str): 처리할 사용자 입력 문자열.
        session (Optional[aiohttp.ClientSession], optional): API 호출 세션. Defaults to None.

    Returns:
        Dict[str, Any]: 라우팅 및 추론 결과 딕셔너리.
                       {'level': str, 'model': str, 'cot_data': Optional[str]} 형식.
                       오류 발생 시 기본값 {'level': 'easy', 'model': <default_model>, 'cot_data': None} 반환.
    """
    # 설정 로드 및 기본 모델 정의
    default_model = 'gpt-3.5-turbo' # 최후의 기본값
    model_map = {'easy': default_model, 'medium': default_model, 'hard': default_model} # 기본 맵

    if not get_config:
        logger.error("Config loader not available in determine_routing_and_reasoning. Using fallback defaults.")
    else:
        try:
            config = get_config()
            if not config: raise ValueError("Configuration could not be loaded.")
            # config에서 라우팅 맵 로드 (오류 발생해도 기본 맵 사용)
            loaded_map = config.get('model_router', {}).get('routing_map', {})
            # easy, medium, hard 키가 모두 있는지, 값은 문자열인지 확인
            if isinstance(loaded_map, dict) and all(k in loaded_map and isinstance(loaded_map[k], str) for k in ['easy', 'medium', 'hard']):
                 model_map = loaded_map
                 default_model = model_map.get('easy', default_model) # easy 모델을 기본 fallback으로 사용
                 logger.debug(f"Loaded routing map from config: {model_map}")
            else:
                 logger.warning("Routing map in config is incomplete or invalid. Using default models.")
                 default_model = model_map.get('easy', default_model) # 혹시 easy만 있어도 사용

        except Exception as e:
            logger.error(f"Error loading config for routing map: {e}. Using default models.", exc_info=True)
            # 오류 발생 시 model_map은 초기 기본값 유지

    logger.info(f"Determining routing and reasoning strategy for: '{user_input[:50]}...'")

    # 1. 복잡도 분류 (비동기 호출)
    # classify_complexity_level 함수는 오류 시 'easy' 반환 보장
    complexity_level = await classify_complexity_level(user_input, session)
    logger.debug(f"Complexity classification result: {complexity_level}")

    # 2. 최종 응답 모델 선택 (분류 결과와 맵 기반)
    # complexity_level이 'easy', 'medium', 'hard' 중 하나임을 가정 (위 함수에서 보장)
    chosen_model = model_map.get(complexity_level, default_model) # 맵에 없거나 잘못된 level이면 fallback
    logger.debug(f"Chosen final response model based on complexity '{complexity_level}': {chosen_model}")

    # 3. CoT 데이터 생성 (필요시, 비동기 호출)
    cot_data: Optional[str] = None
    try:
        if complexity_level == "medium":
            logger.info("Complexity is medium, attempting to generate CoT steps...")
            cot_data = await generate_cot_steps_async(user_input, session)
            if cot_data: logger.info("Medium CoT steps generated.")
            else: logger.warning("Failed to generate medium CoT steps.")
        elif complexity_level == "hard":
            logger.info("Complexity is hard, attempting to generate detailed CoT instructions...")
            cot_data = await generate_hard_cot_instructions_async(user_input, session)
            if cot_data: logger.info("Hard CoT instructions generated.")
            else: logger.warning("Failed to generate hard CoT instructions.")
        # else: easy 레벨은 CoT 생성 안 함
    except Exception as e:
         # CoT 생성 함수 내부 오류 (이미 로그 기록됨), cot_data는 None 유지
         logger.error(f"Error occurred during CoT data generation step: {e}", exc_info=True)


    # 최종 결과 조합
    result = {
        "level": complexity_level,
        "model": chosen_model,
        "cot_data": cot_data # 생성 실패 시 None
    }
    logger.info(f"Routing/Reasoning determination complete: Level='{result['level']}', Model='{result['model']}', CoT Generated={'Yes' if result['cot_data'] else 'No'}")
    return result

# --- 예시 사용법 (기존 유지) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # __name__으로 로거 다시 가져오기
    logger.info("--- Running model_router.py as main script for testing ---")

    async def test_routing_and_reasoning():
        """라우팅 및 추론 로직 테스트 실행"""
        try:
            get_config() # 설정 로드 가능한지 확인
            logger.info("Config loaded successfully for routing test.")
        except Exception as e:
            logger.error(f"Failed to load configuration for test: {e}. Cannot run test.")
            return

        if not os.getenv("OPENAI_API_KEY"):
             logger.error("OPENAI_API_KEY not found. Cannot run API dependent tests.")
             return

        test_inputs = {
            "easy_faq": "매장 전화번호 뭐에요?",
            "medium_comparison": "나이키 270mm랑 아디다스 275mm 신발 특징 비교해주고, 발볼 넓은 사람한테 뭐가 더 나을지 알려줘.",
            "hard_complex_recommendation": "작년에 구매한 퀘차 등산화(MH500)를 신고 겨울 설산 트레킹을 갔는데 발이 너무 시려웠어요. 제 발은 평발에 발볼도 넓은 편인데, 다음 겨울 산행을 위해 보온성과 방수성이 뛰어나면서 제 발에도 편한 다른 데카트론 등산화 모델이 있다면 추천해주시고, MH500과 비교해서 어떤 점이 더 나은지 구체적으로 설명해주세요.",
            "medium_feature_based": "10만원 이하로 방수 기능 좋은 데카트론 트레킹화 찾아줘.",
            "easy_simple_info": "킵런 KS900 무게 알려줘."
        }
        async with aiohttp.ClientSession() as session:
            for test_name, input_text in test_inputs.items():
                print(f"\n--- Testing Routing for ({test_name}): '{input_text[:60]}...' ---")
                logger.info(f"Running full routing and reasoning test for case: {test_name}")
                try:
                    start_t = time.time()
                    routing_result = await determine_routing_and_reasoning(input_text, session=session)
                    duration_t = time.time() - start_t
                    print(f"\nRouting Result (took {duration_t:.3f}s):")

                    # 결과 출력 시 CoT 데이터는 미리보기만
                    result_display = routing_result.copy()
                    if result_display.get('cot_data'):
                        result_display['cot_data_preview'] = result_display['cot_data'][:100].replace('\n', ' ') + '...'
                        del result_display['cot_data'] # 원본 CoT 데이터는 출력에서 제외

                    # ensure_ascii=False 로 한국어 깨짐 방지, default=str로 직렬화 불가 객체 처리
                    print(json.dumps(result_display, indent=2, ensure_ascii=False, default=str))

                except Exception as test_e:
                     logger.error(f"Error during test case '{test_name}': {test_e}", exc_info=True)
                     print(f"ERROR running test case '{test_name}': {test_e}")
                print("-" * 40)

    # 비동기 테스트 실행
    try:
        asyncio.run(test_routing_and_reasoning())
    except Exception as e:
        logger.critical(f"\nAn critical error occurred during testing: {e}", exc_info=True)