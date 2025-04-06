# chatbot/model_router.py

import json
from typing import Optional, Dict, Any, List, Union # 타입 힌트 추가
import aiohttp
import logging

# 필요한 모듈 임포트
from .gpt_interface import call_gpt_async
from .config_loader import get_config

# 로거 설정 및 설정 로드
logger = logging.getLogger(__name__)
config = get_config()
router_config = config.get('model_router', {}) # 라우터 설정 미리 로드

# --- 프롬프트 템플릿 정의 ---

# [수정] 3단계 분류 프롬프트
COMPLEXITY_CLASSIFICATION_PROMPT_TEMPLATE = """
사용자 질문의 복잡도를 분석하여 "easy", "medium", "hard" 중 하나로 분류해주세요.

[분류 기준]
- easy: 단순 정보 검색, FAQ, 단일 의도의 짧은 질문 (예: "매장 위치 알려줘", "반품 규정 뭐야?", "이 신발 재고 있어?")
- medium: 제품 비교, 기능 설명 요구, 여러 조건이 포함된 추천 요청, 약간의 추론 필요 (예: "A랑 B 중에 뭐가 더 나아?", "이 텐트의 장단점 설명해줘", "발볼 넓고 쿠션 좋은 10만원대 러닝화 추천해줘")
- hard: 매우 복잡한 다중 조건, 심층적인 이유 분석 요구, 사용자의 특정 상황에 대한 깊은 이해와 종합적 추론 필요 (예: "지난번 구매한 등산화가 특정 상황에서 불편했는데, 내 등반 스타일과 발 상태(넓고 평발)를 고려해서 대안 제품과 그 이유를 상세히 설명해줘", "내년 국토대장정에 필요한 모든 장비 목록과 각 장비 선택 시 주의사항을 알려줘")

분석 결과는 반드시 다음 JSON 형식으로만 응답하고, 다른 설명은 절대 붙이지 마세요:
```json
{{
  "complexity_level": "easy" | "medium" | "hard"
}}

사용자 질문: "{user_input}"

분석 결과 (JSON):
"""

# [신규] Medium 난이도 CoT 스텝 생성 프롬프트
MEDIUM_COT_STEP_GENERATION_PROMPT_TEMPLATE = """
다음 사용자 질문에 답변하기 위한 단계별 사고(Chain-of-Thought) 과정을 3~5개의 간결한 핵심 단계로 작성해주세요. 각 단계는 답변 생성 모델을 위한 가이드라인 역할을 합니다. 답변 형식은 단계별 목록 형태로 명확하게 작성해주세요.

사용자 질문: "{user_input}"

단계별 사고 과정 (예시:

사용자의 핵심 질문 파악: ...
관련 정보(Slot, RAG) 검토: ...
주요 고려사항 및 비교 분석: ...
최종 답변 구성 방향 설정: ... ): 
"""

# [신규] Hard 난이도 상세 CoT 지침 생성 프롬프트
HARD_COT_INSTRUCTION_GENERATION_PROMPT_TEMPLATE = """
다음 사용자 질문은 복잡도가 높아 답변 생성 시 상세한 단계별 접근이 필요합니다. 이 질문에 답변하기 위한 구체적인 사고 과정 지침(Instruction)을 체계적으로 작성해주세요. 지침은 답변 생성 모델이 따라야 할 명확한 로드맵을 제공해야 합니다. RAG 정보 활용 방안, 추론 과정, 답변 구조 등을 포함하여 상세하게 기술해주세요.

사용자 질문: "{user_input}"

상세 사고 과정 지침:
"""

# --- 함수 구현 ---

async def classify_complexity_level(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    GPT(설정된 모델)를 활용하여 질문 복잡도를 "easy", "medium", "hard"로 분류합니다.
    """
    logger.info(f"Classifying complexity for input: '{user_input[:50]}...'")
    prompt = COMPLEXITY_CLASSIFICATION_PROMPT_TEMPLATE.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]

    classification_model = router_config.get('classification_model', 'gpt-4o') # 설정값 사용
    logger.debug(f"Using model for complexity classification: {classification_model}")

    try:
        response_data = await call_gpt_async(
            messages=messages, model=classification_model, temperature=0.1, max_tokens=100, session=session
        )

        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw complexity classification response: {response_content}")

            # Markdown 코드 블록 및 JSON 파싱 처리 강화
            clean_content = response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:-3].strip()
            elif clean_content.startswith("```"):
                   clean_content = clean_content[3:-3].strip()

            try:
                json_start = clean_content.find('{')
                json_end = clean_content.rfind('}')
                if json_start != -1 and json_end != -1:
                     json_string = clean_content[json_start:json_end+1]
                else:
                     json_string = clean_content

                classification_result = json.loads(json_string)
                level = classification_result.get("complexity_level", "easy").lower()

                if level in ["easy", "medium", "hard"]:
                    logger.info(f"Question complexity classified as: {level}")
                    return level
                else:
                    logger.warning(f"Unexpected classification level value: '{level}'. Defaulting to 'easy'.")
                    return "easy"
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from complexity classification response: {e}. Content: '{clean_content}'")
                return "easy" # 파싱 실패 시 기본값
        else:
            logger.warning("No valid response from complexity classification.")
            return "easy" # API 응답 실패 시 기본값

    except Exception as e:
        logger.error(f"An error occurred during complexity classification: {e}", exc_info=True)
        return "easy" # 예외 발생 시 기본값
    
async def generate_cot_steps_async(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]: # 반환 타입을 문자열로 변경 (리스트 대신)
    """'medium' 난이도 질문에 대한 CoT 스텝을 생성합니다."""
    logger.info("Generating CoT steps for medium complexity question...")
    prompt = MEDIUM_COT_STEP_GENERATION_PROMPT_TEMPLATE.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]
    model = router_config.get('medium_cot_step_generation_model', 'gpt-4o')
    logger.debug(f"Using model for medium CoT step generation: {model}")
    try:
        response_data = await call_gpt_async(messages=messages, model=model, temperature=0.5, max_tokens=200, session=session)
        if response_data and response_data.get("choices"):
            steps_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
            logger.info("CoT steps generated successfully.")
            logger.debug(f"Generated CoT steps:\n{steps_text}")
            # 여러 줄의 스텝을 하나의 문자열로 반환
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
    """'hard' 난이도 질문에 대한 상세 CoT 지침을 생성합니다."""
    logger.info("Generating detailed CoT instructions for hard complexity question...")
    prompt = HARD_COT_INSTRUCTION_GENERATION_PROMPT_TEMPLATE.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]
    model = router_config.get('hard_cot_instruction_generation_model', 'gpt-4o')
    logger.debug(f"Using model for hard CoT instruction generation: {model}")
    try:
        response_data = await call_gpt_async(messages=messages, model=model, temperature=0.5, max_tokens=400, session=session) # 더 긴 지침 위해 토큰 증가
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
    logger.info("Determining routing and reasoning strategy...")
    # 1. 복잡도 분류
    complexity_level = await classify_complexity_level(user_input, session)

    # 2. 최종 응답 모델 선택
    model_map = router_config.get('routing_map', {})
    # config 파일에 easy, medium, hard 키가 없거나 값이 없을 경우 대비 기본값 설정 강화
    default_easy_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
    chosen_model = model_map.get(complexity_level, default_easy_model)

    # 3. CoT 데이터 생성 (필요시)
    cot_data: Optional[str] = None
    if complexity_level == "medium":
        cot_data = await generate_cot_steps_async(user_input, session)
    elif complexity_level == "hard":
        # 사용자와 합의된 상세 CoT 지침 생성 방식 사용
        cot_data = await generate_hard_cot_instructions_async(user_input, session)
        # 만약 하위 질문 분해 방식을 구현한다면 여기서 해당 함수 호출
        # sub_questions = await generate_sub_questions_async(user_input, session)

    result = {
        "level": complexity_level,
        "model": chosen_model,
        "cot_data": cot_data # 생성된 CoT 스텝 또는 상세 지침 (문자열)
        # "sub_questions": sub_questions if complexity_level == "hard" else None # 하위 질문 방식 구현 시
    }
    logger.info(f"Routing decision: Level='{result['level']}', Model='{result['model']}', CoT data generated={'yes' if result['cot_data'] else 'no'}")
    return result

# --- 예시 사용법 (업데이트) ---
if __name__ == "__main__":
    import asyncio
    # 테스트 시 로깅 레벨 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_routing_and_reasoning():
        # 테스트 전에 config.yaml 로드 가능해야 함
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
                # cot_data는 길 수 있으므로 미리보기 또는 길이만 출력
                result_display = routing_result.copy() # 출력용 복사본
                if result_display.get('cot_data'):
                    result_display['cot_data_preview'] = result_display['cot_data'][:100] + '...'
                    del result_display['cot_data'] # 원본 cot_data는 출력에서 제외
                print(json.dumps(result_display, indent=2, ensure_ascii=False, default=str))
                print("-" * 30)

    # 스크립트 직접 실행
    try:
        asyncio.run(test_routing_and_reasoning())
    except FileNotFoundError:
         print("\nError: config.yaml not found. Please ensure it exists.")
    except Exception as e:
         print(f"\nAn error occurred during testing: {e}")