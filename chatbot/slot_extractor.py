# chatbot/slot_extractor.py

import json
from typing import Dict, Any, Optional
import aiohttp
import logging

# 필요한 모듈 임포트
from .gpt_interface import call_gpt_async
from .config_loader import get_config # 설정 로더 임포트

# 로거 설정
logger = logging.getLogger(__name__)
# config는 필요시 함수 내에서 get_config()로 호출

# --- [수정] Slot 추출을 위한 GPT 프롬프트 템플릿 (확장됨) ---
SLOT_EXTRACTION_PROMPT_TEMPLATE = """
다음 사용자 입력 문장에서 아래 정의된 Slot 정보를 JSON 형식으로 추출해줘.

**추출 대상 Slot:**
* `customer_gender`: 사용자 성별 (예: 남성, 여성)
* `customer_age_group`: 사용자 연령대 (예: 20대, 40대, 청소년, 성인)
* `customer_skill_level`: 특정 활동 관련 사용자 숙련도 (예: 러닝 초보, 등산 중급)
* `context_activity`: 사용하려는 활동이나 스포츠 (예: 달리기, 등산, 캠핑, 축구, 일상)
* `context_environment`: 활동 환경 (예: 로드, 트레일, 실내 체육관, 겨울 산, 도심)
* `context_goal`: 활동 목적 (예: 다이어트, 대회 준비, 주말 레저, 편안한 착용감)
* `brand`: 언급된 브랜드 명 (예: 데카트론, 나이키, 킵런, 퀘차, 아디다스)
* `product_category`: 언급된 제품 분류 (예: 러닝화, 등산화, 텐트, 배낭, 자켓, 운동화)
* `size`: 언급된 사이즈 (예: 270mm, M, 95, US 10, L)
* `foot_width`: 발볼 너비 관련 언급 (예: 발볼 넓음, 보통, 좁은 편, 2E)
* `product_feature_preference`: 선호하는 제품 특징 (예: 쿠션 좋은 것, 가벼운 것, 방수 기능, 통기성)
* `product_budget_range`: 희망 예산 범위 (예: 10만원대, 20만원 이하, 5만원 미만)
* `query_type`: 질문의 주된 의도 (예: 제품 추천, 사이즈 문의, 정보 검색, 비교 요청, 재고 확인, 매장 문의)
* `problem_description`: 사용자가 겪고 있는 문제 상황 (예: 신발 특정 부위 통증, 텐트 설치 어려움, 반품 문의)

**규칙:**
- 문장에서 해당 정보를 찾을 수 없으면 값으로 `null`을 사용해줘.
- 값은 최대한 간결하게 핵심 내용만 추출해줘.
- 반드시 아래 예시와 같이 JSON 형식으로만 응답하고, 다른 설명은 절대 붙이지 마.

**출력 형식 예시:**
```json
{{
  "customer_gender": null,
  "customer_age_group": "30대",
  "customer_skill_level": "러닝 초보",
  "context_activity": "러닝",
  "context_environment": "로드",
  "context_goal": "다이어트",
  "brand": "데카트론",
  "product_category": "러닝화",
  "size": "260mm",
  "foot_width": "넓은 편",
  "product_feature_preference": "쿠션 좋은 것",
  "product_budget_range": "10만원 이하",
  "query_type": "제품 추천",
  "problem_description": null
}}

입력 문장: "{user_input}"

추출 결과 (JSON):
"""

async def extract_slots_with_gpt(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    GPT를 사용하여 사용자 입력에서 Slot 정보를 추출합니다. (확장된 Slot 목록 적용)
    """
    config = get_config() # 함수 내에서 config 로드
    logger.info(f"Attempting to extract slots from input: '{user_input[:50]}...'")
    prompt = SLOT_EXTRACTION_PROMPT_TEMPLATE.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]

    # 설정 파일에서 Slot 추출용 모델 및 파라미터 가져오기
    task_config = config.get('tasks', {})
    gen_config = config.get('generation', {})
    model_for_extraction = task_config.get('slot_extraction_model', 'gpt-3.5-turbo')
    # Slot이 많아졌으므로 max_tokens 약간 증가 고려 (config 추가 또는 기본값 사용)
    # config.yaml에 generation 하위에 slot_extraction_max_tokens: 250 추가 가능
    max_tokens_for_extraction = gen_config.get('slot_extraction_max_tokens', 250) # 250으로 증가 예시

    try:
        logger.debug(f"Calling GPT for slot extraction using model: {model_for_extraction}")
        response_data = await call_gpt_async(
            messages=messages,
            model=model_for_extraction,
            temperature=0.1, # 낮은 온도 유지 (정확성)
            max_tokens=max_tokens_for_extraction, # 조정된 max_tokens
            session=session
        )

        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw response content from slot extractor GPT: {response_content[:300]}...") # DEBUG 레벨

            # 마크다운 코드 블록 제거 (정규식 사용 등으로 더 견고하게 만들 수 있음)
            clean_response_content = response_content.strip()
            if clean_response_content.startswith("```json"):
                clean_response_content = clean_response_content[7:-3].strip()
            elif clean_response_content.startswith("```"):
                   clean_response_content = clean_response_content[3:-3].strip()

            # JSON 파싱 시도
            try:
                # 때때로 JSON 앞뒤로 불필요한 텍스트가 붙는 경우가 있어, { } 를 찾는 시도 추가
                json_start = clean_response_content.find('{')
                json_end = clean_response_content.rfind('}')
                if json_start != -1 and json_end != -1:
                     json_string = clean_response_content[json_start:json_end+1]
                else:
                     json_string = clean_response_content # 못 찾으면 원본 사용

                extracted_slots = json.loads(json_string)
                # 추출된 Slot에서 null 값 제거 옵션 (필요시 주석 해제)
                # extracted_slots = {k: v for k, v in extracted_slots.items() if v is not None}
                logger.info(f"Successfully extracted slots: {list(extracted_slots.keys())}") # 로그 간결화
                logger.debug(f"Extracted slot values: {extracted_slots}") # 상세 값은 DEBUG
                return extracted_slots
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from slot extractor response: {e}. Cleaned content: '{clean_response_content}'")
                # 파싱 실패 시, 응답 자체를 반환하는 대신 None 반환 유지 (또는 다른 오류 처리)
                return None
        else:
            logger.warning("Failed to get a valid response from GPT for slot extraction.")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during slot extraction: {e}", exc_info=True)
        return None
    
    # --- 예시 사용법 (테스트 케이스 업데이트) ---
if __name__ == "__main__":
    import asyncio
    # 로깅 기본 설정 (테스트용)
    # 실제 실행 시에는 app.py 등에서 설정된 로거 사용됨
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_slot_extraction():
        # 테스트 실행 전에 config.yaml 파일이 로드 가능해야 함
        try:
             get_config() # 설정 로드 시도
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
        # aiohttp 세션 생성 (테스트용)
        async with aiohttp.ClientSession() as session:
            for test_input in test_inputs:
                print(f"\n--- Testing slot extraction for input: --- \n'{test_input}'")
                logger.info(f"Running test extraction for: '{test_input}'")
                slots = await extract_slots_with_gpt(test_input, session=session)
                if slots:
                    print("\nExtraction Successful:")
                    print(json.dumps(slots, indent=2, ensure_ascii=False))
                else:
                    print("\nExtraction Failed.")
                print("-" * 30)

    # 스크립트 직접 실행 시 이벤트 루프 생성 및 실행
    try:
        asyncio.run(test_slot_extraction())
    except FileNotFoundError:
         print("\nError: config.yaml not found. Please ensure it exists in the project root.")
    except Exception as e:
         print(f"\nAn error occurred during testing: {e}")