# chatbot/slot_extractor.py

import json
from typing import Dict, Any, Optional
import aiohttp # 비동기 HTTP 요청

# gpt_interface 모듈에서 비동기 호출 함수 임포트
from .gpt_interface import call_gpt_async

# --- Slot 추출을 위한 GPT 프롬프트 템플릿 ---
SLOT_EXTRACTION_PROMPT_TEMPLATE = """
다음 문장에서 브랜드, 사이즈, 발볼, 카테고리 정보를 JSON 형식으로 추출해줘.
- 브랜드는 나이키, 호카, 아디다스, 데카트론 등 스포츠 브랜드일 수 있어.
- 사이즈는 "270", "275mm", "US 9" 등의 형태로 표현될 수 있어.
- 발볼은 "좁아요", "넓은 편", "보통", "2E" 등으로 표현될 수 있어.
- 카테고리는 운동화, 러닝화, 트레킹화, 축구화 등 신발이나 스포츠 용품 종류일 수 있어.

찾지 못한 정보는 값으로 `null`을 사용해줘.
반드시 JSON 형식으로만 응답해줘. 다른 설명은 붙이지 마.

입력 문장: "{user_input}"

추출 결과 (JSON):
"""

async def extract_slots_with_gpt(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    GPT를 사용하여 사용자 입력에서 Slot 정보를 추출합니다.

    Args:
        user_input (str): 사용자의 입력 문장.
        session (aiohttp.ClientSession, optional): 재사용할 aiohttp 세션.

    Returns:
        Optional[Dict[str, Any]]: 추출된 Slot 정보 딕셔너리. 실패 시 None.
    """
    prompt = SLOT_EXTRACTION_PROMPT_TEMPLATE.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]

    # Slot 추출은 비교적 간단한 작업이므로 gpt-3.5-turbo 사용 고려 가능
    # 비용 및 속도 측면에서 유리할 수 있음
    model_for_extraction = "gpt-3.5-turbo" # 또는 필요에 따라 다른 모델

    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model_for_extraction,
            temperature=0.1, # 정확한 추출을 위해 낮은 temperature 설정
            max_tokens=150,  # JSON 응답 길이를 고려하여 설정
            session=session
        )

        if response_data and response_data.get("choices"):
            # GPT 응답에서 JSON 내용 추출 시도
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
            # 마크다운 코드 블록 제거 (```json ... ```)
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()
            elif response_content.strip().startswith("```"):
                 response_content = response_content.strip()[3:-3].strip()

            try:
                extracted_slots = json.loads(response_content)
                # null 값을 가진 키 제거 (선택 사항)
                # extracted_slots = {k: v for k, v in extracted_slots.items() if v is not None}
                print(f"Extracted Slots: {extracted_slots}") # 디버깅용
                return extracted_slots
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from GPT response: {e}")
                print(f"Raw response content: {response_content}")
                return None
        else:
            print("Failed to get valid response from GPT for slot extraction.")
            return None

    except Exception as e:
        print(f"An error occurred during slot extraction: {e}")
        return None

# --- 예시 사용법 ---
if __name__ == "__main__":
    import asyncio

    async def test_slot_extraction():
        test_input = "호카는 270mm 신었는데 발볼이 좀 좁았어요. 겨울용 트레킹화 추천해주세요."
        print(f"Testing slot extraction for input: '{test_input}'")
        async with aiohttp.ClientSession() as session:
            slots = await extract_slots_with_gpt(test_input, session=session)
            if slots:
                print("\nExtraction Successful:")
                print(json.dumps(slots, indent=2, ensure_ascii=False))
            else:
                print("\nExtraction Failed.")

        test_input_2 = "나이키 신발 사이즈 문의합니다."
        print(f"\nTesting slot extraction for input: '{test_input_2}'")
        async with aiohttp.ClientSession() as session:
            slots_2 = await extract_slots_with_gpt(test_input_2, session=session)
            if slots_2:
                print("\nExtraction Successful:")
                print(json.dumps(slots_2, indent=2, ensure_ascii=False))
            else:
                print("\nExtraction Failed.")

    asyncio.run(test_slot_extraction())