# chatbot/model_router.py

import json
from typing import Optional
import aiohttp # 비동기 HTTP 요청

# gpt_interface 모듈에서 비동기 호출 함수 임포트
from .gpt_interface import call_gpt_async

# --- 모델 라우팅을 위한 GPT 프롬프트 템플릿 ---
MODEL_ROUTING_PROMPT_TEMPLATE = """
사용자의 질문이 복잡한 질문인지 분석해줘.
다음과 같은 기준으로 판단해:

1. 질문이 단순한 사실 확인(Factual)인지 (예: "반품 정책 알려줘", "매장 위치 어디야?"), 아니면 논리적 사고(Logical Reasoning)나 비교/추천/설명이 필요한지 (예: "A랑 B중에 뭐가 더 나아?", "왜 이 제품을 추천해?", "사이즈 어떻게 선택해야 해?")
2. 단일 질문인지, 여러 내용을 한 번에 묻는 복합 질문인지? (예: "사이즈 추천해주고, 재고도 확인해줘")
3. 제공된 정보(RAG) 외에 추가적인 추론이나 창의적인 답변 생성이 필요한지?

입력된 질문을 아래 기준 중 하나로 분류하고, 반드시 JSON 형식으로만 응답해줘. 다른 설명은 붙이지 마.
- "simple": 단순 질의 (GPT-3.5 사용 권장)
- "complex": 복잡 질의 (GPT-4 사용 권장)

출력 형식 예시:
{{
  "classification": "simple"
}}

사용자 질문: "{user_input}"

분석 결과 (JSON):
"""

async def classify_question_complexity(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    GPT를 활용하여 질문이 단순(simple)한지 복잡(complex)한지 분류합니다.

    Args:
        user_input (str): 사용자의 입력 문장.
        session (aiohttp.ClientSession, optional): 재사용할 aiohttp 세션.

    Returns:
        str: 분류 결과 ("simple" 또는 "complex"). 실패 시 기본값 "simple".
    """
    prompt = MODEL_ROUTING_PROMPT_TEMPLATE.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]

    # 라우팅 자체는 비교적 간단하므로 gpt-3.5-turbo 사용
    model_for_routing = "gpt-3.5-turbo"

    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=model_for_routing,
            temperature=0.1,
            max_tokens=50, # 분류 결과만 받으므로 짧게 설정
            session=session
        )

        if response_data and response_data.get("choices"):
            response_content = response_data["choices"][0].get("message", {}).get("content", "")
             # 마크다운 코드 블록 제거
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()
            elif response_content.strip().startswith("```"):
                 response_content = response_content.strip()[3:-3].strip()

            try:
                classification_result = json.loads(response_content)
                classification = classification_result.get("classification", "simple").lower()
                if classification in ["simple", "complex"]:
                    print(f"Question Complexity Classified as: {classification}") # 디버깅용
                    return classification
                else:
                    print(f"Unexpected classification value: {classification}. Defaulting to simple.")
                    return "simple" # 예상 외의 값이 오면 기본값 처리
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from GPT routing response: {e}")
                print(f"Raw response content: {response_content}")
                return "simple" # JSON 파싱 실패 시 기본값
        else:
            print("Failed to get valid response from GPT for routing.")
            return "simple" # API 응답 실패 시 기본값

    except Exception as e:
        print(f"An error occurred during model routing: {e}")
        return "simple" # 예외 발생 시 기본값

async def route_model_gpt_based(
    user_input: str,
    session: Optional[aiohttp.ClientSession] = None
) -> str:
    """
    GPT가 분류한 결과를 바탕으로 최종 사용할 모델 이름을 반환합니다.

    Args:
        user_input (str): 사용자의 입력 문장.
        session (aiohttp.ClientSession, optional): 재사용할 aiohttp 세션.

    Returns:
        str: 사용할 GPT 모델 이름 ("gpt-3.5-turbo" 또는 "gpt-4").
    """
    classification = await classify_question_complexity(user_input, session=session)
    # TODO: 추후 GPT-4-turbo 등 최신 모델 반영 고려
    return "gpt-4" if classification == "complex" else "gpt-3.5-turbo"

# --- 예시 사용법 ---
if __name__ == "__main__":
    import asyncio

    async def test_model_routing():
        test_inputs = [
            "나이키 270인데 어떤 사이즈 추천해줘?",
            "나이키 270mm랑 아디다스 275mm 신발 특징 비교해주고, 발볼 넓은 사람한테 뭐가 더 나을지 알려줘.",
            "반품 가능한가요?",
            "호카랑 아디다스 러닝화 디자인 차이점 설명해줘.",
            "이 신발이 왜 다른 것보다 더 좋은지 이유를 상세하게 설명해줄 수 있어?",
            "매장 전화번호 뭐에요?"
        ]
        async with aiohttp.ClientSession() as session:
            for input_text in test_inputs:
                print(f"\nTesting routing for: '{input_text}'")
                chosen_model = await route_model_gpt_based(input_text, session=session)
                print(f"--> Chosen Model: {chosen_model}")

    asyncio.run(test_model_routing())