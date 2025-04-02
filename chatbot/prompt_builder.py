# chatbot/prompt_builder.py

from typing import List, Dict, Optional

# 대화 상태 클래스 임포트
from .conversation_state import ConversationState

# --- 프롬프트 템플릿 ---
# 설계 문서에 정의된 내용을 기반으로 구성
# 필요에 따라 동적으로 요소를 추가/제외할 수 있도록 함수로 만듦

def build_final_prompt(
    system_prompt_base: str,
    user_query: str,
    conversation_state: ConversationState,
    rag_results: List[Dict],
    use_cot: bool = False, # Chain-of-Thought 적용 여부
    few_shot_examples: Optional[List[Dict[str, str]]] = None # Few-shot 예시
) -> List[Dict[str, str]]:
    """
    모든 구성요소를 조합하여 최종 GPT API 요청 메시지를 생성합니다.

    Args:
        system_prompt_base (str): 기본적인 시스템 역할/지침.
        user_query (str): 사용자의 현재 질문.
        conversation_state (ConversationState): 현재 대화 상태 객체.
        rag_results (List[Dict]): RAG 검색 결과 리스트.
        use_cot (bool): Chain-of-Thought 프롬프팅 적용 여부.
        few_shot_examples (Optional[List[Dict[str, str]]]): 제공할 Few-shot 예시.

    Returns:
        List[Dict[str, str]]: OpenAI API ChatCompletion 형식의 메시지 리스트.
    """

    # 1. 시스템 프롬프트 구성
    system_prompt_parts = [system_prompt_base]
    system_prompt_parts.append("반드시 제공된 [참고 문서 정보] 내용만 바탕으로 대답하고, 문서에 없는 내용은 상상해서 답변하지 마세요.")
    system_prompt_parts.append("확실하지 않거나 정보가 없으면 '해당 정보는 확인되지 않았습니다' 또는 '제가 알 수 없는 내용입니다' 와 같이 솔직하게 답변하세요.")
    system_prompt_parts.append("고객 응대는 항상 공식적이고 정중한 말투를 사용하며, 2~3문장 이내로 간결하게 핵심만 요약해서 답변해주세요.") # 응답 길이 제한

    # 2. 대화 요약 추가 (있을 경우)
    summary = conversation_state.get_summary()
    if summary:
        system_prompt_parts.append("\n[요약된 이전 대화 맥락]")
        system_prompt_parts.append(summary)

    # 3. 대화 상태(Slot) 정보 추가 (있을 경우)
    slots = conversation_state.get_slots()
    if slots:
        slot_info = "\n[현재까지 파악된 고객 정보 (Slots)]\n"
        slot_info += "\n".join([f"- {key}: {value}" for key, value in slots.items() if value is not None])
        system_prompt_parts.append(slot_info)

    # 4. RAG 검색 결과 추가 (있을 경우)
    if rag_results:
        rag_context = "\n[참고 문서 정보]\n"
        # 상위 N개 결과만 사용 (예: 3개)
        for i, doc in enumerate(rag_results[:3]):
            rag_context += f"- 문서 {i+1}: {doc.get('text', '내용 없음')}\n" # Chunk 텍스트 삽입
            # 필요시 메타데이터(브랜드, 카테고리 등) 추가 정보 포함 가능
            # rag_context += f"  (출처: {doc.get('metadata', {}).get('source_file', '알 수 없음')}, 브랜드: {doc.get('brand', 'N/A')})\n"
        system_prompt_parts.append(rag_context.strip())
    else:
        system_prompt_parts.append("\n[참고 문서 정보]\n- 관련된 문서 정보를 찾지 못했습니다.") # RAG 결과 없을 때 명시

    # 5. Few-shot 예시 추가 (있을 경우)
    if few_shot_examples:
        few_shot_prompt = "\n[답변 형식 예시]\n"
        for example in few_shot_examples[:2]: # 너무 많지 않게 1~2개 정도만
            few_shot_prompt += f"Q: {example.get('question', '')}\nA: {example.get('answer', '')}\n"
        system_prompt_parts.append(few_shot_prompt.strip())

    final_system_prompt = "\n".join(system_prompt_parts)

    # 6. 최종 메시지 리스트 구성
    messages = []
    messages.append({"role": "system", "content": final_system_prompt})

    # 7. CoT 유도 프롬프트 추가 (필요시)
    if use_cot:
        # CoT를 위한 중간 사고 단계를 사용자 메시지 앞에 추가하거나,
        # 시스템 프롬프트 마지막에 "Let's think step by step" 같은 지시 추가 가능
        # 여기서는 사용자 질문 수정 없이 시스템 프롬프트에 지시 추가 형태로 가정
        # (더 복잡한 CoT는 별도 함수나 클래스로 분리 고려)
        # 예: final_system_prompt += "\n\n답변하기 전에, 먼저 단계별로 어떻게 답할지 생각해보세요."
        # 또는 사용자 질문을 수정하는 방식:
        # user_query_with_cot = user_query + "\n\n(답변하기 전에 단계별로 생각해주세요)"
        # messages.append({"role": "user", "content": user_query_with_cot})
        pass # CoT 적용 방식 결정 후 구현

    messages.append({"role": "user", "content": user_query})

    print("\n--- Built Final Prompt Messages ---") # 디버깅용
    # print(json.dumps(messages, indent=2, ensure_ascii=False))
    print(f"System Prompt Length: {len(final_system_prompt)}")
    print(f"User Query: {user_query}")
    print("-" * 30)

    return messages

# --- 기본 시스템 프롬프트 정의 ---
DEFAULT_SYSTEM_PROMPT = "당신은 스포츠 전문 브랜드 '데카트론 코리아'의 AI 고객 서비스 상담원입니다."

# --- 예시 사용법 ---
if __name__ == "__main__":
    # 가상의 데이터 생성
    state = ConversationState()
    state.update_slots({"brand": "나이키", "size": "270mm", "foot_width": "넓은 편", "category": "러닝화"})
    state.update_summary("고객은 나이키 270mm 신발 착용 경험이 있으며, 발볼이 넓어 데카트론 러닝화를 알아보고 있음.")

    rag_data = [
        {"id": "doc1", "text": "데카트론 킵런 시리즈는 발볼이 여유있게 나왔습니다. 정사이즈를 추천합니다.", "brand": "데카트론", "category": "러닝화"},
        {"id": "doc2", "text": "나이키 신발은 일반적으로 발볼이 좁게 디자인되는 경향이 있습니다.", "brand": "나이키", "category": "사이즈 가이드"},
        {"id": "doc3", "text": "사이즈 교환은 구매 후 30일 이내에 가능합니다.", "brand": "데카트론", "category": "정책"}
    ]

    user_input = "제가 발볼이 넓은 편인데, 데카트론 킵런 러닝화는 275mm를 신어야 할까요?"

    few_shots = [
        {"question": "호카 280 신는데 데카트론 트레킹화 사이즈는?", "answer": "호카 280mm 신으시면 데카트론 트레킹화는 280mm를 추천드립니다."},
        {"question": "반품 어떻게 하나요?", "answer": "구매하신 매장 또는 온라인 고객센터를 통해 접수 가능하며, 구매일로부터 30일 이내에 가능합니다."}
    ]

    # CoT 사용, Few-shot 사용 시나리오
    final_messages = build_final_prompt(
        system_prompt_base=DEFAULT_SYSTEM_PROMPT,
        user_query=user_input,
        conversation_state=state,
        rag_results=rag_data,
        use_cot=True, # CoT 사용하도록 설정
        few_shot_examples=few_shots # Few-shot 예시 제공
    )

    # print("\n--- Final Messages for GPT API (with CoT and Few-shots) ---")
    # print(json.dumps(final_messages, indent=2, ensure_ascii=False))

    # RAG 결과가 없을 때 시나리오
    print("\n--- Scenario: No RAG results ---")
    final_messages_no_rag = build_final_prompt(
        system_prompt_base=DEFAULT_SYSTEM_PROMPT,
        user_query="오늘 날씨 어때요?",
        conversation_state=ConversationState(), # 빈 상태
        rag_results=[], # RAG 결과 없음
        use_cot=False,
        few_shot_examples=None
    )
    # print(json.dumps(final_messages_no_rag, indent=2, ensure_ascii=False))