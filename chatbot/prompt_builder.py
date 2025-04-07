# chatbot/prompt_builder.py (오류 수정 및 리팩토링 최종본)

import logging
from typing import List, Dict, Optional, Union

# 필요한 모듈 임포트
try:
    from .conversation_state import ConversationState
    from .config_loader import get_config
except ImportError as ie:
    print(f"ERROR (prompt_builder): Failed to import modules: {ie}. Check relative paths.")
    ConversationState = None
    get_config = None

# 로거 설정
logger = logging.getLogger(__name__)

# --- 프롬프트 구성 함수 ---
def build_final_prompt(
    user_query: str,
    conversation_state: 'ConversationState', # Type hint 수정
    rag_results: List[Dict],
    cot_data: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    모든 구성요소를 조합하여 최종 GPT API 요청 메시지를 생성합니다.
    Config에서 기본 프롬프트, 헤더, 지시문, 정적 메시지 등을 로드합니다.
    """
    if not ConversationState or not get_config:
        logger.error("Required modules not imported correctly in build_final_prompt.")
        # 비상용 메시지 반환
        return [{"role": "system", "content": "System error: Internal configuration failed."},
                {"role": "user", "content": user_query}]

    # config 로드
    try:
        config = get_config()
        prompt_config = config.get('prompts', {})
        prompt_options = config.get('prompt_options', {})
        if not prompt_config or not prompt_options:
             raise ValueError("Prompt configuration ('prompts' or 'prompt_options') is missing or empty.")
    except Exception as e:
        logger.error(f"Error loading configuration in prompt_builder: {e}", exc_info=True)
        return [{"role": "system", "content": "System error: Could not load prompt configuration."},
                {"role": "user", "content": user_query}]

    logger.info("Building final prompt...")

    # 기본 시스템 프롬프트
    system_prompt_base = prompt_config.get('default_system_prompt', "당신은 AI 상담원입니다.")

    # 시스템 프롬프트 구성 시작
    system_prompt_parts = [system_prompt_base]

    # 지시문 추가 (config에 따라)
    # .get() 사용 시 기본값 None 처리 주의
    grounding_instruction = prompt_config.get('grounding_instruction')
    tone_instruction = prompt_config.get('tone_instruction')
    length_constraint_instruction = prompt_config.get('length_constraint_instruction')

    if prompt_options.get('use_rag_grounding_instruction', True) and grounding_instruction:
        system_prompt_parts.append(grounding_instruction)
    if prompt_options.get('use_tone_instruction', True) and tone_instruction:
        system_prompt_parts.append(tone_instruction)
    if prompt_options.get('use_length_constraint_instruction', True) and length_constraint_instruction:
        system_prompt_parts.append(length_constraint_instruction)

    # 대화 요약 추가 (config에 따라)
    summary_context_header = prompt_config.get('summary_context_header', "[요약된 이전 대화 맥락]")
    if prompt_options.get('include_summary_in_prompt', True):
        summary = conversation_state.get_summary()
        if summary:
            logger.debug("Adding conversation summary to prompt.")
            system_prompt_parts.append(f"\n{summary_context_header}")
            system_prompt_parts.append(summary)

    # 대화 상태(Slot) 정보 추가 (config에 따라)
    slot_context_header = prompt_config.get('slot_context_header', "[파악된 사용자 정보]")
    if prompt_options.get('include_slots_in_prompt', True):
        slots = conversation_state.get_slots()
        if slots:
            logger.debug(f"Adding slots to prompt: {list(slots.keys())}")
            slot_info = f"\n{slot_context_header}\n"
            # 값이 있는 슬롯만 추가
            slot_info += "\n".join([f"- {key}: {value}" for key, value in slots.items() if value is not None])
            system_prompt_parts.append(slot_info)

    # RAG 검색 결과 추가 (config에 따라)
    rag_context_header = prompt_config.get('rag_context_header', "[참고 문서 정보]")
    rag_fallback_message = prompt_config.get('rag_fallback_message', "- 관련된 문서 정보를 찾지 못했습니다.")
    if prompt_options.get('include_rag_context_in_prompt', True):
        rag_results_count_limit = prompt_options.get('rag_results_count_limit', 3)
        # rag_results_count_limit이 0보다 큰지 확인
        if rag_results_count_limit > 0 and rag_results:
            # 실제 포함될 결과 수 계산
            num_results_to_add = min(len(rag_results), rag_results_count_limit)
            logger.debug(f"Adding top {num_results_to_add} RAG results to prompt.")
            rag_context = f"\n{rag_context_header}\n"
            for i, doc in enumerate(rag_results[:num_results_to_add]):
                # 문서 텍스트가 비어있지 않은 경우만 추가
                doc_text = doc.get('text', '').strip()
                if doc_text:
                    rag_context += f"- 문서 {i+1}: {doc_text}\n"
                else:
                     logger.debug(f"Skipping empty RAG document (index {i}).")
            # 문서가 하나라도 추가되었는지 확인 후 parts에 추가
            if len(rag_context.strip()) > len(rag_context_header):
                 system_prompt_parts.append(rag_context.strip())
            else: # 유효한 문서가 없었던 경우
                 logger.debug("No valid RAG documents found to add.")
                 system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")
        else:
            logger.debug("No RAG results to add or limit is 0.")
            system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")

    # CoT 데이터 추가 (있을 경우)
    cot_context_header = prompt_config.get('cot_context_header', "[단계별 사고 가이드 (CoT)]")
    cot_follow_instruction = prompt_config.get('cot_follow_instruction', "(중요: 답변 생성 시, CoT 가이드를 따르세요.)")
    if cot_data: # cot_data가 None이나 빈 문자열이 아닌 경우
        logger.debug("Adding CoT data (steps or instructions) to prompt.")
        cot_section = f"\n{cot_context_header}\n"
        cot_section += cot_data # 문자열 형태의 스텝 또는 지침 직접 추가
        system_prompt_parts.append(cot_section)
        # 지시문 추가 (지시문 자체도 None이 아닐 경우)
        if cot_follow_instruction:
             system_prompt_parts.append(f"\n{cot_follow_instruction}")

    # 최종 시스템 프롬프트 조합
    final_system_prompt = "\n\n".join(filter(None, system_prompt_parts)) # None이나 빈 문자열 필터링
    logger.info(f"Final system prompt length: {len(final_system_prompt)} characters.")
    logger.debug(f"Final System Prompt Preview:\n------\n{final_system_prompt[:500]}...\n------")

    # 최종 메시지 리스트 구성
    messages = [{"role": "system", "content": final_system_prompt}]
    # 사용자 질문 추가
    messages.append({"role": "user", "content": user_query})

    logger.info("Final prompt messages constructed.")
    return messages

# --- 예시 사용법 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        get_config()
        logger.info("Configuration loaded successfully for prompt builder test.")
    except Exception as e:
        logger.error(f"Failed to load configuration for test: {e}. Using default values might occur.")

    # ConversationState 임포트 성공 여부 확인
    if not ConversationState:
         print("ConversationState module could not be imported. Exiting test.")
         exit()


    # 가상의 데이터 생성
    state = ConversationState()
    state.update_slots({"brand": "나이키", "size": "270mm", "foot_width": "넓은 편", "product_category": "러닝화"})
    state.update_summary("고객은 나이키 270mm 신발 착용 경험이 있으며, 발볼이 넓어 데카트론 러닝화를 알아보고 있음.")
    rag_data = [
        {"id": "doc1", "text": "데카트론 킵런 시리즈는 발볼이 여유있게 나왔습니다. 정사이즈를 추천합니다.", "metadata": {"source_file": "decathlon_shoes.txt"}},
        {"id": "doc2", "text": "나이키 신발은 일반적으로 발볼이 좁게 디자인되는 경향이 있습니다.", "metadata": {"source_file": "nike_info.txt"}},
        {"id": "doc3", "text": "   ", "metadata": {"source_file": "empty_doc.txt"}}, # 빈 문서 테스트
    ]
    user_input = "제가 발볼이 넓은 편인데, 데카트론 킵런 러닝화는 275mm를 신어야 할까요?"
    medium_cot_steps = "- 1단계: 사용자의 발볼 너비('넓은 편')와 기존 신발 브랜드/사이즈('나이키 270mm') 정보를 확인한다.\n- 2단계: RAG 정보에서 데카트론 킵런 시리즈의 발볼 특징('여유있게 나옴')과 사이즈 추천('정사이즈') 정보를 찾는다.\n- 3단계: 나이키 신발의 발볼 특징('좁게 디자인')을 고려하여 데카트론 정사이즈(270mm)가 적합할 수 있음을 설명한다.\n- 4단계: 사이즈는 개인차가 있을 수 있으므로, 가능하면 매장 방문 착용을 권장하며 답변을 마무리한다."
    hard_cot_instructions = "1. 사용자의 문제 상황(겨울 산행 시 발 시려움, 평발, 넓은 발볼)과 요구사항(보온성, 방수성, 편안함)을 명확히 인지한다.\n2. RAG 정보에서 데카트론 등산화 중 보온/방수 기능이 강화된 모델을 검색한다.\n3. 검색된 모델들의 특징(소재, 기능, 사용자 후기 등)을 분석하고, 특히 평발/넓은 발볼 사용자에게 적합한지 평가한다.\n4. 기존 MH500 모델과 비교하여 새로운 추천 모델의 장점(보온성, 방수성, 착화감 개선 등)을 구체적으로 설명한다.\n5. 최종적으로 1-2개 모델을 추천하고, 선택 이유를 요약하여 제시한다."

    # 시나리오별 테스트
    scenarios = {
        "1_medium_with_cot": (user_input, state, rag_data, medium_cot_steps),
        "2_easy_no_cot": ("반품 규정 알려주세요.", ConversationState(), [], None),
        "3_hard_with_cot": ("작년에 구매한 퀘차 등산화(MH500)...", ConversationState(), [], hard_cot_instructions),
        "4_no_rag": ("푸마 신발 사이즈 가이드 알려줘", ConversationState(), [], None),
        "5_empty_rag": ("킵런 신발 어때요?", ConversationState(), [{"id":"doc_empty", "text": ""}], None), # RAG 결과 텍스트가 비었을 때
    }

    for name, args in scenarios.items():
         print(f"\n--- Scenario: {name} ---")
         final_messages = build_final_prompt(*args)
         print("Generated Messages (System Prompt Preview):")
         # 시스템 프롬프트만 출력 (너무 길 수 있으므로)
         if final_messages and isinstance(final_messages, list) and len(final_messages) > 0:
              print(final_messages[0].get('content', 'ERROR: Could not get system prompt')[:1000] + "...") # 미리보기 길이 조절
         else:
              print("ERROR: Could not generate messages.")
         print("-" * 20)