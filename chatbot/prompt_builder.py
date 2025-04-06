# chatbot/prompt_builder.py

import logging
from typing import List, Dict, Optional, Union

# 필요한 모듈 임포트
from .conversation_state import ConversationState
from .config_loader import get_config # 설정 로더 임포트

# 로거 설정 및 설정 로드
logger = logging.getLogger(__name__)
# config = get_config() # 필요 시점에 함수 내에서 호출

# --- 프롬프트 구성 함수 (수정됨) ---

def build_final_prompt(
    user_query: str,
    conversation_state: ConversationState,
    rag_results: List[Dict],
    cot_data: Optional[str] = None, # 수정: CoT 데이터(문자열) 받음
) -> List[Dict[str, str]]:
    """
    모든 구성요소를 조합하여 최종 GPT API 요청 메시지를 생성합니다.
    Config에서 기본 프롬프트, 헤더, 지시문 등을 로드합니다.
    CoT 데이터(스텝 또는 지침)를 시스템 프롬프트에 포함합니다.
    """
    config = get_config() # 함수 실행 시 최신 설정 로드
    logger.info("Building final prompt...")

    # 설정에서 프롬프트 관련 값들 로드
    prompt_config = config.get('prompts', {})
    prompt_options = config.get('prompt_options', {})
    rag_config = config.get('rag', {})

    # 기본 시스템 프롬프트 (config에서)
    system_prompt_base = prompt_config.get('default_system_prompt', "당신은 AI 상담원입니다.")

    # 1. 시스템 프롬프트 구성 시작
    system_prompt_parts = [system_prompt_base]

    # 지시문 추가 (config에 따라)
    grounding_instruction = prompt_config.get('grounding_instruction')
    tone_instruction = prompt_config.get('tone_instruction')
    length_constraint_instruction = prompt_config.get('length_constraint_instruction')

    if prompt_options.get('use_rag_grounding_instruction', True) and grounding_instruction:
        system_prompt_parts.append(grounding_instruction)
    if prompt_options.get('use_tone_instruction', True) and tone_instruction:
        system_prompt_parts.append(tone_instruction)
    if prompt_options.get('use_length_constraint_instruction', True) and length_constraint_instruction:
        system_prompt_parts.append(length_constraint_instruction)


# 2. 대화 요약 추가 (config에 따라)
    if prompt_options.get('include_summary_in_prompt', True):
        summary = conversation_state.get_summary()
        if summary:
            logger.debug("Adding conversation summary to prompt.")
            system_prompt_parts.append("\n[요약된 이전 대화 맥락]")
            system_prompt_parts.append(summary)

    # 3. 대화 상태(Slot) 정보 추가 (config에 따라)
    slot_context_header = prompt_config.get('slot_context_header', "[파악된 사용자 정보]")
    if prompt_options.get('include_slots_in_prompt', True):
        slots = conversation_state.get_slots()
        if slots:
            logger.debug(f"Adding slots to prompt: {list(slots.keys())}")
            slot_info = f"\n{slot_context_header}\n"
            slot_info += "\n".join([f"- {key}: {value}" for key, value in slots.items() if value is not None])
            system_prompt_parts.append(slot_info)

    # 4. RAG 검색 결과 추가 (config에 따라)
    rag_context_header = prompt_config.get('rag_context_header', "[참고 문서 정보]")
    if prompt_options.get('include_rag_context_in_prompt', True):
        rag_results_count_limit = prompt_options.get('rag_results_count_limit', 3)
        if rag_results and rag_results_count_limit > 0:
            # 실제 포함될 결과 수 계산
            num_results_to_add = min(len(rag_results), rag_results_count_limit)
            logger.debug(f"Adding top {num_results_to_add} RAG results to prompt.")
            rag_context = f"\n{rag_context_header}\n"
            for i, doc in enumerate(rag_results[:num_results_to_add]):
                rag_context += f"- 문서 {i+1}: {doc.get('text', '내용 없음')}\n"
                # logger.debug(f"  RAG Doc {i+1} ID: {doc.get('id', 'N/A')}, Score: {doc.get('similarity_score', 'N/A'):.4f}") # DEBUG
            system_prompt_parts.append(rag_context.strip())
        else:
            logger.debug("No RAG results to add or limit is 0.")
            system_prompt_parts.append(f"\n{rag_context_header}\n- 관련된 문서 정보를 찾지 못했습니다.")

    # 5. [삭제] Few-shot 예시 추가 로직 제거


# 6. [수정] CoT 데이터 추가 (있을 경우)
    if cot_data:
        logger.debug("Adding CoT data (steps or instructions) to prompt.")
        cot_section = "\n[단계별 사고 가이드 (CoT)]\n"
        cot_section += cot_data # 문자열 형태의 스텝 또는 지침 직접 추가
        system_prompt_parts.append(cot_section)
        # 모델이 CoT 가이드를 따르도록 지시 추가
        system_prompt_parts.append("\n(중요: 답변 생성 시, 반드시 위에 제시된 '[단계별 사고 가이드 (CoT)]'를 참고하여 그 논리적 흐름과 지침에 따라 단계적으로 생각하고 답변을 구성하세요.)")


    # 최종 시스템 프롬프트 조합
    final_system_prompt = "\n\n".join(system_prompt_parts) # 문단 구분을 위해 \n\n 사용
    logger.info(f"Final system prompt length: {len(final_system_prompt)} characters.")
    # 시스템 프롬프트가 너무 길 경우 DEBUG 레벨에서만 전체 출력
    logger.debug(f"Final System Prompt Preview:\n------\n{final_system_prompt[:500]}...\n------")

    # 7. 최종 메시지 리스트 구성
    messages = [{"role": "system", "content": final_system_prompt}]
    messages.append({"role": "user", "content": user_query})

    logger.info("Final prompt messages constructed.")
    return messages

# --- 예시 사용법 (수정됨) ---
if __name__ == "__main__":
    # 로깅 기본 설정 (테스트용)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 테스트 실행 전 config.yaml 로드 시도
    try:
        get_config()
        logger.info("Configuration loaded successfully for prompt builder test.")
    except Exception as e:
        logger.error(f"Failed to load configuration for test: {e}. Using default values might occur.")
        # 테스트 계속 진행 (기본값 사용) 또는 여기서 중단 결정 가능


    # 가상의 데이터 생성
    state = ConversationState()
    state.update_slots({"brand": "나이키", "size": "270mm", "foot_width": "넓은 편", "product_category": "러닝화"})
    state.update_summary("고객은 나이키 270mm 신발 착용 경험이 있으며, 발볼이 넓어 데카트론 러닝화를 알아보고 있음.")
    rag_data = [
        {"id": "doc1", "text": "데카트론 킵런 시리즈는 발볼이 여유있게 나왔습니다. 정사이즈를 추천합니다.", "metadata": {"source_file": "decathlon_shoes.txt"}},
        {"id": "doc2", "text": "나이키 신발은 일반적으로 발볼이 좁게 디자인되는 경향이 있습니다.", "metadata": {"source_file": "nike_info.txt"}},
    ]
    user_input = "제가 발볼이 넓은 편인데, 데카트론 킵런 러닝화는 275mm를 신어야 할까요?"
    medium_cot_steps = "- 1단계: 사용자의 발볼 너비('넓은 편')와 기존 신발 브랜드/사이즈('나이키 270mm') 정보를 확인한다.\n- 2단계: RAG 정보에서 데카트론 킵런 시리즈의 발볼 특징('여유있게 나옴')과 사이즈 추천('정사이즈') 정보를 찾는다.\n- 3단계: 나이키 신발의 발볼 특징('좁게 디자인')을 고려하여 데카트론 정사이즈(270mm)가 적합할 수 있음을 설명한다.\n- 4단계: 사이즈는 개인차가 있을 수 있으므로, 가능하면 매장 방문 착용을 권장하며 답변을 마무리한다."
    hard_cot_instructions = "1. 사용자의 문제 상황(겨울 산행 시 발 시려움, 평발, 넓은 발볼)과 요구사항(보온성, 방수성, 편안함)을 명확히 인지한다.\n2. RAG 정보에서 데카트론 등산화 중 보온/방수 기능이 강화된 모델을 검색한다.\n3. 검색된 모델들의 특징(소재, 기능, 사용자 후기 등)을 분석하고, 특히 평발/넓은 발볼 사용자에게 적합한지 평가한다.\n4. 기존 MH500 모델과 비교하여 새로운 추천 모델의 장점(보온성, 방수성, 착화감 개선 등)을 구체적으로 설명한다.\n5. 최종적으로 1-2개 모델을 추천하고, 선택 이유를 요약하여 제시한다."


    # 시나리오 1: Medium 난이도 (CoT 스텝 데이터 제공)
    print("\n--- Scenario 1: Medium Complexity (with CoT Steps) ---")
    final_messages_medium = build_final_prompt(
        user_query=user_input,
        conversation_state=state,
        rag_results=rag_data,
        cot_data=medium_cot_steps # CoT 스텝 문자열 전달
    )
    # print(json.dumps(final_messages_medium, indent=2, ensure_ascii=False)) # 전체 메시지 확인용

    # 시나리오 2: Easy 난이도 (CoT 데이터 없음)
    print("\n--- Scenario 2: Easy Complexity (No CoT data) ---")
    final_messages_easy = build_final_prompt(
        user_query="반품 규정 알려주세요.",
        conversation_state=ConversationState(),
        rag_results=[],
        cot_data=None
    )
    # print(json.dumps(final_messages_easy, indent=2, ensure_ascii=False))

    # 시나리오 3: Hard 난이도 (상세 CoT 지침 제공)
    print("\n--- Scenario 3: Hard Complexity (with Detailed CoT Instructions) ---")
    final_messages_hard = build_final_prompt(
         user_query="작년에 구매한 퀘차 등산화(MH500)...(이하 생략)", # 전체 질문
         conversation_state=ConversationState(),
         rag_results=[], # RAG 결과는 별도 검색 가정
         cot_data=hard_cot_instructions # 상세 CoT 지침 문자열 전달
     )
    # print(json.dumps(final_messages_hard, indent=2, ensure_ascii=False))