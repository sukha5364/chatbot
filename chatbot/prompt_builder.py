# chatbot/prompt_builder.py (요구사항 반영 최종본: 주석 보강 및 확장성 명시)

import logging
from typing import List, Dict, Optional, Union

# --- 필요한 모듈 임포트 ---
try:
    # prompt_builder.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .conversation_state import ConversationState
    from .config_loader import get_config
    logging.info("conversation_state and config_loader imported successfully in prompt_builder.")
except ImportError as ie:
    logging.error(f"ERROR (prompt_builder): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    ConversationState = None
    get_config = None

# --- 로거 설정 (기본 설정 상속) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

# --- 최종 프롬프트 생성 함수 ---
def build_final_prompt(
    user_query: str,
    conversation_state: 'ConversationState', # Type hint 유지
    rag_results: List[Dict],
    cot_data: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    """
    주어진 사용자 질문, 대화 상태, RAG 검색 결과, CoT 데이터를 조합하여
    최종적으로 GPT API에 전달할 메시지 리스트를 생성합니다.
    config.yaml에서 프롬프트 템플릿, 헤더, 지시문, 포함 옵션 등을 로드하여 사용합니다.

    Args:
        user_query (str): 현재 사용자의 입력 질문.
        conversation_state (ConversationState): 현재 대화의 슬롯, 요약 등 상태 정보 객체.
        rag_results (List[Dict]): RAG 검색 결과 리스트 (각 항목은 문서 chunk 정보 포함).
        cot_data (Optional[str], optional): 모델 라우터에서 생성된 CoT 단계 또는 지침 문자열.
                                           'medium' 또는 'hard' 난이도일 때 전달될 수 있음. Defaults to None.

    Returns:
        Optional[List[Dict[str, str]]]: GPT API 요청 형식의 메시지 리스트 ([{"role": "system", ...}, {"role": "user", ...}]).
                                        설정 로드 실패 등 오류 발생 시 None 반환.
    """
    # 필수 모듈 및 설정 로드 확인
    if not ConversationState or not get_config:
        logger.error("Required modules (ConversationState, get_config) not available in build_final_prompt.")
        return None # 필수 모듈 없으면 진행 불가
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
        # 설정 내 필수 섹션 확인
        prompt_config = config.get('prompts', {})
        prompt_options = config.get('prompt_options', {})
        if not prompt_config or not prompt_options:
            raise ValueError("Required configuration sections ('prompts' or 'prompt_options') are missing or empty in config.yaml.")
    except Exception as e:
        logger.error(f"Error loading configuration in prompt_builder: {e}", exc_info=True)
        # 설정 로드 실패 시 진행 불가
        return None

    logger.info("Building final prompt for GPT API call...")
    logger.debug(f"Input User Query: '{user_query[:100]}...'")
    logger.debug(f"RAG Results Count: {len(rag_results)}")
    logger.debug(f"CoT Data Provided: {'Yes' if cot_data else 'No'}")

    # --- 1. 시스템 프롬프트 구성 요소 조합 ---
    system_prompt_parts: List[str] = []

    # 1a. 기본 시스템 프롬프트 로드
    # NOTE: 향후 프롬프트 최적화 도구 연동 시, config 기반 템플릿 대신 최적화된 프롬프트를 동적으로 로드/선택하는 인터페이스 적용 가능
    default_system_prompt = prompt_config.get('default_system_prompt', "You are a helpful AI assistant.") # 기본값 설정
    if default_system_prompt: # 내용이 있을 경우에만 추가
        system_prompt_parts.append(default_system_prompt)
    else:
        logger.warning("Default system prompt is empty or missing in config.")

    # 1b. 일반 지시문 추가 (config 옵션에 따라)
    grounding_instruction = prompt_config.get('grounding_instruction')
    tone_instruction = prompt_config.get('tone_instruction')
    length_constraint_instruction = prompt_config.get('length_constraint_instruction')

    if prompt_options.get('use_rag_grounding_instruction', True) and grounding_instruction:
        system_prompt_parts.append(grounding_instruction)
    if prompt_options.get('use_tone_instruction', True) and tone_instruction:
        system_prompt_parts.append(tone_instruction)
    if prompt_options.get('use_length_constraint_instruction', True) and length_constraint_instruction:
        system_prompt_parts.append(length_constraint_instruction)

    # 1c. 대화 요약 추가 (config 옵션 및 요약 존재 여부에 따라)
    summary_context_header = prompt_config.get('summary_context_header', "[Previous Conversation Summary]")
    if prompt_options.get('include_summary_in_prompt', True):
        summary = conversation_state.get_summary() # 상태 객체에서 요약 가져오기
        if summary: # 요약 내용이 존재할 경우
            logger.debug(f"Adding conversation summary to prompt (Length: {len(summary)} chars).")
            system_prompt_parts.append(f"\n{summary_context_header}")
            system_prompt_parts.append(summary)
        # else: logger.debug("No conversation summary available to add.") # 요약 없으면 추가 안 함

    # 1d. 대화 상태(Slot) 정보 추가 (config 옵션 및 슬롯 존재 여부에 따라)
    slot_context_header = prompt_config.get('slot_context_header', "[User Information / Slots]")
    if prompt_options.get('include_slots_in_prompt', True):
        slots = conversation_state.get_slots() # 상태 객체에서 슬롯 가져오기
        # 값이 있는 슬롯만 필터링
        valid_slots = {key: value for key, value in slots.items() if value is not None}
        if valid_slots: # 유효한 슬롯이 하나라도 있을 경우
            logger.debug(f"Adding slots to prompt: {list(valid_slots.keys())}")
            slot_info_lines = [f"- {key}: {value}" for key, value in valid_slots.items()]
            slot_info = f"\n{slot_context_header}\n" + "\n".join(slot_info_lines)
            system_prompt_parts.append(slot_info)
        # else: logger.debug("No valid slots available to add.") # 슬롯 없으면 추가 안 함

    # 1e. RAG 검색 결과 추가 (config 옵션 및 결과 존재 여부에 따라)
    rag_context_header = prompt_config.get('rag_context_header', "[Reference Document Information]")
    rag_fallback_message = prompt_config.get('rag_fallback_message', "- No relevant document information found.") # RAG 실패/결과 없을 때 메시지
    if prompt_options.get('include_rag_context_in_prompt', True):
        rag_results_count_limit = prompt_options.get('rag_results_count_limit', 3)
        # RAG 결과가 있고, 제한 개수가 0보다 클 때만 처리
        if rag_results and isinstance(rag_results_count_limit, int) and rag_results_count_limit > 0:
            num_results_to_add = min(len(rag_results), rag_results_count_limit)
            logger.debug(f"Adding top {num_results_to_add} RAG results to prompt (limit: {rag_results_count_limit}).")
            rag_context_lines = [f"\n{rag_context_header}"]
            added_doc_count = 0
            for i, doc in enumerate(rag_results[:num_results_to_add]):
                doc_text = doc.get('text', '').strip()
                if doc_text: # 문서 내용이 있을 경우에만 추가
                    # 문서 출처 등 메타데이터 추가 가능 (선택 사항)
                    # source = doc.get('metadata', {}).get('source_file', 'Unknown')
                    # rag_context_lines.append(f"- Document {i+1} (Source: {source}): {doc_text}")
                    rag_context_lines.append(f"- Document {i+1}: {doc_text}")
                    added_doc_count += 1
                else:
                    logger.debug(f"Skipping empty RAG document at index {i}.")

            # 유효한 문서가 하나라도 추가되었으면 시스템 프롬프트에 포함
            if added_doc_count > 0:
                system_prompt_parts.append("\n".join(rag_context_lines))
            else: # 추가된 문서가 없으면 fallback 메시지 사용
                 logger.debug("No valid RAG documents to add, adding fallback message.")
                 system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")
        else:
            # RAG 결과가 없거나 제한 개수가 0 이하인 경우 fallback 메시지 사용
            logger.debug("No RAG results to add or RAG count limit is zero/negative, adding fallback message.")
            system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")

    # 1f. CoT 데이터 추가 (cot_data가 제공된 경우)
    # Multi-Stage CoT는 구현되지 않았으므로, 'medium' 또는 'hard' 레벨에서 생성된
    # CoT 스텝 또는 상세 지침(cot_data)이 있으면 그대로 포함시킴.
    cot_context_header = prompt_config.get('cot_context_header', "[Step-by-Step Thinking Guide (CoT)]")
    cot_follow_instruction = prompt_config.get('cot_follow_instruction', "(Important: Follow the CoT guide above when generating the response.)")
    if cot_data and isinstance(cot_data, str) and cot_data.strip(): # None이나 빈 문자열이 아닐 경우
        logger.debug(f"Adding CoT data (steps or instructions) to prompt. Length: {len(cot_data)} chars.")
        cot_section = f"\n{cot_context_header}\n{cot_data.strip()}" # 앞뒤 공백 제거
        system_prompt_parts.append(cot_section)
        # CoT 따르라는 지시문 추가 (설정에 따라)
        if cot_follow_instruction:
            system_prompt_parts.append(f"\n{cot_follow_instruction}")

    # 1g. Few-shot 예제 동적 삽입 (향후 확장 지점)
    # TODO: 향후 Few-shot 예제를 동적으로 검색하여 삽입하는 로직 추가 고려
    # 예: 사용자 질문 유형(query_type 슬롯 등)에 따라 관련된 Q&A 예시를 RAG로 검색하여 추가
    # if few_shot_examples:
    #     logger.debug(f"Adding {len(few_shot_examples)} Few-shot examples to prompt.")
    #     few_shot_section = "\n[Examples]\n" + "\n".join([f"Q: {ex['q']}\nA: {ex['a']}" for ex in few_shot_examples])
    #     system_prompt_parts.append(few_shot_section)

    # --- 2. 최종 시스템 프롬프트 조합 ---
    # 각 파트 사이에 두 번의 줄바꿈(\n\n)으로 구분하고, None이나 빈 문자열은 제외
    final_system_prompt = "\n\n".join(filter(None, system_prompt_parts))
    logger.info(f"Final system prompt constructed. Total length: {len(final_system_prompt)} characters.")
    # DEBUG 레벨일 때만 전체 시스템 프롬프트 로깅 (매우 길 수 있음 주의)
    logger.debug(f"Final System Prompt Preview:\n------\n{final_system_prompt[:1000]}...\n------")

    # --- 3. 최종 메시지 리스트 생성 (System + User) ---
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_query}
    ]

    # 대화 히스토리 추가 (선택 사항 - 현재 구조에서는 요약본을 사용)
    # 이전 턴들을 직접 추가해야 한다면 conversation_state.get_history() 활용
    # messages.extend(conversation_state.get_history()[-N:]) # 예: 최근 N개 턴 추가

    logger.info("Final prompt message list constructed for API call.")
    return messages

# --- 예시 사용법 (기존 유지) ---
if __name__ == "__main__":
    # 로깅 레벨 강제 DEBUG
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # __name__으로 로거 다시 가져오기

    try:
        get_config() # 설정 로드 가능한지 확인
        logger.info("Configuration loaded successfully for prompt builder test.")
    except Exception as e:
        logger.error(f"Failed to load configuration for test: {e}. Using default values might occur.", exc_info=True)

    # ConversationState 임포트 성공 여부 확인
    if not ConversationState:
        print("CRITICAL: ConversationState module could not be imported. Exiting test.")
        exit()

    # 가상의 데이터 생성
    print("\n--- Running Prompt Builder Test Scenarios ---")
    state = ConversationState()
    state.update_slots({"brand": "나이키", "size": "270mm", "foot_width": "넓은 편", "product_category": "러닝화"})
    state.update_summary("고객은 나이키 270mm 신발 착용 경험이 있으며, 발볼이 넓어 데카트론 러닝화를 알아보고 있음. 이전 요약 테스트.")
    rag_data = [
        {"id": "doc1", "text": "데카트론 킵런 시리즈는 발볼이 여유있게 나왔습니다. 정사이즈를 추천합니다.", "metadata": {"source_file": "decathlon_shoes.txt"}, "similarity_score": 0.85},
        {"id": "doc2", "text": "나이키 신발은 일반적으로 발볼이 좁게 디자인되는 경향이 있습니다.", "metadata": {"source_file": "nike_info.txt"}, "similarity_score": 0.78},
        {"id": "doc3", "text": " ", "metadata": {"source_file": "empty_doc.txt"}, "similarity_score": 0.6}, # 빈 문서 테스트
        {"id": "doc4", "text": "킵런 KS900 모델은 장거리 러닝에 적합한 쿠셔닝을 제공합니다.", "metadata": {"source_file": "kiprun_ks900.txt"}, "similarity_score": 0.88} # 추가 문서
    ]
    user_input = "제가 발볼이 넓은 편인데, 데카트론 킵런 러닝화는 275mm를 신어야 할까요?"
    medium_cot_steps = "- 1단계: 사용자의 발볼('넓은 편')과 기존 사이즈('나이키 270mm') 확인.\n- 2단계: RAG 정보에서 킵런 특징('발볼 여유', '정사이즈 추천') 확인.\n- 3단계: 나이키 특징('좁음') 고려하여 데카트론 정사이즈 가능성 설명.\n- 4단계: 매장 착용 권장."
    hard_cot_instructions = "1. 문제점(겨울 산행 시 발 시려움, 평발, 넓은 발볼) 및 요구사항(보온, 방수, 편안함) 파악.\n2. RAG에서 데카트론 등산화 중 보온/방수 강화 모델 검색.\n3. 검색된 모델 특징(소재, 기능, 후기) 분석 및 발 상태 적합성 평가.\n4. 기존 MH500과 비교하여 추천 모델 장점(보온성, 착화감 등) 구체적 설명.\n5. 최종 1~2개 모델 추천 및 이유 요약."

    # 시나리오별 테스트
    scenarios = {
        "1_medium_with_cot": (user_input, state, rag_data, medium_cot_steps),
        "2_easy_no_cot": ("반품 규정 알려주세요.", ConversationState(), [], None), # 새 상태 객체 사용
        "3_hard_with_cot": ("작년에 구매한 퀘차 등산화(MH500)...(질문 생략)", ConversationState(), rag_data[:1], hard_cot_instructions), # RAG 결과 1개만 제공
        "4_no_rag": ("푸마 신발 사이즈 가이드 알려줘", ConversationState(), [], None), # RAG 결과 없음
        "5_empty_rag_results": ("킵런 신발 어때요?", ConversationState(), [{"id":"doc_empty", "text": ""}], None), # RAG 결과 텍스트가 비었을 때
        "6_no_summary_or_slots": ("영업 시간 알려줘", ConversationState(), rag_data[0:1], None), # 요약/슬롯 없는 상태
    }

    for name, args in scenarios.items():
        print(f"\n--- Scenario: {name} ---")
        logger.info(f"Building prompt for scenario: {name}")
        final_messages = build_final_prompt(*args)
        print("\nGenerated Messages (System Prompt Preview - First 1000 chars):")
        if final_messages and isinstance(final_messages, list) and len(final_messages) > 0 and 'content' in final_messages[0]:
            print(final_messages[0]['content'][:1000] + ("..." if len(final_messages[0]['content']) > 1000 else ""))
        else:
            print("ERROR: Could not generate messages or system prompt.")
        print("-" * 30)