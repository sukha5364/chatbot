# chatbot/prompt_builder.py (요구사항 반영 최종본: 요약/CoT 개선, 최근 K턴 추가)

import logging
from typing import List, Dict, Optional, Union, Any

# --- 필요한 모듈 임포트 ---
try:
    # [수정] ConversationState 임포트 제거 (직접 사용 안 함)
    # from .conversation_state import ConversationState
    from .config_loader import get_config
    logging.info("config_loader imported successfully in prompt_builder.")
except ImportError as ie:
    logging.error(f"ERROR (prompt_builder): Failed to import config_loader: {ie}. Check relative paths.", exc_info=True)
    get_config = None

# --- 로거 설정 (기본 설정 상속) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

# --- 최종 프롬프트 생성 함수 ([수정됨] 시그니처 변경) ---
def build_final_prompt(
    user_query: str,
    summary: Optional[str],          # 요약 정보 (scheduler 결과)
    history: List[Dict[str, str]], # 현재까지의 대화 기록 (state에서 가져옴)
    slots: Dict[str, Any],           # 현재 슬롯 정보 (state에서 가져옴)
    rag_results: List[Dict],         # RAG 검색 결과
    cot_data: Optional[str] = None,  # CoT 데이터 (scheduler 결과)
) -> Optional[List[Dict[str, str]]]:
    """
    주어진 사용자 질문, 대화 컨텍스트(요약, 최근 기록, 슬롯), RAG 검색 결과,
    CoT 데이터를 조합하여 최종적으로 GPT API에 전달할 메시지 리스트를 생성합니다.

    Args:
        user_query (str): 현재 사용자의 입력 질문.
        summary (Optional[str]): 이전 턴까지의 대화 요약 (scheduler가 생성).
        history (List[Dict[str, str]]): 현재까지의 전체 대화 기록 리스트.
        slots (Dict[str, Any]): 현재까지 파악된 슬롯 정보.
        rag_results (List[Dict]): RAG 검색 결과 리스트 (파싱된 메타데이터 포함).
        cot_data (Optional[str], optional): CoT 단계 또는 지침 문자열. Defaults to None.

    Returns:
        Optional[List[Dict[str, str]]]: GPT API 요청 형식의 메시지 리스트. 오류 시 None.
    """
    # 필수 모듈 및 설정 로드 확인
    if not get_config:
        logger.error("Required module (get_config) not available in build_final_prompt.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
        prompt_config = config.get('prompts', {})
        prompt_options = config.get('prompt_options', {})
        if not prompt_config or not prompt_options:
            raise ValueError("Required config sections ('prompts', 'prompt_options') missing.")
    except Exception as e:
        logger.error(f"Error loading configuration in prompt_builder: {e}", exc_info=True)
        return None

    logger.info("Building final prompt for GPT API call...")
    logger.debug(f"Input User Query: '{user_query[:100]}...'")
    logger.debug(f"RAG Results Count Received: {len(rag_results)}")
    logger.debug(f"CoT Data Provided: {'Yes' if cot_data else 'No'}")
    logger.debug(f"Summary Provided: {'Yes' if summary else 'No'}")
    logger.debug(f"History Length: {len(history)}")
    logger.debug(f"Slots Provided: {list(slots.keys())}")

    # --- 1. 시스템 프롬프트 구성 요소 조합 ---
    system_prompt_parts: List[str] = []

    # 1a. 기본 시스템 프롬프트
    default_system_prompt = prompt_config.get('default_system_prompt', "You are a helpful AI assistant.")
    if default_system_prompt: system_prompt_parts.append(default_system_prompt)
    else: logger.warning("Default system prompt is empty or missing in config.")

    # 1b. 일반 지시문 (Grounding, Tone, Length)
    grounding_instruction = prompt_config.get('grounding_instruction')
    tone_instruction = prompt_config.get('tone_instruction')
    length_constraint_instruction = prompt_config.get('length_constraint_instruction')
    if prompt_options.get('use_rag_grounding_instruction', True) and grounding_instruction:
        system_prompt_parts.append(grounding_instruction)
    if prompt_options.get('use_tone_instruction', True) and tone_instruction:
        system_prompt_parts.append(tone_instruction)
    if prompt_options.get('use_length_constraint_instruction', True) and length_constraint_instruction:
        system_prompt_parts.append(length_constraint_instruction)

    # 1c. 대화 요약 추가 ([수정] 인자로 받은 summary 사용)
    summary_context_header = prompt_config.get('summary_context_header', "[Previous Conversation Summary]")
    if prompt_options.get('include_summary_in_prompt', True) and summary: # None 이 아니고 비어있지 않을 때
        logger.debug(f"Adding conversation summary to prompt (Length: {len(summary)} chars).")
        system_prompt_parts.append(f"\n{summary_context_header}")
        system_prompt_parts.append(summary)

    # --- [신규 추가] 1d. 최근 K턴 대화 기록 추가 ---
    recent_conv_header = prompt_config.get('recent_conversation_header', "[Recent Conversation Turns]")
    if prompt_options.get('include_recent_history_in_prompt', True):
        k_turns = prompt_options.get('recent_history_turns_to_include', 0)
        if isinstance(k_turns, int) and k_turns > 0 and history:
            num_messages_to_include = k_turns * 2 # 1턴 = user + assistant
            # history 리스트의 마지막 K*2개 메시지 슬라이싱
            recent_history_turns = history[-num_messages_to_include:]
            if recent_history_turns:
                logger.debug(f"Adding last {len(recent_history_turns)} messages ({k_turns} turns) to prompt.")
                formatted_recent_history_lines = []
                for turn in recent_history_turns:
                    role = turn.get("role", "unknown").capitalize()
                    content = turn.get("content", "")
                    formatted_recent_history_lines.append(f"{role}: {content}")
                recent_history_str = "\n".join(formatted_recent_history_lines)
                system_prompt_parts.append(f"\n{recent_conv_header}")
                system_prompt_parts.append(recent_history_str)
            else:
                logger.debug("No recent history turns to add (history might be shorter than K*2).")
        elif k_turns <= 0:
             logger.debug("Recent history inclusion is enabled but turns_to_include is 0 or invalid.")
        elif not history:
             logger.debug("Recent history inclusion is enabled but history is empty.")


    # 1e. 대화 상태(Slot) 정보 추가 ([수정] 인자로 받은 slots 사용)
    slot_context_header = prompt_config.get('slot_context_header', "[User Information / Slots]")
    if prompt_options.get('include_slots_in_prompt', True):
        valid_slots = {key: value for key, value in slots.items() if value is not None} # None 값 제외
        if valid_slots:
            logger.debug(f"Adding slots to prompt: {list(valid_slots.keys())}")
            slot_info_lines = [f"- {key}: {value}" for key, value in valid_slots.items()]
            slot_info = f"\n{slot_context_header}\n" + "\n".join(slot_info_lines)
            system_prompt_parts.append(slot_info)

    # 1f. RAG 검색 결과 추가 (구조화된 정보 요약 방식 - 변경 없음)
    rag_context_header = prompt_config.get('rag_context_header', "[참고 문서 정보]")
    rag_fallback_message = prompt_config.get('rag_fallback_message', "- 관련된 문서 정보를 찾지 못했습니다.")
    if prompt_options.get('include_rag_context_in_prompt', True):
        rag_results_count_limit = prompt_options.get('rag_results_count_limit', 3)
        formatted_rag_results = []
        if rag_results and isinstance(rag_results_count_limit, int) and rag_results_count_limit > 0:
            num_results_to_add = min(len(rag_results), rag_results_count_limit)
            logger.debug(f"Formatting top {num_results_to_add} RAG results for prompt.")
            for i, doc_meta in enumerate(rag_results[:num_results_to_add]):
                doc_info_lines = []
                name = doc_meta.get('product_name', 'N/A')
                brand = doc_meta.get('brand', 'Unknown')
                category = doc_meta.get('category', 'N/A')
                price = doc_meta.get('price', 'N/A')
                features = doc_meta.get('features', [])
                overall_review = doc_meta.get('reviews_overall')
                pros = doc_meta.get('reviews_pros', [])
                cons = doc_meta.get('reviews_cons', [])

                doc_info_lines.append(f"  - 제품 {i+1}: {name} (브랜드: {brand}, 카테고리: {category}, 가격: {price})")
                if features:
                    features_str = ", ".join(features[:3]) + ('...' if len(features) > 3 else '')
                    doc_info_lines.append(f"    특징: {features_str}")
                review_parts = []
                if overall_review: review_parts.append(f"종합: {overall_review}")
                if pros: review_parts.append(f"장점: {', '.join(pros[:2])}{'...' if len(pros) > 2 else ''}")
                if cons: review_parts.append(f"단점: {', '.join(cons[:2])}{'...' if len(cons) > 2 else ''}")
                if review_parts: doc_info_lines.append(f"    리뷰 요약: {' / '.join(review_parts)}")
                formatted_rag_results.append("\n".join(doc_info_lines))

            if formatted_rag_results:
                rag_context_string = f"\n{rag_context_header}\n" + "\n\n".join(formatted_rag_results)
                system_prompt_parts.append(rag_context_string)
                logger.debug(f"Added formatted RAG context to prompt.")
            else:
                logger.debug("No valid RAG results to format, adding fallback message.")
                system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")
        else: # RAG 결과 없거나 limit <= 0
            logger.debug("RAG results are empty or limit is not positive, adding fallback message.")
            system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")

    # 1g. CoT 데이터 추가 (cot_data 인자 사용 - 변경 없음)
    cot_context_header = prompt_config.get('cot_context_header', "[Step-by-Step Thinking Guide (CoT)]")
    cot_follow_instruction = prompt_config.get('cot_follow_instruction', "(Important: Follow the CoT guide above.)")
    if cot_data and isinstance(cot_data, str) and cot_data.strip():
        logger.debug(f"Adding CoT data to prompt. Length: {len(cot_data)} chars.")
        cot_section = f"\n{cot_context_header}\n{cot_data.strip()}"
        system_prompt_parts.append(cot_section)
        if cot_follow_instruction:
            system_prompt_parts.append(f"\n{cot_follow_instruction}")

    # --- 2. 최종 시스템 프롬프트 조합 ---
    final_system_prompt = "\n\n".join(filter(None, system_prompt_parts)) # 빈 문자열 제거하고 결합
    logger.info(f"Final system prompt constructed. Total length: {len(final_system_prompt)} characters.")
    if logger.getEffectiveLevel() <= logging.DEBUG: # DEBUG 레벨일 때만 상세 로깅
        logger.debug(f"Final System Prompt Preview:\n------\n{final_system_prompt[:1000]}...\n------")


    # --- 3. 최종 메시지 리스트 생성 (System + User) ([수정] user_query 인자 사용) ---
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_query} # 현재 사용자 질문을 마지막 user 메시지로 추가
    ]

    logger.info("Final prompt message list constructed for API call.")
    return messages

# --- 예시 사용법 ([수정됨] 함수 호출 방식 변경) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        config = get_config() # 설정 로드 가능한지 확인
        logger.info("Configuration loaded successfully for prompt builder test.")
    except Exception as e:
        logger.error(f"Failed to load configuration for test: {e}. Using default values might occur.", exc_info=True)
        config = {} # 테스트 계속 위해 빈 dict 할당

    print("\n--- Running Prompt Builder Test Scenario (Structured RAG, Recent History) ---")
    # 테스트용 데이터 생성
    test_user_query = "푸강자 승마 바지랑 헬멧 같이 사려고 하는데, 특징이랑 리뷰 좀 요약해줄 수 있을까요?"
    test_summary = "고객은 승마용품에 관심이 있으며, 푸강자 브랜드에 대해 질문함. 이전 대화에서 승마 조끼 관련 문의가 있었음."
    test_history = [
        {"role": "user", "content": "안녕하세요, 승마할 때 입을만한 조끼 있나요?"},
        {"role": "assistant", "content": "네, 푸강자 브랜드에 다양한 승마 조끼가 준비되어 있습니다. 어떤 스타일을 선호하시나요?"},
        {"role": "user", "content": "음.. 그냥 기본적인 거로 보여주세요."},
        {"role": "assistant", "content": "네, 푸강자 500 모델은 기본적인 보호 기능과 활동성을 제공합니다."},
        # 여기에 test_user_query가 마지막으로 추가될 것임 (build_final_prompt 내부 아님)
    ]
    test_slots = {"brand": "푸강자", "product_category": "승마용품", "context_activity": "승마", "previous_product": "승마 조끼"}
    test_rag_data_structured = [ # 이전 예시 데이터 재사용
        { "id": "p1", "product_name": "푸강자 (FOUGANZA) 100 승마 조퍼스 - 블랙", "brand": "FOUGANZA", "category": "승마 의류", "price": "50,000원", "features": ["신축성 우수", "가성비"], "reviews_overall": "가격 대비 만족", "reviews_pros": ["저렴함", "편함"], "reviews_cons": ["사이즈 불확실", "핏 관련 의견"] },
        { "id": "p2", "product_name": "푸강자 (FOUGANZA) 140 벨벳 승마 헬멧", "brand": "FOUGANZA", "category": "승마 보호 장비", "price": "49,000원", "features": ["벨벳 디자인", "사이즈 조절"], "reviews_overall": "가성비 좋음", "reviews_pros": ["합리적 가격", "안전 기능"], "reviews_cons": ["벨벳 마모 빠름", "더움"] }
    ]
    test_cot_steps = "- 1단계: 사용자가 찾는 '푸강자 승마 바지'(조퍼스)와 '헬멧' 정보 파악.\n- 2단계: 제공된 RAG 정보에서 각 제품의 특징 및 리뷰 요약 추출.\n- 3단계: 추출된 정보를 바탕으로 각 제품의 핵심 특징과 장단점 요약.\n- 4단계: 두 제품 정보를 종합하여 답변 구성."

    # 함수 호출 테스트 (변경된 시그니처 사용)
    final_messages = build_final_prompt(
        user_query=test_user_query,
        summary=test_summary,
        history=test_history,
        slots=test_slots,
        rag_results=test_rag_data_structured,
        cot_data=test_cot_steps
    )

    # 결과 출력
    if final_messages and isinstance(final_messages, list) and len(final_messages) > 1:
        print("\nGenerated System Prompt (Preview - First 1500 chars):")
        system_content = final_messages[0].get('content', 'ERROR: No system content')
        print(system_content[:1500] + ("..." if len(system_content) > 1500 else ""))
        print("-" * 30)
        print("\nGenerated User Prompt:")
        print(final_messages[1].get('content', 'ERROR: No user content'))
    else:
        print("ERROR: Could not generate messages.")

    print("\n--- Prompt Builder Test Scenario Finished ---")