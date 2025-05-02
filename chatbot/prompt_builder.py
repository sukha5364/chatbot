# chatbot/prompt_builder.py (RAG 결과 형식 변경 반영 - 구조화된 정보 활용)

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
    rag_results: List[Dict], # 이제 각 Dict는 파싱된 메타데이터 포함
    cot_data: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    """
    주어진 사용자 질문, 대화 상태, RAG 검색 결과(구조화된 제품 정보 포함),
    CoT 데이터를 조합하여 최종적으로 GPT API에 전달할 메시지 리스트를 생성합니다.
    RAG 결과는 단순히 text를 나열하는 대신, 주요 메타데이터를 요약하여 제공합니다.

    Args:
        user_query (str): 현재 사용자의 입력 질문.
        conversation_state (ConversationState): 현재 대화의 슬롯, 요약 등 상태 정보 객체.
        rag_results (List[Dict]): RAG 검색 결과 리스트 (각 항목은 파싱된 제품 메타데이터 포함).
        cot_data (Optional[str], optional): 모델 라우터에서 생성된 CoT 단계 또는 지침 문자열.
                                           'medium' 또는 'hard' 난이도일 때 전달될 수 있음. Defaults to None.

    Returns:
        Optional[List[Dict[str, str]]]: GPT API 요청 형식의 메시지 리스트 ([{"role": "system", ...}, {"role": "user", ...}]).
                                        설정 로드 실패 등 오류 발생 시 None 반환.
    """
    # 필수 모듈 및 설정 로드 확인
    if not ConversationState or not get_config:
        logger.error("Required modules (ConversationState, get_config) not available in build_final_prompt.")
        return None
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
        prompt_config = config.get('prompts', {})
        prompt_options = config.get('prompt_options', {})
        if not prompt_config or not prompt_options:
            raise ValueError("Required configuration sections ('prompts' or 'prompt_options') are missing or empty in config.yaml.")
    except Exception as e:
        logger.error(f"Error loading configuration in prompt_builder: {e}", exc_info=True)
        return None

    logger.info("Building final prompt for GPT API call (using structured RAG info)...")
    logger.debug(f"Input User Query: '{user_query[:100]}...'")
    logger.debug(f"RAG Results Count Received: {len(rag_results)}")
    logger.debug(f"CoT Data Provided: {'Yes' if cot_data else 'No'}")

    # --- 1. 시스템 프롬프트 구성 요소 조합 ---
    system_prompt_parts: List[str] = []

    # 1a. 기본 시스템 프롬프트 로드
    default_system_prompt = prompt_config.get('default_system_prompt', "You are a helpful AI assistant.")
    if default_system_prompt:
        system_prompt_parts.append(default_system_prompt)
    else: logger.warning("Default system prompt is empty or missing in config.")

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
        summary = conversation_state.get_summary()
        if summary:
            logger.debug(f"Adding conversation summary to prompt (Length: {len(summary)} chars).")
            system_prompt_parts.append(f"\n{summary_context_header}")
            system_prompt_parts.append(summary)

    # 1d. 대화 상태(Slot) 정보 추가 (config 옵션 및 슬롯 존재 여부에 따라)
    slot_context_header = prompt_config.get('slot_context_header', "[User Information / Slots]")
    if prompt_options.get('include_slots_in_prompt', True):
        slots = conversation_state.get_slots()
        valid_slots = {key: value for key, value in slots.items() if value is not None}
        if valid_slots:
            logger.debug(f"Adding slots to prompt: {list(valid_slots.keys())}")
            slot_info_lines = [f"- {key}: {value}" for key, value in valid_slots.items()]
            slot_info = f"\n{slot_context_header}\n" + "\n".join(slot_info_lines)
            system_prompt_parts.append(slot_info)

    # *** 1e. RAG 검색 결과 추가 (구조화된 정보 요약 방식) ***
    rag_context_header = prompt_config.get('rag_context_header', "[참고 문서 정보]")
    rag_fallback_message = prompt_config.get('rag_fallback_message', "- 관련된 문서 정보를 찾지 못했습니다.")
    if prompt_options.get('include_rag_context_in_prompt', True):
        rag_results_count_limit = prompt_options.get('rag_results_count_limit', 3)
        formatted_rag_results = []

        if rag_results and isinstance(rag_results_count_limit, int) and rag_results_count_limit > 0:
            num_results_to_add = min(len(rag_results), rag_results_count_limit)
            logger.debug(f"Formatting top {num_results_to_add} RAG results (product blocks) for prompt (limit: {rag_results_count_limit}).")

            for i, doc_meta in enumerate(rag_results[:num_results_to_add]):
                doc_info_lines = []
                # 주요 메타데이터 추출 및 포맷팅 (get 사용으로 안전하게 접근)
                name = doc_meta.get('product_name', 'N/A')
                brand = doc_meta.get('brand', 'Unknown')
                category = doc_meta.get('category', 'N/A')
                price = doc_meta.get('price', 'N/A')
                features = doc_meta.get('features', []) # 리스트
                overall_review = doc_meta.get('reviews_overall')
                pros = doc_meta.get('reviews_pros', []) # 리스트
                cons = doc_meta.get('reviews_cons', []) # 리스트
                # description = doc_meta.get('description', '') # 상세 설명은 너무 길 수 있어 일단 제외

                # 기본 정보
                doc_info_lines.append(f"  - 제품 {i+1}: {name} (브랜드: {brand}, 카테고리: {category}, 가격: {price})")

                # 주요 특징 (최대 3개)
                if features:
                    features_str = ", ".join(features[:3]) + ('...' if len(features) > 3 else '')
                    doc_info_lines.append(f"    특징: {features_str}")

                # 리뷰 요약 (종합 평가 + 장단점 각 최대 2개)
                review_parts = []
                if overall_review: review_parts.append(f"종합: {overall_review}")
                if pros: review_parts.append(f"장점: {', '.join(pros[:2])}{'...' if len(pros) > 2 else ''}")
                if cons: review_parts.append(f"단점: {', '.join(cons[:2])}{'...' if len(cons) > 2 else ''}")
                if review_parts:
                    doc_info_lines.append(f"    리뷰 요약: {' / '.join(review_parts)}")

                # 상세 설명 미리보기 (선택 사항)
                # if description:
                #     desc_preview = description[:100] + ('...' if len(description) > 100 else '')
                #     doc_info_lines.append(f"    상세 설명(일부): {desc_preview}")

                formatted_rag_results.append("\n".join(doc_info_lines))

        # 최종 RAG 컨텍스트 문자열 생성
        if formatted_rag_results:
            rag_context_string = f"\n{rag_context_header}\n" + "\n\n".join(formatted_rag_results) # 각 제품 정보 사이에 빈 줄 추가
            system_prompt_parts.append(rag_context_string)
            logger.debug(f"Added formatted RAG context to prompt:\n{rag_context_string}")
        else:
            # 결과가 없거나 포맷팅된 내용이 없을 경우 fallback 메시지 사용
            logger.debug("No valid RAG results to format or RAG count limit is zero/negative, adding fallback message.")
            system_prompt_parts.append(f"\n{rag_context_header}\n{rag_fallback_message}")


    # 1f. CoT 데이터 추가 (cot_data가 제공된 경우)
    cot_context_header = prompt_config.get('cot_context_header', "[Step-by-Step Thinking Guide (CoT)]")
    cot_follow_instruction = prompt_config.get('cot_follow_instruction', "(Important: Follow the CoT guide above when generating the response.)")
    if cot_data and isinstance(cot_data, str) and cot_data.strip():
        logger.debug(f"Adding CoT data (steps or instructions) to prompt. Length: {len(cot_data)} chars.")
        cot_section = f"\n{cot_context_header}\n{cot_data.strip()}"
        system_prompt_parts.append(cot_section)
        if cot_follow_instruction:
            system_prompt_parts.append(f"\n{cot_follow_instruction}")

    # --- 2. 최종 시스템 프롬프트 조합 ---
    final_system_prompt = "\n\n".join(filter(None, system_prompt_parts))
    logger.info(f"Final system prompt constructed. Total length: {len(final_system_prompt)} characters.")
    logger.debug(f"Final System Prompt Preview:\n------\n{final_system_prompt[:1000]}...\n------")

    # --- 3. 최종 메시지 리스트 생성 (System + User) ---
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_query}
    ]

    logger.info("Final prompt message list constructed for API call.")
    return messages

# --- 예시 사용법 (테스트용) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        get_config() # 설정 로드 가능한지 확인
        logger.info("Configuration loaded successfully for prompt builder test.")
    except Exception as e:
        logger.error(f"Failed to load configuration for test: {e}. Using default values might occur.", exc_info=True)

    if not ConversationState:
        print("CRITICAL: ConversationState module could not be imported. Exiting test.")
        exit()

    print("\n--- Running Prompt Builder Test Scenario (Structured RAG) ---")
    state = ConversationState()
    state.update_slots({"brand": "푸강자", "product_category": "승마용품", "context_activity": "승마"})
    state.update_summary("고객은 승마용품에 관심이 있으며, 푸강자 브랜드에 대해 질문함.")

    # RAG 결과 예시 (rag_generator.py의 parse_product_block 결과 형식 가정)
    rag_data_structured = [
        {
            "id": "some_file.txt-block0", "source_file": "some_file.txt", "block_index": 0,
            "product_name": "푸강자 (FOUGANZA) 100 승마 조퍼스 - 블랙", "brand": "FOUGANZA",
            "category": "승마 (기타 스포츠 > 승마 의류)", "price": "50,000원", "price_numeric": 50000,
            "target_audience": "승마 활동을 하는 성인 및 청소년",
            "description": "이 제품은 승마 시 안장에서의 편안함과 자유로운 움직임을 제공하도록 설계된 100 승마 조퍼스 모델입니다...",
            "features": ["신축성이 우수한 소재", "다리 쓸림 최소화 설계", "가격 대비 뛰어난 성능 (가성비)", "운동복처럼 편하게 착용 가능"],
            "size_fit_analysis": "사용자 리뷰는 사이즈 및 핏에 대해 다소 엇갈린 의견을 보입니다...",
            "reviews_summary": "", # 이 예시에서는 하위 섹션으로 분리됨
            "reviews_overall": "가격 대비 매우 만족스럽다는 평가가 지배적입니다. 승마 입문용으로 긍정적인 피드백이 많습니다.",
            "reviews_pros": ["저렴한 가격", "편안한 착용감", "우수한 신축성", "기본적인 내구성", "가성비"],
            "reviews_cons": ["사이즈 선택의 불확실성 (개인차)", "허리/골반 부위 핏 관련 의견", "주머니 기능성 부재 아쉬움", "초기 사용 후 단추 탈락 등"],
            "reviews_languages": ["한국어", "프랑스어", "터키어", "중국어", "폴란드어", "..."],
            "usage_recommendation": ["승마 훈련 (초급)", "일상적인 편안한 운동복", "가벼운 외출복"],
            "care_tips": ["세탁 시 원단 변형이나 물 빠짐에 유의 (리뷰 기반)"],
            "text": "푸강자 (FOUGANZA) 100 승마 조퍼스 - 블랙\n제품 정보\n카테고리: ... (블록 전체 텍스트)", # 원본 텍스트 필드
            "similarity_score": 0.88 # searcher에서 추가된 점수
        },
        {
            "id": "another_file.txt-block1", "source_file": "another_file.txt", "block_index": 1,
            "product_name": "푸강자 (FOUGANZA) 140 벨벳 승마 헬멧", "brand": "FOUGANZA",
            "category": "승마 (기타 스포츠 > 승마 보호 장비)", "price": "49,000원", "price_numeric": 49000,
            "target_audience": "승마 활동을 하는 라이더 (성인 및 청소년)",
            "description": "이 140 승마 헬멧은 승마 시 필수적인 보호 장비입니다...",
            "features": ["전통적인 벨벳 외관 디자인", "머리 보호 기능", "머리 사이즈 조절 가능", "가격 대비 우수한 품질"],
            "size_fit_analysis": "이 헬멧은 조절 기능을 통해 사용자의 머리 크기에 맞춰 착용할 수 있으나...",
            "reviews_summary": "",
            "reviews_overall": "뛰어난 가성비와 기본적인 안전 기능에 대해 매우 긍정적인 평가를 받습니다.",
            "reviews_pros": ["합리적인 가격", "안전 기능 (낙마 보호)", "멋진 디자인"],
            "reviews_cons": ["벨벳 소재 마모 빠름", "더운 날씨에 덥다는 의견", "내부 패딩 불만 (소수)"],
            "reviews_languages": ["한국어", "터키어", "프랑스어", "..."],
            "usage_recommendation": ["기본적인 승마 훈련 및 활동 (필수 착용)"],
            "care_tips": ["벨벳 소재 특성상 외부 오염 및 마모에 주의"],
            "text": "푸강자 (FOUGANZA) 140 벨벳 승마 헬멧\n제품 정보\n카테고리: ... (블록 전체 텍스트)",
            "similarity_score": 0.85
        }
    ]
    user_input = "푸강자 승마 바지랑 헬멧 같이 사려고 하는데, 특징이랑 리뷰 좀 요약해줄 수 있을까요?"
    cot_steps = "- 1단계: 사용자가 찾는 '푸강자 승마 바지'(조퍼스)와 '헬멧' 정보 파악.\n- 2단계: 제공된 RAG 정보에서 각 제품의 특징 및 리뷰 요약 추출.\n- 3단계: 추출된 정보를 바탕으로 각 제품의 핵심 특징과 장단점 요약.\n- 4단계: 두 제품 정보를 종합하여 답변 구성."

    # 시나리오 실행
    final_messages = build_final_prompt(user_input, state, rag_data_structured, cot_steps)

    print("\nGenerated Messages (System Prompt Preview - First 1500 chars):")
    if final_messages and isinstance(final_messages, list) and len(final_messages) > 0 and 'content' in final_messages[0]:
        # 시스템 프롬프트 내용 출력 (길이 제한하여)
        print(final_messages[0]['content'][:1500] + ("..." if len(final_messages[0]['content']) > 1500 else ""))
    else:
        print("ERROR: Could not generate messages or system prompt.")
    print("-" * 30)

    print("\nGenerated Messages (User Prompt):")
    if final_messages and isinstance(final_messages, list) and len(final_messages) > 1 and 'content' in final_messages[1]:
        print(final_messages[1]['content'])
    else:
        print("ERROR: Could not generate messages or user prompt.")

    print("\n--- Prompt Builder Test Scenario Finished ---")