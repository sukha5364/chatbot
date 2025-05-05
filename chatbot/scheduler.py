# chatbot/scheduler.py (Tool Use 워크플로우 및 후 필터링 적용 버전)

import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Union
import aiohttp
import numpy as np

# --- 필요한 모듈 임포트 ---
try:
    from .gpt_interface import call_gpt_async, get_openai_embedding_async, api_logger # api_logger 임포트 추가
    from .searcher import RagSearcher
    from .conversation_state import ConversationState
    from .config_loader import get_config
    # Slot/Summary는 이제 app.py에서 백그라운드로 호출
    # from .slot_extractor import extract_slots_with_gpt
    # from .summarizer import summarize_conversation_async
    # model_router 관련 임포트 제거
    logging.info("Required modules imported successfully in scheduler.")
except ImportError as ie:
    logging.error(f"ERROR (scheduler): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 스케줄러 기능 사용 불가
    call_gpt_async = None
    get_openai_embedding_async = None
    RagSearcher = None
    ConversationState = None
    get_config = None
    api_logger = None

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

# --- RAG 검색 비동기 실행 함수 (변경 없음) ---
async def run_rag_search_async(
    query_embedding: Union[List[float], np.ndarray, None],
    k: int,
    rag_searcher: Optional[RagSearcher]
) -> List[Dict]:
    """
    주어진 쿼리 임베딩과 RagSearcher 객체를 사용하여 RAG 검색을 비동기적으로 실행합니다.
    FAISS 검색은 동기 함수이므로 asyncio.get_running_loop().run_in_executor를 사용합니다.
    """
    # 1. 입력 유효성 검사
    if rag_searcher is None or rag_searcher.index is None or not rag_searcher.metadata:
        logger.warning("RagSearcher instance is not available or not properly initialized. Skipping RAG search.")
        return []
    if query_embedding is None:
        logger.warning("Query embedding is None, cannot perform RAG search.")
        return []
    if k <= 0:
        logger.warning(f"Invalid value for k ({k}). Skipping RAG search.")
        return []

    # 2. 임베딩 벡터 형식 변환 및 검증
    try:
        if isinstance(query_embedding, list):
            query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        elif isinstance(query_embedding, np.ndarray):
            query_embedding_np = query_embedding.astype(np.float32).reshape(1, -1)
        else:
            logger.error(f"Invalid query embedding type: {type(query_embedding)}. Skipping RAG search.")
            return []

        index_dim = getattr(rag_searcher.index, 'd', None)
        if index_dim is None or index_dim != query_embedding_np.shape[1]:
            logger.error(f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match index dimension ({index_dim}). Skipping RAG search.")
            return []
    except Exception as e:
        logger.error(f"Error processing query embedding for RAG search: {e}", exc_info=True)
        return []

    # 3. 비동기 실행 준비
    loop = asyncio.get_running_loop()
    logger.debug(f"Running RAG search asynchronously in executor with k={k}, embedding shape {query_embedding_np.shape}")

    # 4. FAISS 검색 실행
    try:
        start_t = time.time()
        results = await loop.run_in_executor(
            None, # 기본 스레드 풀 사용
            rag_searcher.search, # 호출할 동기 함수
            query_embedding_np, # 인자 1
            k                   # 인자 2
        )
        duration_t = time.time() - start_t
        logger.debug(f"Async RAG search execution finished in {duration_t:.4f}s. Found {len(results)} initial results.")
        return results
    except RuntimeError as re:
        logger.error(f"RuntimeError during async RAG search execution: {re}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error during async RAG search execution via executor: {e}", exc_info=True)
        return []


# --- [신규] 메타데이터 후 필터링 함수 ---
def apply_metadata_filters(
    rag_results: List[Dict],
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    RAG 검색 결과(메타데이터 리스트)에 대해 LLM이 생성한 필터 조건을 적용하여 결과를 필터링합니다.

    Args:
        rag_results (List[Dict]): RAG 검색으로 얻은 초기 메타데이터 딕셔너리 리스트.
        filters (Optional[Dict]): LLM이 생성한 필터 조건 객체.
            예: {"logic": "AND", "conditions": [{"field": "brand", "operator": "==", "value": "Kiprun"}, ...]}

    Returns:
        List[Dict]: 필터 조건에 맞는 메타데이터 딕셔너리 리스트.
    """
    if not filters or not isinstance(filters, dict) or 'conditions' not in filters or not filters['conditions']:
        logger.debug("No valid filters provided or filters are empty, returning all initial RAG results.")
        return rag_results # 필터 없으면 원본 반환

    logic = filters.get('logic', 'AND').upper() # 기본값 AND
    conditions = filters['conditions']
    filtered_results = []

    logger.info(f"Applying post-filtering with logic '{logic}' and {len(conditions)} conditions...")

    for item in rag_results:
        match = False # 개별 아이템의 매치 여부
        if logic == 'AND':
            match = True # AND는 모든 조건 만족해야 하므로 True로 시작
            for condition in conditions:
                if not isinstance(condition, dict) or not all(k in condition for k in ['field', 'operator', 'value']):
                    logger.warning(f"Skipping invalid filter condition format: {condition}")
                    continue

                field = condition['field']
                operator = condition['operator']
                value = condition['value']
                item_value = item.get(field) # 메타데이터에서 값 가져오기 (없으면 None)

                # 필드 값이 메타데이터에 없는 경우 AND 조건에서는 무조건 실패
                if item_value is None:
                    logger.debug(f"Item ID {item.get('id', 'N/A')} missing filter field '{field}', fails AND condition.")
                    match = False
                    break # 더 이상 다른 조건 볼 필요 없음

                # 조건 검사
                condition_match = False
                try:
                    if operator == '==':
                        # 타입 고려 (가격은 숫자 비교, 나머지는 문자열 비교 - 필요시 타입 변환)
                        if field == 'price_numeric':
                            condition_match = int(item_value) == int(value)
                        else: # brand, category 등 문자열 비교 (대소문자 무시 고려 가능)
                            condition_match = str(item_value).lower() == str(value).lower()
                    elif operator == '<=':
                        if field == 'price_numeric':
                            condition_match = int(item_value) <= int(value)
                        else: logger.warning(f"Operator '<=' not supported for field '{field}'.")
                    elif operator == '>=':
                        if field == 'price_numeric':
                            condition_match = int(item_value) >= int(value)
                        else: logger.warning(f"Operator '>=' not supported for field '{field}'.")
                    elif operator == 'contains':
                        # features 필드는 list, value는 string 또는 list 일 수 있음
                        if field == 'features' and isinstance(item_value, list):
                            if isinstance(value, list): # value가 리스트면 모든 항목 포함 여부 (AND)
                                condition_match = all(str(v).lower() in [str(f).lower() for f in item_value] for v in value)
                            else: # value가 문자열이면 해당 문자열 포함 여부
                                condition_match = any(str(value).lower() in str(f).lower() for f in item_value)
                        else: logger.warning(f"Operator 'contains' requires field '{field}' to be a list in metadata.")
                    else:
                        logger.warning(f"Unsupported operator '{operator}' for field '{field}'.")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error comparing field '{field}' (value: {item_value}) with filter value '{value}': {e}")
                    condition_match = False

                if not condition_match:
                    match = False
                    break # AND 조건 중 하나라도 실패하면 break

            # 모든 AND 조건을 통과한 경우
            if match:
                filtered_results.append(item)

        elif logic == 'OR':
            match = False # OR는 하나라도 만족하면 되므로 False로 시작
            for condition in conditions:
                if not isinstance(condition, dict) or not all(k in condition for k in ['field', 'operator', 'value']):
                    logger.warning(f"Skipping invalid filter condition format: {condition}")
                    continue

                field = condition['field']
                operator = condition['operator']
                value = condition['value']
                item_value = item.get(field) # 메타데이터에서 값 가져오기

                # 필드 값이 없어도 OR 조건은 다른 조건으로 통과 가능하므로 바로 실패 아님
                if item_value is None:
                    logger.debug(f"Item ID {item.get('id', 'N/A')} missing filter field '{field}', skipping this condition for OR logic.")
                    continue # 이 조건은 건너뛰고 다음 OR 조건 확인

                # 조건 검사 (AND와 동일 로직)
                condition_match = False
                try:
                    # (AND 로직과 동일한 비교 로직 적용)
                    if operator == '==':
                        if field == 'price_numeric': condition_match = int(item_value) == int(value)
                        else: condition_match = str(item_value).lower() == str(value).lower()
                    elif operator == '<=':
                        if field == 'price_numeric': condition_match = int(item_value) <= int(value)
                        else: logger.warning(f"Operator '<=' not supported for field '{field}'.")
                    elif operator == '>=':
                        if field == 'price_numeric': condition_match = int(item_value) >= int(value)
                        else: logger.warning(f"Operator '>=' not supported for field '{field}'.")
                    elif operator == 'contains':
                        if field == 'features' and isinstance(item_value, list):
                            if isinstance(value, list): condition_match = all(str(v).lower() in [str(f).lower() for f in item_value] for v in value)
                            else: condition_match = any(str(value).lower() in str(f).lower() for f in item_value)
                        else: logger.warning(f"Operator 'contains' requires field '{field}' to be a list.")
                    else: logger.warning(f"Unsupported operator '{operator}' for field '{field}'.")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error comparing field '{field}' (value: {item_value}) with filter value '{value}': {e}")
                    condition_match = False

                if condition_match:
                    match = True
                    break # OR 조건 중 하나라도 성공하면 break

            # OR 조건 중 하나라도 만족한 경우
            if match:
                filtered_results.append(item)
        else:
            logger.error(f"Unsupported filter logic: {logic}. Defaulting to AND.")
            # AND 로직과 동일하게 처리 (Fallback)
            match = True
            # ... (AND 로직 반복) ...
            if match: filtered_results.append(item)


    logger.info(f"Post-filtering complete. {len(filtered_results)} results passed the filters.")
    return filtered_results


# --- [수정됨] 메인 오케스트레이션 함수 (Tool Use 워크플로우) ---
async def orchestrate_chatbot_turn(
    user_input: str,
    conversation_state: ConversationState,
    session: aiohttp.ClientSession,
    rag_searcher: Optional[RagSearcher] # RAG 검색기 인스턴스
) -> Dict[str, Any]:
    """
    Tool Use 기반 챗봇 응답 생성 오케스트레이션 함수.
    1. 1차 LLM 호출로 Tool Use 여부 및 파라미터 결정.
    2. Tool 호출 시: RAG 검색 -> 후 필터링 -> 2차 LLM 호출로 최종 답변 생성.
    3. Tool 미호출 시: 1차 LLM 응답 사용.
    4. 오류 처리 포함.

    Args:
        user_input (str): 현재 사용자 입력 텍스트.
        conversation_state (ConversationState): 현재 대화 상태 객체.
        session (aiohttp.ClientSession): API 호출에 사용할 공유 aiohttp 세션.
        rag_searcher (Optional[RagSearcher]): 초기화된 RagSearcher 인스턴스.

    Returns:
        Dict[str, Any]: 최종 결과 딕셔너리.
            {'response': str, 'debug_info': {...}} 또는
            {'error': str} 또는
            {'error_message_for_user': str} (재검색 유도 메시지)
    """
    # 필수 모듈 확인
    if not all([call_gpt_async, get_openai_embedding_async, get_config, api_logger]):
        logger.error("CRITICAL: Required functions/modules not available in scheduler.")
        return {"error": "Scheduler dependencies missing"}

    # --- config 로드 ---
    try:
        config = get_config()
        if not config: raise ValueError("Configuration could not be loaded.")
        # Tool Use 관련 설정
        tool_use_config = config.get('tasks', {}).get('tool_use', {})
        tool_use_model = tool_use_config.get('model', 'gpt-4o')
        decision_temp = tool_use_config.get('decision_temperature', 0.2)
        decision_max_tokens = tool_use_config.get('decision_max_tokens', 1500)
        generation_temp = tool_use_config.get('generation_temperature', 0.7)
        generation_max_tokens = tool_use_config.get('generation_max_tokens', 3000)
        # RAG 설정
        rag_config = config.get('rag', {})
        rag_k = rag_config.get('retrieval_k', 5)
        # 프롬프트
        prompts_config = config.get('prompts', {})
        tool_use_prompt_template = prompts_config.get('tool_use_system_prompt')
        tool_arg_error_prompt = prompts_config.get('tool_argument_error_prompt')
        # Tools 정의
        tools_definition = config.get('tools')

        if not all([tool_use_model, tool_use_prompt_template, tool_arg_error_prompt, tools_definition]):
            raise ValueError("Required Tool Use configurations missing in config.yaml")

    except Exception as conf_e:
        logger.error(f"CRITICAL: Failed to load configuration within scheduler: {conf_e}", exc_info=True)
        return {"error": "Configuration load failed"}

    start_time_scheduler = time.time()
    logger.info("--- Starting Tool Use Orchestration ---")
    debug_info = {"orchestration_start_time": start_time_scheduler} # 디버그 정보 기록 시작

    # --- 1. 1차 LLM 호출 준비 ---
    try:
        # 컨텍스트 가져오기 (이전 턴 완료 후 업데이트된 상태)
        history = conversation_state.get_history()
        previous_summary = conversation_state.get_summary()
        previous_slots = conversation_state.get_slots()
        debug_info['context_history_turns'] = len(history)
        debug_info['context_summary_present'] = bool(previous_summary)
        debug_info['context_slots_present'] = bool(previous_slots)

        # 메시지 구성 (시스템 프롬프트 + 기록 + 현재 입력)
        messages = [{"role": "system", "content": tool_use_prompt_template}]
        # TODO: 프롬프트에 요약/슬롯 정보를 어떻게 포함할지 전략 필요 (예: 시스템 프롬프트 내에 삽입)
        # 예: messages = [{"role": "system", "content": tool_use_prompt_template.format(summary=previous_summary or 'N/A', slots=str(previous_slots or {}))}]
        # 주의: history 와 중복될 수 있으므로 프롬프트 튜닝 필요

        messages.extend(history) # 이전 대화 기록 추가
        messages.append({"role": "user", "content": user_input}) # 현재 사용자 입력 추가

    except Exception as e:
        logger.error(f"Error preparing messages for 1st LLM call: {e}", exc_info=True)
        return {"error": "Failed to prepare messages"}

    # --- 2. 1차 LLM 호출 (Tool Use 결정) ---
    logger.info("Executing 1st LLM call for Tool Use decision...")
    llm_call_1_start = time.time()
    response_1 = await call_gpt_async(
        messages=messages,
        model=tool_use_model,
        temperature=decision_temp,
        max_tokens=decision_max_tokens,
        session=session,
        tools=tools_definition,
        tool_choice="auto"
    )
    llm_call_1_duration = time.time() - llm_call_1_start
    debug_info['llm_call_1_duration_ms'] = int(llm_call_1_duration * 1000)
    debug_info['llm_call_1_model'] = tool_use_model

    if response_1 is None:
        logger.error("1st LLM call failed.")
        debug_info['status'] = 'failed_llm_call_1'
        # 사용자 요청대로 오류 메시지 반환
        return {"error_message_for_user": "대화 출력이 실패했습니다.", "debug_info": debug_info}

    # --- 3. LLM 응답 처리 ---
    message_1 = response_1["choices"][0].get("message", {})
    tool_calls = message_1.get("tool_calls")
    content_1 = message_1.get("content")

    if tool_calls:
        # --- 4. Tool 실행 (product_search 호출) ---
        logger.info(f"LLM requested tool call(s): {[tc.get('function', {}).get('name') for tc in tool_calls]}")
        debug_info['tool_called'] = True
        tool_results = [] # 여러 tool_call 이 있을 경우 대비 (현재는 하나만 예상)
        final_response = None # 최종 응답 초기화

        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            tool_call_id = tool_call.get("id")
            debug_info['tool_function_name'] = function_name
            debug_info['tool_call_id'] = tool_call_id

            if function_name == "product_search":
                # 4a. 인수 파싱
                logger.debug(f"Parsing arguments for tool call ID: {tool_call_id}")
                try:
                    arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                    arguments = json.loads(arguments_str)
                    search_keywords = arguments.get("search_keywords")
                    filters = arguments.get("filters") # Optional
                    num_results_req = arguments.get("num_results", 3) # Tool 정의의 default 사용
                    debug_info['tool_arguments'] = arguments # 파싱된 인수 기록

                    if not search_keywords:
                        raise ValueError("Required argument 'search_keywords' is missing.")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse arguments or missing required 'search_keywords' for tool call {tool_call_id}: {e}")
                    debug_info['tool_execution_status'] = 'failed_arg_parsing'
                    debug_info['error'] = f"Argument parsing error: {e}"
                    # 사용자 요청대로 재검색 유도 메시지 생성 요청
                    return {"error": "invalid_tool_arguments", "debug_info": debug_info}

                # 4b. 임베딩 생성
                logger.debug(f"Generating embedding for keywords: '{search_keywords[:50]}...'")
                embedding_start = time.time()
                query_embedding = await get_openai_embedding_async(search_keywords, session)
                embedding_duration = time.time() - embedding_start
                debug_info['embedding_duration_ms'] = int(embedding_duration * 1000)

                if query_embedding is None:
                    logger.error("Failed to generate query embedding.")
                    debug_info['tool_execution_status'] = 'failed_embedding'
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": json.dumps({"error": "Embedding generation failed", "results_found": False}),
                    })
                    continue # 다음 tool_call (있다면) 처리

                # 4c. RAG 검색
                logger.debug(f"Performing RAG search with k={rag_k}")
                rag_start = time.time()
                initial_rag_results = await run_rag_search_async(query_embedding, rag_k, rag_searcher)
                rag_duration = time.time() - rag_start
                debug_info['rag_search_duration_ms'] = int(rag_duration * 1000)
                debug_info['rag_results_initial_count'] = len(initial_rag_results)

                # 4d. 후 필터링
                logger.debug(f"Applying post-filtering based on LLM filters: {filters}")
                filtering_start = time.time()
                filtered_rag_results = apply_metadata_filters(initial_rag_results, filters)
                filtering_duration = time.time() - filtering_start
                debug_info['filtering_duration_ms'] = int(filtering_duration * 1000)
                debug_info['rag_results_filtered_count'] = len(filtered_rag_results)

                # 4e. Tool 결과 포맷팅
                tool_result_content = ""
                if filtered_rag_results:
                    # num_results_req 에 맞춰 결과 개수 제한
                    results_to_include = filtered_rag_results[:num_results_req]
                    # 상세 정보 포함 JSON 생성 (예시 - 실제 필요한 필드 선택)
                    formatted_results = [
                        {
                            "product_name": r.get("product_name"),
                            "brand": r.get("brand"),
                            "category": r.get("category"),
                            "price": r.get("price"),
                            "features": r.get("features", [])[:3], # 상위 3개 특징
                            "reviews_overall": r.get("reviews_overall"),
                            "similarity_score": round(r.get("similarity_score", 0.0), 4)
                        } for r in results_to_include
                    ]
                    tool_result_content = json.dumps({"results": formatted_results, "results_found": True}, ensure_ascii=False)
                else:
                    # 결과 없음 처리 (사용자 요청 반영)
                    tool_result_content = json.dumps({"results_found": False})
                    # 재검색 유도를 위한 메시지는 2차 LLM 호출 프롬프트에서 처리

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": tool_result_content,
                })
                debug_info['tool_execution_status'] = 'success'

            else:
                logger.warning(f"Received unhandled tool call function name: {function_name}")
                # 알 수 없는 함수 호출 시 처리 (예: 에러 메시지 반환)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps({"error": f"Unknown function name: {function_name}"}),
                })
                debug_info['tool_execution_status'] = f'unhandled_function_{function_name}'


        # --- 5. 2차 LLM 호출 (최종 답변 생성) ---
        if tool_results: # Tool 실행 결과가 하나라도 있으면 2차 호출 진행
            logger.info("Executing 2nd LLM call to generate final response based on tool results...")
            # 이전 메시지 + 1차 LLM 응답(tool_calls 포함) + tool 결과 메시지
            messages_for_call_2 = messages + [message_1] + tool_results
            llm_call_2_start = time.time()
            response_2 = await call_gpt_async(
                messages=messages_for_call_2,
                model=tool_use_model, # 동일 모델 사용
                temperature=generation_temp,
                max_tokens=generation_max_tokens,
                session=session
                # tools 파라미터는 전달하지 않음
            )
            llm_call_2_duration = time.time() - llm_call_2_start
            debug_info['llm_call_2_duration_ms'] = int(llm_call_2_duration * 1000)
            debug_info['llm_call_2_model'] = tool_use_model

            if response_2 and response_2["choices"]:
                final_response = response_2["choices"][0].get("message", {}).get("content")
                debug_info['status'] = 'completed_with_tool_use'
            else:
                logger.error("2nd LLM call failed or returned no content.")
                debug_info['status'] = 'failed_llm_call_2'
                # 사용자 요청대로 오류 메시지 반환
                return {"error_message_for_user": "대화 출력이 실패했습니다.", "debug_info": debug_info}
        else:
             logger.error("No tool results generated despite tool call request.")
             debug_info['status'] = 'failed_tool_execution_no_result'
             return {"error_message_for_user": "정보 검색 중 오류가 발생했습니다.", "debug_info": debug_info}

    elif content_1:
        # --- 6. Tool 미호출 시 (1차 LLM 호출로 종료) ---
        logger.info("Tool use not required by LLM. Using direct response.")
        final_response = content_1
        debug_info['tool_called'] = False
        debug_info['status'] = 'completed_without_tool_use'
        # TODO: 여기에 LLM이 명백히 검색 필요성을 놓친 경우 Fallback 로직 추가 가능 (선택 사항)

    else:
        # 1차 LLM 응답에 content도, tool_calls도 없는 경우
        logger.error("1st LLM call returned neither content nor tool_calls.")
        debug_info['status'] = 'failed_llm_call_1_empty'
        return {"error_message_for_user": "대화 응답 생성에 실패했습니다.", "debug_info": debug_info}


    # --- 7. 최종 결과 반환 ---
    end_time_scheduler = time.time()
    total_duration = end_time_scheduler - start_time_scheduler
    debug_info['total_orchestration_time_ms'] = int(total_duration * 1000)
    logger.info(f"--- Scheduler Orchestration Finished in {total_duration:.3f} seconds (Status: {debug_info.get('status', 'unknown')}) ---")

    return {"response": final_response, "debug_info": debug_info}


# --- 예시 사용법 (직접 실행 어려움, 통합 테스트 필요) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- Running scheduler.py as main script (placeholder) ---")
    print("Scheduler module contains the core orchestration logic using Tool Use.")
    print("Direct execution is complex. Please test via app.py or test_runner.py.")
    # async def test_orchestration():
    #     # Requires setting up ConversationState, RagSearcher, aiohttp.ClientSession, and mocks/live API calls
    #     pass
    # asyncio.run(test_orchestration())