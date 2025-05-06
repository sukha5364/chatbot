# chatbot/scheduler.py (Summary+K-Turns, Detailed Logging, Periodic Summary, Configurable Iterations 적용 최종 버전)

import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Union
import aiohttp
import numpy as np

# --- 필요한 모듈 임포트 ---
try:
    # gpt_interface 모듈에서 api_logger 임포트 유지 (로깅용)
    from .gpt_interface import call_gpt_async, get_openai_embedding_async, api_logger
    from .searcher import RagSearcher
    from .conversation_state import ConversationState
    from .config_loader import get_config
    # Slot/Summary/ModelRouter/PromptBuilder 관련 임포트는 제거됨
    logging.info("Required modules imported successfully in scheduler.")
except ImportError as ie:
    logging.error(f"ERROR (scheduler): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 스케줄러 기능 사용 불가 처리
    call_gpt_async = None
    get_openai_embedding_async = None
    RagSearcher = None
    ConversationState = None
    get_config = None
    api_logger = None

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

# --- RAG 검색 비동기 실행 함수 ---
async def run_rag_search_async(
    query_embedding: Union[List[float], np.ndarray, None],
    k: int,
    rag_searcher: Optional[RagSearcher]
) -> List[Dict]:
    """
    주어진 쿼리 임베딩과 RagSearcher 객체를 사용하여 RAG 검색을 비동기적으로 실행합니다.
    (FAISS 검색은 동기 함수이므로 run_in_executor 사용)
    """
    if rag_searcher is None or rag_searcher.index is None or not rag_searcher.metadata:
        logger.warning("RagSearcher not available/initialized. Skipping RAG search.")
        return []
    if query_embedding is None:
        logger.warning("Query embedding is None. Skipping RAG search.")
        return []
    if k <= 0:
        logger.warning(f"Invalid k ({k}) for RAG search. Skipping.")
        return []

    try:
        # 입력 임베딩 타입 처리 (리스트 또는 NumPy 배열)
        if isinstance(query_embedding, list):
            query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        elif isinstance(query_embedding, np.ndarray):
            query_embedding_np = query_embedding.astype(np.float32).reshape(1, -1)
        else:
            raise TypeError(f"Invalid query embedding type: {type(query_embedding)}")

        # FAISS 인덱스 차원과 쿼리 임베딩 차원 일치 확인
        # RagSearcher 생성 시점에 이미 확인하지만, 방어적으로 한 번 더 체크
        index_dim = getattr(rag_searcher.index, 'd', None)
        if index_dim is None or index_dim != query_embedding_np.shape[1]:
            raise ValueError(f"Query embedding dim ({query_embedding_np.shape[1]}) mismatch with index dim ({index_dim}).")

    except (TypeError, ValueError) as e:
        logger.error(f"Error processing query embedding for RAG: {e}", exc_info=False)
        return []

    loop = asyncio.get_running_loop()
    logger.debug(f"Running async RAG search in executor (k={k}, emb_shape={query_embedding_np.shape})")

    try:
        start_t = time.time()
        # RagSearcher.search는 동기 함수이므로 run_in_executor 사용
        results = await loop.run_in_executor(
            None, rag_searcher.search, query_embedding_np, k
        )
        duration_t = time.time() - start_t
        logger.debug(f"Async RAG search finished in {duration_t:.4f}s. Found {len(results)} initial results.")
        return results # 검색 결과 (메타데이터 리스트) 반환
    except Exception as e:
        logger.error(f"Error during async RAG search execution: {e}", exc_info=True)
        return []

# --- 메타데이터 후 필터링 함수 ---
def apply_metadata_filters(
    rag_results: List[Dict],
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    RAG 검색 결과(메타데이터 리스트)에 대해 LLM이 생성한 필터 조건을 적용하여 결과를 필터링합니다.
    기본적으로 OR 논리를 사용하고, 필드 부재 시 처리 로직을 명확히 합니다.

    Args:
        rag_results (List[Dict]): RAG 검색으로 얻은 초기 메타데이터 딕셔너리 리스트.
        filters (Optional[Dict]): LLM이 생성한 필터 조건 객체 (logic, conditions 포함).

    Returns:
        List[Dict]: 필터 조건에 맞는 메타데이터 딕셔너리 리스트.
    """
    if not filters or not isinstance(filters, dict) or 'conditions' not in filters or not filters['conditions']:
        logger.debug("No valid filters provided or filters empty, returning all initial RAG results.")
        return rag_results # 필터 없으면 원본 반환

    logic = filters.get('logic', 'OR').upper() # [수정됨] 기본값 OR 사용 및 대문자 변환
    conditions = filters['conditions']
    filtered_results = []

    logger.info(f"Applying post-filtering with logic '{logic}' and {len(conditions)} conditions...")
    if logic not in ['AND', 'OR']:
        logger.warning(f"Unsupported filter logic '{logic}'. Defaulting to OR.")
        logic = 'OR'

    for item in rag_results:
        item_metadata = item # searcher 결과가 바로 메타데이터 딕셔너리 리스트
        item_id = item.get('id', 'Unknown') # 로깅용 ID

        conditions_met = [] # 각 조건 충족 여부 저장 (OR/AND 로직 적용 위해)
        for condition in conditions:
            if not isinstance(condition, dict) or not all(k in condition for k in ['field', 'operator', 'value']):
                logger.warning(f"Skipping invalid filter condition format: {condition}")
                conditions_met.append(False) # 잘못된 조건은 False 처리
                continue

            field = condition['field']
            operator = condition['operator']
            filter_value = condition['value']
            item_value = item_metadata.get(field) # 메타데이터에서 값 가져오기

            condition_match = False # 현재 조건 매칭 여부

            # 필드 값 존재 여부 확인
            if item_value is None:
                logger.debug(f"Filter field '{field}' not found in metadata for item '{item_id}'.")
                # OR 논리에서는 이 조건은 False지만, 다른 조건으로 통과 가능
                # AND 논리에서는 이 조건 때문에 전체 아이템이 탈락
                condition_match = False
            else:
                # 필드 값 존재 시 조건 검사
                try:
                    if operator == '==':
                        # 대소문자 구분 없이 비교 (브랜드, 카테고리 등)
                        condition_match = str(item_value).lower() == str(filter_value).lower()
                    elif operator == '<=':
                        # 숫자 필드(price_numeric)만 지원 가정
                        if field == 'price_numeric':
                            # 타입 변환 시도 및 비교
                            condition_match = int(item_value) <= int(filter_value)
                        else: logger.warning(f"Operator '<=' not applicable to field '{field}'. Condition fails.")
                    elif operator == '>=':
                        if field == 'price_numeric':
                            condition_match = int(item_value) >= int(filter_value)
                        else: logger.warning(f"Operator '>=' not applicable to field '{field}'. Condition fails.")
                    elif operator == 'contains':
                        # features 필드 (list) 또는 문자열 필드 지원
                        if isinstance(item_value, list): # features 필드
                            if isinstance(filter_value, list): # 필터 값이 리스트면 모든 값 포함 (AND)
                                condition_match = all(str(fv).lower() in [str(iv).lower() for iv in item_value] for fv in filter_value)
                            else: # 필터 값이 문자열이면 하나라도 포함 (OR)
                                condition_match = any(str(filter_value).lower() in str(iv).lower() for iv in item_value)
                        elif isinstance(item_value, str): # 일반 문자열 필드
                            condition_match = str(filter_value).lower() in item_value.lower()
                        else: logger.warning(f"Operator 'contains' not supported for item_value type {type(item_value)} in field '{field}'.")
                    else:
                        logger.warning(f"Unsupported operator '{operator}' for field '{field}'.")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error comparing field '{field}' (value: {item_value}, type: {type(item_value)}) with filter value '{filter_value}': {e}")
                    condition_match = False

            conditions_met.append(condition_match)
            logger.debug(f" - Item '{item_id}', Condition: {condition}, ItemValue: {item_value}, Match: {condition_match}")

        # 최종 매치 여부 판단 (OR / AND)
        final_match = False
        if logic == 'OR':
            final_match = any(conditions_met) # 하나라도 True면 통과
        elif logic == 'AND':
            # 모든 조건이 True여야 하고, 조건 리스트가 비어있지 않아야 함
            final_match = all(conditions_met) and bool(conditions_met)

        if final_match:
            filtered_results.append(item)
            logger.debug(f" => Item '{item_id}' PASSED filtering (Logic: {logic})")
        # else: logger.debug(f" => Item '{item_id}' FAILED filtering (Logic: {logic})")

    logger.info(f"Post-filtering complete. {len(filtered_results)} results passed the filters.")
    return filtered_results


# --- [수정됨] 메인 오케스트레이션 함수 (Summary + K Turns, Configurable Max Iterations 적용) ---
async def orchestrate_chatbot_turn(
    user_input: str,
    conversation_state: ConversationState,
    session: aiohttp.ClientSession,
    rag_searcher: Optional[RagSearcher]
) -> Dict[str, Any]:
    """
    'Summary + K Turns' 컨텍스트와 Tool Use 기반 챗봇 응답 생성 오케스트레이션 함수
    (다중/순차 호출 지원, 최대 반복 횟수 설정 가능).
    """
    # --- 필수 모듈 및 설정 로드 ---
    if not all([call_gpt_async, get_openai_embedding_async, get_config, api_logger]):
        logger.critical("CRITICAL: Required scheduler dependencies missing.")
        return {"error_message_for_user": "죄송합니다, 시스템 설정 오류로 답변을 드릴 수 없습니다."}

    try:
        config = get_config()
        if not config: raise ValueError("Config load failed.")
        tool_use_config = config.get('tasks', {}).get('tool_use', {})
        rag_config = config.get('rag', {})
        prompts_config = config.get('prompts', {})
        tools_definition = config.get('tools')

        # 필요한 설정값 추출 및 검증
        tool_use_model = tool_use_config.get('model', 'gpt-4o')
        decision_temp = tool_use_config.get('decision_temperature', 0.2)
        decision_max_tokens = tool_use_config.get('decision_max_tokens', 1500)
        # [수정됨] generation 파라미터는 Tool 결과 후 최종 답변 생성 시 사용하도록 분리
        generation_temp = tool_use_config.get('generation_temperature', 0.7)
        generation_max_tokens = tool_use_config.get('generation_max_tokens', 3000)
        rag_k = rag_config.get('retrieval_k', 15)
        tool_use_prompt_template = prompts_config.get('tool_use_system_prompt')
        tool_arg_error_prompt = prompts_config.get('tool_argument_error_prompt')
        context_k = prompts_config.get('context_recent_k_turns', 3) # [신규] K 값 읽기
        # [신규] 최대 Tool 반복 횟수 읽기 (기본값 3)
        max_tool_iterations = tool_use_config.get('max_iterations', 3)
        logger.debug(f"Max Tool Iterations set to: {max_tool_iterations}")

        # 필수 설정값 존재 및 타입 검증
        if not all([tool_use_model, tool_use_prompt_template, tool_arg_error_prompt, tools_definition, isinstance(rag_k, int), isinstance(context_k, int), isinstance(max_tool_iterations, int)]):
            raise ValueError("Essential configurations for Tool Use, RAG, or Context Management are missing or invalid.")
        if context_k < 0: # K=0 허용 (요약만 사용)
            logger.warning(f"Invalid context_recent_k_turns ({context_k}). Using 0.")
            context_k = 0
        if max_tool_iterations <= 0:
            logger.warning(f"Invalid max_iterations ({max_tool_iterations}). Using default 1.")
            max_tool_iterations = 1

    except Exception as conf_e:
        logger.critical(f"Critical configuration error in scheduler: {conf_e}", exc_info=True)
        return {"error_message_for_user": "죄송합니다, 시스템 설정 오류로 답변을 드릴 수 없습니다."}

    # --- 오케스트레이션 시작 ---
    start_time_scheduler = time.time()
    logger.info("--- Starting Tool Use Orchestration Cycle (Summary + K Turns) ---")
    debug_info = {"orchestration_start_time": start_time_scheduler, "steps": []}

    # --- [수정됨] 컨텍스트 구성 (Summary + K Turns) ---
    messages = []
    # 1. 시스템 프롬프트 준비 (요약 포함)
    current_summary = conversation_state.get_summary()
    # 요약본이 있으면 프롬프트에 섹션 추가, 없으면 최근 대화만 표시
    summary_section_text = f"[이전 대화 요약]:\n{current_summary}\n\n[최근 대화 기록]:" if current_summary else "[최근 대화 기록]:"
    try:
        # 프롬프트 템플릿에 summary_section 플레이스홀더가 있다고 가정
        system_prompt_content = tool_use_prompt_template.format(summary_section=summary_section_text)
    except KeyError:
        logger.warning("'{summary_section}' placeholder not found in 'tool_use_system_prompt'. Appending summary separately.")
        # Fallback: 요약 정보를 별도 메시지로 추가하거나, 시스템 프롬프트에 단순히 덧붙임
        system_prompt_content = tool_use_prompt_template + "\n\n" + summary_section_text # 간단하게 뒤에 붙이는 방식
    except Exception as fmt_e:
        logger.error(f"Error formatting tool_use_system_prompt: {fmt_e}. Using base prompt.")
        system_prompt_content = tool_use_prompt_template # 포맷팅 실패 시 원본 사용

    messages.append({"role": "system", "content": system_prompt_content})

    # 2. 최근 K턴 기록 추가
    full_history = conversation_state.get_history(copy=True) # 원본 보호 위해 복사본 사용
    if context_k > 0:
        # K턴에 해당하는 메시지 수 추정 (User + Assistant + Tool Calls + Tool Results 등 고려)
        # 더 정확한 로직은 턴 단위로 그룹화하는 것이지만, 여기서는 메시지 개수로 근사
        # 각 턴은 최소 user+assistant(tool_call) 이므로 2개, tool 결과까지 하면 3개 이상 가능
        approx_messages_per_turn = 3 # 평균 턴당 메시지 수 (조정 가능)
        num_messages_to_get = context_k * approx_messages_per_turn
        last_k_messages = full_history[-num_messages_to_get:] # 뒤에서부터 N개 메시지 추출
        messages.extend(last_k_messages)
        logger.debug(f"Added last {len(last_k_messages)} messages (approximating K={context_k} turns) to context.")
        # 디버깅 위해 K턴 내용 로깅 (길이 제한)
        k_turn_preview = json.dumps([{"role": m.get("role"), "content": str(m.get("content"))[:50] + "..."} for m in last_k_messages], ensure_ascii=False, indent=2)
        logger.debug(f"Last K turns preview:\n{k_turn_preview}")
    else:
        logger.debug("context_recent_k_turns is 0, not adding recent history.")

    # 3. 현재 사용자 입력 추가
    messages.append({"role": "user", "content": user_input})

    # --- LLM 호출 및 Tool 실행 반복 루프 ---
    current_iteration = 0
    final_response_content = None

    while current_iteration < max_tool_iterations: # [수정됨] 설정값 사용
        current_iteration += 1
        step_debug = {"iteration": current_iteration, "start_time": time.time()}
        logger.info(f"--- Iteration {current_iteration}/{max_tool_iterations} ---")

        # 1. LLM 호출 (Tool 결정 또는 답변 생성)
        logger.info(f"Executing LLM call #{current_iteration}...")
        step_debug["llm_call_start"] = time.time()

        # --- 수정된 부분 시작 ---
        # 마지막 메시지가 tool 역할인지 확인하여 파라미터 결정
        use_generation_params = False
        if messages and messages[-1].get("role") == "tool":
            logger.debug("Previous message was from a tool. Using 'generation' parameters.")
            current_temp = generation_temp
            current_max_tokens = generation_max_tokens
            use_generation_params = True # 최종 답변 생성 단계임을 표시
        else:
            logger.debug("Previous message not from a tool (or first call). Using 'decision' parameters.")
            current_temp = decision_temp
            current_max_tokens = decision_max_tokens
        # --- 수정된 부분 끝 ---

        # [!중요!] 도구 사용 후 답변 생성 시에는 도구 사용을 강제하지 않도록 tool_choice 조정 필요
        current_tool_choice = "auto" if not use_generation_params else None # Tool 결과 받은 후엔 강제 도구 호출 방지
        if use_generation_params:
            logger.debug("Setting tool_choice to None for final response generation.")

        llm_response = await call_gpt_async(
            messages=messages, # 현재까지 누적된 메시지 전달
            model=tool_use_model,
            temperature=current_temp, # 결정된 파라미터 사용
            max_tokens=current_max_tokens, # 결정된 파라미터 사용
            session=session,
            tools=tools_definition,
            tool_choice=current_tool_choice # 수정된 tool_choice 적용
        )
        step_debug["llm_call_end"] = time.time()
        step_debug["llm_call_duration_ms"] = int((step_debug["llm_call_end"] - step_debug["llm_call_start"]) * 1000)
        step_debug["llm_model_used"] = tool_use_model
        step_debug["llm_params_used"] = {"temperature": current_temp, "max_tokens": current_max_tokens, "tool_choice": current_tool_choice} # 수정된 파라미터 로깅


        if not llm_response or not llm_response.get("choices"):
            logger.error(f"LLM call #{current_iteration} failed or returned no choices.")
            step_debug["status"] = "failed_llm_call"
            step_debug["error"] = "LLM API call failed"
            debug_info["steps"].append(step_debug)
            # 사용자에게 표시될 수 있는 안전한 오류 메시지 반환
            return {"error_message_for_user": "죄송합니다, 답변 생성 중 오류가 발생했습니다 (LLM 호출 실패).", "debug_info": debug_info}

        # LLM 응답 메시지 추출
        assistant_message = llm_response["choices"][0].get("message", {})
        messages.append(assistant_message) # 다음 호출 또는 최종 히스토리 저장을 위해 어시스턴트 응답 추가
        step_debug["llm_response_raw"] = assistant_message # 디버깅용 (전체 저장)

        tool_calls = assistant_message.get("tool_calls")
        response_content = assistant_message.get("content")

        if tool_calls:
            # 2. Tool 실행 (product_search)
            logger.info(f"LLM requested {len(tool_calls)} tool call(s).")
            step_debug["tool_calls_requested"] = tool_calls
            tool_results_for_next_call = [] # 이번 이터레이션의 Tool 결과 저장

            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                function_call = tool_call.get("function")
                if not tool_call_id or not function_call:
                    logger.warning(f"Skipping invalid tool call object: {tool_call}")
                    continue

                function_name = function_call.get("name")
                logger.info(f"Processing tool call ID: {tool_call_id}, Function: {function_name}")
                # 각 Tool 호출별 디버그 정보 저장용
                tool_step_debug = {"tool_call_id": tool_call_id, "function_name": function_name}

                if function_name == "product_search":
                    # 인수 파싱
                    arguments = {}
                    try:
                        arguments_str = function_call.get("arguments", "{}")
                        arguments = json.loads(arguments_str)
                        search_keywords = arguments.get("search_keywords")
                        filters = arguments.get("filters") # Optional
                        num_results_req = arguments.get("num_results", 3)
                        tool_step_debug["arguments"] = arguments

                        if not search_keywords: raise ValueError("Missing 'search_keywords'")
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.error(f"Failed to parse args for tool {tool_call_id}: {e}")
                        tool_step_debug["status"] = "failed_arg_parsing"
                        tool_step_debug["error"] = str(e)
                        # 파싱 실패 시 Tool 결과 메시지에 에러 포함
                        tool_results_for_next_call.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": function_name,
                            "content": json.dumps({"error": f"Argument parsing error: {e}", "results_found": False})
                        })
                        step_debug.setdefault("tool_executions", []).append(tool_step_debug) # 스텝 디버그에 추가
                        continue # 다음 Tool 호출 처리

                    # 임베딩 생성
                    tool_step_debug["embedding_start"] = time.time()
                    query_embedding = await get_openai_embedding_async(search_keywords, session)
                    tool_step_debug["embedding_end"] = time.time()
                    tool_step_debug["embedding_duration_ms"] = int((tool_step_debug["embedding_end"] - tool_step_debug["embedding_start"]) * 1000)

                    if query_embedding is None:
                        logger.error(f"Failed to get embedding for tool {tool_call_id}")
                        tool_step_debug["status"] = "failed_embedding"
                        tool_results_for_next_call.append({
                            "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                            "content": json.dumps({"error": "Embedding generation failed", "results_found": False})
                        })
                        step_debug.setdefault("tool_executions", []).append(tool_step_debug)
                        continue

                    # RAG 검색
                    tool_step_debug["rag_search_start"] = time.time()
                    initial_rag_results = await run_rag_search_async(query_embedding, rag_k, rag_searcher)
                    tool_step_debug["rag_search_end"] = time.time()
                    tool_step_debug["rag_search_duration_ms"] = int((tool_step_debug["rag_search_end"] - tool_step_debug["rag_search_start"]) * 1000)
                    tool_step_debug["rag_initial_count"] = len(initial_rag_results)

                    # 후 필터링
                    tool_step_debug["filtering_start"] = time.time()
                    filtered_rag_results = apply_metadata_filters(initial_rag_results, filters)
                    tool_step_debug["filtering_end"] = time.time()
                    tool_step_debug["filtering_duration_ms"] = int((tool_step_debug["filtering_end"] - tool_step_debug["filtering_start"]) * 1000)
                    tool_step_debug["rag_filtered_count"] = len(filtered_rag_results)
                    tool_step_debug["filters_applied"] = filters # 적용된 필터 기록

                    # Tool 결과 포맷팅
                    tool_result_content_obj = {} # [수정됨] JSON 객체로 생성 후 마지막에 dump
                    if filtered_rag_results:
                        results_to_include = filtered_rag_results[:num_results_req]
                        # 필요한 정보만 선택적으로 포함 (간결화)
                        formatted_results = [{
                                "product_name": r.get("product_name"), "brand": r.get("brand"),
                                "category": r.get("category"), "price": r.get("price"),
                                # [결정필요] features 를 얼마나 포함할지? 여기서는 예시로 5개
                                "features": r.get("features", [])[:5],
                                "similarity_score": round(r.get("similarity_score", 0.0), 4)
                            } for r in results_to_include]
                        tool_result_content_obj = {"results": formatted_results, "results_found": True}
                    else:
                        tool_result_content_obj = {"results_found": False}

                    # 최종 Tool 결과 문자열 생성
                    tool_result_content = json.dumps(tool_result_content_obj, ensure_ascii=False)

                    tool_results_for_next_call.append({
                        "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                        "content": tool_result_content # JSON 문자열 전달
                    })
                    tool_step_debug["status"] = "success"
                    tool_step_debug["result_summary"] = tool_result_content_obj # 디버깅용 객체 저장

                else: # 정의되지 않은 함수 호출 시
                    logger.warning(f"Received unhandled tool function name: {function_name}")
                    tool_step_debug["status"] = "unhandled_function"
                    tool_results_for_next_call.append({
                        "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                        "content": json.dumps({"error": f"Unknown function: {function_name}"})
                    })

                step_debug.setdefault("tool_executions", []).append(tool_step_debug) # 각 tool_call 디버그 정보 추가

            # 다음 LLM 호출을 위해 Tool 결과 메시지 추가
            messages.extend(tool_results_for_next_call)
            # 루프 계속 (다음 LLM 호출로)

        elif response_content:
            # 3. LLM이 Tool 호출 없이 직접 답변 생성
            logger.info(f"LLM generated final response directly in iteration {current_iteration}.")
            final_response_content = response_content # 최종 응답 저장
            step_debug["status"] = "completed_direct_response"
            debug_info["steps"].append(step_debug)
            break # 루프 종료

        else:
            # 4. LLM 응답에 content도 tool_calls도 없는 경우 (오류)
            logger.error(f"LLM response in iteration {current_iteration} had neither content nor tool_calls.")
            step_debug["status"] = "failed_llm_empty_response"
            debug_info["steps"].append(step_debug)
            # 사용자에게 표시될 수 있는 안전한 오류 메시지 반환
            return {"error_message_for_user": "죄송합니다, 응답 생성 중 예상치 못한 오류가 발생했습니다.", "debug_info": debug_info}

        # 루프 계속 전 현재 스텝 정보 저장
        debug_info["steps"].append(step_debug)

    # --- 루프 종료 후 처리 ---
    if final_response_content is None:
        # 최대 반복 도달 또는 다른 이유로 답변 생성 실패
        logger.warning(f"Failed to get final response content after {current_iteration} iterations (max: {max_tool_iterations}).")
        # 마지막 스텝 상태 업데이트 (실패 명시)
        if debug_info["steps"]:
            # 마지막 스텝이 이미 실패 상태가 아니면 max_iterations 실패로 기록
            if not debug_info["steps"][-1].get("status", "").startswith("failed"):
                debug_info["steps"][-1]["status"] = "failed_max_iterations"
        else: # 스텝 정보가 아예 없는 경우 (루프 진입 실패 등)
            debug_info["steps"].append({"status": "failed_unknown_before_loop"})

        # 사용자 오류 메시지 반환
        # 만약 마지막 Tool 실행 결과가 있다면 그걸 보여주는게 나을 수도? (여기서는 일단 일반 오류 메시지)
        # last_tool_result = messages[-1]['content'] if messages and messages[-1].get('role') == 'tool' else None
        return {"error_message_for_user": "죄송합니다, 요청을 처리하는 데 시간이 너무 오래 걸리거나 오류가 발생했습니다.", "debug_info": debug_info}

    # --- 최종 결과 반환 ---
    end_time_scheduler = time.time()
    total_duration = end_time_scheduler - start_time_scheduler
    debug_info['total_orchestration_time_ms'] = int(total_duration * 1000)
    debug_info['final_status'] = 'success'
    debug_info['total_iterations'] = current_iteration # 총 반복 횟수 추가
    logger.info(f"--- Scheduler Orchestration Cycle Finished in {total_duration:.3f} seconds ({current_iteration} iterations) ---")

    # 최종 히스토리에는 assistant 응답이 이미 messages 리스트 마지막에 추가되어 있음
    # app.py에서 이 assistant_message를 conversation_state에 저장할 때 tool_calls 정보도 함께 저장해야 함
    # scheduler 결과에 tool_calls 정보를 포함시켜 전달하는 방안 고려
    # orchestration_debug_info['llm_response_raw'] 안에 이미 포함되어 있음

    return {"response": final_response_content, "debug_info": debug_info}


# --- 예시 사용법 (직접 실행 어려움) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 기본 로깅 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running scheduler.py as main script (placeholder) ---")
    print("Scheduler module contains the core Tool Use orchestration logic.")
    print("Direct execution requires setting up dependencies (ConversationState, RagSearcher, etc.).")
    print("Please test via app.py or test_runner.py.")
    # 테스트 코드 추가 시, ConversationState, RagSearcher 목(Mock) 객체 및
    # aiohttp.ClientSession 설정 필요