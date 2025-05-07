# chatbot/chatbot/scheduler.py (o3 모델 토큰 파라미터 처리 및 gpt_interface.py 호출부 수정 반영)

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
    logging.info("Required modules imported successfully in scheduler.") # 일반 로그는 logging 사용 유지
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


# --- RAG 검색 비동기 실행 함수 (기존과 동일) ---
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
        if isinstance(query_embedding, list):
            query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        elif isinstance(query_embedding, np.ndarray):
            query_embedding_np = query_embedding.astype(np.float32).reshape(1, -1)
        else:
            raise TypeError(f"Invalid query embedding type: {type(query_embedding)}")

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
        results = await loop.run_in_executor(
            None, rag_searcher.search, query_embedding_np, k
        )
        duration_t = time.time() - start_t
        logger.debug(f"Async RAG search finished in {duration_t:.4f}s. Found {len(results)} initial results.")

        print("-" * 20 + " RAG Initial Search Results (Names Only) " + "-" * 20)
        if results:
            print(f"[RAG Initial Results] Found {len(results)} raw results (Product names & scores shown, max 5):")
            for i, res_meta in enumerate(results[:5]):
                try:
                    prod_name = res_meta.get('product_name', 'N/A')
                    score = res_meta.get('similarity_score', 0.0)
                    print(f"  [{i+1}] Product Name: {prod_name} (Score: {score:.4f})")
                except Exception as log_err:
                    print(f"  Error printing product name for initial result {i+1}: {log_err}")
        else:
            print("[RAG Initial Results] No results found from FAISS search.")
        print("-" * 70)
        return results
    except Exception as e:
        logger.error(f"Error during async RAG search execution: {e}", exc_info=True)
        return []

# --- 메타데이터 후 필터링 함수 (기존과 동일) ---
def apply_metadata_filters(
    rag_results: List[Dict],
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    RAG 검색 결과(메타데이터 리스트)에 대해 LLM이 생성한 필터 조건을 적용하여 결과를 필터링합니다.
    """
    if not filters or not isinstance(filters, dict) or 'conditions' not in filters or not filters['conditions']:
        logger.debug("No valid filters provided or filters empty, returning all initial RAG results.")
        return rag_results

    logic = filters.get('logic', 'OR').upper()
    conditions = filters['conditions']
    filtered_results = []

    logger.info(f"Applying post-filtering with logic '{logic}' and {len(conditions)} conditions...")
    if logic not in ['AND', 'OR']:
        logger.warning(f"Unsupported filter logic '{logic}'. Defaulting to OR.")
        logic = 'OR'

    for item_metadata in rag_results:
        item_id = item_metadata.get('id', 'Unknown')
        conditions_met = []
        for condition in conditions:
            if not isinstance(condition, dict) or not all(k in condition for k in ['field', 'operator', 'value']):
                logger.warning(f"Skipping invalid filter condition format: {condition}")
                conditions_met.append(False)
                continue

            field = condition['field']
            operator = condition['operator']
            filter_value = condition['value']
            item_value = item_metadata.get(field)

            condition_match = False
            if item_value is None:
                logger.debug(f"Filter field '{field}' not found or is None in metadata for item '{item_id}'. Condition considered False.")
                condition_match = False
            else:
                try:
                    if operator == '==':
                        condition_match = str(item_value).lower() == str(filter_value).lower()
                    elif operator == '!=':
                        condition_match = str(item_value).lower() != str(filter_value).lower()
                    elif operator == '<=' and field == 'price_numeric': # 예시: price_numeric 필드가 있다고 가정
                        condition_match = int(item_value) <= int(filter_value)
                    elif operator == '>=' and field == 'price_numeric': # 예시: price_numeric 필드가 있다고 가정
                        condition_match = int(item_value) >= int(filter_value)
                    elif operator == 'contains':
                        if isinstance(item_value, list):
                            if isinstance(filter_value, list):
                                condition_match = all(str(fv).lower() in [str(iv).lower() for iv in item_value] for fv in filter_value)
                            else:
                                condition_match = any(str(filter_value).lower() in str(iv).lower() for iv in item_value)
                        elif isinstance(item_value, str):
                            condition_match = str(filter_value).lower() in item_value.lower()
                        else: logger.warning(f"Operator 'contains' not supported for item_value type {type(item_value)} in field '{field}'.")
                    else:
                        logger.warning(f"Unsupported operator '{operator}' for field '{field}'.")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error comparing field '{field}' (value: {item_value}, type: {type(item_value)}) with filter value '{filter_value}': {e}")
                    condition_match = False

            conditions_met.append(condition_match)
            logger.debug(f" - Item '{item_id}', Condition: {condition}, ItemValue: {item_value}, Match: {condition_match}")

        final_match = False
        if logic == 'OR':
            final_match = any(conditions_met) if conditions_met else False
        elif logic == 'AND':
            final_match = all(conditions_met) if conditions_met else False

        if final_match:
            filtered_results.append(item_metadata)
            logger.debug(f" => Item '{item_id}' PASSED filtering (Logic: {logic})")

    logger.info(f"Post-filtering complete. {len(filtered_results)} results passed the filters.")
    return filtered_results


# --- 메인 오케스트레이션 함수 (gpt_interface.py 호출부 수정) ---
async def orchestrate_chatbot_turn(
    user_input: str,
    conversation_state: ConversationState,
    session: aiohttp.ClientSession,
    rag_searcher: Optional[RagSearcher]
) -> Dict[str, Any]:
    """
    'Summary + K Turns' 컨텍스트와 Tool Use 기반 챗봇 응답 생성 오케스트레이션 함수
    (다중/순차 호출 지원, 최대 반복 횟수 설정 가능, gpt_interface.py 호출 시 max_output_tokens 사용).
    """
    if not all([call_gpt_async, get_openai_embedding_async, get_config, api_logger]):
        logger.critical("CRITICAL: Required scheduler dependencies missing.")
        return {"error_message_for_user": "죄송합니다, 시스템 설정 오류로 답변을 드릴 수 없습니다."}

    try:
        config_data = get_config() # config -> config_data로 변경 (전역 config와 구분)
        if not config_data: raise ValueError("Config load failed.")
        tool_use_config = config_data.get('tasks', {}).get('tool_use', {})
        rag_config = config_data.get('rag', {})
        prompts_config = config_data.get('prompts', {})
        tools_definition = config_data.get('tools')

        tool_use_model = tool_use_config.get('model', 'o3')
        decision_temp = tool_use_config.get('decision_temperature', 0.2)
        # [수정] config.yaml에서 decision_max_completion_tokens 읽기
        decision_max_output_tokens_val = tool_use_config.get('decision_max_completion_tokens', 1500)
        generation_temp = tool_use_config.get('generation_temperature', 0.7)
        # [수정] config.yaml에서 generation_max_completion_tokens 읽기
        generation_max_output_tokens_val = tool_use_config.get('generation_max_completion_tokens', 3000)

        rag_k = rag_config.get('retrieval_k', 15)
        tool_use_prompt_template = prompts_config.get('tool_use_system_prompt')
        tool_arg_error_prompt = prompts_config.get('tool_argument_error_prompt')
        context_k = prompts_config.get('context_recent_k_turns', 3)
        max_tool_iterations = tool_use_config.get('max_iterations', 3)
        logger.debug(f"Max Tool Iterations set to: {max_tool_iterations}")

        if not all([tool_use_model, tool_use_prompt_template, tool_arg_error_prompt, tools_definition,
                    isinstance(rag_k, int), isinstance(context_k, int), isinstance(max_tool_iterations, int),
                    # [수정] 검사 대상 변수명 변경
                    isinstance(decision_max_output_tokens_val, int), isinstance(generation_max_output_tokens_val, int)]):
            logger.error(f"Essential configurations for Tool Use, RAG, or Context Management are missing or invalid. Check model, prompt, tools, k values, and max_output_tokens limits.")
            raise ValueError("Essential configurations missing or invalid.")
        if context_k < 0:
            logger.warning(f"Invalid context_recent_k_turns ({context_k}). Using 0.")
            context_k = 0
        if max_tool_iterations <= 0:
            logger.warning(f"Invalid max_iterations ({max_tool_iterations}). Using default 1.")
            max_tool_iterations = 1

    except Exception as conf_e:
        logger.critical(f"Critical configuration error in scheduler: {conf_e}", exc_info=True)
        return {"error_message_for_user": "죄송합니다, 시스템 설정 오류로 답변을 드릴 수 없습니다."}

    start_time_scheduler = time.time()
    logger.info("--- Starting Tool Use Orchestration Cycle (Summary + K Turns) ---")
    debug_info = {"orchestration_start_time": start_time_scheduler, "steps": []}

    messages = []
    current_summary = conversation_state.get_summary()
    summary_section_text = f"[이전 대화 요약]:\n{current_summary}\n\n[최근 대화 기록]:" if current_summary else "[최근 대화 기록]:"
    try:
        system_prompt_content = tool_use_prompt_template.format(summary_section=summary_section_text)
    except KeyError:
        logger.warning("'{summary_section}' placeholder not found in 'tool_use_system_prompt'. Appending summary separately.")
        system_prompt_content = tool_use_prompt_template + "\n\n" + summary_section_text
    except Exception as fmt_e:
        logger.error(f"Error formatting tool_use_system_prompt: {fmt_e}. Using base prompt.")
        system_prompt_content = tool_use_prompt_template

    messages.append({"role": "system", "content": system_prompt_content})

    full_history = conversation_state.get_history(copy=True)
    if context_k > 0:
        approx_messages_per_turn = 3
        num_messages_to_get = context_k * approx_messages_per_turn
        last_k_messages = full_history[-num_messages_to_get:]
        messages.extend(last_k_messages)
        logger.debug(f"Added last {len(last_k_messages)} messages (approximating K={context_k} turns) to context.")
        k_turn_preview = json.dumps([{"role": m.get("role"), "content": str(m.get("content"))[:50] + "..."} for m in last_k_messages], ensure_ascii=False, indent=2)
        logger.debug(f"Last K turns preview:\n{k_turn_preview}")
    else:
        logger.debug("context_recent_k_turns is 0, not adding recent history.")

    messages.append({"role": "user", "content": user_input})

    current_iteration = 0
    final_response_content = None

    while current_iteration < max_tool_iterations:
        current_iteration += 1
        step_debug = {"iteration": current_iteration, "start_time": time.time()}
        logger.info(f"--- Iteration {current_iteration}/{max_tool_iterations} ---")

        step_debug["llm_call_start"] = time.time()
        use_generation_params = False
        # [수정] 사용할 최대 토큰 수 변수명 변경 및 초기화
        current_max_output_tokens_to_use = 0

        if messages and messages[-1].get("role") == "tool":
            logger.debug("Previous message was from a tool. Using 'generation' parameters.")
            current_temp = generation_temp
            # [수정] generation_max_output_tokens_val 사용
            current_max_output_tokens_to_use = generation_max_output_tokens_val
            use_generation_params = True
        else:
            logger.debug("Previous message not from a tool (or first call). Using 'decision' parameters.")
            current_temp = decision_temp
            # [수정] decision_max_output_tokens_val 사용
            current_max_output_tokens_to_use = decision_max_output_tokens_val

        current_tool_choice = "auto" if not use_generation_params else None
        if use_generation_params:
            logger.debug("Setting tool_choice to None for final response generation.")

        # [수정] call_gpt_async 호출 시 max_output_tokens 인자 사용
        llm_response = await call_gpt_async(
            messages=messages,
            model=tool_use_model,
            temperature=current_temp,
            max_output_tokens=current_max_output_tokens_to_use, # [수정] 인자명 변경
            session=session,
            tools=tools_definition,
            tool_choice=current_tool_choice
        )
        step_debug["llm_call_end"] = time.time()
        step_debug["llm_call_duration_ms"] = int((step_debug["llm_call_end"] - step_debug["llm_call_start"]) * 1000)
        step_debug["llm_model_used"] = tool_use_model
        # [수정] 로깅 파라미터명 변경
        step_debug["llm_params_used"] = {"temperature": current_temp, "max_output_tokens": current_max_output_tokens_to_use, "tool_choice": current_tool_choice}


        if not llm_response or not llm_response.get("choices"):
            logger.error(f"LLM call #{current_iteration} failed or returned no choices.")
            step_debug["status"] = "failed_llm_call"
            step_debug["error"] = "LLM API call failed"
            debug_info["steps"].append(step_debug)
            return {"error_message_for_user": "죄송합니다, 답변 생성 중 오류가 발생했습니다 (LLM 호출 실패).", "debug_info": debug_info}

        assistant_message = llm_response["choices"][0].get("message", {})
        messages.append(assistant_message)
        step_debug["llm_response_raw"] = assistant_message

        tool_calls = assistant_message.get("tool_calls")
        response_content = assistant_message.get("content")

        if tool_calls:
            logger.info(f"LLM requested {len(tool_calls)} tool call(s).")
            step_debug["tool_calls_requested"] = tool_calls
            tool_results_for_next_call = []

            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                function_call = tool_call.get("function")
                if not tool_call_id or not function_call:
                    logger.warning(f"Skipping invalid tool call object: {tool_call}")
                    continue

                function_name = function_call.get("name")
                logger.info(f"Processing tool call ID: {tool_call_id}, Function: {function_name}")
                tool_step_debug = {"tool_call_id": tool_call_id, "function_name": function_name}

                if function_name == "product_search":
                    arguments = {}
                    try:
                        arguments_str = function_call.get("arguments", "{}")
                        arguments = json.loads(arguments_str)
                        search_keywords = arguments.get("search_keywords")
                        filters = arguments.get("filters") # LLM이 생성한 필터 조건
                        num_results_req = arguments.get("num_results", 3)
                        tool_step_debug["arguments"] = arguments
                        if not search_keywords: raise ValueError("Missing 'search_keywords'")
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.error(f"Failed to parse args for tool {tool_call_id}: {e}")
                        tool_step_debug["status"] = "failed_arg_parsing"
                        tool_step_debug["error"] = str(e)
                        tool_results_for_next_call.append({
                            "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                            "content": json.dumps({"error": f"Argument parsing error: {e}", "results_found": False})
                        })
                        step_debug.setdefault("tool_executions", []).append(tool_step_debug)
                        continue

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

                    tool_step_debug["rag_search_start"] = time.time()
                    initial_rag_results = await run_rag_search_async(query_embedding, rag_k, rag_searcher)
                    tool_step_debug["rag_search_end"] = time.time()
                    tool_step_debug["rag_search_duration_ms"] = int((tool_step_debug["rag_search_end"] - tool_step_debug["rag_search_start"]) * 1000)
                    tool_step_debug["rag_initial_count"] = len(initial_rag_results)

                    tool_step_debug["filtering_start"] = time.time()
                    filtered_rag_results = apply_metadata_filters(initial_rag_results, filters) # LLM이 생성한 필터 적용
                    tool_step_debug["filtering_end"] = time.time()
                    tool_step_debug["filtering_duration_ms"] = int((tool_step_debug["filtering_end"] - tool_step_debug["filtering_start"]) * 1000)
                    tool_step_debug["rag_filtered_count"] = len(filtered_rag_results)
                    tool_step_debug["filters_applied"] = filters

                    print("-" * 20 + " RAG Filtered Results (Names Only, for LLM Context) " + "-" * 20)
                    print(f"[RAG Filtered Results] Tool Call ID: {tool_call_id}")
                    print(f"  - Search Keywords: '{search_keywords}'")
                    print(f"  - Filters Applied (if any): {json.dumps(filters, ensure_ascii=False) if filters else 'None'}")
                    print(f"  - Initial Results Count from FAISS: {len(initial_rag_results)}")
                    print(f"  - Filtered Results Count: {len(filtered_rag_results)}")

                    tool_result_content_parts = []
                    if filtered_rag_results:
                        results_to_use_and_log = filtered_rag_results[:num_results_req]
                        print(f"  - Product Names to LLM (Top {len(results_to_use_and_log)} of {num_results_req} requested):")

                        for i, res_meta in enumerate(results_to_use_and_log):
                            try:
                                prod_name = res_meta.get('product_name', 'N/A')
                                raw_text = res_meta.get('raw_block_text', '') # `doc_meta.jsonl`에 저장된 원본 텍스트
                                print(f"    [{i+1}] Product Name: {prod_name}")
                                tool_result_content_parts.append({
                                    "product_name": prod_name,
                                    "details": raw_text # 원본 텍스트를 전달
                                })
                            except Exception as log_err:
                                print(f"    Error processing/printing filtered product name {i+1}: {log_err}")

                        tool_result_content_obj = {"results": tool_result_content_parts, "results_found": True}
                    else:
                        print("  - No results passed the filters or found for LLM.")
                        tool_result_content_obj = {"results_found": False}

                    print("-" * 70)

                    tool_result_content = json.dumps(tool_result_content_obj, ensure_ascii=False)
                    tool_results_for_next_call.append({
                        "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                        "content": tool_result_content
                    })
                    tool_step_debug["status"] = "success"
                    tool_step_debug["result_summary"] = tool_result_content_obj
                else:
                    logger.warning(f"Received unhandled tool function name: {function_name}")
                    tool_step_debug["status"] = "unhandled_function"
                    tool_results_for_next_call.append({
                        "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                        "content": json.dumps({"error": f"Unknown function: {function_name}"})
                    })
                step_debug.setdefault("tool_executions", []).append(tool_step_debug)
            messages.extend(tool_results_for_next_call)

        elif response_content:
            logger.info(f"LLM generated final response directly in iteration {current_iteration}.")
            final_response_content = response_content
            step_debug["status"] = "completed_direct_response"
            debug_info["steps"].append(step_debug)
            break
        else:
            logger.error(f"LLM response in iteration {current_iteration} had neither content nor tool_calls.")
            step_debug["status"] = "failed_llm_empty_response"
            debug_info["steps"].append(step_debug)
            return {"error_message_for_user": "죄송합니다, 응답 생성 중 예상치 못한 오류가 발생했습니다.", "debug_info": debug_info}

        debug_info["steps"].append(step_debug)

    if final_response_content is None:
        logger.warning(f"Failed to get final response content after {current_iteration} iterations (max: {max_tool_iterations}).")
        if debug_info["steps"]:
            if not debug_info["steps"][-1].get("status", "").startswith("failed"):
                debug_info["steps"][-1]["status"] = "failed_max_iterations"
        else:
            debug_info["steps"].append({"status": "failed_unknown_before_loop"})
        return {"error_message_for_user": "죄송합니다, 요청을 처리하는 데 시간이 너무 오래 걸리거나 오류가 발생했습니다.", "debug_info": debug_info}

    end_time_scheduler = time.time()
    total_duration = end_time_scheduler - start_time_scheduler
    debug_info['total_orchestration_time_ms'] = int(total_duration * 1000)
    debug_info['final_status'] = 'success'
    debug_info['total_iterations'] = current_iteration
    logger.info(f"--- Scheduler Orchestration Cycle Finished in {total_duration:.3f} seconds ({current_iteration} iterations) ---")

    return {"response": final_response_content, "debug_info": debug_info}

# --- 예시 사용법 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running scheduler.py as main script (placeholder, max_output_tokens handling updated) ---")
    print("Scheduler module contains the core Tool Use orchestration logic.")
    print("Direct execution requires setting up dependencies (ConversationState, RagSearcher, etc.).")
    print("Please test via app.py or test_runner.py.")