# chatbot/scheduler.py (요구사항 반영 최종본: 요약/범용CoT 병렬 처리)

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union
import aiohttp
import numpy as np

# --- 필요한 모듈 임포트 (파일 상단에 이미 정의되어 있어야 함) ---
try:
    from .slot_extractor import extract_slots_with_gpt
    # [수정] CoT 생성 함수 및 라우팅 함수 임포트 변경
    from .model_router import determine_routing_and_reasoning, generate_general_cot_async, classify_complexity_level
    from .searcher import RagSearcher # 클래스 임포트
    from .conversation_state import ConversationState
    from .gpt_interface import get_openai_embedding_async
    # [추가] 요약 함수 임포트
    from .summarizer import summarize_conversation_async
    from .config_loader import get_config
    logging.info("Required modules imported successfully in scheduler.")
except ImportError as ie:
    logging.error(f"ERROR (scheduler): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 스케줄러 기능 사용 불가
    extract_slots_with_gpt = None
    determine_routing_and_reasoning = None
    generate_general_cot_async = None
    classify_complexity_level = None # 추가됨
    RagSearcher = None
    ConversationState = None
    get_openai_embedding_async = None
    summarize_conversation_async = None # 추가됨
    get_config = None

# --- 로거 설정 (파일 상단에 이미 정의되어 있어야 함) ---
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
    (이 함수의 코드는 이전과 동일하게 유지됩니다)
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

    # 2. 임베딩 벡터 형식 변환 (Numpy float32) 및 검증
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

    # 4. FAISS 검색 실행 (run_in_executor 사용)
    try:
        start_t = time.time()
        # RagSearcher.search는 (distances, indices) 튜플 대신 메타데이터 리스트를 반환해야 함 (searcher.py 확인 필요)
        # -> searcher.py의 search 함수는 메타데이터 리스트를 반환하는 것으로 확인됨.
        results = await loop.run_in_executor(
            None, # 기본 스레드 풀 사용
            rag_searcher.search, # 호출할 동기 함수
            query_embedding_np, # 인자 1
            k                   # 인자 2
        )
        duration_t = time.time() - start_t
        logger.debug(f"Async RAG search execution finished in {duration_t:.4f}s. Found {len(results)} results.")
        return results
    except RuntimeError as re:
        logger.error(f"RuntimeError during async RAG search execution: {re}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error during async RAG search execution via executor: {e}", exc_info=True)
        return []


# --- [수정됨] 병렬/순차 작업 실행 메인 함수 ---
async def run_parallel_tasks(
    user_input: str,
    conversation_state: ConversationState,
    session: aiohttp.ClientSession,
    rag_searcher: Optional[RagSearcher] # RAG 검색기 인스턴스를 인자로 받음
) -> Dict[str, Any]:
    """
    챗봇 응답 생성에 필요한 여러 작업들을 효율적으로 실행합니다.
    - Slot 추출, 복잡도 분류/모델선택, 쿼리 임베딩 생성, 범용 CoT 생성, 요약 생성을 병렬로 실행.
    - 쿼리 임베딩 완료 후 RAG 검색을 순차적으로 실행.
    모든 작업의 결과를 종합하여 딕셔너리로 반환합니다.

    Args:
        user_input (str): 현재 사용자 입력 텍스트.
        conversation_state (ConversationState): 현재 대화 상태 객체.
        session (aiohttp.ClientSession): API 호출에 사용할 공유 aiohttp 세션.
        rag_searcher (Optional[RagSearcher]): 초기화된 RagSearcher 인스턴스.

    Returns:
        Dict[str, Any]: 각 작업의 결과 또는 오류 정보를 포함하는 딕셔너리.
                       {'slots': {...}, 'routing_info': {'level':..., 'model':..., 'cot_data':...}, 'rag_results': [...], 'summary': ...}.
                       오류 발생 시 해당 키의 값은 None 또는 기본값일 수 있음.
    """
    # 필수 모듈 확인
    required_funcs = [
        extract_slots_with_gpt, determine_routing_and_reasoning,
        get_openai_embedding_async, generate_general_cot_async,
        summarize_conversation_async, get_config
    ]
    if not all(required_funcs):
        logger.error("CRITICAL: Required functions or config loader not available in scheduler. Cannot run tasks.")
        return {"error": "Scheduler dependencies missing", "slots": None, "routing_info": None, "rag_results": [], "summary": None}

    # --- config 로드 ---
    try:
        config = get_config()
        if not config:
            raise ValueError("Configuration could not be loaded in scheduler.")
    except Exception as conf_e:
        logger.error(f"CRITICAL: Failed to load configuration within scheduler: {conf_e}", exc_info=True)
        return {"error": "Configuration load failed in scheduler", "slots": None, "routing_info": None, "rag_results": [], "summary": None}

    start_time_scheduler = time.time()
    logger.info("--- Running Parallel Tasks in Scheduler (v2: Summary/General CoT included) ---")
    logger.debug(f"Input user query: '{user_input[:70]}...'")

    # 결과 저장용 딕셔너리 초기화
    final_results: Dict[str, Any] = {
        "slots": None,
        "routing_info": None, # level, model, cot_data 포함 예정
        "rag_results": [],
        "summary": None       # 요약 결과 포함 예정
    }
    tasks: Dict[str, asyncio.Task] = {}
    task_start_times: Dict[str, float] = {}

    # --- 1. 병렬 실행 태스크 정의 및 생성 ---
    # 이전 요약/슬롯 정보 가져오기 (CoT, 요약 태스크에 필요)
    previous_summary = conversation_state.get_summary()
    previous_slots = conversation_state.get_slots()
    # 현재 히스토리 (현재 사용자 입력 전까지)
    current_history = conversation_state.get_history()

    # 병렬 실행할 코루틴 정의
    parallel_coroutines = {
        "slots": extract_slots_with_gpt(user_input, session),
        "routing": determine_routing_and_reasoning(user_input, session), # CoT 생성 제외됨
        "embedding": get_openai_embedding_async(user_input, session),
        # [수정] 범용 CoT 생성 (컨텍스트 전달)
        "general_cot": generate_general_cot_async(user_input, previous_summary, previous_slots, session),
        # [추가] 요약 생성 (현재 히스토리와 이전 요약 전달)
        "summarization": summarize_conversation_async(current_history, previous_summary, session),
    }

    logger.info(f"Creating {len(parallel_coroutines)} parallel tasks: {list(parallel_coroutines.keys())}")
    for name, coro in parallel_coroutines.items():
        if coro is None: # 함수 임포트 실패 시
             logger.error(f"Coroutine for task '{name}' is None. Skipping task creation.")
             continue
        task_start_times[name] = time.time()
        tasks[name] = asyncio.create_task(coro)

    # --- 2. 병렬 작업 완료 대기 및 결과 처리 ---
    active_task_names = list(tasks.keys()) # 실제 생성된 태스크 이름만 사용
    if not active_task_names:
        logger.error("No parallel tasks were created. Aborting scheduler run.")
        return {"error": "No parallel tasks created", **final_results} # 초기화된 결과 반환

    logger.info(f"Waiting for {len(active_task_names)} parallel tasks to complete...")
    # return_exceptions=True 로 설정하여 개별 태스크 예외 발생해도 gather는 완료됨
    task_results_list = await asyncio.gather(*[tasks[name] for name in active_task_names], return_exceptions=True)
    logger.info("All parallel tasks finished (or raised exceptions).")

    # 각 태스크 결과 처리
    extracted_slots: Optional[Dict] = None
    routing_result: Optional[Dict] = None # level, model 포함
    query_embedding_vector: Union[List[float], np.ndarray, None] = None
    general_cot_result: Optional[str] = None
    new_summary_result: Optional[str] = None

    for i, task_result in enumerate(task_results_list):
        task_name = active_task_names[i]
        start_time = task_start_times.get(task_name)
        duration = time.time() - start_time if start_time else 0

        if isinstance(task_result, Exception):
            logger.error(f"Task '{task_name}' failed after {duration:.3f}s with Exception: {task_result}", exc_info=(logger.getEffectiveLevel() <= logging.DEBUG))
            # 실패 시 각 변수는 초기값(None) 유지
        else:
            logger.info(f"Task '{task_name}' completed successfully in {duration:.3f}s.")
            # 성공 시 결과 할당
            if task_name == "slots":
                extracted_slots = task_result
            elif task_name == "routing":
                routing_result = task_result
            elif task_name == "embedding":
                query_embedding_vector = task_result
            elif task_name == "general_cot":
                general_cot_result = task_result
            elif task_name == "summarization":
                new_summary_result = task_result

    # --- 결과 조합 및 후처리 ---

    # 최종 슬롯 결과 할당
    final_results["slots"] = extracted_slots if extracted_slots else {} # 실패 시 빈 dict

    # 최종 요약 결과 할당
    final_results["summary"] = new_summary_result # 실패 시 None

    # 라우팅 정보 및 CoT 데이터 구성
    if routing_result and isinstance(routing_result, dict):
        level = routing_result.get('level', 'easy') # 기본값 easy
        chosen_model = routing_result.get('model')
        cot_data = None
        if level in ['medium', 'hard']:
            if general_cot_result:
                cot_data = general_cot_result
                logger.info(f"Using generated CoT data for complexity level '{level}'.")
            else:
                logger.warning(f"Complexity level is '{level}' but CoT generation failed or returned None.")
        else: # easy 레벨
            if general_cot_result:
                 logger.debug("Complexity level is 'easy', ignoring generated CoT data.")
            # else: CoT 생성 실패했어도 easy 레벨은 상관 없음

        final_results["routing_info"] = {
            "level": level,
            "model": chosen_model,
            "cot_data": cot_data # level에 따라 None일 수 있음
        }
    else:
        # 라우팅 분류 실패 시 fallback
        logger.warning("Routing task failed or returned invalid result. Applying default routing (easy, default model, no CoT).")
        try:
            default_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
        except Exception: default_model = 'gpt-3.5-turbo'
        final_results["routing_info"] = {"level": "easy", "model": default_model, "cot_data": None}

    # --- RAG 검색 실행 (순차적 실행: 쿼리 임베딩 필요) ---
    if query_embedding_vector is not None and rag_searcher is not None:
        logger.info("Query embedding generated, proceeding with RAG search task...")
        try:
            rag_k = config.get('rag', {}).get('retrieval_k', 3)
            if not isinstance(rag_k, int) or rag_k <= 0:
                logger.warning(f"Invalid 'retrieval_k' value ({rag_k}). Using default k=3.")
                rag_k = 3

            rag_search_start_time = time.time()
            final_results['rag_results'] = await run_rag_search_async(
                query_embedding=query_embedding_vector,
                k=rag_k,
                rag_searcher=rag_searcher
            )
            rag_search_duration = time.time() - rag_search_start_time
            logger.info(f"RAG search task finished in {rag_search_duration:.3f}s. Found {len(final_results['rag_results'])} results.")

        except Exception as e:
            logger.error(f"Unexpected error initiating or awaiting RAG search task: {e}", exc_info=True)
            final_results['rag_results'] = []
    else:
        if query_embedding_vector is None:
            logger.warning("Skipping RAG search because query embedding generation failed or returned None.")
        if rag_searcher is None:
            logger.warning("Skipping RAG search because RagSearcher instance is not available.")

    # --- 최종 결과 반환 ---
    end_time_scheduler = time.time()
    total_duration = end_time_scheduler - start_time_scheduler
    logger.info(f"--- Scheduler Finished All Tasks in {total_duration:.3f} seconds ---")

    if logger.getEffectiveLevel() <= logging.DEBUG:
        summary_log = {
            "slots_found": bool(final_results.get("slots")),
            "routing_level": final_results.get("routing_info", {}).get("level"),
            "routing_model": final_results.get("routing_info", {}).get("model"),
            "cot_generated_and_used": bool(final_results.get("routing_info", {}).get("cot_data")),
            "summary_generated": bool(final_results.get("summary")),
            "rag_results_count": len(final_results.get("rag_results", [])),
        }
        logger.debug(f"Scheduler final results summary: {summary_log}")

    return final_results


# --- [수정됨] 예시 사용법 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- Running scheduler.py as main script for testing ---")

    async def test_scheduler():
        """스케줄러 로직 테스트 실행"""
        required_modules = [ConversationState, RagSearcher, get_config] + required_funcs
        if not all(required_modules):
            logger.error("Required modules/functions not available. Cannot run scheduler test.")
            return
        try:
            get_config() # 설정 로드 확인
            logger.info("Config loaded for scheduler test.")
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Cannot run test.")
            return
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY missing. Cannot run API dependent tests.")
            return

        # 테스트용 RagSearcher (실제 앱에서는 app.state 사용)
        test_rag_searcher: Optional[RagSearcher] = None
        try:
            logger.info("Attempting to create temporary RagSearcher instance for test...")
            test_rag_searcher = RagSearcher()
            if not test_rag_searcher.index or not test_rag_searcher.metadata:
                 logger.warning("Temporary RagSearcher created but resources might be missing.")
            else: logger.info("Temporary RagSearcher created and resources seem loaded.")
        except Exception as e:
            logger.warning(f"Could not create temporary RagSearcher for test: {e}. RAG search will be skipped.")

        # 테스트 입력 및 상태
        test_input = "나이키 에어맥스 270mm 신고 있는데, 발볼이 좀 넓은 편이에요. 데카트론에서 비슷한 러닝화 추천해주세요."
        state = ConversationState()
        state.add_to_history("user", "안녕하세요")
        state.add_to_history("assistant", "안녕하세요! 데카트론 AI 챗봇입니다.")
        state.update_summary("사용자와 챗봇이 인사를 나눔.") # 이전 요약 가정
        state.update_slots({"user_preference": ["발볼 넓음"]}) # 이전 슬롯 가정

        logger.info("Starting scheduler test run...")
        start_run_time = time.time()
        async with aiohttp.ClientSession() as session:
            scheduler_output = await run_parallel_tasks(test_input, state, session, test_rag_searcher)
        end_run_time = time.time()
        logger.info(f"Scheduler test run finished in {end_run_time - start_run_time:.3f} seconds.")

        # 결과 출력 (JSON 직렬화 가능하도록)
        print("\n--- Scheduler Test Output (JSON Serializable) ---")
        def default_serializer(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (datetime, time.struct_time)): return str(obj) # 시간 관련 객체 문자열화
            try: return str(obj)
            except Exception: return f"<Object of type {type(obj).__name__} not serializable>"

        try:
            output_to_print = {}
            for key, value in scheduler_output.items():
                if key == 'rag_results':
                     output_to_print[key] = [ {k:v for k,v in item.items() if k != 'text'} for item in value[:2] ] # 상위 2개 결과 (text 제외)
                     output_to_print['rag_results_count'] = len(value)
                elif key == 'routing_info' and value and 'cot_data' in value and value['cot_data']:
                     output_to_print[key] = value.copy()
                     output_to_print[key]['cot_data_preview'] = value['cot_data'][:100] + '...'
                     del output_to_print[key]['cot_data']
                elif key == 'summary' and value:
                     output_to_print[key] = value[:200] + "..." # 요약 미리보기
                else:
                     output_to_print[key] = value

            print(json.dumps(output_to_print, indent=2, ensure_ascii=False, default=default_serializer))
        except Exception as json_e:
            logger.error(f"Error serializing scheduler output to JSON: {json_e}")
            print(f"Error displaying results as JSON: {json_e}")

    # 비동기 테스트 실행
    try:
        import os # os 임포트 추가 (getenv 사용 위함)
        asyncio.run(test_scheduler())
    except Exception as e:
        logger.critical(f"\nAn critical error occurred during scheduler testing: {e}", exc_info=True)

    logger.info("--- scheduler.py test finished ---")