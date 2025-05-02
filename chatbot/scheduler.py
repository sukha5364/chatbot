# chatbot/scheduler.py (요구사항 반영 최종본: 주석 보강 및 품질 개선)

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union
import aiohttp
import numpy as np

# --- 필요한 모듈 임포트 (파일 상단에 이미 정의되어 있어야 함) ---
try:
    from .slot_extractor import extract_slots_with_gpt
    from .model_router import determine_routing_and_reasoning
    from .searcher import RagSearcher # 클래스 임포트
    from .conversation_state import ConversationState
    from .gpt_interface import get_openai_embedding_async
    from .config_loader import get_config
    # logging.info("Required modules imported successfully in scheduler.") # 이미 로깅됨
except ImportError as ie:
    logging.error(f"ERROR (scheduler): Failed to import modules: {ie}. Check relative paths.", exc_info=True)
    # 필수 모듈 실패 시 스케줄러 기능 사용 불가 (이 함수 호출 전에 체크될 수도 있음)
    extract_slots_with_gpt = None
    determine_routing_and_reasoning = None
    RagSearcher = None
    ConversationState = None
    get_openai_embedding_async = None
    get_config = None

# --- 로거 설정 (파일 상단에 이미 정의되어 있어야 함) ---
logger = logging.getLogger(__name__)

# --- RAG 검색 비동기 실행 함수 (이 함수 위에 정의되어 있어야 함) ---
async def run_rag_search_async(
    query_embedding: Union[List[float], np.ndarray, None],
    k: int,
    rag_searcher: Optional[RagSearcher]
) -> List[Dict]:
    """
    주어진 쿼리 임베딩과 RagSearcher 객체를 사용하여 RAG 검색을 비동기적으로 실행합니다.
    FAISS 검색은 동기 함수이므로 asyncio.get_running_loop().run_in_executor를 사용합니다.

    Args:
        query_embedding (Union[List[float], np.ndarray, None]): 검색할 쿼리의 임베딩 벡터. 생성 실패 시 None일 수 있음.
        k (int): 검색할 상위 결과의 개수 (config에서 읽어옴).
        rag_searcher (Optional[RagSearcher]): 초기화된 RagSearcher 인스턴스. None이면 검색 건너뜀.

    Returns:
        List[Dict]: 검색된 문서 Chunk 메타데이터 리스트. 오류 발생 또는 결과 없음 시 빈 리스트.
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


# --- 병렬/순차 작업 실행 메인 함수 (수정된 버전) ---
async def run_parallel_tasks(
    user_input: str,
    conversation_state: ConversationState,
    session: aiohttp.ClientSession,
    rag_searcher: Optional[RagSearcher] # RAG 검색기 인스턴스를 인자로 받음
) -> Dict[str, Any]:
    """
    챗봇 응답 생성에 필요한 여러 작업들을 효율적으로 실행합니다.
    - Slot 추출, 모델 라우팅(복잡도 분석 및 CoT 생성), 쿼리 임베딩 생성을 병렬로 실행합니다.
    - 쿼리 임베딩이 완료되면 RAG 검색을 순차적으로 실행합니다.
    모든 작업의 결과를 종합하여 딕셔너리로 반환합니다.

    Args:
        user_input (str): 현재 사용자 입력 텍스트.
        conversation_state (ConversationState): 현재 대화 상태 객체 (히스토리, 슬롯, 요약 포함).
        session (aiohttp.ClientSession): API 호출에 사용할 공유 aiohttp 세션.
        rag_searcher (Optional[RagSearcher]): 초기화된 RagSearcher 인스턴스.

    Returns:
        Dict[str, Any]: 각 작업의 결과 또는 오류 정보를 포함하는 딕셔너리.
                       예: {'slots': {...}, 'routing_info': {...}, 'rag_results': [...]}.
                       오류 발생 시 해당 키의 값은 None 또는 기본값일 수 있음.
    """
    # 필수 모듈 확인 (함수 시작 시)
    if not all([extract_slots_with_gpt, determine_routing_and_reasoning, get_openai_embedding_async, get_config]):
        logger.error("CRITICAL: Required functions or config loader not available in scheduler. Cannot run tasks.")
        # 필수 기능 누락 시 비정상 상태 반환
        return {"error": "Scheduler dependencies missing", "slots": None, "routing_info": None, "rag_results": []}

    # --- <<< 여기에 config 로드 코드 추가 >>> ---
    try:
        config = get_config() # 함수 시작 시 config 로드
        if not config:
            raise ValueError("Configuration could not be loaded in scheduler.")
    except Exception as conf_e:
        logger.error(f"CRITICAL: Failed to load configuration within scheduler: {conf_e}", exc_info=True)
        # 설정 로드 실패 시 오류 반환
        return {"error": "Configuration load failed in scheduler", "slots": None, "routing_info": None, "rag_results": []}
    # --- <<< config 로드 코드 끝 >>> ---

    start_time_scheduler = time.time()
    logger.info("--- Running Parallel/Sequential Tasks in Scheduler ---")
    logger.debug(f"Input user query: '{user_input[:70]}...'")

    # 결과 저장용 딕셔너리
    final_results: Dict[str, Any] = {
        "slots": None,
        "routing_info": None,
        "rag_results": []
    }
    tasks: Dict[str, asyncio.Task] = {}
    task_start_times: Dict[str, float] = {}

    # --- 1. 병렬 실행 가능 작업 정의 ---
    parallel_task_definitions = {
        "slots": extract_slots_with_gpt(user_input, session),
        "routing_info": determine_routing_and_reasoning(user_input, session),
        "query_embedding": get_openai_embedding_async(user_input, session),
    }

    logger.info(f"Creating parallel tasks: {list(parallel_task_definitions.keys())}")
    for name, coro in parallel_task_definitions.items():
        task_start_times[name] = time.time()
        tasks[name] = asyncio.create_task(coro)

    # --- 2. 병렬 작업 완료 대기 및 결과 처리 ---
    parallel_task_keys = list(tasks.keys())
    logger.info(f"Waiting for {len(parallel_task_keys)} parallel tasks to complete...")
    parallel_results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
    logger.info("All parallel tasks finished.")

    query_embedding_vector: Union[List[float], np.ndarray, None] = None
    routing_info_result: Optional[Dict] = None

    for i, task_result in enumerate(parallel_results_list):
        task_name = parallel_task_keys[i]
        start_time = task_start_times.get(task_name)
        duration = time.time() - start_time if start_time else 0

        if isinstance(task_result, Exception):
            logger.error(f"Task '{task_name}' failed after {duration:.3f}s with Exception: {task_result}", exc_info=(logger.getEffectiveLevel() <= logging.DEBUG))
            if task_name == "routing_info":
                routing_info_result = None
        else:
            logger.info(f"Task '{task_name}' completed successfully in {duration:.3f}s.")
            if task_name == "slots":
                final_results["slots"] = task_result
            elif task_name == "routing_info":
                routing_info_result = task_result
            elif task_name == "query_embedding":
                query_embedding_vector = task_result

    # --- 라우팅 결과 최종 처리 (실패 시 fallback 적용) ---
    if routing_info_result is None:
        logger.warning("Routing info task failed or returned None. Applying default routing.")
        # 이제 config는 이미 로드되어 있으므로 바로 사용 가능
        try:
            # config = get_config() # <- 이 줄 제거!
            default_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
            final_results['routing_info'] = {"level": "easy", "model": default_model, "cot_data": None}
        except Exception as conf_e: # config 접근 시 오류 발생 가능성 대비
            logger.error(f"Failed to get default routing model from already loaded config: {conf_e}. Using hardcoded default.")
            final_results['routing_info'] = {"level": "easy", "model": 'gpt-3.5-turbo', "cot_data": None}
    else:
        final_results['routing_info'] = routing_info_result

    # --- 3. RAG 검색 실행 (순차적 실행: 쿼리 임베딩 필요) ---
    if query_embedding_vector is not None and rag_searcher is not None:
        logger.info("Query embedding generated, proceeding with RAG search task...")
        try:
            # config에서 검색할 k값 읽기 (이제 config 변수 사용 가능)
            rag_k = config.get('rag', {}).get('retrieval_k', 3) # 기본값 3
            if not isinstance(rag_k, int) or rag_k <= 0:
                logger.warning(f"Invalid 'retrieval_k' value ({rag_k}) in config. Using default k=3.")
                rag_k = 3

            rag_search_start_time = time.time()
            final_results['rag_results'] = await run_rag_search_async(
                query_embedding=query_embedding_vector,
                k=rag_k,
                rag_searcher=rag_searcher
            )
            rag_search_duration = time.time() - rag_search_start_time
            logger.info(f"RAG search task finished in {rag_search_duration:.3f}s. Found {len(final_results['rag_results'])} results.")

        except Exception as e: # run_rag_search_async 호출 자체의 오류 (거의 발생 안 함)
            logger.error(f"Unexpected error initiating or awaiting RAG search task: {e}", exc_info=True)
            final_results['rag_results'] = [] # 오류 시 빈 결과

    else:
        if query_embedding_vector is None:
            logger.warning("Skipping RAG search because query embedding generation failed or returned None.")
        if rag_searcher is None:
            logger.warning("Skipping RAG search because RagSearcher instance is not available.")

    # --- 4. 최종 결과 반환 ---
    end_time_scheduler = time.time()
    total_duration = end_time_scheduler - start_time_scheduler
    logger.info(f"--- Scheduler Finished All Tasks in {total_duration:.3f} seconds ---")

    if logger.getEffectiveLevel() <= logging.DEBUG:
        summary_log = {
            "slots_found": bool(final_results.get("slots")),
            "routing_level": final_results.get("routing_info", {}).get("level"),
            "routing_model": final_results.get("routing_info", {}).get("model"),
            "cot_generated": bool(final_results.get("routing_info", {}).get("cot_data")),
            "rag_results_count": len(final_results.get("rag_results", [])),
        }
        logger.debug(f"Scheduler final results summary: {summary_log}")

    return final_results


# --- 예시 사용법 (기존 유지, 로깅 확인용) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # __name__으로 로거 다시 가져오기
    logger.info("--- Running scheduler.py as main script for testing ---")

    async def test_scheduler():
        """스케줄러 로직 테스트 실행"""
        if not get_config or not ConversationState or not RagSearcher:
             logger.error("Required modules not available. Cannot run scheduler test.")
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

        # 테스트를 위해 임시 RagSearcher 인스턴스 생성 (실제 앱에서는 app.state 사용)
        # 이 테스트는 data/ 디렉토리에 index.faiss 와 doc_meta.jsonl 파일이 필요
        test_rag_searcher: Optional[RagSearcher] = None
        try:
            logger.info("Attempting to create temporary RagSearcher instance for test...")
            test_rag_searcher = RagSearcher() # 초기화 시 리소스 로드 시도
            if not test_rag_searcher.index or not test_rag_searcher.metadata:
                 logger.warning("Temporary RagSearcher created but resources (index/metadata) might be missing. RAG search may not work.")
            else:
                 logger.info("Temporary RagSearcher created and resources seem loaded.")
        except Exception as e:
            logger.warning(f"Could not create temporary RagSearcher for test: {e}. RAG search will be skipped.", exc_info=True)

        # 테스트 입력 및 상태
        test_input = "나이키 에어맥스 270mm 신고 있는데, 발볼이 좀 넓은 편이에요. 데카트론에서 비슷한 러닝화 추천해주세요."
        state = ConversationState() # 빈 상태로 시작

        logger.info("Starting scheduler test run...")
        start_run_time = time.time()
        async with aiohttp.ClientSession() as session:
            # 테스트 시 생성된 rag_searcher 전달
            scheduler_output = await run_parallel_tasks(test_input, state, session, test_rag_searcher)
        end_run_time = time.time()
        logger.info(f"Scheduler test run finished in {end_run_time - start_run_time:.3f} seconds.")

        # 결과 출력 (JSON 직렬화 가능한 형태로)
        print("\n--- Scheduler Test Output (JSON Serializable) ---")
        def default_serializer(obj):
            if isinstance(obj, np.ndarray): return obj.tolist() # Numpy 배열은 리스트로
            if isinstance(obj, (datetime, time)): return obj.isoformat() # 시간 객체는 ISO 포맷
            try: return str(obj) # 그 외는 문자열로 시도
            except Exception: return f"<Object of type {type(obj).__name__} not serializable>"

        try:
            # 순환 참조 방지 및 직렬화 가능하도록 처리
            output_to_print = {}
            for key, value in scheduler_output.items():
                if key == 'rag_results': # RAG 결과는 일부만 표시
                     output_to_print[key] = [ {k:v for k,v in item.items() if k != 'text'} for item in value[:2] ] # 상위 2개 결과의 text 제외 메타데이터
                     output_to_print['rag_results_count'] = len(value)
                elif key == 'routing_info' and value and 'cot_data' in value and value['cot_data']: # CoT 데이터 미리보기
                     output_to_print[key] = value.copy()
                     output_to_print[key]['cot_data_preview'] = value['cot_data'][:100] + '...'
                     del output_to_print[key]['cot_data']
                else:
                     output_to_print[key] = value

            print(json.dumps(output_to_print, indent=2, ensure_ascii=False, default=default_serializer))
        except Exception as json_e:
             logger.error(f"Error serializing scheduler output to JSON: {json_e}")
             print(f"Error displaying results as JSON: {json_e}")
             # print(scheduler_output) # 원본 객체 출력 (디버깅용)

    # 비동기 테스트 실행
    try:
        asyncio.run(test_scheduler())
    except Exception as e:
        logger.critical(f"\nAn critical error occurred during scheduler testing: {e}", exc_info=True)

    logger.info("--- scheduler.py test finished ---")