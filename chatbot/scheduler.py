# chatbot/scheduler.py

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union
import aiohttp
import numpy as np

# 필요한 모듈 임포트
from .slot_extractor import extract_slots_with_gpt
from .model_router import determine_routing_and_reasoning
# [수정] searcher 모듈에서 클래스만 임포트 (인스턴스는 app.py에서 받음)
from .searcher import RagSearcher
from .conversation_state import ConversationState
from .gpt_interface import get_openai_embedding_async
from .config_loader import get_config

# 로거 설정 및 설정 로드
logger = logging.getLogger(__name__)
config = get_config()
log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# --- RAG 검색 함수 ([수정] rag_searcher 인자 추가) ---
async def run_rag_search_async(
    query_embedding: Union[List[float], np.ndarray],
    k: int,
    rag_searcher: Optional[RagSearcher] # [수정] 검색기 인스턴스를 인자로 받음
) -> List[Dict]:
    """
    사전 계산된 쿼리 임베딩과 RagSearcher 객체를 사용하여 RAG 검색을 비동기 실행합니다.
    """
    loop = asyncio.get_running_loop()

    # [수정] rag_searcher 인스턴스 유효성 검사 추가
    if rag_searcher is None or rag_searcher.index is None or not rag_searcher.metadata:
        logger.warning("RagSearcher instance is not available or not initialized. Skipping RAG search.")
        return []
    if query_embedding is None:
        logger.warning("Query embedding is None, skipping RAG search.")
        return []

    # FAISS는 NumPy 배열 필요
    if isinstance(query_embedding, list):
        query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)
    elif isinstance(query_embedding, np.ndarray):
        query_embedding_np = query_embedding.astype('float32').reshape(1, -1)
    else:
        logger.error(f"Invalid query embedding type: {type(query_embedding)}")
        return []

    # FAISS 인덱스 차원과 맞는지 확인
    expected_dim = config.get('rag', {}).get('embedding_dimension')
    # [수정] rag_searcher.index.d 사용
    if expected_dim and rag_searcher.index.d != query_embedding_np.shape[1]:
        logger.error(f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match index dimension ({rag_searcher.index.d}). Skipping RAG search.")
        return []

    logger.debug(f"Running RAG search with k={k} and embedding shape {query_embedding_np.shape}")
    try:
        # [수정] 인자로 받은 rag_searcher 객체의 search 메서드 사용
        results = await loop.run_in_executor(
            None,
            rag_searcher.search, # 인스턴스의 메서드 호출
            query_embedding_np,
            k
        )
        logger.debug(f"RAG search returned {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during async RAG search execution: {e}", exc_info=True)
        return []

# --- 병렬/순차 작업 실행 함수 ([수정] rag_searcher 인자 추가) ---
async def run_parallel_tasks(
    user_input: str,
    conversation_state: ConversationState,
    session: aiohttp.ClientSession,
    rag_searcher: Optional[RagSearcher] # [수정] RAG 검색기 인스턴스를 인자로 받음
) -> Dict[str, Any]:
    """
    Slot 추출, 모델 라우팅, 쿼리 임베딩을 병렬 실행 후, RAG 검색을 실행합니다.
    """
    start_time_scheduler = time.time()
    logger.info("--- Running Parallel/Sequential Tasks in Scheduler ---")

    task_start_times = {}
    tasks = {}
    final_results = {}

    # --- 병렬 실행 가능 작업 정의 ---
    parallel_task_definitions = {
        "slots": extract_slots_with_gpt(user_input, session),
        "routing_info": determine_routing_and_reasoning(user_input, session),
        "query_embedding": get_openai_embedding_async(user_input, session),
        # "summary": ...
    }

    logger.info(f"Creating parallel tasks: {list(parallel_task_definitions.keys())}")
    for name, coro in parallel_task_definitions.items():
        task_start_times[name] = time.time()
        tasks[name] = asyncio.create_task(coro)

    # --- 병렬 작업 완료 대기 ---
    parallel_task_keys = list(tasks.keys())
    parallel_results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

    logger.info("Processing parallel task results...")
    query_embedding_vector = None
    for i, result in enumerate(parallel_results_list):
        task_name = parallel_task_keys[i]
        start_time = task_start_times.get(task_name)
        duration = time.time() - start_time if start_time else 0

        if isinstance(result, Exception):
            logger.error(f"Task '{task_name}' failed in {duration:.3f}s: {result}", exc_info=(log_level <= logging.DEBUG))
            if task_name == "slots": final_results[task_name] = None
            elif task_name == "routing_info":
                default_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
                final_results['routing_info'] = {"level": "easy", "model": default_model, "cot_data": None}
            elif task_name == "query_embedding": query_embedding_vector = None
        else:
            logger.info(f"Task '{task_name}' completed successfully in {duration:.3f}s.")
            if task_name == "slots": final_results[task_name] = result
            elif task_name == "routing_info": final_results[task_name] = result
            elif task_name == "query_embedding": query_embedding_vector = result

    # --- RAG 검색 실행 (쿼리 임베딩 완료 & 검색기 준비 완료 후) ---
    # [수정] rag_searcher 유효성 검사 추가
    if query_embedding_vector and rag_searcher and rag_searcher.index:
        logger.info("Starting RAG search task using the obtained query embedding...")
        rag_k = config.get('rag', {}).get('retrieval_k', 3)
        task_start_times["rag_search"] = time.time()
        # [수정] run_rag_search_async에 rag_searcher 인스턴스 전달
        rag_search_task = run_rag_search_async(query_embedding_vector, rag_k, rag_searcher)
        rag_result_list = await asyncio.gather(rag_search_task, return_exceptions=True)
        rag_result = rag_result_list[0]
        duration = time.time() - task_start_times["rag_search"]

        if isinstance(rag_result, Exception):
             logger.error(f"Task 'rag_search' failed in {duration:.3f}s: {rag_result}", exc_info=(log_level <= logging.DEBUG))
             final_results['rag_results'] = []
        else:
             logger.info(f"Task 'rag_search' completed successfully in {duration:.3f}s.")
             final_results['rag_results'] = rag_result
    else:
        if not query_embedding_vector:
             logger.warning("Skipping RAG search because query embedding failed or was not generated.")
        elif not rag_searcher or not rag_searcher.index:
             logger.warning("Skipping RAG search because RagSearcher is not ready.")
        final_results['rag_results'] = []


    end_time_scheduler = time.time()
    total_duration = end_time_scheduler - start_time_scheduler
    logger.info(f"--- Scheduler Finished in {total_duration:.3f} seconds ---")
    # ... (DEBUG 로그 부분은 이전과 동일) ...
    if log_level <= logging.DEBUG:
        # ... (상세 결과 로깅) ...
        pass

    return final_results


# --- 예시 사용법 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_scheduler():
        try: get_config(); logger.info("Config loaded for scheduler test.")
        except Exception as e: logger.error(f"Failed to load config: {e}"); return

        # 테스트를 위해 임시 RagSearcher 인스턴스 생성 (실제 앱에서는 app.state 사용)
        # 이 테스트는 FAISS 인덱스와 메타데이터가 있어야 의미 있음
        test_rag_searcher = None
        try:
             test_rag_searcher = RagSearcher()
             logger.info("Temporary RagSearcher created for test.")
        except Exception as e:
             logger.warning(f"Could not create temporary RagSearcher for test: {e}. RAG search will be skipped.")

        test_input = "나이키 에어맥스 270mm 신고 있는데, 발볼이 좀 넓은 편이에요. 데카트론에서 비슷한 러닝화 추천해주세요."
        state = ConversationState()
        logger.info("Starting scheduler test...")
        async with aiohttp.ClientSession() as session:
            # 테스트 시 생성된 rag_searcher 전달
            scheduler_output = await run_parallel_tasks(test_input, state, session, test_rag_searcher)
            print("\n--- Scheduler Test Output (JSON) ---")
            def default_serializer(obj):
                 if isinstance(obj, np.ndarray): return obj.tolist()
                 try: return str(obj)
                 except Exception: return f"<Object of type {type(obj).__name__} not serializable>"
            print(json.dumps(scheduler_output, indent=2, ensure_ascii=False, default=default_serializer))

    try: asyncio.run(test_scheduler())
    except FileNotFoundError: print("\nError: config.yaml not found.")
    except Exception as e: print(f"\nAn error occurred during testing: {e}")