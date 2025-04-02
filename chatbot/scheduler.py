# chatbot/scheduler.py

import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List
import aiohttp # 비동기 HTTP 요청

# 필요한 모듈 임포트
from .slot_extractor import extract_slots_with_gpt
from .model_router import route_model_gpt_based
from .searcher import rag_searcher_instance # 로드된 RAG 검색기 인스턴스
from .conversation_state import ConversationState

# RAG 검색 함수를 비동기적으로 실행하기 위한 래퍼
# sentence_transformers나 faiss가 내부적으로 blocking I/O를 사용할 수 있으므로,
# asyncio.to_thread (Python 3.9+) 또는 run_in_executor 사용 권장
async def run_rag_search_async(query: str, k: int = 3) -> List[Dict]:
    """RAG 검색을 비동기적으로 실행합니다."""
    loop = asyncio.get_running_loop()
    try:
        # rag_searcher_instance.search가 CPU 집약적이거나 동기 I/O를 포함할 수 있으므로
        # 별도 스레드에서 실행하여 이벤트 루프 차단 방지
        results = await loop.run_in_executor(
            None, # 기본 ThreadPoolExecutor 사용
            rag_searcher_instance.search,
            query,
            k
        )
        return results
    except Exception as e:
        print(f"Error during async RAG search: {e}")
        return []

async def run_parallel_tasks(
    user_input: str,
    conversation_state: ConversationState, # 현재 상태는 필요 없을 수 있으나, 추후 요약 등에 사용 가능
    session: aiohttp.ClientSession # 공유 세션 사용
) -> Dict[str, Any]:
    """
    Slot 추출, 모델 라우팅, RAG 검색 등의 작업을 병렬로 실행합니다.

    Args:
        user_input (str): 사용자 입력.
        conversation_state (ConversationState): 현재 대화 상태.
        session (aiohttp.ClientSession): 비동기 HTTP 요청 세션.

    Returns:
        Dict[str, Any]: 각 작업의 결과를 담은 딕셔너리.
                       {'slots': ..., 'chosen_model': ..., 'rag_results': ...}
                       실패 시 해당 키 값은 None 또는 빈 리스트.
    """
    start_time = time.time()
    print("\n--- Running Parallel Tasks ---")

    # 병렬로 실행할 비동기 작업 목록 생성
    tasks = {
        "slots": asyncio.create_task(extract_slots_with_gpt(user_input, session)),
        "routing": asyncio.create_task(route_model_gpt_based(user_input, session)),
        "rag": asyncio.create_task(run_rag_search_async(user_input, k=3)) # RAG 검색 (상위 3개)
        # TODO: 추후 대화 요약 작업 추가 가능
        # "summary": asyncio.create_task(summarize_conversation_async(conversation_state.get_history(), session))
    }

    # asyncio.gather를 사용하여 모든 작업이 완료될 때까지 대기
    # return_exceptions=True : 작업 중 예외가 발생해도 다른 작업은 계속 진행하고, 예외 객체를 결과로 받음
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # 결과 매핑
    task_keys = list(tasks.keys())
    final_results = {}
    for i, result in enumerate(results):
        task_name = task_keys[i]
        if isinstance(result, Exception):
            print(f"Error in parallel task '{task_name}': {result}")
            # 실패 시 기본값 설정
            if task_name == "slots":
                final_results[task_name] = None
            elif task_name == "routing":
                final_results['chosen_model'] = "gpt-3.5-turbo" # 실패 시 기본 모델
            elif task_name == "rag":
                final_results['rag_results'] = []
            # elif task_name == "summary":
            #     final_results['summary'] = None
        else:
            # 성공 시 결과 저장
            if task_name == "slots":
                final_results[task_name] = result
            elif task_name == "routing":
                final_results['chosen_model'] = result
            elif task_name == "rag":
                final_results['rag_results'] = result
            # elif task_name == "summary":
            #     final_results['summary'] = result


    end_time = time.time()
    print(f"--- Parallel Tasks Finished in {end_time - start_time:.2f} seconds ---")
    print(f"Scheduler Results: ")
    print(f"  - Slots: {final_results.get('slots')}")
    print(f"  - Chosen Model: {final_results.get('chosen_model')}")
    print(f"  - RAG Results Count: {len(final_results.get('rag_results', []))}")
    # print(f"  - Summary: {final_results.get('summary')}")

    return final_results

# --- 예시 사용법 ---
if __name__ == "__main__":
    import asyncio

    async def test_scheduler():
        test_input = "나이키 에어맥스 270mm 신고 있는데, 발볼이 좀 넓은 편이에요. 데카트론에서 비슷한 러닝화 추천해주세요."
        # conversation_state는 필요시 생성하여 전달
        state = ConversationState()
        # state.add_to_history(...) # 이전 대화 기록 추가 가능

        # aiohttp 세션 생성
        async with aiohttp.ClientSession() as session:
            scheduler_output = await run_parallel_tasks(test_input, state, session)

            print("\n--- Scheduler Test Output ---")
            print(json.dumps(scheduler_output, indent=2, ensure_ascii=False))

            # 결과 활용 예시
            if scheduler_output.get("slots"):
                state.update_slots(scheduler_output["slots"])
                print("\nUpdated Conversation State Slots:")
                print(state.get_slots())

            chosen_model = scheduler_output.get("chosen_model", "gpt-3.5-turbo")
            rag_docs = scheduler_output.get("rag_results", [])
            print(f"\nNext step: Use model '{chosen_model}' with {len(rag_docs)} RAG documents.")

    # RAG Searcher가 제대로 초기화되지 않으면 에러 발생 가능성 있음
    # 실행 전 rag_generator.py 실행 및 필요 라이브러리 설치 확인 필요
    asyncio.run(test_scheduler())