# chatbot/app.py (**수정됨**: run_summarization_bg에서 슬롯/설정값 전달 로직 추가)

import asyncio
import time
import json
import logging
import traceback
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiohttp
from datetime import datetime

# --- 필요한 모듈 임포트 ---
try:
    # app.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .conversation_state import ConversationState
    from .scheduler import orchestrate_chatbot_turn
    from .config_loader import get_config
    from .searcher import RagSearcher
    # [수정됨] 백그라운드 작업을 위한 개별 함수 임포트
    from .slot_extractor import extract_slots_with_gpt
    from .summarizer import summarize_conversation_async # 직접 임포트

    logging.info("Required chatbot modules imported successfully in app.py.")
except ImportError as ie:
    logging.error(f"CRITICAL ERROR (app.py): Failed to import required modules: {ie}. Check relative paths and file existence.", exc_info=True)
    exit(1) # 필수 모듈 임포트 실패 시 즉시 종료

# --- 로깅 설정 ---
# 기본 로깅 설정은 다른 모듈에서 이미 수행되었을 수 있으므로, 여기서는 로거만 가져옴
logger = logging.getLogger(__name__)
# 로거 레벨은 루트 또는 다른 설정 파일에서 결정된 것을 따름
logger.info("FastAPI application logger retrieved.")


# --- 설정 로드 ---
try:
    config = get_config()
    if not config:
        raise ValueError("Configuration could not be loaded.")
    logger.info("Configuration loaded successfully in app.py.")
except Exception as e:
    logger.critical(f"CRITICAL ERROR (app.py): Failed to load configuration: {e}. Exiting.", exc_info=True)
    exit(1)

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="Decathlon Chatbot API", version="1.4.0") # [수정됨] 버전 업데이트 반영 (Summary Context Enhanced)

# --- 정적 파일 마운트 ---
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
logger.debug(f"Attempting to mount static files from: {static_dir}")
if not os.path.isdir(static_dir):
    logger.warning(f"Static directory not found at {static_dir}, creating one.")
    try:
        os.makedirs(static_dir, exist_ok=True)
        index_html_default_path = os.path.join(static_dir, 'index.html')
        if not os.path.exists(index_html_default_path):
            # Placeholder index.html 내용 (실제 프로젝트에서는 제공된 index.html 사용)
            with open(index_html_default_path, 'w', encoding='utf-8') as f:
                 f.write("<!DOCTYPE html><html><head><title>Chatbot</title></head><body><h1>Chatbot UI Placeholder</h1></body></html>")
            logger.info(f"Created placeholder index.html at {index_html_default_path}")
    except Exception as e:
        logger.error(f"Failed to create static directory or placeholder index.html: {e}")
if os.path.isdir(static_dir):
    try:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Successfully mounted static files from: {static_dir}")
    except Exception as e:
        logger.error(f"Failed to mount static directory {static_dir}: {e}. Static file serving might not work.", exc_info=True)
else:
    logger.error(f"Static directory {static_dir} not found or is not a directory. Static file serving disabled.")


# --- 인메모리 대화 상태 관리 (싱글톤 인스턴스) ---
try:
    if not ConversationState: # 임포트 실패 체크
        raise ImportError("ConversationState class not available.")
    # 애플리케이션 레벨에서 단일 인스턴스 유지 (실제 서비스에서는 세션별 관리 필요)
    conversation_handler = ConversationState()
    logger.info("Initialized in-memory conversation handler (SINGLE INSTANCE).")
except Exception as cs_e:
    logger.critical(f"Failed to initialize ConversationState: {cs_e}. Exiting.", exc_info=True)
    exit(1)


# --- 요청 본문 모델 ---
class ChatRequest(BaseModel):
    """/chat 엔드포인트 요청 본문 모델"""
    user_input: str

# --- 공유 aiohttp 세션 및 RAG 검색기 인스턴스 ---
app.state.http_session: Optional[aiohttp.ClientSession] = None
app.state.rag_searcher_instance: Optional[RagSearcher] = None

# --- Startup / Shutdown 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 AIOHTTP 세션 생성 및 RAG 검색기 인스턴스 초기화."""
    logger.info("Executing FastAPI startup event...")
    # 1. AIOHTTP 세션 생성
    try:
        app.state.http_session = aiohttp.ClientSession()
        logger.info("AIOHTTP ClientSession created for application lifetime.")
    except Exception as session_e:
        logger.error(f"Failed to create AIOHTTP ClientSession during startup: {session_e}", exc_info=True)

    # 2. RAG 검색기 인스턴스 초기화 (백그라운드)
    logger.info("Initializing RagSearcher instance in background executor...")
    loop = asyncio.get_running_loop()
    try:
        if not RagSearcher: raise ImportError("RagSearcher class not available.")
        # RagSearcher 초기화는 동기 작업이므로 run_in_executor 사용
        app.state.rag_searcher_instance = await loop.run_in_executor(None, RagSearcher)
        if app.state.rag_searcher_instance and app.state.rag_searcher_instance.index and app.state.rag_searcher_instance.metadata:
            index_size = getattr(app.state.rag_searcher_instance.index, 'ntotal', 'N/A')
            metadata_size = len(app.state.rag_searcher_instance.metadata)
            logger.info(f"RagSearcher instance initialized successfully in background. Index size: {index_size}, Metadata size: {metadata_size}")
        else:
            logger.error("RagSearcher instance initialization failed or incomplete in background! RAG features will be unavailable.")
            app.state.rag_searcher_instance = None # 실패 시 명시적으로 None 설정
    except ImportError:
        logger.error("RagSearcher class not imported correctly. RAG features will be unavailable.")
        app.state.rag_searcher_instance = None
    except Exception as e:
        logger.error(f"Error during RagSearcher initialization in startup_event: {e}", exc_info=True)
        app.state.rag_searcher_instance = None
    logger.info("FastAPI startup event finished.")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 AIOHTTP 세션 종료."""
    logger.info("Executing FastAPI shutdown event...")
    if app.state.http_session and not app.state.http_session.closed:
        await app.state.http_session.close()
        logger.info("AIOHTTP ClientSession closed.")
    app.state.http_session = None
    logger.info("FastAPI application shutdown.")

# --- 루트 경로 (HTML UI 제공) ---
@app.get("/", response_class=HTMLResponse, summary="챗봇 UI 페이지 제공")
async def read_root(request: Request):
    """루트 경로('/') 접근 시 static/index.html 파일을 읽어 HTML 응답으로 반환합니다."""
    client_host = request.client.host if request.client else "Unknown"
    logger.info(f"GET request received for '/' from {client_host}")
    index_html_path = os.path.join(static_dir, 'index.html')
    if not os.path.exists(index_html_path):
        logger.error(f"index.html not found at {index_html_path}")
        return HTMLResponse(content="<html><body><h1>Error: Chatbot UI not found</h1></body></html>", status_code=404)
    try:
        with open(index_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading or serving index.html: {e}", exc_info=True)
        return HTMLResponse(content="<html><body><h1>Internal Server Error</h1></body></html>", status_code=500)


# --- 백그라운드 작업 함수 분리 ---

async def run_slot_extraction_bg(
    handler: ConversationState,
    current_input: str,
    http_session: Optional[aiohttp.ClientSession],
    parent_run_id: str # 디버깅용 부모 ID
):
    """슬롯 추출을 백그라운드에서 비동기 실행합니다."""
    task_start_time = time.time()
    task_run_id = f"bg_slot_{parent_run_id}"
    logger.info(f"[{task_run_id}] Starting background task: slot_extraction...")

    try:
        # extract_slots_with_gpt 함수가 임포트되었는지 확인
        if 'extract_slots_with_gpt' in globals() and extract_slots_with_gpt:
            slots = await extract_slots_with_gpt(current_input, http_session)
            if slots is not None: # 추출 실패(None) 또는 성공(dict)
                handler.update_slots(slots) # 상태 업데이트
                logger.debug(f"[{task_run_id}] Slot extraction finished. Slots updated: {list(slots.keys()) if isinstance(slots, dict) else 'None'}")
            else:
                logger.warning(f"[{task_run_id}] Slot extraction returned None (failed).")
        else:
            logger.error(f"[{task_run_id}] Function 'extract_slots_with_gpt' not available or not imported.")
    except Exception as e:
        logger.error(f"[{task_run_id}] Error in slot_extraction background task: {e}", exc_info=True)
    finally:
        duration = time.time() - task_start_time
        logger.info(f"[{task_run_id}] Background task slot_extraction finished. Duration: {duration:.3f}s")
        # 개별 완료 이벤트는 설정하지 않음. Wrapper에서 처리.

# --- [수정됨] 요약 백그라운드 작업 함수 (슬롯/설정값 전달 추가) ---
async def run_summarization_bg(
    handler: ConversationState,
    history: List[Dict[str, Any]], # 타입 Any 허용
    http_session: Optional[aiohttp.ClientSession],
    parent_run_id: str # 디버깅용 부모 ID
):
    """대화 요약을 백그라운드에서 비동기 실행합니다. (슬롯 및 설정값 전달)"""
    task_start_time = time.time()
    task_run_id = f"bg_summary_{parent_run_id}"
    logger.info(f"[{task_run_id}] Starting background task: summarization...")

    try:
        global config # 설정 접근
        summarization_cfg = config.get('tasks', {}).get('summarization', {})
        summarization_enabled = summarization_cfg.get('enabled', False)

        if summarization_enabled:
            # summarize_conversation_async 함수가 임포트되었는지 확인
            if 'summarize_conversation_async' in globals() and summarize_conversation_async:
                current_summary = handler.get_summary()
                current_slots = handler.get_slots() # <-- *** 현재 슬롯 정보 가져오기 ***

                # config 에서 관련 설정 읽기
                include_slots_cfg = summarization_cfg.get('include_slots_in_summary_prompt', False)
                include_tools_cfg = summarization_cfg.get('include_tool_results_in_summary', 'none')

                # summarize_conversation_async 호출 시 추가 인자 전달
                new_summary = await summarize_conversation_async(
                    history=history,
                    previous_summary=current_summary,
                    session=http_session,
                    current_slots=current_slots, # <-- 전달
                    include_slots=include_slots_cfg, # <-- 전달
                    include_tool_results=include_tools_cfg # <-- 전달
                )

                # 요약 결과 처리
                if new_summary is not None:
                    handler.update_summary(new_summary) # 상태 업데이트
                    logger.debug(f"[{task_run_id}] Summarization finished successfully. Summary updated.")
                else:
                    logger.warning(f"[{task_run_id}] Summarization returned None (failed).")
            else:
                 logger.error(f"[{task_run_id}] Function 'summarize_conversation_async' not available or not imported.")
        else:
             logger.debug(f"[{task_run_id}] Summarization is disabled in config, task skipped.")

    except Exception as e:
        logger.error(f"[{task_run_id}] Error in summarization background task: {e}", exc_info=True)
    finally:
        duration = time.time() - task_start_time
        logger.info(f"[{task_run_id}] Background task summarization finished. Duration: {duration:.3f}s")
        # 개별 완료 이벤트는 설정하지 않음. Wrapper에서 처리.

# --- [기존] 백그라운드 작업 전체 관리 함수 (변경 없음) ---
async def run_all_background_tasks(
    tasks_to_run: List[asyncio.Task],
    handler: ConversationState,
    run_id: str # 전체 실행 식별용
):
    """주어진 비동기 작업들을 병렬 실행하고 완료 시 이벤트를 설정합니다."""
    if not tasks_to_run:
        logger.info(f"[{run_id}] No background tasks scheduled for this turn. Setting completion event immediately.")
        handler.update_complete_event.set() # 실행할 태스크 없으면 즉시 완료 처리
        return

    logger.info(f"[{run_id}] Starting execution of {len(tasks_to_run)} background tasks.")
    start_time = time.time()
    try:
        # asyncio.gather를 사용하여 모든 작업이 완료될 때까지 기다림
        await asyncio.gather(*tasks_to_run)
        logger.info(f"[{run_id}] All scheduled background tasks completed successfully.")
    except Exception as bg_e:
        # gather 자체에서 예외 발생 또는 개별 태스크 예외 처리 (개별 태스크 내에서 로깅됨)
        logger.error(f"[{run_id}] Error during asyncio.gather for background tasks: {bg_e}", exc_info=True)
    finally:
        # 성공/실패 여부와 관계없이 완료 시그널 전송
        handler.update_complete_event.set() # 완료 이벤트 설정
        duration = time.time() - start_time
        logger.info(f"[{run_id}] Background task execution finished (event set). Total Duration: {duration:.3f}s")


# --- [기존] 챗봇 응답 API 엔드포인트 (호출 로직 변경 없음) ---
@app.post("/chat", response_class=JSONResponse, summary="챗봇 응답 생성 (Summary+K Turns, Periodic Summary, Enhanced Context)")
async def handle_chat(
    chat_request: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks # FastAPI의 BackgroundTasks 주입
):
    """
    사용자 입력을 받아 'Summary + K Turns' 기반 챗봇 파이프라인을 실행하고,
    최종 응답 및 디버그 정보(테스트 모드 시)를 반환합니다.
    응답 반환 후, 슬롯 추출(매번) 및 요약 업데이트(주기적, 강화된 컨텍스트 사용)를 백그라운드로 수행합니다.
    다음 요청 처리 전, 이전 턴의 백그라운드 작업 완료를 대기합니다.
    """
    request_received_time = time.time()
    user_input = chat_request.user_input
    if not user_input or not user_input.strip():
        logger.warning("Received empty or whitespace-only user input.")
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    client_host = request.client.host if request.client else "Unknown"
    test_config = config.get('testing', {}) # 테스트 설정 읽기
    test_mode_header_name = test_config.get('test_mode_header', 'X-Test-Mode')
    is_test_mode = request.headers.get(test_mode_header_name, 'false').lower() == 'true'
    request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    logger.info(f"[{request_id}] Received POST '/chat' from {client_host}. TestMode={is_test_mode}.")

    # --- [신규] 턴 카운터 증가 ---
    conversation_handler.increment_turn_counter()
    current_turn = conversation_handler.get_turn_counter()
    logger.info(f"[{request_id}] Current Turn: {current_turn}")
    logger.info(f"[{request_id}] User Input: '{user_input[:70]}...'")

    # --- 이전 턴 백그라운드 작업 완료 대기 ---
    wait_start_time = time.time()
    logger.debug(f"[{request_id}] Checking if previous background update is complete...")
    try:
        await asyncio.wait_for(conversation_handler.update_complete_event.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Timeout waiting for previous background tasks! Proceeding, but state might be inconsistent.")
    except Exception as wait_e:
         logger.error(f"[{request_id}] Error waiting for background task event: {wait_e}", exc_info=True)
    wait_duration = time.time() - wait_start_time
    if wait_duration > 0.1:
        logger.info(f"[{request_id}] Waited {wait_duration:.3f}s for previous background tasks.")
    logger.debug(f"[{request_id}] Previous background update check finished.")

    # --- 필수 서비스 확인 ---
    session = app.state.http_session
    rag_searcher = app.state.rag_searcher_instance

    if not session or session.closed:
        logger.error(f"[{request_id}] AIOHTTP session not available or closed.")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: HTTP session not ready")

    # --- 오케스트레이션 실행 (Scheduler 호출 - 변경 없음) ---
    final_response = None
    error_response = None
    orchestration_debug_info = {}
    try:
        logger.info(f"[{request_id}] Calling scheduler (orchestrate_chatbot_turn)...")
        orchestration_start_time = time.time()
        orchestration_result = await orchestrate_chatbot_turn(
            user_input=user_input,
            conversation_state=conversation_handler, # 현재 상태 전달
            session=session,
            rag_searcher=rag_searcher
        )
        orchestration_duration = time.time() - orchestration_start_time
        logger.info(f"[{request_id}] Scheduler finished in {orchestration_duration:.3f}s.")

        # 스케줄러 결과 처리 (기존과 동일)
        orchestration_debug_info = orchestration_result.get('debug_info', {})
        if "response" in orchestration_result:
            final_response = orchestration_result["response"]
            if not final_response or not isinstance(final_response, str):
                 logger.warning(f"[{request_id}] Orchestrator returned success but response content is empty/invalid. Type: {type(final_response)}")
                 final_response = "(죄송합니다, 답변을 생성하는 데 문제가 발생했습니다.)" # Fallback
            else:
                 logger.info(f"[{request_id}] Received successful response from orchestrator.")
        elif "error_message_for_user" in orchestration_result:
             error_response = orchestration_result["error_message_for_user"]
             logger.warning(f"[{request_id}] Orchestrator returned user-facing error: {error_response}")
        elif "error" in orchestration_result:
             internal_error_msg = orchestration_result["error"]
             logger.error(f"[{request_id}] Orchestrator returned internal error: {internal_error_msg}")
             error_response = "죄송합니다, 요청을 처리하는 중 오류가 발생했습니다."
        else:
             logger.error(f"[{request_id}] Orchestrator returned unexpected result: {list(orchestration_result.keys())}")
             error_response = "죄송합니다, 예상치 못한 오류가 발생했습니다."

    except HTTPException as http_exc:
        logger.warning(f"[{request_id}] HTTPException from scheduler: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        orchestration_duration = time.time() - orchestration_start_time if 'orchestration_start_time' in locals() else 0
        logger.error(f"[{request_id}] Unexpected error calling orchestrator after {orchestration_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during orchestration.")


    # --- 최종 응답 결정 및 히스토리 업데이트 (기존과 동일) ---
    assistant_message = final_response if final_response else error_response
    if not assistant_message: # 최종 Fallback
        assistant_message = "(죄송합니다, 답변을 드릴 수 없습니다.)"
    logger.info(f"[{request_id}] Final assistant message determined. Length: {len(assistant_message)}")

    # 응답 반환 *전* 히스토리 업데이트
    final_history_for_background = [] # 백그라운드 전달용 변수 초기화
    try:
        conversation_handler.add_to_history("user", user_input)
        # scheduler가 tool_calls를 반환했다면 여기서 추가해야 함 (orchestration_result에서 추출)
        assistant_tool_calls = orchestration_debug_info.get('llm_response_raw', {}).get('tool_calls')
        # 히스토리 추가 시 Tool 호출 정보도 함께 저장 (기존 주석 해제 및 수정)
        if assistant_tool_calls:
             conversation_handler.add_to_history("assistant", assistant_message, tool_calls=assistant_tool_calls)
        else:
             conversation_handler.add_to_history("assistant", assistant_message)

        final_history_for_background = conversation_handler.get_history(copy=True) # 복사본 사용
        logger.info(f"[{request_id}] Updated conversation history. Total items: {len(final_history_for_background)}")
    except Exception as history_e:
        logger.error(f"[{request_id}] Failed to update conversation history: {history_e}", exc_info=True)


    # --- [기존] 백그라운드 작업 예약 (주기적 요약 로직은 run_summarization_bg 내부에 있음 - 변경 없음) ---
    bg_tasks_to_schedule = [] # 이번 턴에 실행할 실제 백그라운드 Task 객체 리스트
    bg_run_id = f"bg_run_{request_id}" # 백그라운드 실행 식별자

    # 1. 슬롯 추출은 항상 실행
    if 'run_slot_extraction_bg' in globals() and run_slot_extraction_bg:
        slot_task = asyncio.create_task(run_slot_extraction_bg(conversation_handler, user_input, session, bg_run_id))
        bg_tasks_to_schedule.append(slot_task)
        logger.info(f"[{request_id}] Scheduled background task: Slot Extraction.")
    else:
         logger.warning(f"[{request_id}] Slot extraction background function not available.")

    # 2. 요약은 주기적으로 실행 (조건 확인은 run_summarization_bg 에서)
    try:
        summarization_cfg = config.get('tasks', {}).get('summarization', {})
        summarization_enabled = summarization_cfg.get('enabled', False)
        summarize_every_n = summarization_cfg.get('summarize_every_n_turns', 1)
        if summarize_every_n <= 0: summarize_every_n = 1

        # 요약 실행 조건: 기능 활성화, 함수 존재, 현재 턴이 주기(m)의 배수
        if summarization_enabled and 'run_summarization_bg' in globals() and run_summarization_bg and (current_turn % summarize_every_n == 0):
            logger.info(f"[{request_id}] Current turn ({current_turn}) is a multiple of {summarize_every_n}. Scheduling summarization.")
            # run_summarization_bg 함수 자체를 태스크로 만듦 (내부에서 summarize_conversation_async 호출)
            summary_task = asyncio.create_task(run_summarization_bg(conversation_handler, final_history_for_background, session, bg_run_id))
            bg_tasks_to_schedule.append(summary_task)
        elif summarization_enabled:
             logger.debug(f"[{request_id}] Skipping summarization trigger for turn {current_turn} (summarize every {summarize_every_n} turns or function unavailable).")

    except Exception as cfg_err:
         logger.error(f"[{request_id}] Error reading summarization config for scheduling: {cfg_err}. Skipping summarization trigger.")

    # 3. 실제 백그라운드 실행 등록
    if bg_tasks_to_schedule:
        try:
            conversation_handler.update_complete_event.clear()
            logger.debug(f"[{request_id}] Cleared update completion event, scheduling {len(bg_tasks_to_schedule)} background tasks...")
            background_tasks.add_task(
                run_all_background_tasks,
                bg_tasks_to_schedule,
                conversation_handler,
                bg_run_id
            )
            logger.info(f"[{request_id}] Scheduled background task manager to run {len(bg_tasks_to_schedule)} tasks.")
        except Exception as bg_schedule_e:
            logger.error(f"[{request_id}] Failed to schedule background tasks manager: {bg_schedule_e}", exc_info=True)
            conversation_handler.update_complete_event.set()
            logger.warning(f"[{request_id}] Reset update completion event due to background scheduling failure.")
    else:
         logger.debug(f"[{request_id}] No background tasks to schedule for this turn.")


    # --- 최종 응답 페이로드 구성 (기존과 동일) ---
    end_time = time.time()
    total_request_time = end_time - request_received_time
    logger.info(f"[{request_id}] Request processing finished in {total_request_time:.3f} seconds.")

    response_payload = {"response": assistant_message}
    if is_test_mode:
        response_payload["debug_info"] = {
            "request_id": request_id,
            "current_turn": current_turn,
            "total_processing_time_ms": int(total_request_time * 1000),
            "wait_for_previous_update_ms": int(wait_duration * 1000),
            **orchestration_debug_info # 스케줄러 디버그 정보
        }
        logger.info(f"[{request_id}] Returning response with debug info for test mode.")

    # --- 최종 응답 반환 (기존과 동일) ---
    return JSONResponse(content=response_payload, background=background_tasks)


# --- 서버 실행 (기존과 동일) ---
if __name__ == "__main__":
    import uvicorn
    try:
        server_config = config.get('server', {})
        server_host = server_config.get('host', '127.0.0.1')
        server_port = server_config.get('port', 8000)
        server_reload = server_config.get('reload', True)
        uvicorn_log_level = config.get('logging', {}).get('log_level', 'info').lower()

        logger.info(f"Starting FastAPI server via uvicorn (host={server_host}, port={server_port}, reload={server_reload}, uvicorn_log_level={uvicorn_log_level})...")
        uvicorn.run(
            "chatbot.app:app", # 모듈 경로 확인 필요
            host=server_host,
            port=server_port,
            reload=server_reload,
            log_level=uvicorn_log_level
        )
    except KeyError as ke:
        logger.critical(f"Missing config key for starting server: {ke}")
    except Exception as e:
        logger.critical(f"CRITICAL Error starting uvicorn server: {e}", exc_info=True)