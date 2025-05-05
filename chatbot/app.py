# chatbot/app.py (Tool Use, 백그라운드 작업 및 완료 대기 적용)

import asyncio
import time
import json
import logging
import traceback
import os
from typing import Optional, List, Dict, Any # 타입 힌트 추가
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks # BackgroundTasks 임포트
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiohttp
from datetime import datetime

# --- 필요한 모듈 임포트 ---
try:
    # app.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .conversation_state import ConversationState
    # [수정] scheduler 함수 이름 변경 (예시: orchestrate_chatbot_turn) 및 관련 함수 임포트
    from .scheduler import orchestrate_chatbot_turn # run_parallel_tasks 대신 새 함수 임포트
    # [삭제] prompt_builder 는 scheduler 내부에서 사용되거나 역할 변경
    # from .prompt_builder import build_final_prompt
    # [삭제] gpt_interface 는 scheduler 통해서 호출됨
    # from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    from .searcher import RagSearcher
    # [신규] 백그라운드 작업 실행을 위한 함수 임포트 (scheduler.py에서 이동했거나 새로 정의)
    from .slot_extractor import extract_slots_with_gpt
    from .summarizer import summarize_conversation_async

    logging.info("Required chatbot modules imported successfully in app.py.")
except ImportError as ie:
    logging.error(f"CRITICAL ERROR (app.py): Failed to import required modules: {ie}. Check relative paths and file existence.", exc_info=True)
    exit(1) # 필수 모듈 임포트 실패 시 즉시 종료

# --- 로깅 설정 (DEBUG 고정) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("FastAPI application logger initialized with DEBUG level.")


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
app = FastAPI(title="Decathlon Chatbot API", version="1.2.0") # 버전 업데이트 (Tool Use 반영)

# --- 정적 파일 마운트 (변경 없음) ---
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
logger.debug(f"Attempting to mount static files from: {static_dir}")
if not os.path.isdir(static_dir):
    logger.warning(f"Static directory not found at {static_dir}, creating one.")
    try:
        os.makedirs(static_dir, exist_ok=True)
        index_html_default_path = os.path.join(static_dir, 'index.html')
        if not os.path.exists(index_html_default_path):
            with open(index_html_default_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html><html><head><title>Chatbot</title></head><body><h1>Chatbot UI Placeholder</h1><p>Connect your UI here.</p></body></html>")
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


# --- 인메모리 대화 상태 관리 (변경 없음) ---
try:
    if not ConversationState: # 임포트 실패 체크
        raise ImportError("ConversationState class not available.")
    # 애플리케이션 레벨에서 단일 인스턴스 유지 (실제 서비스에서는 세션별 관리 필요)
    conversation_handler = ConversationState()
    logger.info("Initialized in-memory conversation handler (SINGLE INSTANCE).")
except Exception as cs_e:
    logger.critical(f"Failed to initialize ConversationState: {cs_e}. Exiting.", exc_info=True)
    exit(1)


# --- 요청 본문 모델 (변경 없음) ---
class ChatRequest(BaseModel):
    """/chat 엔드포인트 요청 본문 모델"""
    user_input: str

# --- 공유 aiohttp 세션 및 RAG 검색기 인스턴스 (변경 없음) ---
app.state.http_session: Optional[aiohttp.ClientSession] = None
app.state.rag_searcher_instance: Optional[RagSearcher] = None

# --- Startup / Shutdown 이벤트 핸들러 (변경 없음) ---
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
        app.state.rag_searcher_instance = await loop.run_in_executor(None, RagSearcher)
        if app.state.rag_searcher_instance and app.state.rag_searcher_instance.index and app.state.rag_searcher_instance.metadata:
            index_size = getattr(app.state.rag_searcher_instance.index, 'ntotal', 'N/A')
            metadata_size = len(app.state.rag_searcher_instance.metadata)
            logger.info(f"RagSearcher instance initialized successfully in background. Index size: {index_size}, Metadata size: {metadata_size}")
        else:
            logger.error("RagSearcher instance initialization failed or incomplete in background! RAG features will be unavailable.")
            app.state.rag_searcher_instance = None
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

# --- 루트 경로 (HTML UI 제공) (변경 없음) ---
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


# --- [신규] 백그라운드 작업 실행 함수 ---
async def run_post_response_tasks(
    handler: ConversationState,
    current_input: str,
    history: List[Dict[str, str]], # 최종 업데이트된 히스토리
    http_session: Optional[aiohttp.ClientSession]
):
    """
    응답 전송 후 슬롯 추출 및 요약 업데이트를 백그라운드에서 비동기 실행합니다.
    완료 후 ConversationState의 완료 이벤트를 설정합니다.
    """
    task_start_time = time.time()
    logger.info("Starting post-response background tasks (slot extraction, summarization)...")

    tasks_to_run = []

    # 슬롯 추출 태스크 정의
    async def slot_task_wrapper():
        task_name = "background_slot_extraction"
        logger.debug(f"Running {task_name}...")
        start = time.time()
        try:
            if extract_slots_with_gpt: # 함수 존재 확인
                slots = await extract_slots_with_gpt(current_input, http_session)
                if slots is not None:
                    handler.update_slots(slots) # 상태 업데이트
                    logger.debug(f"{task_name} finished successfully. Slots: {list(slots.keys())}")
                else:
                    logger.warning(f"{task_name} returned None.")
            else:
                logger.error(f"Function 'extract_slots_with_gpt' not available for {task_name}.")
        except Exception as e:
            logger.error(f"Error in {task_name}: {e}", exc_info=True)
        finally:
            logger.debug(f"{task_name} execution took {time.time() - start:.3f}s")

    tasks_to_run.append(asyncio.create_task(slot_task_wrapper()))

    # 요약 업데이트 태스크 정의
    async def summary_task_wrapper():
        task_name = "background_summarization"
        logger.debug(f"Running {task_name}...")
        start = time.time()
        try:
            # 설정에서 요약 기능 활성화 여부 확인 (선택적)
            summarization_enabled = config.get('tasks', {}).get('summarization', {}).get('enabled', False)
            if summarization_enabled and summarize_conversation_async: # 함수 존재 확인
                # 현재 상태의 요약을 previous_summary로 전달
                summary = await summarize_conversation_async(history, handler.get_summary(), http_session)
                if summary is not None:
                    handler.update_summary(summary) # 상태 업데이트
                    logger.debug(f"{task_name} finished successfully. Summary length: {len(summary)}")
                else:
                    logger.warning(f"{task_name} returned None.")
            elif not summarization_enabled:
                 logger.debug(f"Skipping {task_name} as it's disabled in config.")
            else:
                 logger.error(f"Function 'summarize_conversation_async' not available for {task_name}.")

        except Exception as e:
            logger.error(f"Error in {task_name}: {e}", exc_info=True)
        finally:
            logger.debug(f"{task_name} execution took {time.time() - start:.3f}s")

    tasks_to_run.append(asyncio.create_task(summary_task_wrapper()))

    # 두 백그라운드 작업 병렬 실행 및 대기
    try:
        await asyncio.gather(*tasks_to_run)
        logger.info("All post-response background tasks completed successfully.")
    except Exception as bg_e:
        # gather 자체에서 예외 발생 시 (거의 없음) 또는 개별 태스크 예외는 이미 로깅됨
        logger.error(f"Error during asyncio.gather for background tasks: {bg_e}", exc_info=True)
    finally:
        # 성공/실패 여부와 관계없이 완료 시그널 전송!
        handler.update_complete_event.set()
        task_duration = time.time() - task_start_time
        logger.info(f"Post-response background task execution finished (event set). Duration: {task_duration:.3f}s")


# --- 챗봇 응답 API 엔드포인트 ([수정됨]) ---
@app.post("/chat", response_class=JSONResponse, summary="챗봇 응답 생성 (Tool Use 및 백그라운드 처리 적용)")
async def handle_chat(
    chat_request: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks # FastAPI의 BackgroundTasks 주입
):
    """
    사용자 입력을 받아 Tool Use 기반 챗봇 파이프라인을 실행하고,
    최종 응답 및 디버그 정보(테스트 모드 시)를 반환합니다.
    응답 반환 후, 슬롯 추출 및 요약 업데이트를 백그라운드로 수행합니다.
    다음 요청 처리 전, 이전 턴의 백그라운드 작업 완료를 대기합니다.

    Args:
        chat_request (ChatRequest): 사용자 입력을 포함하는 요청 본문.
        request (Request): FastAPI 요청 객체 (헤더 등 접근용).
        background_tasks (BackgroundTasks): 백그라운드 작업 실행을 위한 FastAPI 객체.

    Returns:
        JSONResponse: 챗봇 응답 또는 오류 정보를 포함하는 JSON 응답.

    Raises:
        HTTPException: 잘못된 요청(400), 서비스 불가(503), 내부 서버 오류(500) 등.
    """
    request_received_time = time.time()
    user_input = chat_request.user_input
    if not user_input or not user_input.strip():
        logger.warning("Received empty or whitespace-only user input.")
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    client_host = request.client.host if request.client else "Unknown"
    test_config = config.get('testing', {})
    test_mode_header_name = test_config.get('test_mode_header', 'X-Test-Mode')
    is_test_mode = request.headers.get(test_mode_header_name, 'false').lower() == 'true'
    request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    logger.info(f"[{request_id}] Received POST '/chat' from {client_host}. TestMode={is_test_mode}. Input: '{user_input[:50]}...'")

    # --- [신규] 이전 턴 백그라운드 작업 완료 대기 ---
    wait_start_time = time.time()
    logger.debug(f"[{request_id}] Checking if previous background update is complete...")
    # ConversationState의 이벤트 대기 (이미 set 상태면 즉시 통과)
    await conversation_handler.update_complete_event.wait()
    wait_duration = time.time() - wait_start_time
    if wait_duration > 0.1: # 100ms 이상 대기 시 로그 기록
        logger.info(f"[{request_id}] Waited {wait_duration:.3f}s for previous background tasks to complete.")
    logger.debug(f"[{request_id}] Previous background update complete. Proceeding with current request.")


    # --- 필수 서비스 (세션, RAG 검색기) 확인 ---
    session = app.state.http_session
    rag_searcher = app.state.rag_searcher_instance

    if not session or session.closed:
        logger.error(f"[{request_id}] AIOHTTP session is not available or closed.")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: HTTP session not ready")

    # RAG 검색기는 None일 수 있음 (scheduler에서 처리)
    if rag_searcher is None:
        logger.warning(f"[{request_id}] RAG searcher instance is not available. RAG search will be skipped if requested.")

    # --- 오케스트레이션 실행 (Scheduler 호출) ---
    final_response = None
    error_response = None
    orchestration_debug_info = {} # 스케줄러의 디버그 정보 저장용

    try:
        logger.info(f"[{request_id}] Calling scheduler (orchestrate_chatbot_turn)...")
        orchestration_start_time = time.time()

        # [수정] 새로운 오케스트레이션 함수 호출
        orchestration_result = await orchestrate_chatbot_turn(
            user_input=user_input,
            conversation_state=conversation_handler, # 현재 상태 전달
            session=session,
            rag_searcher=rag_searcher
        )
        orchestration_duration = time.time() - orchestration_start_time
        logger.info(f"[{request_id}] Scheduler finished in {orchestration_duration:.3f}s.")
        orchestration_debug_info = orchestration_result.get('debug_info', {}) # 디버그 정보 추출

        # --- 오케스트레이션 결과 처리 ---
        if "response" in orchestration_result:
            final_response = orchestration_result["response"]
            if not final_response or not isinstance(final_response, str):
                logger.warning(f"[{request_id}] Orchestrator returned success but response content is empty or invalid. Type: {type(final_response)}")
                final_response = "(죄송합니다, 답변을 생성하는 데 문제가 발생했습니다.)" # Fallback
            else:
                logger.info(f"[{request_id}] Received successful response from orchestrator. Length: {len(final_response)} chars.")
        elif "error_message_for_user" in orchestration_result:
            # 스케줄러가 사용자에게 직접 전달할 오류 메시지를 반환한 경우 (예: 재검색 유도)
            error_response = orchestration_result["error_message_for_user"]
            logger.warning(f"[{request_id}] Orchestrator returned a user-facing error message: {error_response}")
        elif "error" in orchestration_result:
            # 스케줄러 내부 오류 발생 시
            internal_error_msg = orchestration_result["error"]
            logger.error(f"[{request_id}] Orchestrator returned an internal error: {internal_error_msg}")
            error_response = "죄송합니다, 요청을 처리하는 중 오류가 발생했습니다." # 일반 오류 메시지
        else:
            # 예상치 못한 결과 구조
            logger.error(f"[{request_id}] Orchestrator returned an unexpected result structure: {list(orchestration_result.keys())}")
            error_response = "죄송합니다, 예상치 못한 오류가 발생했습니다."

    except HTTPException as http_exc: # 스케줄러 내부에서 발생한 HTTPException 처리
        logger.warning(f"[{request_id}] HTTPException raised from scheduler: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        # 오케스트레이션 호출 자체에서 예외 발생
        orchestration_duration = time.time() - orchestration_start_time
        logger.error(f"[{request_id}] An unexpected error occurred calling orchestrator after {orchestration_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during orchestration.")


    # --- 최종 응답 결정 및 기록 업데이트 ---
    assistant_message = final_response if final_response else error_response
    if not assistant_message: # 만약 둘 다 None이면 최종 Fallback
        assistant_message = "(죄송합니다, 답변을 드릴 수 없습니다.)"

    logger.info(f"[{request_id}] Final assistant message determined. Length: {len(assistant_message)}")

    # [수정] 응답 반환 *전*에 현재 턴 히스토리 업데이트
    try:
        conversation_handler.add_to_history("user", user_input)
        conversation_handler.add_to_history("assistant", assistant_message)
        final_history_for_summary = conversation_handler.get_history() # 백그라운드 요약에 사용될 최종 히스토리
        logger.info(f"[{request_id}] Updated conversation history. Total turns: {len(final_history_for_summary)}")
    except Exception as history_e:
        logger.error(f"[{request_id}] Failed to update conversation history: {history_e}", exc_info=True)
        # 히스토리 업데이트 실패는 치명적이지 않을 수 있으므로 계속 진행하되 로깅


    # --- [신규] 백그라운드 작업 예약 ---
    try:
        # 다음 요청을 위해 현재 업데이트 시작 상태로 변경
        conversation_handler.update_complete_event.clear()
        logger.debug(f"[{request_id}] Cleared update completion event, scheduling background tasks...")

        # 백그라운드 작업 함수 등록 (슬롯 추출, 요약 업데이트 동시 실행)
        background_tasks.add_task(
            run_post_response_tasks,
            conversation_handler,
            user_input, # 현재 사용자 입력
            final_history_for_summary, # 최종 업데이트된 히스토리
            session # 공유 http 세션 전달
        )
        logger.info(f"[{request_id}] Scheduled post-response background tasks.")
    except Exception as bg_schedule_e:
         logger.error(f"[{request_id}] Failed to schedule background tasks: {bg_schedule_e}", exc_info=True)
         # 백그라운드 작업 예약 실패 시 이벤트 다시 set (다음 요청 막지 않도록)
         conversation_handler.update_complete_event.set()
         logger.warning(f"[{request_id}] Set update completion event due to background scheduling failure.")


    # --- 최종 응답 페이로드 구성 ---
    end_time = time.time()
    total_request_time = end_time - request_received_time
    logger.info(f"[{request_id}] Request processing finished in {total_request_time:.3f} seconds.")

    response_payload = {"response": assistant_message}
    if is_test_mode:
        # [수정] 스케줄러에서 받은 디버그 정보 통합
        response_payload["debug_info"] = {
            "request_id": request_id,
            "total_processing_time_ms": int(total_request_time * 1000),
            "wait_for_previous_update_ms": int(wait_duration * 1000),
            **orchestration_debug_info # 스케줄러의 디버그 정보 합치기
        }
        logger.info(f"[{request_id}] Returning response with debug info for test mode.")

    # --- 최종 응답 반환 (백그라운드 작업 실행 예약 포함) ---
    # background 인자에 tasks 객체 전달
    return JSONResponse(content=response_payload, background=background_tasks)


# --- 서버 실행 (uvicorn 사용) (변경 없음) ---
if __name__ == "__main__":
    import uvicorn
    try:
        main_config = get_config()
        server_host = main_config.get('server', {}).get('host', '127.0.0.1')
        server_port = main_config.get('server', {}).get('port', 8000)
        server_reload = main_config.get('server', {}).get('reload', True)
        # logging 설정에서 log_level을 읽어 uvicorn에 반영
        uvicorn_log_level = main_config.get('logging', {}).get('log_level', 'info').lower()

        logger.info(f"Starting FastAPI server using uvicorn (host={server_host}, port={server_port}, reload={server_reload}, uvicorn_log_level={uvicorn_log_level})...")
        uvicorn.run(
            "chatbot.app:app", # 모듈 경로 확인 'chatbot' 패키지 내의 'app' 모듈의 'app' 객체
            host=server_host,
            port=server_port,
            reload=server_reload,
            log_level=uvicorn_log_level
        )
    except KeyError as ke:
        logger.critical(f"Missing configuration key required for starting server: {ke}")
    except Exception as e:
        logger.critical(f"CRITICAL Error starting uvicorn server: {e}", exc_info=True)