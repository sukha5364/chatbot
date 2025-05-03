# chatbot/app.py (요구사항 반영 최종본: 요약/CoT 개선 반영)

import asyncio
import time
import json
import logging
import traceback
import os
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiohttp
from datetime import datetime

# --- 필요한 모듈 임포트 ---
try:
    # app.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .conversation_state import ConversationState
    from .scheduler import run_parallel_tasks
    # [수정] prompt_builder 임포트
    from .prompt_builder import build_final_prompt
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    from .searcher import RagSearcher
    # [삭제] summarizer 모듈 직접 임포트 불필요 (scheduler가 처리)
    # from . import summarizer
    logging.info("Required chatbot modules imported successfully in app.py.")
except ImportError as ie:
    logging.error(f"CRITICAL ERROR (app.py): Failed to import required modules: {ie}. Check relative paths and file existence.", exc_info=True)
    # 필수 모듈 임포트 실패 시 앱 실행 불가
    exit(1) # 즉시 종료

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
app = FastAPI(title="Decathlon Chatbot API", version="1.1.0") # 버전 업데이트

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
    conversation_handler = ConversationState()
    logger.info("Initialized in-memory conversation handler.")
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

# --- 챗봇 응답 API 엔드포인트 ([수정됨]) ---
@app.post("/chat", response_class=JSONResponse, summary="챗봇 응답 생성")
async def handle_chat(chat_request: ChatRequest, request: Request):
    """
    사용자 입력을 받아 챗봇 파이프라인(병렬처리: 슬롯,라우팅,임베딩,CoT,요약 -> 순차: RAG)을
    실행하고, 최종 응답 및 디버그 정보(테스트 모드 시)를 반환합니다.
    요약은 병렬 처리된 결과를 사용하고, 상태 업데이트는 응답 후 수행합니다.

    Args:
        chat_request (ChatRequest): 사용자 입력을 포함하는 요청 본문.
        request (Request): FastAPI 요청 객체 (헤더 등 접근용).

    Returns:
        JSONResponse: 챗봇 응답 또는 오류 정보를 포함하는 JSON 응답.

    Raises:
        HTTPException: 잘못된 요청(400), 서비스 불가(503), 내부 서버 오류(500) 등.
    """
    start_time = time.time()
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

    session = app.state.http_session
    rag_searcher = app.state.rag_searcher_instance

    if not session or session.closed:
        logger.error(f"[{request_id}] AIOHTTP session is not available or closed.")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: HTTP session not ready")

    if rag_searcher is None:
        logger.warning(f"[{request_id}] RAG searcher instance is not available. Proceeding without RAG search.")

    try:
        # --- 1. 병렬/순차 작업 실행 (Scheduler) [수정됨: 요약/CoT 병렬 포함] ---
        logger.info(f"[{request_id}] Calling scheduler to run parallel tasks...")
        scheduler_start_time = time.time()
        scheduler_results = await run_parallel_tasks(
            user_input=user_input,
            conversation_state=conversation_handler, # 현재 대화 상태 전달
            session=session,
            rag_searcher=rag_searcher
        )
        scheduler_duration = time.time() - scheduler_start_time
        logger.info(f"[{request_id}] Scheduler finished in {scheduler_duration:.3f}s.")
        logger.debug(f"[{request_id}] Scheduler results keys: {list(scheduler_results.keys())}")

        # --- 2. 스케줄러 결과 추출 및 상태 업데이트 (Slot) ---
        # [수정] 요약 결과도 추출
        extracted_slots = scheduler_results.get("slots")
        routing_info = scheduler_results.get("routing_info") # level, model, cot_data 포함
        rag_results = scheduler_results.get("rag_results", [])
        new_summary = scheduler_results.get("summary") # 병렬 생성된 요약 결과

        # 슬롯 정보 먼저 업데이트
        if isinstance(extracted_slots, dict):
            logger.info(f"[{request_id}] Updating conversation state with extracted slots: {list(extracted_slots.keys())}")
            conversation_handler.update_slots(extracted_slots)
            logger.debug(f"[{request_id}] Current slots after update: {conversation_handler.get_slots()}")
        elif extracted_slots is not None:
            logger.warning(f"[{request_id}] Received unexpected data type for slots: {type(extracted_slots)}. Skipping slot update.")

        # --- 3. 최종 프롬프트 구성 요소 준비 ---
        # 라우팅 정보 처리 (기본값 설정 강화)
        if not isinstance(routing_info, dict) or not routing_info.get('model'):
            default_model_name = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
            logger.warning(f"[{request_id}] Invalid routing info ({routing_info}), falling back to default: level='easy', model='{default_model_name}', cot_data=None")
            routing_info = {"level": "easy", "model": default_model_name, "cot_data": None}

        complexity_level = routing_info.get("level", "easy")
        chosen_model = routing_info.get("model") # 이제 확실히 존재
        cot_data = routing_info.get("cot_data")

        logger.info(f"[{request_id}] Preparing final prompt components: Complexity='{complexity_level}', ChosenModel='{chosen_model}', RAG Results={len(rag_results)}, CoT Data Present={'Yes' if cot_data else 'No'}, Summary Present={'Yes' if new_summary else 'No'}")

        # --- 4. 최종 프롬프트 생성 (Prompt Builder) [수정됨: 인자 전달 방식 변경] ---
        logger.info(f"[{request_id}] Building final prompt...")
        prompt_build_start_time = time.time()
        try:
            # [수정] build_final_prompt 호출 시 필요한 정보 직접 전달
            # (prompt_builder.py의 함수 시그니처 변경 필요)
            current_history_for_prompt = conversation_handler.get_history() # 현재까지의 히스토리
            current_slots_for_prompt = conversation_handler.get_slots() # 업데이트된 슬롯

            final_messages = build_final_prompt(
                user_query=user_input,
                summary=new_summary, # 스케줄러에서 생성된 (이전 턴까지의) 요약
                history=current_history_for_prompt, # 현재까지의 히스토리 (마지막 user 입력 포함 X)
                slots=current_slots_for_prompt,     # 현재 업데이트된 슬롯
                rag_results=rag_results,
                cot_data=cot_data
            )
            prompt_build_duration = time.time() - prompt_build_start_time
            logger.info(f"[{request_id}] Final prompt built in {prompt_build_duration:.3f}s.")

            if final_messages is None or not isinstance(final_messages, list) or not final_messages:
                 logger.error(f"[{request_id}] Failed to build final prompt (returned None or empty list).")
                 raise HTTPException(status_code=500, detail="Internal server error: Failed to construct AI prompt.")

        except Exception as build_e:
            logger.error(f"[{request_id}] Error during final prompt building: {build_e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error: Error building prompt.")
        
# --- 5. 최종 GPT 모델 호출 (GPT Interface) ---
        logger.info(f"[{request_id}] Calling final GPT model: {chosen_model}")
        final_call_start_time = time.time()
        gpt_response_data = None # 초기화
        try:
            gen_config = config.get('generation', {})
            final_temp = gen_config.get('final_response_temperature')
            final_max_tokens = gen_config.get('final_response_max_tokens')

            # 설정값 유효성 확인 강화
            if not isinstance(final_temp, (int, float)):
                logger.warning(f"[{request_id}] Invalid final_response_temperature in config. Using default 0.7.")
                final_temp = 0.7
            if not isinstance(final_max_tokens, int) or final_max_tokens <= 0:
                logger.warning(f"[{request_id}] Invalid final_response_max_tokens in config. Using default 500.")
                final_max_tokens = 500

            # call_gpt_async 호출
            gpt_response_data = await call_gpt_async(
                messages=final_messages,
                model=chosen_model,
                temperature=final_temp,
                max_tokens=final_max_tokens,
                session=session
            )
            final_call_duration = time.time() - final_call_start_time
            logger.info(f"[{request_id}] Final GPT call completed in {final_call_duration:.3f}s.")

        except Exception as final_call_e:
            final_call_duration = time.time() - final_call_start_time
            logger.error(f"[{request_id}] Unexpected error during final GPT call setup or execution: {final_call_e}", exc_info=True)
            # gpt_response_data는 None 상태 유지

        # --- 6. 응답 처리 및 반환 ---
        if gpt_response_data and gpt_response_data.get("choices"):
            assistant_message = gpt_response_data["choices"][0].get("message", {}).get("content", "").strip()
            if not assistant_message:
                logger.warning(f"[{request_id}] Final GPT response content is empty from model {chosen_model}. Returning fallback message.")
                assistant_message = "(죄송합니다, 답변을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요.)"
            else:
                logger.info(f"[{request_id}] Received successful response from {chosen_model}. Length: {len(assistant_message)} chars.")

            # --- 7. 대화 기록 저장 (사용자 입력 + 챗봇 응답) ---
            conversation_handler.add_to_history("user", user_input)
            conversation_handler.add_to_history("assistant", assistant_message)
            logger.info(f"[{request_id}] Updated conversation history. Total turns: {len(conversation_handler.get_history())}")

            # --- [수정됨] 8. 대화 상태 업데이트 (요약) ---
            # 스케줄러에서 병렬로 생성된 요약 결과로 상태 업데이트
            if new_summary is not None:
                logger.info(f"[{request_id}] Updating conversation summary state with the result from scheduler.")
                conversation_handler.update_summary(new_summary)
                logger.debug(f"[{request_id}] New summary length: {len(new_summary)} chars.")
            else:
                logger.warning(f"[{request_id}] Summarization task in scheduler did not return a valid summary. Summary state not updated for this turn.")

            # --- [삭제됨] 기존의 순차적 요약 실행 로직 제거 ---
            # (이전에 여기에 있던 주기적 요약 실행 코드 블록 삭제됨)

            # --- 9. 최종 응답 반환 ---
            end_time = time.time()
            total_request_time = end_time - start_time
            logger.info(f"[{request_id}] Request processing finished successfully in {total_request_time:.3f} seconds.")

            response_payload = {"response": assistant_message}
            # 테스트 모드일 경우 디버그 정보 추가
            if is_test_mode:
                debug_info = {
                    "request_id": request_id,
                    "model_chosen": chosen_model,
                    "complexity_level": complexity_level,
                    "cot_data_present": bool(cot_data),
                    "slots_extracted": extracted_slots if isinstance(extracted_slots, dict) else {},
                    "rag_results_count": len(rag_results),
                    # [수정] 현재 *상태*의 요약 (방금 업데이트됨) 포함
                    "current_summary": conversation_handler.get_summary(),
                    "total_processing_time_ms": int(total_request_time * 1000),
                    # TODO: 토큰 사용량 정보는 gpt_interface 로그에서 확인 또는 call_gpt_async 반환값 활용 필요
                }
                # 스케줄러 자체 소요 시간 등 추가 정보 포함 가능
                debug_info["scheduler_duration_ms"] = int(scheduler_duration * 1000)
                response_payload["debug_info"] = debug_info
                logger.info(f"[{request_id}] Returning response with debug info for test mode.")

            return JSONResponse(content=response_payload)
        else:
            # 최종 GPT 호출 실패 시
            logger.error(f"[{request_id}] Failed to get valid response from the final GPT call using {chosen_model}. Check previous logs for API errors.")
            raise HTTPException(status_code=502, detail="Failed to generate response from AI model (Bad Gateway)")

    # 핸들러 내 전역 예외 처리
    except HTTPException as http_exc:
        logger.warning(f"[{request_id}] Raising HTTPException: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        end_time = time.time()
        total_request_time = end_time - start_time
        logger.error(f"[{request_id}] An unexpected error occurred in '/chat' after {total_request_time:.3f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- 서버 실행 (uvicorn 사용) (변경 없음) ---
if __name__ == "__main__":
    import uvicorn
    try:
        main_config = get_config()
        server_host = main_config.get('server', {}).get('host', '127.0.0.1')
        server_port = main_config.get('server', {}).get('port', 8000)
        server_reload = main_config.get('server', {}).get('reload', True)
        uvicorn_log_level = config.get('logging', {}).get('log_level', 'info').lower()

        logger.info(f"Starting FastAPI server using uvicorn (host={server_host}, port={server_port}, reload={server_reload}, uvicorn_log_level={uvicorn_log_level})...")
        uvicorn.run(
            "chatbot.app:app", # 모듈 경로 확인
            host=server_host,
            port=server_port,
            reload=server_reload,
            log_level=uvicorn_log_level
        )
    except KeyError as ke:
        logger.critical(f"Missing configuration key required for starting server: {ke}")
    except Exception as e:
        logger.critical(f"CRITICAL Error starting uvicorn server: {e}", exc_info=True)