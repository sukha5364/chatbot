# chatbot/app.py (요구사항 반영 최종본: 주기적 요약 실행 로직 추가)

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

# --- 필요한 모듈 임포트 ---
try:
    # app.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .conversation_state import ConversationState
    from .scheduler import run_parallel_tasks
    from .prompt_builder import build_final_prompt
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    from .searcher import RagSearcher
    from . import summarizer # summarizer 모듈 임포트 추가
    logging.info("Required chatbot modules imported successfully in app.py.")
except ImportError as ie:
    logging.error(f"CRITICAL ERROR (app.py): Failed to import required modules: {ie}. Check relative paths and file existence.", exc_info=True)
    # 필수 모듈 임포트 실패 시 앱 실행 불가
    exit(1) # 즉시 종료

# --- 로깅 설정 (DEBUG 고정) ---
# 기본 로깅 설정 (Uvicorn 등 외부 서버에서 설정할 수도 있음)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 추가적으로 특정 로거 레벨 설정 가능
# logger.setLevel(logging.DEBUG)
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
app = FastAPI(title="Decathlon Chatbot API", version="1.0.0") # API 문서 정보 추가

# --- 정적 파일 마운트 ---
# static 디렉토리 경로 계산 (app.py 기준 상위 -> 상위)
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
logger.debug(f"Attempting to mount static files from: {static_dir}")

if not os.path.isdir(static_dir):
    logger.warning(f"Static directory not found at {static_dir}, creating one.")
    try:
        os.makedirs(static_dir, exist_ok=True)
        # 기본 index.html 생성 (없을 경우)
        index_html_default_path = os.path.join(static_dir, 'index.html')
        if not os.path.exists(index_html_default_path):
            with open(index_html_default_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html><html><head><title>Chatbot</title></head><body><h1>Chatbot UI Placeholder</h1><p>Connect your UI here.</p></body></html>")
            logger.info(f"Created placeholder index.html at {index_html_default_path}")
    except Exception as e:
         logger.error(f"Failed to create static directory or placeholder index.html: {e}")

# StaticFiles 마운트 (디렉토리 존재 시)
if os.path.isdir(static_dir):
    try:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Successfully mounted static files from: {static_dir}")
    except Exception as e:
        logger.error(f"Failed to mount static directory {static_dir}: {e}. Static file serving might not work.", exc_info=True)
else:
    logger.error(f"Static directory {static_dir} not found or is not a directory. Static file serving disabled.")


# --- 인메모리 대화 상태 관리 (단일 사용자) ---
try:
    if not ConversationState: # 임포트 실패 체크
         raise ImportError("ConversationState class not available.")
    conversation_handler = ConversationState()
    logger.info("Initialized in-memory conversation handler (suitable for single user/session).")
except Exception as cs_e:
    logger.critical(f"Failed to initialize ConversationState: {cs_e}. Exiting.", exc_info=True)
    exit(1)


# --- 요청 본문 모델 ---
class ChatRequest(BaseModel):
    """/chat 엔드포인트 요청 본문 모델"""
    user_input: str

# --- 공유 aiohttp 세션 및 RAG 검색기 인스턴스 ---
# app.state를 사용하여 애플리케이션 생명주기 동안 객체 공유
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
        # 세션 생성 실패 시 앱 실행 중단 또는 다른 복구 로직 고려
        # raise HTTPException(status_code=500, detail="Failed to initialize HTTP session")

    # 2. RAG 검색기 인스턴스 초기화 (백그라운드에서 실행하여 블로킹 방지)
    logger.info("Initializing RagSearcher instance in background executor...")
    loop = asyncio.get_running_loop()
    try:
        if not RagSearcher: raise ImportError("RagSearcher class not available.")
        # run_in_executor 사용 (기본 스레드 풀 사용)
        # RagSearcher() 초기화 시 파일 I/O, 모델 로딩 등 블로킹 작업 포함될 수 있음
        app.state.rag_searcher_instance = await loop.run_in_executor(None, RagSearcher)

        # 초기화 성공/실패 로깅 강화
        if app.state.rag_searcher_instance and app.state.rag_searcher_instance.index and app.state.rag_searcher_instance.metadata:
            index_size = getattr(app.state.rag_searcher_instance.index, 'ntotal', 'N/A')
            metadata_size = len(app.state.rag_searcher_instance.metadata)
            logger.info(f"RagSearcher instance initialized successfully in background. Index size: {index_size}, Metadata size: {metadata_size}")
        else:
            logger.error("RagSearcher instance initialization failed or incomplete in background! RAG features will be unavailable.")
            app.state.rag_searcher_instance = None # 명시적으로 None 설정
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
    # RAG 인스턴스 등 다른 리소스 정리 필요 시 여기에 추가
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
        # 사용자에게 보여줄 기본 에러 페이지 반환
        return HTMLResponse(content="<html><body><h1>Error: Chatbot UI not found</h1><p>Please check server configuration.</p></body></html>", status_code=404)
    try:
        with open(index_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading or serving index.html: {e}", exc_info=True)
        # 내부 서버 오류 응답
        return HTMLResponse(content="<html><body><h1>Internal Server Error</h1><p>Could not serve the chatbot UI.</p></body></html>", status_code=500)

# --- 챗봇 응답 API 엔드포인트 ---
@app.post("/chat", response_class=JSONResponse, summary="챗봇 응답 생성")
async def handle_chat(chat_request: ChatRequest, request: Request):
    """
    사용자 입력을 받아 챗봇 파이프라인을 실행하고, 최종 응답 및 디버그 정보(테스트 모드 시)를 반환합니다.
    주기적으로 대화 요약을 실행하여 상태를 업데이트합니다.

    Args:
        chat_request (ChatRequest): 사용자 입력을 포함하는 요청 본문.
        request (Request): FastAPI 요청 객체 (헤더 등 접근용).

    Returns:
        JSONResponse: 챗봇 응답 또는 오류 정보를 포함하는 JSON 응답.

    Raises:
        HTTPException: 잘못된 요청(400), 서비스 불가(503), 내부 서버 오류(500) 등.
    """
    start_time = time.time() # 요청 처리 시작 시간 기록
    user_input = chat_request.user_input
    if not user_input or not user_input.strip():
        logger.warning("Received empty or whitespace-only user input.")
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    client_host = request.client.host if request.client else "Unknown"
    # 설정에서 테스트 모드 헤더 이름 읽기 (config는 전역 로드됨)
    test_config = config.get('testing', {})
    test_mode_header_name = test_config.get('test_mode_header', 'X-Test-Mode') # 기본값 설정
    # 헤더 값 비교 시 소문자로 통일
    is_test_mode = request.headers.get(test_mode_header_name, 'false').lower() == 'true'

    request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}" # 간단한 요청 ID 생성
    logger.info(f"[{request_id}] Received POST '/chat' from {client_host}. TestMode={is_test_mode}. Input: '{user_input[:50]}...'")

    # 공유된 세션 및 RAG 검색기 가져오기
    session = app.state.http_session
    rag_searcher = app.state.rag_searcher_instance

    # 세션 유효성 검사
    if not session or session.closed:
        logger.error(f"[{request_id}] AIOHTTP session is not available or closed. Cannot process request.")
        # 503 Service Unavailable 반환
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: HTTP session not ready")

    # RAG 준비 상태 확인 (오류 대신 경고 및 RAG 없이 진행)
    if rag_searcher is None:
        logger.warning(f"[{request_id}] RAG searcher instance is not available. Proceeding without RAG search.")
        # rag_searcher는 None이므로 이후 로직에서 None으로 전달됨

    try:
        # --- 1. 병렬/순차 작업 실행 (Scheduler) ---
        logger.info(f"[{request_id}] Calling scheduler to run parallel/sequential tasks...")
        scheduler_start_time = time.time()
        scheduler_results = await run_parallel_tasks(
            user_input=user_input,
            conversation_state=conversation_handler, # 현재 대화 상태 전달
            session=session,
            rag_searcher=rag_searcher # None 전달 가능
        )
        scheduler_duration = time.time() - scheduler_start_time
        logger.info(f"[{request_id}] Scheduler finished in {scheduler_duration:.3f}s.")
        logger.debug(f"[{request_id}] Scheduler results keys: {list(scheduler_results.keys())}")

        # --- 2. 대화 상태 업데이트 (Slot) ---
        extracted_slots = scheduler_results.get("slots")
        if isinstance(extracted_slots, dict):
            logger.info(f"[{request_id}] Updating conversation state with extracted slots: {list(extracted_slots.keys())}")
            conversation_handler.update_slots(extracted_slots)
            logger.debug(f"[{request_id}] Current slots after update: {conversation_handler.get_slots()}")
        elif extracted_slots is not None: # slots 키는 있지만 dict가 아닌 경우
            logger.warning(f"[{request_id}] Received unexpected data type for slots: {type(extracted_slots)}. Skipping slot update.")

        # --- 3. 최종 프롬프트 구성 요소 준비 ---
        # 기본 라우팅 정보 설정 강화
        default_routing_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
        default_routing_info = {"level": "easy", "model": default_routing_model, "cot_data": None}
        routing_info = scheduler_results.get("routing_info", default_routing_info) # 스케줄러 결과 없으면 기본값 사용

        # 라우팅 결과 유효성 검사 강화
        if not isinstance(routing_info, dict) or not routing_info.get('model'):
            logger.warning(f"[{request_id}] Routing info from scheduler is invalid or missing model ({routing_info}), falling back to default: {default_routing_info}")
            routing_info = default_routing_info # 기본값으로 재설정

        complexity_level = routing_info.get("level", "easy") # 기본값 easy
        chosen_model = routing_info.get("model") # 이제 확실히 존재
        cot_data = routing_info.get("cot_data")
        rag_results = scheduler_results.get("rag_results", []) # 기본값 빈 리스트
        logger.info(f"[{request_id}] Preparing final prompt components: Complexity='{complexity_level}', ChosenModel='{chosen_model}', RAG Results={len(rag_results)}, CoT Data Present={'Yes' if cot_data else 'No'}")

        # --- 4. 최종 프롬프트 생성 (Prompt Builder) ---
        logger.info(f"[{request_id}] Building final prompt...")
        prompt_build_start_time = time.time()
        # build_final_prompt 호출 시 conversation_state 전달
        final_messages = build_final_prompt(
            user_query=user_input,
            conversation_state=conversation_handler, # 현재 슬롯/요약 정보 포함
            rag_results=rag_results,
            cot_data=cot_data
        )
        prompt_build_duration = time.time() - prompt_build_start_time
        logger.info(f"[{request_id}] Final prompt built in {prompt_build_duration:.3f}s.")

        # final_messages 유효성 검사
        if final_messages is None or not isinstance(final_messages, list) or not final_messages:
            logger.error(f"[{request_id}] Failed to build final prompt (returned None or empty list). Check prompt_builder logic and configuration.")
            raise HTTPException(status_code=500, detail="Internal server error: Failed to construct AI prompt.")

        # --- 5. 최종 GPT 모델 호출 (GPT Interface) ---
        logger.info(f"[{request_id}] Calling final GPT model: {chosen_model}")
        final_call_start_time = time.time()
        gpt_response_data = None # 초기화
        try:
            # generation 설정 읽기
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
                # 필요 시 추가 파라미터 전달 가능
            )
            final_call_duration = time.time() - final_call_start_time
            logger.info(f"[{request_id}] Final GPT call completed in {final_call_duration:.3f}s.")

        except Exception as final_call_e:
            # call_gpt_async 내부에서 예외 발생 시 (거의 발생 안 함, 내부에서 처리됨)
            final_call_duration = time.time() - final_call_start_time
            logger.error(f"[{request_id}] Unexpected error during final GPT call setup or execution: {final_call_e}", exc_info=True)
            # gpt_response_data는 None 상태 유지

        # --- 6. 응답 처리 및 반환 ---
        if gpt_response_data and gpt_response_data.get("choices"):
            # assistant 메시지 추출 (None이거나 빈 문자열일 경우 처리)
            assistant_message = gpt_response_data["choices"][0].get("message", {}).get("content", "").strip()
            if not assistant_message:
                logger.warning(f"[{request_id}] Final GPT response content is empty from model {chosen_model}. Returning fallback message.")
                # 사용자에게 전달할 대체 메시지 정의
                assistant_message = "(죄송합니다, 답변을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요.)"
            else:
                 logger.info(f"[{request_id}] Received successful response from {chosen_model}. Length: {len(assistant_message)} chars.")

            # --- 7. 대화 기록 저장 (사용자 입력 + 챗봇 응답) ---
            conversation_handler.add_to_history("user", user_input)
            conversation_handler.add_to_history("assistant", assistant_message)
            logger.info(f"[{request_id}] Updated conversation history. Total turns: {len(conversation_handler.get_history())}")

            # --- 8. 주기적 대화 요약 실행 [수정됨] ---
            try:
                # 요약 설정 읽기
                summarization_cfg = config.get('tasks', {}).get('summarization', {})
                summarizer_enabled = summarization_cfg.get('enabled', False)
                summarize_every_n = summarization_cfg.get('summarize_every_n_turns', 0) # 0 또는 음수면 비활성화

                # 현재 총 턴 수 (user + assistant 합쳐서 카운트)
                current_total_turns = len(conversation_handler.get_history())

                # 요약 실행 조건 확인
                if summarizer_enabled and isinstance(summarize_every_n, int) and summarize_every_n > 0 and current_total_turns > 0 and current_total_turns % summarize_every_n == 0:
                    logger.info(f"[{request_id}] Triggering summarization (Turns: {current_total_turns}, Trigger every: {summarize_every_n})")
                    summary_start_time = time.time()
                    try:
                        # summarizer 모듈 임포트 재확인 (코드 상단에서 했지만 안전 차원)
                        if not summarizer: raise ImportError("Summarizer module is not available.")

                        # 이전 요약 가져오기
                        previous_summary = conversation_handler.get_summary()
                        logger.debug(f"[{request_id}] Previous summary length: {len(previous_summary) if previous_summary else 0} chars.")

                        # 요약 함수 호출 (전체 히스토리와 이전 요약 전달)
                        new_summary = await summarizer.summarize_conversation_async(
                            history=conversation_handler.get_history(),
                            previous_summary=previous_summary, # 이전 요약 전달
                            session=session
                        )
                        summary_duration = time.time() - summary_start_time

                        if new_summary:
                            conversation_handler.update_summary(new_summary)
                            logger.info(f"[{request_id}] Conversation summary updated successfully in {summary_duration:.3f}s. New length: {len(new_summary)} chars.")
                        else:
                            # 요약 API 호출은 성공했으나 빈 결과 반환
                            logger.warning(f"[{request_id}] Summarization attempt did not return a valid summary text (took {summary_duration:.3f}s).")
                    except ImportError:
                         logger.error(f"[{request_id}] Summarizer module not imported correctly, skipping summarization.")
                    except Exception as summary_e:
                        summary_duration = time.time() - summary_start_time
                        # 요약 실패는 전체 응답 실패로 이어지지 않도록 함
                        logger.error(f"[{request_id}] Error during conversation summarization call (took {summary_duration:.3f}s): {summary_e}", exc_info=True)
                else:
                    # 요약 조건 미충족 또는 비활성화 시 로깅 (DEBUG 레벨)
                    if summarizer_enabled and summarize_every_n > 0:
                        logger.debug(f"[{request_id}] Skipping summarization (Turns: {current_total_turns}, Trigger every: {summarize_every_n})")
                    # else: logger.debug(f"[{request_id}] Summarization is disabled or trigger interval is not positive.")

            except Exception as outer_summary_e:
                # 요약 로직 자체의 오류 (설정 읽기 등)
                logger.error(f"[{request_id}] Error in summarization handling logic: {outer_summary_e}", exc_info=True)
            # --- 요약 로직 끝 ---

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
                    "slots_extracted": extracted_slots if isinstance(extracted_slots, dict) else {}, # 타입 확인 후 저장
                    "rag_results_count": len(rag_results),
                    "current_summary": conversation_handler.get_summary(), # 현재 요약 상태 포함
                    "total_processing_time_ms": int(total_request_time * 1000), # 밀리초 단위 처리 시간
                    # TODO: 토큰 사용량 정보 추가 필요 (call_gpt_async 반환값 또는 gpt_interface 로그 활용)
                }
                response_payload["debug_info"] = debug_info
                logger.info(f"[{request_id}] Returning response with debug info for test mode.")

            return JSONResponse(content=response_payload)
        else:
            # 최종 GPT 호출 실패 시 (call_gpt_async가 None 반환)
            # call_gpt_async 내부에서 이미 상세 에러 로깅됨
            logger.error(f"[{request_id}] Failed to get valid response from the final GPT call using {chosen_model}. Check previous logs for API errors.")
            raise HTTPException(status_code=502, detail="Failed to generate response from AI model (Bad Gateway)") # 502 Bad Gateway가 더 적절할 수 있음

    # 핸들러 내 전역 예외 처리
    except HTTPException as http_exc:
        # FastAPI가 자동으로 처리하도록 다시 raise
        logger.warning(f"[{request_id}] Raising HTTPException: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        # 예상치 못한 내부 서버 오류
        end_time = time.time()
        total_request_time = end_time - start_time
        logger.error(f"[{request_id}] An unexpected error occurred in '/chat' after {total_request_time:.3f}s: {e}", exc_info=True)
        # 사용자에게는 일반적인 오류 메시지 반환
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- 서버 실행 (uvicorn 사용) ---
if __name__ == "__main__":
    import uvicorn
    # uvicorn 실행 전 설정 로드 재확인 (선택 사항)
    try:
        main_config = get_config()
        # config에서 호스트, 포트, 리로드 설정 읽기 (선택 사항)
        server_host = main_config.get('server', {}).get('host', '127.0.0.1')
        server_port = main_config.get('server', {}).get('port', 8000)
        server_reload = main_config.get('server', {}).get('reload', True) # 개발 중에는 True 권장

        # 로그 레벨은 코드에서 DEBUG로 고정했으므로 uvicorn 설정은 참고용
        uvicorn_log_level = config.get('logging', {}).get('log_level', 'info').lower()

        logger.info(f"Starting FastAPI server using uvicorn (host={server_host}, port={server_port}, reload={server_reload}, uvicorn_log_level={uvicorn_log_level})...")
        uvicorn.run(
            # "chatbot.app:app" - chatbot 패키지 내 app 모듈의 app 객체 지정
            "chatbot.app:app",
            host=server_host,
            port=server_port,
            reload=server_reload, # 개발 완료 후 False로 변경
            log_level=uvicorn_log_level # Uvicorn 자체 로그 레벨
        )
    except KeyError as ke:
         logger.critical(f"Missing configuration key required for starting server: {ke}")
    except Exception as e:
        logger.critical(f"CRITICAL Error starting uvicorn server: {e}", exc_info=True)