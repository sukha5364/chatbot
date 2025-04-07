# chatbot/app.py (오류 수정 및 리팩토링 최종본)

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
    from .conversation_state import ConversationState
    from .scheduler import run_parallel_tasks
    from .prompt_builder import build_final_prompt
    from .gpt_interface import call_gpt_async
    from .config_loader import get_config
    from .searcher import RagSearcher
    # [신규] summarizer 모듈 임포트
    from . import summarizer
except ImportError as ie:
    print(f"ERROR (app.py): Failed to import modules: {ie}. Check relative paths and ensure all files exist.")
    # 필수 모듈 실패 시 앱 실행 불가
    exit(1) # 또는 적절한 오류 처리

# --- 설정 및 로거 초기화 ---
try:
    config = get_config()
except Exception as e:
    print(f"CRITICAL ERROR (app.py): Failed to load configuration: {e}. Exiting.")
    exit(1)

logger = logging.getLogger(__name__)
# 설정 파일에서 로그 레벨 읽기 (실패 시 INFO)
log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("FastAPI application starting...")
logger.info(f"Log level set to: {logging.getLevelName(logger.getEffectiveLevel())}") # 실제 적용된 레벨 로깅

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- 정적 파일 마운트 ---
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
# print(f"DEBUG (app.py): Static directory path: {static_dir}") # 경로 확인용
if not os.path.isdir(static_dir):
     logger.warning(f"Static directory not found at {static_dir}, creating one.")
     os.makedirs(static_dir, exist_ok=True)
     index_html_default_path = os.path.join(static_dir, 'index.html')
     if not os.path.exists(index_html_default_path):
         try:
             # 간단한 기본 HTML 생성
             with open(index_html_default_path, 'w', encoding='utf-8') as f:
                  f.write("<!DOCTYPE html><html><head><title>Chatbot</title></head><body><h1>Chatbot UI Placeholder</h1></body></html>")
             logger.info(f"Created placeholder index.html at {index_html_default_path}")
         except Exception as e:
             logger.error(f"Failed to create placeholder index.html: {e}")

# StaticFiles 마운트 (오류 발생 가능성 있음, 방어적으로 처리)
if os.path.isdir(static_dir):
    try:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Mounted static files from: {static_dir}")
    except Exception as e:
        logger.error(f"Failed to mount static directory {static_dir}: {e}")
        # 정적 파일 서빙 없이 계속 진행하거나, 앱 실행 중단 결정 가능
else:
     logger.error(f"Static directory {static_dir} not found or is not a directory. Static file serving disabled.")


# --- 인메모리 대화 상태 관리 ---
try:
    # ConversationState 임포트 확인
    if not ConversationState:
         raise ImportError("ConversationState class not available.")
    conversation_handler = ConversationState()
    logger.info("Initialized in-memory conversation handler (single user).")
except Exception as cs_e:
     logger.critical(f"Failed to initialize ConversationState: {cs_e}. Exiting.")
     exit(1)


# --- 요청 본문 모델 ---
class ChatRequest(BaseModel):
    user_input: str

# --- 공유 aiohttp 세션 및 RAG 검색기 인스턴스 ---
app.state.http_session: Optional[aiohttp.ClientSession] = None
app.state.rag_searcher_instance: Optional[RagSearcher] = None

# --- Startup / Shutdown 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 aiohttp 세션 및 RAG 검색기 인스턴스 생성."""
    try:
        app.state.http_session = aiohttp.ClientSession()
        logger.info("AIOHTTP ClientSession created for application lifetime.")
    except Exception as session_e:
        logger.error(f"Failed to create AIOHTTP ClientSession: {session_e}")
        # 세션 생성 실패 시 앱 실행 중단 또는 다른 처리 필요
        # raise HTTPException(status_code=500, detail="Failed to initialize HTTP session")

    # RAG 검색기 인스턴스 생성 (오류 발생해도 서버는 계속 실행되도록)
    logger.info("Initializing RagSearcher instance in background thread...")
    loop = asyncio.get_running_loop()
    try:
        # RagSearcher 임포트 확인
        if not RagSearcher:
             raise ImportError("RagSearcher class not available.")
        # run_in_executor 사용하여 블로킹 방지
        app.state.rag_searcher_instance = await loop.run_in_executor(None, RagSearcher)

        # 초기화 성공/실패 로깅 강화
        if app.state.rag_searcher_instance and app.state.rag_searcher_instance.index and app.state.rag_searcher_instance.metadata:
             logger.info(f"RagSearcher instance initialized successfully in background. Index size: {app.state.rag_searcher_instance.index.ntotal}")
        else:
             logger.error("RagSearcher instance initialization failed in background! RAG features will be unavailable.")
             app.state.rag_searcher_instance = None # 명시적으로 None 설정
    except ImportError:
         logger.error("RagSearcher class not imported. RAG features will be unavailable.")
         app.state.rag_searcher_instance = None
    except Exception as e:
         logger.error(f"Error during RagSearcher initialization in startup_event: {e}", exc_info=True)
         app.state.rag_searcher_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 aiohttp 세션 종료."""
    if app.state.http_session and not app.state.http_session.closed:
        await app.state.http_session.close()
        logger.info("AIOHTTP ClientSession closed.")
    app.state.http_session = None
    logger.info("FastAPI application shutdown.")


# --- 루트 경로 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info(f"GET request received for root path '/' from {request.client.host if request.client else 'Unknown'}")
    index_html_path = os.path.join(static_dir, 'index.html')
    if not os.path.exists(index_html_path):
         logger.error(f"index.html not found at {index_html_path}")
         # 사용자에게 보여줄 기본 에러 페이지 반환 고려
         return HTMLResponse(content="<html><body><h1>Chatbot UI not found</h1></body></html>", status_code=404)
    try:
        with open(index_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
         logger.error(f"Error reading index.html: {e}", exc_info=True)
         return HTMLResponse(content="<html><body><h1>Internal Server Error serving UI</h1></body></html>", status_code=500)


# --- 챗봇 응답 API 엔드포인트 ---
@app.post("/chat", response_class=JSONResponse)
async def handle_chat(chat_request: ChatRequest, request: Request):
    """사용자 입력을 받아 챗봇 응답을 반환합니다."""
    start_time = time.time()
    user_input = chat_request.user_input
    if not user_input or not user_input.strip():
         logger.warning("Received empty user input.")
         raise HTTPException(status_code=400, detail="User input cannot be empty.")

    client_host = request.client.host if request.client else "Unknown"
    # config는 전역에서 로드됨 (또는 여기서 get_config() 호출)
    test_config = config.get('testing', {})
    test_mode_header = test_config.get('test_mode_header', 'X-Test-Mode')
    is_test_mode = request.headers.get(test_mode_header, 'false').lower() == 'true'

    logger.info(f"POST '/chat' from {client_host}. TestMode={is_test_mode}. Input: '{user_input[:50]}...'")

    session = app.state.http_session
    rag_searcher = app.state.rag_searcher_instance

    if not session or session.closed:
         logger.error("AIOHTTP session is not available or closed.")
         raise HTTPException(status_code=503, detail="Service temporarily unavailable: Session not ready") # 503 Service Unavailable

    # RAG 준비 상태 확인 (오류 대신 경고만)
    if rag_searcher is None:
         logger.warning("RAG searcher is not available. Proceeding without RAG.")
         # rag_searcher는 이미 None이므로 별도 처리 불필요

    try:
        # 1. 병렬/순차 작업 실행
        logger.info("Calling scheduler to run tasks...")
        scheduler_results = await run_parallel_tasks(
            user_input=user_input,
            conversation_state=conversation_handler,
            session=session,
            rag_searcher=rag_searcher # None 전달 가능
        )
        logger.info("Scheduler finished.")

        # 2. 대화 상태 업데이트 (Slot)
        extracted_slots = scheduler_results.get("slots")
        if extracted_slots and isinstance(extracted_slots, dict):
            logger.info(f"Updating conversation state with extracted slots: {list(extracted_slots.keys())}")
            conversation_handler.update_slots(extracted_slots)
        elif extracted_slots is not None: # slots 키는 있지만 dict가 아닌 경우
             logger.warning(f"Received unexpected data type for slots: {type(extracted_slots)}. Skipping update.")


        # 3. 최종 프롬프트 구성 요소 준비
        # 기본 라우팅 정보 설정 강화
        default_routing_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
        default_routing = {"level": "easy", "model": default_routing_model, "cot_data": None}
        routing_info = scheduler_results.get("routing_info", default_routing)
        # 라우팅 결과 유효성 검사 강화
        if not isinstance(routing_info, dict) or not routing_info.get('model'):
             logger.warning(f"Routing info is invalid or missing model ({routing_info}), falling back to default: {default_routing_model}")
             routing_info = default_routing # 기본값으로 재설정

        complexity_level = routing_info.get("level", "easy")
        chosen_model = routing_info.get("model") # 이제 확실히 존재
        cot_data = routing_info.get("cot_data")
        rag_results = scheduler_results.get("rag_results", []) # 기본값 빈 리스트
        logger.info(f"Preparing final prompt. Complexity='{complexity_level}', ChosenModel='{chosen_model}', RAG={len(rag_results)}, CoTData={'Yes' if cot_data else 'No'}")

        # 4. 최종 프롬프트 생성
        logger.info("Building final prompt...")
        final_messages = build_final_prompt(
            user_query=user_input, conversation_state=conversation_handler,
            rag_results=rag_results, cot_data=cot_data
        )
        # build_final_prompt가 None 반환 가능성 체크 (설정 오류 등)
        if final_messages is None:
             logger.error("Failed to build final prompt due to configuration errors.")
             raise HTTPException(status_code=500, detail="Internal server error: Prompt generation failed.")

        logger.info("Final prompt built.")

        # 5. 최종 GPT 모델 호출 - 온도, 최대 토큰 명시적 전달
        logger.info(f"Calling final GPT model: {chosen_model}")
        gpt_response_data = None # 초기화
        try:
            # generation 설정 읽기
            gen_config = config.get('generation', {})
            final_temp = gen_config.get('final_response_temperature')
            final_max_tokens = gen_config.get('final_response_max_tokens')

            # 설정값 유효성 확인
            if not isinstance(final_temp, (int, float)) or not isinstance(final_max_tokens, int):
                logger.error("Final response temperature or max_tokens not found or invalid in generation config.")
                # 비상용 기본값 사용 (또는 에러 발생)
                final_temp = 0.7
                final_max_tokens = 500
                logger.warning(f"Using fallback temperature={final_temp}, max_tokens={final_max_tokens} for final response.")

            # call_gpt_async 호출
            gpt_response_data = await call_gpt_async(
                messages=final_messages,
                model=chosen_model,
                temperature=final_temp, # 명시적 전달
                max_tokens=final_max_tokens, # 명시적 전달
                session=session
            )
        except Exception as final_call_e:
             logger.error(f"Error during final GPT call setup or execution: {final_call_e}", exc_info=True)
             # gpt_response_data는 None 유지

        # 6. 응답 처리 및 반환
        if gpt_response_data and gpt_response_data.get("choices"):
            # assistant 메시지 추출 (None일 경우 빈 문자열)
            assistant_message = gpt_response_data["choices"][0].get("message", {}).get("content", "")
            if not assistant_message.strip():
                 logger.warning(f"Final GPT response content is empty from model {chosen_model}.")
                 # 빈 응답도 기록하거나, 사용자에게 다른 메시지 전달 고려
                 assistant_message = "(응답 생성에 실패했습니다. 다시 시도해주세요.)" # 예시 fallback 메시지

            logger.info(f"Received response from {chosen_model}. Length: {len(assistant_message)}")

            # 대화 기록 저장
            conversation_handler.add_to_history("user", user_input)
            conversation_handler.add_to_history("assistant", assistant_message)
            logger.info("Updated conversation history.")

            # --- [신규] 대화 요약 로직 ---
            try:
                # summarizer 모듈 임포트 확인
                if not summarizer:
                     raise ImportError("Summarizer module is not available.")

                summarization_cfg = config.get('tasks', {}).get('summarization', {})
                summarizer_enabled = summarization_cfg.get('enabled', False)
                trigger_turn = summarization_cfg.get('trigger_turn_count', 6) # 기본값 6

                # history 길이는 user + assistant 턴 수의 합
                current_total_turns = len(conversation_handler.get_history())

                if summarizer_enabled and isinstance(trigger_turn, int) and trigger_turn > 0 and current_total_turns >= trigger_turn:
                    logger.info(f"Triggering summarization (current turns: {current_total_turns} >= trigger: {trigger_turn})")
                    # 요약 실행 (백그라운드 작업 아님, 응답 시간에 포함됨)
                    try:
                        new_summary = await summarizer.summarize_conversation_async(
                            conversation_handler.get_history(), # 전체 히스토리 전달
                            session=session
                        )
                        if new_summary:
                            conversation_handler.update_summary(new_summary)
                            logger.info("Conversation summary updated successfully.")
                        else:
                            logger.warning("Summarization attempt did not return a valid summary.")
                    except Exception as summary_e:
                        # 요약 실패는 전체 응답 실패로 이어지지 않도록 함
                        logger.error(f"Error during conversation summarization call: {summary_e}", exc_info=True)
                else:
                    # 요약 조건 미충족 또는 비활성화 시 로깅
                    if summarizer_enabled:
                         logger.debug(f"Skipping summarization (current turns: {current_total_turns}, trigger: {trigger_turn})")
                    # else: # 비활성화 시 로깅은 너무 빈번할 수 있음
                    #      logger.debug("Summarization is disabled.")

            except ImportError:
                 logger.error("Summarizer module not imported correctly, skipping summarization.")
            except Exception as outer_summary_e:
                 # 요약 로직 자체의 오류 (설정 읽기 등)
                 logger.error(f"Error in summarization handling logic: {outer_summary_e}", exc_info=True)
            # --- 요약 로직 끝 ---


            # 최종 응답 반환 준비
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Request processing finished successfully in {total_time:.3f} seconds.")

            response_payload = {"response": assistant_message}
            if is_test_mode:
                 debug_info = {
                     "model_chosen": chosen_model,
                     "complexity_level": complexity_level,
                     "cot_data_present": bool(cot_data),
                     "slots_extracted": extracted_slots if isinstance(extracted_slots, dict) else {}, # 타입 확인
                     "rag_results_count": len(rag_results),
                     "current_summary": conversation_handler.get_summary() # 요약 내용 포함
                 }
                 response_payload["debug_info"] = debug_info
                 logger.info("Returning response with debug info for test mode.")

            return JSONResponse(content=response_payload)
        else:
            # 최종 GPT 호출 실패 시
            logger.error(f"Failed to get valid response from the final GPT call using {chosen_model}.")
            raise HTTPException(status_code=500, detail="Failed to generate response from AI model")

    # 핸들러 내 전역 예외 처리
    except HTTPException as http_exc:
        # 이미 로깅된 HTTP 예외는 다시 raise
        raise http_exc
    except Exception as e:
        # 예상치 못한 내부 서버 오류
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"An unexpected error in '/chat' after {total_time:.3f}s: {e}", exc_info=True)
        # 사용자에게는 일반적인 오류 메시지 반환
        raise HTTPException(status_code=500, detail="Internal server error.")


# --- 서버 실행 ---
if __name__ == "__main__":
    import uvicorn
    # uvicorn 실행 전 설정 로드 재확인
    try:
        main_config = get_config()
        server_log_level = main_config.get('logging', {}).get('log_level', 'info').lower()
        logger.info(f"Starting FastAPI server using uvicorn (host=127.0.0.1, port=8000, reload=True, log_level={server_log_level})...")
        uvicorn.run(
            # "chatbot.app:app" -> 현재 파일이 app.py 이므로 "app:app" 사용 가능
            # 단, uvicorn을 프로젝트 루트에서 실행 시 "chatbot.app:app" 사용
            "chatbot.app:app",
            host="127.0.0.1",
            port=8000,
            reload=True, # 개발 중에는 True, 배포 시에는 False
            log_level=server_log_level
        )
    except Exception as e:
         print(f"Error starting uvicorn server: {e}")