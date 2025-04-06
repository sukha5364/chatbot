# chatbot/app.py

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
from .conversation_state import ConversationState
from .scheduler import run_parallel_tasks
from .prompt_builder import build_final_prompt
from .gpt_interface import call_gpt_async
from .config_loader import get_config
from .searcher import RagSearcher # [수정] RagSearcher 클래스 임포트

# --- 설정 및 로거 초기화 ---
config = get_config()
logger = logging.getLogger(__name__)
log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("FastAPI application starting...")

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- 정적 파일 마운트 ---
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
# ... (Static 파일 디렉토리 생성 및 마운트 로직) ...
if not os.path.isdir(static_dir):
     logger.warning(f"Static directory not found at {static_dir}, creating one.")
     os.makedirs(static_dir, exist_ok=True)
     index_html_default_path = os.path.join(static_dir, 'index.html')
     if not os.path.exists(index_html_default_path):
          try:
              with open(index_html_default_path, 'w', encoding='utf-8') as f: f.write("<html><body>Chatbot UI</body></html>")
              logger.info(f"Created placeholder index.html at {index_html_default_path}")
          except Exception as e: logger.error(f"Failed to create placeholder index.html: {e}")
if os.path.isdir(static_dir):
    try: app.mount("/static", StaticFiles(directory=static_dir), name="static"); logger.info(f"Mounted static files from: {static_dir}")
    except Exception as e: logger.error(f"Failed to mount static directory {static_dir}: {e}")
else: logger.error(f"Static directory {static_dir} not found. Static file serving disabled.")

# --- 인메모리 대화 상태 관리 ---
conversation_handler = ConversationState()
logger.info("Initialized in-memory conversation handler (single user).")

# --- 요청 본문 모델 ---
class ChatRequest(BaseModel):
    user_input: str

# --- 공유 aiohttp 세션 및 [신규] RAG 검색기 인스턴스 ---
# app.state를 사용하여 FastAPI 애플리케이션 상태 관리
app.state.http_session: Optional[aiohttp.ClientSession] = None
app.state.rag_searcher_instance: Optional[RagSearcher] = None # [신규] RAG 인스턴스 저장 공간

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 aiohttp 세션 및 RAG 검색기 인스턴스 생성."""
    # 1. aiohttp 세션 생성
    app.state.http_session = aiohttp.ClientSession()
    logger.info("AIOHTTP ClientSession created for application lifetime.")

    # 2. RAG 검색기 인스턴스 생성 (백그라운드 스레드에서)
    logger.info("Initializing RagSearcher instance in background thread...")
    loop = asyncio.get_running_loop()
    try:
        # RagSearcher() 인스턴스화 자체 (내부 _load_resources 포함)가 블로킹 가능하므로 executor 사용
        app.state.rag_searcher_instance = await loop.run_in_executor(None, RagSearcher)
        # 초기화 성공/실패 로깅
        if app.state.rag_searcher_instance and app.state.rag_searcher_instance.index and app.state.rag_searcher_instance.metadata:
             logger.info("RagSearcher instance initialized successfully in background.")
        else:
             logger.error("RagSearcher instance initialization failed in background!")
             app.state.rag_searcher_instance = None # 실패 시 None으로 설정
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
    # RagSearcher 관련 정리 작업은 필요 없음 (파일 핸들 등은 __del__ 이나 컨텍스트 매니저 필요 시 추가)
    logger.info("FastAPI application shutdown.")


# --- 루트 경로 (HTML 페이지 반환) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # (이전 코드와 동일)
    logger.info(f"GET request received for root path '/' from {request.client.host if request.client else 'Unknown'}")
    index_html_path = os.path.join(static_dir, 'index.html')
    if not os.path.exists(index_html_path):
          logger.error(f"index.html not found at {index_html_path}")
          raise HTTPException(status_code=404, detail="index.html not found")
    try:
        with open(index_html_path, "r", encoding="utf-8") as f: html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
         logger.error(f"Error reading index.html: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Error serving UI")


# --- 챗봇 응답 API 엔드포인트 (수정됨) ---
@app.post("/chat", response_class=JSONResponse)
async def handle_chat(chat_request: ChatRequest, request: Request):
    """사용자 입력을 받아 챗봇 응답을 반환합니다."""
    start_time = time.time()
    user_input = chat_request.user_input
    client_host = request.client.host if request.client else "Unknown"
    test_config = config.get('testing', {})
    test_mode_header = test_config.get('test_mode_header', 'X-Test-Mode')
    is_test_mode = request.headers.get(test_mode_header, 'false').lower() == 'true'

    logger.info(f"POST '/chat' from {client_host}. TestMode={is_test_mode}. Input: '{user_input[:50]}...'")

    # [수정] 공유 세션 및 RAG 인스턴스 가져오기 (app.state 사용)
    session = app.state.http_session
    rag_searcher = app.state.rag_searcher_instance

    if not session or session.closed:
         logger.error("AIOHTTP session is not available or closed.")
         raise HTTPException(status_code=500, detail="Internal server error: Session not available")

    # [수정] RAG 검색기 준비 상태 확인
    if rag_searcher is None or rag_searcher.index is None or not rag_searcher.metadata:
         logger.warning("RAG searcher is not ready (still initializing or failed). Proceeding without RAG.")
         # RAG 없이 진행하거나, 사용자에게 잠시 후 시도하라는 메시지 반환 등 정책 결정 필요
         # 여기서는 RAG 결과가 비어있는 상태로 진행
         rag_searcher = None # RAG 기능 비활성화 의미로 None 전달

    try:
        # 1. 병렬/순차 작업 실행 ([수정] rag_searcher 전달)
        logger.info("Calling scheduler to run tasks...")
        scheduler_results = await run_parallel_tasks(
            user_input=user_input,
            conversation_state=conversation_handler,
            session=session,
            rag_searcher=rag_searcher # 생성된 RAG 검색기 인스턴스 전달
        )
        logger.info("Scheduler finished.")

        # 2. 대화 상태 업데이트 (Slot)
        # ... (기존 코드 유지) ...
        extracted_slots = scheduler_results.get("slots")
        if extracted_slots:
            logger.info(f"Updating conversation state with extracted slots: {list(extracted_slots.keys())}")
            conversation_handler.update_slots(extracted_slots)

        # 3. 최종 프롬프트 구성 요소 준비 (라우터 결과 사용)
        # ... (기존 코드 유지) ...
        default_routing_model = config.get('model_router', {}).get('routing_map', {}).get('easy', 'gpt-3.5-turbo')
        default_routing = {"level": "easy", "model": default_routing_model, "cot_data": None}
        routing_info = scheduler_results.get("routing_info", default_routing)
        complexity_level = routing_info.get("level", "easy")
        chosen_model = routing_info.get("model", default_routing_model)
        cot_data = routing_info.get("cot_data")
        rag_results = scheduler_results.get("rag_results", []) # 스케줄러가 RAG 결과 반환
        logger.info(f"Preparing final prompt. Complexity='{complexity_level}', ChosenModel='{chosen_model}', RAG={len(rag_results)}, CoTData={'Yes' if cot_data else 'No'}")

        # 4. 최종 프롬프트 생성
        # ... (기존 코드 유지) ...
        logger.info("Building final prompt...")
        final_messages = build_final_prompt(
            user_query=user_input, conversation_state=conversation_handler,
            rag_results=rag_results, cot_data=cot_data
        )
        logger.info("Final prompt built.")

        # 5. 최종 GPT 모델 호출
        # ... (기존 코드 유지) ...
        logger.info(f"Calling final GPT model: {chosen_model}")
        gpt_response_data = await call_gpt_async(
            messages=final_messages, model=chosen_model, session=session
            # temperature, max_tokens는 call_gpt_async 내부에서 config 참조
        )

        # 6. 응답 처리 및 반환
        # ... (테스트 모드 응답 포함하여 기존 코드 유지) ...
        if gpt_response_data and gpt_response_data.get("choices"):
            assistant_message = gpt_response_data["choices"][0].get("message", {}).get("content", "")
            logger.info(f"Received successful response from {chosen_model}. Length: {len(assistant_message)}")
            conversation_handler.add_to_history("user", user_input)
            conversation_handler.add_to_history("assistant", assistant_message)
            logger.info("Updated conversation history.")
            # TODO: Summarization logic
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Request processing finished successfully in {total_time:.3f} seconds.")

            response_payload = {"response": assistant_message}
            if is_test_mode:
                 debug_info = {
                     "model_chosen": chosen_model, "complexity_level": complexity_level,
                     "cot_data_present": bool(cot_data),
                     "slots_extracted": extracted_slots if extracted_slots is not None else {},
                     "rag_results_count": len(rag_results),
                 }
                 response_payload["debug_info"] = debug_info
                 logger.info("Returning response with debug info for test mode.")
            return JSONResponse(content=response_payload)
        else:
            logger.error(f"Failed to get valid response from the final GPT call using {chosen_model}.")
            raise HTTPException(status_code=500, detail="Failed to generate response from AI model")

    except HTTPException as http_exc:
        logger.warning(f"HTTP Exception occurred: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"An unexpected error in '/chat' after {total_time:.3f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error.")


# --- 서버 실행 (개발용) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server using uvicorn for development...")
    server_log_level = config.get('logging', {}).get('log_level', 'info').lower()
    uvicorn.run("chatbot.app:app", host="127.0.0.1", port=8000, reload=True, log_level=server_log_level)