# chatbot/app.py

import asyncio
import time
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiohttp # 비동기 HTTP 요청

# --- 필요한 모듈 임포트 ---
from .conversation_state import ConversationState
from .scheduler import run_parallel_tasks
from .prompt_builder import build_final_prompt, DEFAULT_SYSTEM_PROMPT
from .gpt_interface import call_gpt_async

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- 정적 파일 마운트 (static 폴더의 HTML, CSS, JS 등) ---
# 'static' 이름의 디렉토리를 '/static' 경로에 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 인메모리 대화 상태 관리 (단일 사용자 가정) ---
# 실제 서비스에서는 세션별 또는 사용자별 상태 관리가 필요 (예: Redis, DB 사용)
conversation_handler = ConversationState()

# --- 요청 본문 모델 ---
class ChatRequest(BaseModel):
    user_input: str

# --- 공유 aiohttp 세션 (애플리케이션 생명 주기 동안 유지) ---
app.state.http_session = None

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 aiohttp 세션 생성."""
    app.state.http_session = aiohttp.ClientSession()
    print("AIOHTTP ClientSession created.")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 aiohttp 세션 종료."""
    if app.state.http_session:
        await app.state.http_session.close()
        print("AIOHTTP ClientSession closed.")

# --- 루트 경로 (HTML 페이지 반환) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """기본 HTML 인터페이스를 제공합니다."""
    # static/index.html 파일을 읽어서 반환
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

# --- 챗봇 응답 API 엔드포인트 ---
@app.post("/chat", response_class=JSONResponse)
async def handle_chat(chat_request: ChatRequest):
    """사용자 입력을 받아 챗봇 응답을 반환합니다."""
    start_time = time.time()
    user_input = chat_request.user_input
    print(f"\nReceived user input: {user_input}")

    # 공유 세션 가져오기
    session = app.state.http_session
    if not session:
        raise HTTPException(status_code=500, detail="HTTP session not available")

    try:
        # 1. 병렬 작업 실행 (Slot 추출, 모델 라우팅, RAG 검색)
        scheduler_results = await run_parallel_tasks(user_input, conversation_handler, session)

        # 2. 대화 상태 업데이트 (Slot)
        extracted_slots = scheduler_results.get("slots")
        if extracted_slots:
            conversation_handler.update_slots(extracted_slots)
            print(f"Updated slots: {conversation_handler.get_slots()}")

        # 3. 최종 프롬프트 구성 요소 준비
        chosen_model = scheduler_results.get("chosen_model", "gpt-3.5-turbo")
        rag_results = scheduler_results.get("rag_results", [])

        # 모델 라우팅 결과에 따라 CoT, Few-shot 적용 여부 결정 (예시)
        use_cot = True if chosen_model == "gpt-4" else False
        # TODO: Few-shot 예제 로딩 로직 추가 (필요시)
        few_shot_examples = None # 현재는 사용 안 함

        # 4. 최종 프롬프트 생성
        final_messages = build_final_prompt(
            system_prompt_base=DEFAULT_SYSTEM_PROMPT,
            user_query=user_input,
            conversation_state=conversation_handler,
            rag_results=rag_results,
            use_cot=use_cot,
            few_shot_examples=few_shot_examples
        )

        # 5. 최종 GPT 모델 호출
        print(f"Calling final GPT model: {chosen_model}")
        gpt_response_data = await call_gpt_async(
            messages=final_messages,
            model=chosen_model,
            temperature=0.7, # 일반적인 응답 온도
            max_tokens=500, # 응답 길이 제한 (프롬프트 빌더와 연계)
            session=session
        )

        # 6. 응답 처리 및 반환
        if gpt_response_data and gpt_response_data.get("choices"):
            assistant_message = gpt_response_data["choices"][0].get("message", {}).get("content", "")
            print(f"Assistant response: {assistant_message}")

            # 대화 기록 업데이트
            conversation_handler.add_to_history("user", user_input)
            conversation_handler.add_to_history("assistant", assistant_message)
            # TODO: 필요시 대화 요약 업데이트 로직 추가
            # if len(conversation_handler.get_history()) > 4: # 예: 대화 2턴 이상 시 요약
            #    summary = await summarize_conversation_async(...)
            #    conversation_handler.update_summary(summary)

            end_time = time.time()
            print(f"Total processing time: {end_time - start_time:.2f} seconds")

            return JSONResponse(content={"response": assistant_message})
        else:
            print("Failed to get response from the final GPT call.")
            raise HTTPException(status_code=500, detail="Failed to generate response")

    except Exception as e:
        print(f"An error occurred in /chat endpoint: {e}")
        # 스택 트레이스 로깅 추가 가능
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- 서버 실행 (개발용) ---
# 터미널에서 uvicorn chatbot.app:app --reload 실행 권장
if __name__ == "__main__":
    import uvicorn
    # 이 파일 자체를 직접 실행할 때 (보통 개발 시에는 uvicorn 명령 사용)
    print("Starting FastAPI server using uvicorn...")
    print("Access the UI at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)