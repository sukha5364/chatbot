# tests/interactive_tester.py
# CLI 도구: 10+턴 대화 시나리오 수동/반자동 테스트 실행 및 로깅

import asyncio
import aiohttp
import argparse
import json
import time
import logging
import os
import sys
from datetime import datetime
import random # test_run_id 생성용

# --- 경로 설정 및 모듈 임포트 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # tests 폴더의 상위 -> 프로젝트 루트
    chatbot_module_path = os.path.join(project_root, 'chatbot')
    if chatbot_module_path not in sys.path:
        sys.path.insert(0, chatbot_module_path)
    # 필요한 chatbot 모듈 임포트
    from config_loader import get_config
    from gpt_interface import call_gpt_async, evaluate_satisfaction_async # gpt_interface.py에 evaluate_satisfaction_async가 있다고 가정, 없으면 직접 구현 필요
    # ConversationState는 이 스크립트에서 직접 사용하지 않고 히스토리 리스트 관리
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import chatbot modules. Ensure paths are correct. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during module import setup: {e}")
    sys.exit(1)

# --- 로깅 설정 (파일 및 콘솔) ---
# 기본 로깅 설정 (스크립트 실행 시 파일 로깅도 포함하도록 설정 가능)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # 콘솔에는 INFO 레벨 이상만 표시
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# 파일 핸들러 (선택적: 상세 로그 파일 저장 원할 시)
try:
    log_file_path = os.path.join(project_root, 'logs', f'interactive_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # 파일에는 DEBUG 레벨까지 모두 기록
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Detailed logs will be saved to: {log_file_path}")
except Exception as e:
    logger.error(f"Failed to set up file logging: {e}")

logger.info("Interactive Tester logger initialized.")


# --- 설정 로드 및 상수 정의 ---
try:
    config = get_config()
    if not config: raise ValueError("Configuration could not be loaded.")

    # 필수 설정값 로드
    testing_config = config.get('testing', {})
    tasks_config = config.get('tasks', {})
    prompts_config = config.get('prompts', {})
    gen_config = config.get('generation', {}) # Mode 1, 2 파라미터용

    CHATBOT_API_URL = testing_config.get('chatbot_api_url')
    DEFAULT_BASELINE_MODEL = testing_config.get('default_baseline_model') # Mode 1/2 용
    SATISFACTION_MODEL = tasks_config.get('satisfaction_evaluation', {}).get('model') # 만족도 평가용
    API_TIMEOUT_SECONDS = testing_config.get('api_timeout', 60)
    DEFAULT_SYSTEM_PROMPT = prompts_config.get("default_system_prompt", "You are an AI assistant.") # Mode 2 용
    # Mode 1/2 파라미터 (config 없으면 기본값)
    BASELINE_TEMP = gen_config.get('baseline_temperature', 0.7)
    BASELINE_MAX_TOKENS = gen_config.get('baseline_max_tokens', 500)

    # 필수 값 검증
    if not all([CHATBOT_API_URL, DEFAULT_BASELINE_MODEL, SATISFACTION_MODEL]):
        raise ValueError("Essential configuration values missing (chatbot_api_url, default_baseline_model, tasks.satisfaction_evaluation.model).")

    logger.info("Configuration loaded successfully for Interactive Tester.")
    logger.info(f"Using Chatbot API URL: {CHATBOT_API_URL}")
    logger.info(f"Using Baseline Model (A/B): {DEFAULT_BASELINE_MODEL}")
    logger.info(f"Using Satisfaction Model: {SATISFACTION_MODEL}")

except (ValueError, KeyError, Exception) as e:
    logger.critical(f"CRITICAL: Failed to load or validate configuration: {e}", exc_info=True)
    sys.exit(1)


# --- Helper Functions ---

async def run_model_a_turn(history: list, question: str, session: aiohttp.ClientSession) -> dict:
    """Model A (Baseline 1) 실행"""
    logger.debug(f"Running Model A turn with question: '{question[:50]}...'")
    start_time = time.time()
    messages = history + [{"role": "user", "content": question}]
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=DEFAULT_BASELINE_MODEL,
            temperature=BASELINE_TEMP,
            max_tokens=BASELINE_MAX_TOKENS,
            session=session
        )
        latency = time.time() - start_time
        if response_data and response_data.get("choices"):
            response_text = response_data["choices"][0].get("message", {}).get("content", "")
            token_usage = response_data.get("usage")
            logger.debug(f"Model A success. Latency: {latency:.3f}s")
            return {"response": response_text, "latency_seconds": round(latency, 4), "token_usage": token_usage, "error": None}
        else:
            logger.warning("Model A API call failed or returned no choices.")
            return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "error": "API call failed or no choices", "raw_response": response_data}
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Error running Model A turn: {e}", exc_info=True)
        return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "error": f"Exception: {str(e)}"}

async def run_model_b_turn(history: list, question: str, session: aiohttp.ClientSession) -> dict:
    """Model B (Baseline 2) 실행"""
    logger.debug(f"Running Model B turn with question: '{question[:50]}...'")
    start_time = time.time()
    # 시스템 프롬프트 포함
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + history + [{"role": "user", "content": question}]
    try:
        response_data = await call_gpt_async(
            messages=messages,
            model=DEFAULT_BASELINE_MODEL,
            temperature=BASELINE_TEMP,
            max_tokens=BASELINE_MAX_TOKENS,
            session=session
        )
        latency = time.time() - start_time
        if response_data and response_data.get("choices"):
            response_text = response_data["choices"][0].get("message", {}).get("content", "")
            token_usage = response_data.get("usage")
            logger.debug(f"Model B success. Latency: {latency:.3f}s")
            return {"response": response_text, "latency_seconds": round(latency, 4), "token_usage": token_usage, "error": None}
        else:
            logger.warning("Model B API call failed or returned no choices.")
            return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "error": "API call failed or no choices", "raw_response": response_data}
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Error running Model B turn: {e}", exc_info=True)
        return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "error": f"Exception: {str(e)}"}

async def run_model_c_turn(question: str, session: aiohttp.ClientSession) -> dict:
    """Model C (Target Pipeline) 실행 via /chat API"""
    logger.debug(f"Running Model C turn with question: '{question[:50]}...'")
    start_time = time.time()
    payload = {"user_input": question}
    # 테스트 모드 헤더 포함 (debug_info 받기 위해)
    headers = {config['testing']['test_mode_header']: 'true', 'Content-Type': 'application/json'}
    api_timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
    token_usage_c = None
    debug_info_c = None

    try:
        async with session.post(CHATBOT_API_URL, json=payload, headers=headers, timeout=api_timeout) as response:
            latency = time.time() - start_time
            status_code = response.status
            response_text_content = await response.text()

            if response.ok:
                try:
                    api_response = json.loads(response_text_content)
                    response_c = api_response.get("response")
                    debug_info_c = api_response.get("debug_info")

                    # 토큰 사용량 추출 시도 (scheduler.py 반환 구조 확인 필요!)
                    if debug_info_c and isinstance(debug_info_c, dict):
                        # --- [가정] debug_info 안에 'total_token_usage' 키로 토큰 정보가 있다고 가정 ---
                        # --- 실제 키 이름/경로는 scheduler.py 구현에 맞춰 반드시 수정해야 함! ---
                        token_usage_c = debug_info_c.get('total_token_usage')
                        if token_usage_c and isinstance(token_usage_c, dict):
                             if all(k in token_usage_c for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']):
                                 logger.debug(f"Extracted Model C token usage: {token_usage_c}")
                             else:
                                 logger.warning(f"Mode C token usage dict missing required keys. Found: {list(token_usage_c.keys())}")
                                 token_usage_c = None # 불완전하면 None 처리
                        else:
                             logger.warning(f"Mode C debug_info key 'total_token_usage' not found or not dict. Found keys: {list(debug_info_c.keys())}")
                             token_usage_c = None
                    else:
                        logger.warning("Mode C debug_info missing or not a dictionary, cannot extract token usage.")

                    logger.debug(f"Model C success. Latency: {latency:.3f}s")
                    return {"response": response_c or "", "latency_seconds": round(latency, 4), "token_usage": token_usage_c, "debug_info": debug_info_c or {}, "error": None, "status_code": status_code}

                except json.JSONDecodeError:
                    logger.error(f"Mode C API call OK (status {status_code}) but failed to decode JSON: {response_text_content[:200]}...")
                    return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "debug_info": None, "error": "JSON Decode Error", "status_code": status_code, "response_text": response_text_content}
            else:
                logger.warning(f"Mode C API call failed with status {status_code}: {response_text_content[:200]}...")
                return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "debug_info": None, "error": f"API Error: {status_code}", "status_code": status_code, "response_text": response_text_content}

    except asyncio.TimeoutError:
        latency = time.time() - start_time
        logger.error(f"Mode C API call timed out after {API_TIMEOUT_SECONDS} seconds.")
        return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "debug_info": None, "error": "Request Timeout"}
    except aiohttp.ClientConnectorError as e:
        latency = time.time() - start_time
        logger.error(f"Mode C API call connection error: {e}. Is server running at {CHATBOT_API_URL}?", exc_info=False)
        return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "debug_info": None, "error": f"Connection Error: {e}"}
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Unexpected error running Model C turn: {e}", exc_info=True)
        return {"response": None, "latency_seconds": round(latency, 4), "token_usage": None, "debug_info": None, "error": f"Exception: {str(e)}"}

async def get_satisfaction_scores(question: str, response: str, session: aiohttp.ClientSession) -> dict:
    """Model C 응답에 대한 자동 만족도 평가 (점수만 추출)"""
    if not response:
        return {"status": "skipped_no_response"}

    logger.debug("Running satisfaction evaluation...")
    start_time = time.time()
    try:
        # evaluate_satisfaction_async 가 점수와 이유가 포함된 dict를 반환한다고 가정
        evaluation_result = await evaluate_satisfaction_async(question, response, session)
        latency = time.time() - start_time
        logger.debug(f"Satisfaction evaluation finished in {latency:.3f}s")

        if evaluation_result and evaluation_result.get("status") == "success":
            # 점수 키만 추출
            score_keys = ["relevance_score", "accuracy_score", "completeness_score",
                          "conciseness_score", "tone_score", "overall_satisfaction_score"]
            scores = {key: evaluation_result.get(key) for key in score_keys if key in evaluation_result}
            scores["status"] = "success"
            scores["latency_seconds"] = round(latency, 4)
            logger.debug(f"Extracted satisfaction scores: {scores}")
            return scores
        elif evaluation_result: # 평가 실패 또는 경고 상태
             logger.warning(f"Satisfaction evaluation returned non-success status: {evaluation_result.get('status')}, Error: {evaluation_result.get('error')}")
             return {"status": evaluation_result.get('status', 'evaluation_failed'), "error": evaluation_result.get('error', 'Unknown evaluation error'), "latency_seconds": round(latency, 4)}
        else: # 함수 자체가 None 반환 (심각한 오류)
             logger.error("evaluate_satisfaction_async returned None.")
             return {"status": "evaluation_failed_unexpected", "error": "evaluate_satisfaction_async returned None", "latency_seconds": round(latency, 4)}

    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Exception during satisfaction evaluation: {e}", exc_info=True)
        return {"status": "evaluation_failed_exception", "error": str(e), "latency_seconds": round(latency, 4)}

def append_log(filepath: str, data: dict):
    """JSON Lines 파일에 데이터 추가"""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            # numpy 타입 등이 있을 경우 str로 변환
            f.write(json.dumps(data, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        logger.error(f"Failed to append to log file {filepath}: {e}")

def append_text_log(filepath: str, turn: int, role: str, text: str):
    """텍스트 로그 파일에 데이터 추가"""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            # 응답이 여러 줄일 경우 들여쓰기 등 가독성 처리 추가 가능
            f.write(f"[Turn {turn}] {role}:\n{text}\n\n")
    except Exception as e:
        logger.error(f"Failed to append to text log file {filepath}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Interactive Chatbot Tester for Multi-Turn Scenarios")
    parser.add_argument("--start-question", required=True, help="Initial question for the conversation.")
    parser.add_argument("--probe-turns", type=int, nargs='*', default=[], help="Turn numbers for probing questions (e.g., 5 10).")
    args = parser.parse_args()

    # --- 초기화 ---
    history_a, history_b = [], [] # Model A, B 용 로컬 히스토리 (Model C는 서버에서 관리)
    output_dir = os.path.join(project_root, 'tests', 'interactive_results') # 결과 저장 폴더
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"interactive_test_{datetime.now().strftime('%Y%m%d%H%M%S')}" # 실행 ID 포함
    jsonl_log_path = os.path.join(output_dir, f"{run_id}_results.jsonl")
    text_log_a_path = os.path.join(output_dir, f"{run_id}_model_a.txt")
    text_log_b_path = os.path.join(output_dir, f"{run_id}_model_b.txt")
    text_log_c_path = os.path.join(output_dir, f"{run_id}_model_c.txt")
    probe_turns_set = set(args.probe_turns)

    num_turns = 10
    logger.info(f"Starting interactive test run: {run_id}")
    logger.info(f"Target turns: {num_turns}")
    logger.info(f"Saving results to directory: {output_dir}")
    logger.info(f"Probe turns: {probe_turns_set if probe_turns_set else 'None'}")

    # 이전 로그 파일 확인 (덮어쓰기 경고 또는 새 파일 이름 사용)
    if os.path.exists(jsonl_log_path):
        logger.warning(f"Result file {jsonl_log_path} already exists. Will be appended to.")

    async with aiohttp.ClientSession() as session:
        try:
            for turn in range(1, num_turns + 1):
                print(f"\n===== Turn {turn}/{num_turns} =====")
                is_probe = turn in probe_turns_set
                if is_probe: print("***** This is a PROBE turn *****")

                # --- 질문 입력 ---
                if turn == 1:
                    q_a = q_b = q_c = args.start_question
                    print(f"Start Question: {q_a}")
                else:
                    print("\n--- Enter Next Questions ---")
                    q_a = input(f"Model A Q: ")
                    q_b = input(f"Model B Q: ")
                    q_c = input(f"Model C Q: ")

                # --- 텍스트 로그 기록 (질문) ---
                append_text_log(text_log_a_path, turn, "User", q_a)
                append_text_log(text_log_b_path, turn, "User", q_b)
                append_text_log(text_log_c_path, turn, "User", q_c)

                # --- 로컬 히스토리 업데이트 (Model A, B 용) ---
                history_a.append({"role": "user", "content": q_a})
                history_b.append({"role": "user", "content": q_b})

                # --- 모델 실행 ---
                start_turn_time = time.time()
                # 병렬 실행 가능 (선택적)
                # tasks = [
                #     run_model_a_turn(history_a, q_a, session),
                #     run_model_b_turn(history_b, q_b, session),
                #     run_model_c_turn(q_c, session)
                # ]
                # results = await asyncio.gather(*tasks)
                # res_a, res_b, res_c = results

                # 순차 실행 (간단)
                res_a = await run_model_a_turn(history_a, q_a, session)
                res_b = await run_model_b_turn(history_b, q_b, session)
                res_c = await run_model_c_turn(q_c, session)

                # --- 로컬 히스토리 업데이트 (Model A, B 응답 추가) ---
                if res_a.get("response"): history_a.append({"role": "assistant", "content": res_a["response"]})
                if res_b.get("response"): history_b.append({"role": "assistant", "content": res_b["response"]})

                # --- 결과 출력 ---
                print("\n--- Responses ---")
                print(f"[Model A ({res_a.get('latency_seconds', -1):.2f}s)] {res_a.get('response', f'ERROR: {res_a.get('error', 'Unknown')}')}")
                print(f"[Model B ({res_b.get('latency_seconds', -1):.2f}s)] {res_b.get('response', f'ERROR: {res_b.get('error', 'Unknown')}')}")
                print(f"[Model C ({res_c.get('latency_seconds', -1):.2f}s)] {res_c.get('response', f'ERROR: {res_c.get('error', 'Unknown')}')}")

                # --- Model C 자동 만족도 평가 ---
                satisfaction_scores = {}
                if res_c.get("response"):
                    satisfaction_scores = await get_satisfaction_scores(q_c, res_c["response"], session)
                else:
                    satisfaction_scores = {"status": "skipped_no_response"}
                print(f"[Model C Auto-Satisfaction] Status: {satisfaction_scores.get('status', 'error')}, Overall Score: {satisfaction_scores.get('overall_satisfaction_score', 'N/A')}")


                # --- 수동 평가 입력 ---
                print("\n--- Manual Evaluation ---")
                manual_notes = input(f"Enter notes for Turn {turn} {'(PROBE)' if is_probe else ''} (e.g., accuracy, context issues): ")

                # --- 데이터 로깅 ---
                turn_data = {
                    "run_id": run_id,
                    "turn": turn,
                    "probe_turn": is_probe,
                    "model_a": {"question": q_a, **res_a},
                    "model_b": {"question": q_b, **res_b},
                    "model_c": {"question": q_c, **res_c, "auto_satisfaction_scores": satisfaction_scores}, # 점수만 포함
                    "manual_notes": manual_notes,
                    "timestamp": datetime.now().isoformat()
                }
                append_log(jsonl_log_path, turn_data)

                # --- 텍스트 로그 저장 (응답) ---
                append_text_log(text_log_a_path, turn, "Assistant", res_a.get('response', f"ERROR: {res_a.get('error', 'Unknown')}"))
                append_text_log(text_log_b_path, turn, "Assistant", res_b.get('response', f"ERROR: {res_b.get('error', 'Unknown')}"))
                append_text_log(text_log_c_path, turn, "Assistant", res_c.get('response', f"ERROR: {res_c.get('error', 'Unknown')}"))

                end_turn_time = time.time()
                logger.info(f"Turn {turn} completed. Duration: {end_turn_time - start_turn_time:.2f} seconds.")

                # --- 계속 진행 여부 확인 (선택적) ---
                if turn < num_turns:
                    cont = input("Press Enter to continue to next turn, or type 'q' to quit: ")
                    if cont.lower() == 'q':
                        logger.info("User requested to quit.")
                        break

        except KeyboardInterrupt:
             logger.info("User interrupted the test run.")
        except Exception as e:
             logger.error(f"An error occurred during the test loop: {e}", exc_info=True)
        finally:
             logger.info("Closing aiohttp session.")
             # 세션은 async with 구문으로 자동 관리됨

    print(f"\n===== Interactive Test Finished: {run_id} =====")
    print(f"Structured results logged to: {jsonl_log_path}")
    print(f"Text conversation logs saved to:")
    print(f" - Model A: {text_log_a_path}")
    print(f" - Model B: {text_log_b_path}")
    print(f" - Model C: {text_log_c_path}")

if __name__ == "__main__":
    # Python 3.7+ required for asyncio.run
    try:
        asyncio.run(main())
    except Exception as e:
         logger.critical(f"Failed to run the interactive tester: {e}", exc_info=True)