# tests/test_runner.py

import os
import json
import time
import asyncio # 비동기 실행 위해 추가
import argparse # 커맨드 라인 인자 처리 위해 추가
# import requests # 동기 호출 제거
import aiohttp # 비동기 HTTP 호출 위해 추가
from datetime import datetime
import logging
from typing import List, Dict, Optional, Any

# --- 필요한 모듈 임포트 (상대 경로 주의) ---
# test_runner.py를 프로젝트 루트에서 python -m tests.test_runner 로 실행한다고 가정
try:
    from chatbot.config_loader import get_config
    from chatbot.gpt_interface import call_gpt_async
except ImportError:
    # 스크립트를 tests 폴더에서 직접 실행하는 경우 등 경로 문제 발생 시
    import sys
    # 현재 파일의 상위 디렉토리(tests)의 상위 디렉토리(프로젝트 루트)를 경로에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    from chatbot.config_loader import get_config
    from chatbot.gpt_interface import call_gpt_async

# --- 로거 설정 ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 설정 로드 ---
try:
    config = get_config()
    test_config = config.get('testing', {})
    prompt_config = config.get('prompts', {})
    task_config = config.get('tasks', {})
    gen_config = config.get('generation', {})
    log_config = config.get('logging', {})
    model_config = config.get('model_router', {}) # 모델 정보 필요
except Exception as e:
     logger.error(f"Failed to load configuration: {e}. Exiting.")
     exit(1)

# --- 설정 값 사용 ---
TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'test_cases')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')
CHATBOT_API_URL = test_config.get("chatbot_api_url", "http://127.0.0.1:8000/chat")
TEST_MODE_HEADER_NAME = test_config.get("test_mode_header", "X-Test-Mode")
SATISFACTION_MODEL = task_config.get("satisfaction_evaluation_model", "gpt-4o")
DEFAULT_SYSTEM_PROMPT = prompt_config.get("default_system_prompt", "You are an AI assistant.")
# Mode 1, 2에서 사용할 기본 모델 (easy 라우팅 모델 사용)
DEFAULT_BASELINE_MODEL = model_config.get('routing_map', {}).get('easy', 'gpt-3.5-turbo')

# --- 테스트 결과 저장 함수 (개선: default=str 추가) ---
def save_test_results(results: List[Dict], output_dir: str, filename: str):
    """테스트 결과를 JSON Lines 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    logger.info(f"Saving test results to: {filepath}")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                # NumPy 객체 등 직렬화 불가능한 타입 처리 추가
                f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
        logger.info(f"Successfully saved {len(results)} results.")
    except Exception as e:
        logger.error(f"Error saving test results: {e}", exc_info=True)

# --- 테스트 케이스 로드 함수 (수정됨) ---
def load_test_cases(test_type: str, set_number: Optional[int] = None) -> List[Dict]:
    """지정된 타입과 세트 번호에 맞는 테스트 케이스 파일들을 로드합니다."""
    all_test_cases = []
    filenames_to_load = []
    # test_generator.py에서 사용하는 설정을 가져옴
    gen_test_config = TEST_SET_CONFIG # test_generator.py의 설정을 직접 참조하거나 config.yaml로 옮겨야 함

    if test_type == 'function':
        func_config = gen_test_config.get('function', {})
        filename = func_config.get('filename_template')
        if filename: filenames_to_load.append(filename)
        else: logger.error("Functional test filename template not found."); return []
    elif test_type == 'overall':
        if set_number is None: logger.error("Set number is required for 'overall' test type."); return []
        overall_config = gen_test_config.get('overall', {})
        filename_tpl = overall_config.get('filename_template')
        num_sets = overall_config.get('num_sets', 0)
        if filename_tpl and 1 <= set_number <= num_sets:
             filename = filename_tpl.format(set_num=set_number)
             filenames_to_load.append(filename)
        elif not filename_tpl: logger.error("Overall test filename template not found."); return []
        else: logger.error(f"Invalid set number: {set_number}. Available sets: 1 to {num_sets}"); return []
    else:
        logger.error(f"Invalid test type: {test_type}. Choose 'function' or 'overall'.")
        return []

    logger.info(f"Attempting to load test cases from: {filenames_to_load} in {TEST_CASES_DIR}")

    for filename in filenames_to_load:
        filepath = os.path.join(TEST_CASES_DIR, filename)
        if not os.path.exists(filepath):
             logger.error(f"Test case file not found: {filepath}")
             continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_count = 0
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try:
                        test_case = json.loads(line)
                        # 기본 필드(question) 확인
                        if 'question' not in test_case:
                             logger.warning(f"Skipping line {i+1} in {filename} due to missing 'question' field.")
                             continue
                        test_case['_source_file'] = filename # 소스 파일 정보 추가
                        all_test_cases.append(test_case)
                        loaded_count += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {filename}: {line[:100]}...")
                logger.info(f"Loaded {loaded_count} test cases from {filename}")
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}", exc_info=True)

    logger.info(f"Total test cases loaded: {len(all_test_cases)}")
    return all_test_cases


# --- [신규] 만족도 평가 함수 ---
async def evaluate_satisfaction_async(
    question: str,
    response: str,
    session: aiohttp.ClientSession
) -> Optional[Dict[str, Any]]:
    """GPT-4o를 사용하여 응답 만족도를 평가합니다."""
    if not response or not isinstance(response, str) or len(response.strip()) == 0:
        logger.warning("Empty or invalid response received for satisfaction evaluation.")
        return {"error": "Empty response, cannot evaluate."}

    # 평가 프롬프트 (내용은 이전 답변 참조)
    evaluation_prompt = f"""
    다음은 사용자 질문과 챗봇의 답변입니다. 아래 평가 기준에 따라 답변의 만족도를 1점에서 5점 사이로 평가하고, 각 항목별 점수와 간결한 평가 이유를 JSON 형식으로 제공해주세요.

    [평가 기준]
    - 관련성(Relevance): 질문의 의도에 얼마나 부합하는 답변인가? (1-5)
    - 정확성(Accuracy): 제공된 정보나 사실에 기반하여 얼마나 정확한가? 환각은 없는가? (1-5)
    - 완결성(Completeness): 질문에 대해 충분하고 만족스러운 정보를 제공하는가? (1-5)
    - 간결성(Conciseness): 불필요하게 길거나 장황하지 않고 핵심 정보를 잘 전달하는가? (1-5)
    - 어조(Tone): 데카트론 상담원으로서 전문적이고 친절한 어조인가? (1-5)
    - 종합 만족도(Overall): 위 항목들을 고려한 전반적인 만족도는? (1-5)

    [사용자 질문]
    {question}

    [챗봇 답변]
    {response}

    [평가 결과 (JSON 형식)]
    ```json
    {{
      "relevance_score": <점수>,
      "accuracy_score": <점수>,
      "completeness_score": <점수>,
      "conciseness_score": <점수>,
      "tone_score": <점수>,
      "overall_satisfaction_score": <점수>,
      "evaluation_reason": "<간결한 총평>"
    }}
    ```
    """
    messages = [{"role": "user", "content": evaluation_prompt}]
    logger.debug(f"Requesting satisfaction evaluation from {SATISFACTION_MODEL}")
    try:
        # call_gpt_async 재사용하여 평가 모델 호출
        eval_response_data = await call_gpt_async(
            messages=messages,
            model=SATISFACTION_MODEL,
            temperature=0.1, # 일관된 평가
            max_tokens=300,
            session=session
        )
        # ... (JSON 파싱 및 반환 로직 - 이전 답변 참조) ...
        if eval_response_data and eval_response_data.get("choices"):
            eval_content = eval_response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw satisfaction evaluation response: {eval_content[:100]}...")
            clean_eval_content = eval_content.strip()
            if clean_eval_content.startswith("```json"): clean_eval_content = clean_eval_content[7:-3].strip()
            elif clean_eval_content.startswith("```"): clean_eval_content = clean_eval_content[3:-3].strip()
            try:
                json_start = clean_eval_content.find('{'); json_end = clean_eval_content.rfind('}')
                json_string = clean_eval_content[json_start:json_end+1] if json_start != -1 and json_end != -1 else clean_eval_content
                evaluation_result = json.loads(json_string)
                # 점수 타입 변환 시도 (오류 발생 가능성 있음)
                for key in evaluation_result:
                    if key.endswith("_score"):
                        try: evaluation_result[key] = int(evaluation_result[key])
                        except (ValueError, TypeError): logger.warning(f"Could not convert score '{key}' to int: {evaluation_result[key]}")
                logger.info("Satisfaction evaluation successful.")
                return evaluation_result
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from satisfaction evaluation: {e}. Content: '{clean_eval_content}'")
                return {"error": f"JSON Decode Error: {e}"}
        else:
            logger.warning("Failed to get valid response from satisfaction evaluation model.")
            return {"error": "No valid response from evaluation model."}

    except Exception as e:
        logger.error(f"Error during satisfaction evaluation call: {e}", exc_info=True)
        return {"error": f"Exception during evaluation: {e}"}

# --- [수정] 단일 테스트 실행 함수 (비동기, 3 모드, 평가) ---
async def run_test_async(
    test_case: Dict,
    session: aiohttp.ClientSession # 비동기 호출 위해 세션 필요
) -> Dict:
    """단일 테스트 케이스를 비동기적으로 실행하고 결과를 반환합니다 (3 모드 비교)."""
    question = test_case.get("question")
    if not question:
        return {"error": "Missing 'question' field", **test_case}

    result_data = test_case.copy() # 원본 데이터 복사
    result_data['test_start_time'] = datetime.now().isoformat() # 테스트 시작 시간
    result_data['results'] = {} # 각 모드별 결과 저장

    api_timeout_seconds = test_config.get('api_timeout', 60)
    api_timeout = aiohttp.ClientTimeout(total=api_timeout_seconds) # aiohttp 타임아웃 객체
    test_mode_header_dict = {TEST_MODE_HEADER_NAME: 'true'} # 테스트 모드 헤더

    # 공통 파라미터
    common_params = {
        "session": session,
        "temperature": gen_config.get('default_temperature', 0.7),
        "max_tokens": gen_config.get('default_max_tokens', 500)
    }

    # --- Mode 1: User Prompt Only ---
    mode1_start = time.time()
    result_data['results']['mode1'] = {"success": False} # 초기화
    try:
        logger.debug("Running Mode 1: User Prompt Only")
        messages_m1 = [{"role": "user", "content": question}]
        model_m1 = DEFAULT_BASELINE_MODEL
        response_m1_data = await call_gpt_async(messages=messages_m1, model=model_m1, **common_params)
        mode1_latency = time.time() - mode1_start
        if response_m1_data and response_m1_data.get("choices"):
            result_data['results']['mode1'].update({
                "success": True,
                "response": response_m1_data["choices"][0].get("message", {}).get("content", ""),
                "latency_seconds": round(mode1_latency, 4),
                "model_used": model_m1 # 사용된 모델 기록
            })
        else: result_data['results']['mode1'].update({"error": "API call failed or no choices", "latency_seconds": round(mode1_latency, 4)})
    except Exception as e:
        mode1_latency = time.time() - mode1_start
        logger.error(f"Error in Mode 1 execution: {e}", exc_info=True)
        result_data['results']['mode1'].update({"error": f"Exception: {e}", "latency_seconds": round(mode1_latency, 4)})

    # --- Mode 2: User + System Prompt ---
    mode2_start = time.time()
    result_data['results']['mode2'] = {"success": False} # 초기화
    try:
        logger.debug("Running Mode 2: User + System Prompt")
        messages_m2 = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": question}]
        model_m2 = DEFAULT_BASELINE_MODEL # 동일 모델 사용
        response_m2_data = await call_gpt_async(messages=messages_m2, model=model_m2, **common_params)
        mode2_latency = time.time() - mode2_start
        if response_m2_data and response_m2_data.get("choices"):
             result_data['results']['mode2'].update({
                "success": True,
                "response": response_m2_data["choices"][0].get("message", {}).get("content", ""),
                "latency_seconds": round(mode2_latency, 4),
                "model_used": model_m2
            })
        else: result_data['results']['mode2'].update({"error": "API call failed or no choices", "latency_seconds": round(mode2_latency, 4)})
    except Exception as e:
        mode2_latency = time.time() - mode2_start
        logger.error(f"Error in Mode 2 execution: {e}", exc_info=True)
        result_data['results']['mode2'].update({"error": f"Exception: {e}", "latency_seconds": round(mode2_latency, 4)})

    # --- Mode 3: Full Pipeline (via Chatbot API) ---
    mode3_start = time.time()
    result_data['results']['mode3'] = {"success": False} # 초기화
    try:
        logger.debug(f"Running Mode 3: Full Pipeline via POST to {CHATBOT_API_URL}")
        async with session.post(CHATBOT_API_URL,
                                json={"user_input": question},
                                headers=test_mode_header_dict, # 테스트 모드 헤더 전달
                                timeout=api_timeout) as response:
            mode3_latency = time.time() - mode3_start
            status_code = response.status
            response_text = await response.text()

            result_data['results']['mode3'].update({
                 "latency_seconds": round(mode3_latency, 4),
                 "status_code": status_code
            })

            if response.ok:
                try:
                     api_response = json.loads(response_text)
                     response_m3 = api_response.get("response") # 기본 응답
                     debug_info_m3 = api_response.get("debug_info") # 디버그 정보

                     # 응답 또는 디버그 정보가 없으면 경고
                     if response_m3 is None: logger.warning("Mode 3 response missing 'response' field.")
                     if debug_info_m3 is None: logger.warning("Mode 3 response missing 'debug_info' field (ensure test mode header worked).")

                     result_data['results']['mode3'].update({
                          "success": True,
                          "response": response_m3 or "", # None 대신 빈 문자열
                          "debug_info": debug_info_m3 or {} # None 대신 빈 딕셔너리
                          })

                     # 성공 시 만족도 평가 호출
                     if response_m3: # 응답이 있을 때만 평가
                          logger.info("Requesting satisfaction evaluation for Mode 3 response...")
                          satisfaction_result = await evaluate_satisfaction_async(question, response_m3, session)
                          result_data['satisfaction_evaluation'] = satisfaction_result
                          logger.info("Finished satisfaction evaluation.")
                     else:
                           result_data['satisfaction_evaluation'] = {"error": "Mode 3 response was empty, skipped evaluation."}

                except json.JSONDecodeError:
                     logger.error(f"Mode 3 API call successful (status {status_code}) but failed to decode JSON response: {response_text[:200]}...")
                     result_data['results']['mode3'].update({"success": False, "error": "JSON Decode Error", "response_text": response_text})
            else:
                logger.warning(f"Mode 3 API call failed with status {status_code}: {response_text[:200]}...")
                result_data['results']['mode3'].update({"success": False, "error": f"API Error: {status_code}", "response_text": response_text})

    except asyncio.TimeoutError:
         mode3_latency = time.time() - mode3_start
         logger.error(f"Mode 3 API call timed out after {api_timeout_seconds} seconds.")
         result_data['results']['mode3'].update({"success": False, "error": "Request Timeout", "latency_seconds": round(mode3_latency, 4)})
    except aiohttp.ClientConnectorError as e: # ClientError 보다 구체적인 연결 오류
        mode3_latency = time.time() - mode3_start
        logger.error(f"Mode 3 API call connection error: {e}. Is the chatbot server running at {CHATBOT_API_URL}?", exc_info=False) # 스택 트레이스 제외
        result_data['results']['mode3'].update({"success": False, "error": f"Connection Error: {e}", "latency_seconds": round(mode3_latency, 4)})
    except Exception as e:
        mode3_latency = time.time() - mode3_start
        logger.error(f"Error in Mode 3 execution: {e}", exc_info=True)
        result_data['results']['mode3'].update({"success": False, "error": f"Exception: {e}", "latency_seconds": round(mode3_latency, 4)})

    result_data['test_end_time'] = datetime.now().isoformat()
    return result_data


# --- [수정] 메인 실행 로직 (비동기 및 인자 처리) ---
async def main():
    """메인 비동기 실행 함수"""
    parser = argparse.ArgumentParser(description="Run chatbot tests.")
    parser.add_argument('--test-type', type=str, required=True, choices=['function', 'overall'], help="Type of test to run ('function' or 'overall').")
    parser.add_argument('--set', type=int, help="Set number for 'overall' test type (e.g., 1, 2, 3). Required if --test-type is 'overall'.")
    # 동시 실행 개수 제한 옵션 (API 부하 조절)
    parser.add_argument('--concurrency', type=int, default=5, help="Number of tests to run concurrently.")
    args = parser.parse_args()

    if args.test_type == 'overall' and args.set is None:
         parser.error("--set <number> is required when --test-type is 'overall'.")
    if args.test_type == 'overall' and args.set not in range(1, TEST_SET_CONFIG['overall']['num_sets'] + 1):
         parser.error(f"Invalid set number. Please choose between 1 and {TEST_SET_CONFIG['overall']['num_sets']}.")


    logger.info(f"--- Starting Chatbot Test Runner ---")
    logger.info(f"Test Type: {args.test_type}, Set Number: {args.set if args.set else 'N/A'}, Concurrency: {args.concurrency}")
    overall_start_time = time.time()

    # 1. Load Test Cases based on args
    test_cases = load_test_cases(args.test_type, args.set)
    if not test_cases:
        logger.error("No test cases loaded. Exiting.")
        return

    # 결과 저장 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"test_run_{args.test_type}{f'_set{args.set}' if args.set else ''}_{timestamp}.jsonl"

    # 2. Run Tests using asyncio semaphore for concurrency control
    test_results = []
    semaphore = asyncio.Semaphore(args.concurrency) # 동시 실행 개수 제어

    async with aiohttp.ClientSession() as session:
         logger.info(f"Running {len(test_cases)} test cases against {CHATBOT_API_URL} with concurrency limit {args.concurrency}...")

         async def run_with_semaphore(test_case):
             async with semaphore:
                 # 각 테스트 사이에 약간의 지연 추가 (API 부하 분산)
                 await asyncio.sleep(0.1)
                 logger.info(f"Running test ID: {test_case.get('id', 'N/A')} - {test_case['question'][:30]}...")
                 return await run_test_async(test_case, session)

         tasks = [run_with_semaphore(tc) for tc in test_cases]
         # tqdm 같은 라이브러리로 진행률 표시 가능
         test_results = await asyncio.gather(*tasks)
         logger.info(f"Finished running {len(test_results)} test cases.")

    # 3. Save Results
    save_test_results(test_results, RESULTS_DIR, result_filename)

    # 4. Print Summary (개선)
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    success_count_mode3 = sum(1 for r in test_results if r.get('results', {}).get('mode3', {}).get('success'))
    total_latency_mode3 = sum(r.get('results', {}).get('mode3', {}).get('latency_seconds', 0) for r in test_results if r.get('results', {}).get('mode3', {}).get('success'))
    avg_latency_mode3 = (total_latency_mode3 / success_count_mode3) if success_count_mode3 > 0 else 0
    # 만족도 평균 계산 (존재 및 숫자형 점수 확인)
    valid_evals = []
    for r in test_results:
        eval_data = r.get('satisfaction_evaluation')
        if isinstance(eval_data, dict) and isinstance(eval_data.get('overall_satisfaction_score'), (int, float)):
            valid_evals.append(eval_data['overall_satisfaction_score'])
    avg_satisfaction = sum(valid_evals) / len(valid_evals) if valid_evals else 'N/A'

    print("\n--- Test Run Summary ---")
    print(f"Test Type: {args.test_type}, Set: {args.set if args.set else 'N/A'}")
    print(f"Total Test Cases Run: {len(test_cases)}")
    print(f"Mode 3 (Full Pipeline) Successful Runs: {success_count_mode3}")
    print(f"Mode 3 Failed Runs: {len(test_cases) - success_count_mode3}")
    if success_count_mode3 > 0:
        print(f"Mode 3 Average Latency (successful runs): {avg_latency_mode3:.4f} seconds")
    print(f"Average Overall Satisfaction (Mode 3, evaluated): {avg_satisfaction if avg_satisfaction != 'N/A' else 'Not Evaluated'}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Results saved in: {os.path.join(RESULTS_DIR, result_filename)}")
    print("--- Test Runner Finished ---")

if __name__ == "__main__":
    # 비동기 메인 함수 실행
    try:
      asyncio.run(main())
    except KeyboardInterrupt:
         logger.info("Test run interrupted by user.")
    except Exception as e:
         logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)