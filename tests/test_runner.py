# tests/test_runner.py (요구사항 반영 최종본)

import os
import json
import time
import asyncio
import argparse
import aiohttp
from datetime import datetime
import logging
from typing import List, Dict, Optional, Any

# --- 로깅 설정 (DEBUG 레벨 고정) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Test Runner logger initialized with DEBUG level.")

# --- 설정 로더 임포트 및 설정 로드 ---
try:
    # 스크립트 실행 위치에 따라 프로젝트 루트 경로 설정 필요
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # tests 폴더의 상위 -> 프로젝트 루트
    # chatbot 모듈 경로 추가
    import sys
    chatbot_module_path = os.path.join(project_root, 'chatbot')
    if chatbot_module_path not in sys.path:
        sys.path.insert(0, chatbot_module_path)
    from config_loader import get_config
    from gpt_interface import call_gpt_async # gpt_interface도 임포트
    logger.info("Chatbot modules (config_loader, gpt_interface) imported successfully.")

    config = get_config()
    logger.info("Configuration loaded successfully via get_config().")

    # --- 필수 설정값 로드 및 검증 ---
    testing_config = config.get('testing', {})
    tasks_config = config.get('tasks', {})
    prompts_config = config.get('prompts', {}) # 필요시 사용
    gen_config = config.get('generation', {}) # 필요시 사용

    # 필수 키 검증
    required_testing_keys = ['chatbot_api_url', 'test_mode_header', 'default_baseline_model', 'api_timeout']
    required_tasks_keys = ['satisfaction_evaluation'] # model은 하위 키
    missing_keys = []

    for key in required_testing_keys:
        if key not in testing_config: missing_keys.append(f"testing.{key}")
    if 'satisfaction_evaluation' not in tasks_config or 'model' not in tasks_config['satisfaction_evaluation']:
        missing_keys.append("tasks.satisfaction_evaluation.model")

    # 테스트 케이스 설정을 위한 키 확인 (function_test_set, overall_test_set 가정)
    if 'function_test_set' not in testing_config: missing_keys.append("testing.function_test_set")
    if 'overall_test_set' not in testing_config: missing_keys.append("testing.overall_test_set")

    if missing_keys:
        raise KeyError(f"Missing required configuration keys in config.yaml: {', '.join(missing_keys)}")

    # 설정값 변수화 (검증 후)
    CHATBOT_API_URL = testing_config['chatbot_api_url']
    TEST_MODE_HEADER_NAME = testing_config['test_mode_header']
    DEFAULT_BASELINE_MODEL = testing_config['default_baseline_model'] # Mode 1/2 용
    SATISFACTION_MODEL = tasks_config['satisfaction_evaluation']['model'] # 만족도 평가용
    API_TIMEOUT_SECONDS = testing_config.get('api_timeout', 60) # 기본값 60초
    DEFAULT_SYSTEM_PROMPT = prompts_config.get("default_system_prompt", "You are an AI assistant.") # Mode 2 용

    logger.info(f"Test Runner Config: API URL='{CHATBOT_API_URL}', Baseline Model='{DEFAULT_BASELINE_MODEL}', Satisfaction Model='{SATISFACTION_MODEL}'")

except (ImportError, KeyError, FileNotFoundError, Exception) as e:
    logger.error(f"CRITICAL: Failed to load configuration or required sections: {e}", exc_info=True)
    exit(1)

# --- 경로 설정 ---
TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'test_cases')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')


# --- 테스트 결과 저장 함수 ---
def save_test_results(results: List[Dict], output_dir: str, filename: str):
    """
    테스트 결과를 JSON Lines 파일로 저장합니다.

    Args:
        results (List[Dict]): 저장할 테스트 결과 딕셔너리 리스트.
        output_dir (str): 결과를 저장할 디렉토리 경로.
        filename (str): 저장할 파일 이름.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    logger.info(f"Saving test results to: {filepath}")
    saved_count = 0
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                try:
                    # NumPy 객체 등 직렬화 불가능한 타입 처리 추가
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
                    saved_count += 1
                except TypeError as te:
                     logger.warning(f"Could not serialize result item due to TypeError: {te}. Item keys: {list(result.keys())}. Skipping item.")
                     # 부분적인 데이터라도 기록하고 싶다면 아래처럼 처리 가능
                     # simplified_result = {k: str(v) for k, v in result.items()} # 모든 값을 문자열로 변환 (정보 손실 가능)
                     # f.write(json.dumps(simplified_result, ensure_ascii=False) + '\n')
        logger.info(f"Successfully saved {saved_count}/{len(results)} results.")
    except Exception as e:
        logger.error(f"Error saving test results to {filepath}: {e}", exc_info=True)

# --- 테스트 케이스 로드 함수 ---
def load_test_cases(test_type: str, set_number: Optional[int] = None) -> List[Dict]:
    """
    지정된 타입과 세트 번호에 맞는 테스트 케이스 파일들을 로드합니다.
    config.yaml에서 파일명 템플릿 등의 설정을 읽어옵니다.

    Args:
        test_type (str): 로드할 테스트 타입 ('function' 또는 'overall').
        set_number (Optional[int]): 'overall' 타입일 경우 로드할 세트 번호.

    Returns:
        List[Dict]: 로드된 테스트 케이스 딕셔너리 리스트.
    """
    global testing_config # 전역 설정 사용

    all_test_cases = []
    filenames_to_load = []

    try:
        if test_type == 'function':
            # config.yaml의 'testing.function_test_set' 섹션 참조
            func_config = testing_config.get('function_test_set', {})
            filename = func_config.get('filename_template')
            if filename:
                filenames_to_load.append(filename)
            else:
                logger.error("Config key 'testing.function_test_set.filename_template' not found.")
                return []
        elif test_type == 'overall':
            if set_number is None:
                logger.error("Set number is required for 'overall' test type.")
                return []
            # config.yaml의 'testing.overall_test_set' 섹션 참조
            overall_config = testing_config.get('overall_test_set', {})
            filename_tpl = overall_config.get('filename_template')
            num_sets = overall_config.get('num_sets', 0)
            if filename_tpl and 1 <= set_number <= num_sets:
                filename = filename_tpl.format(set_num=set_number)
                filenames_to_load.append(filename)
            elif not filename_tpl:
                logger.error("Config key 'testing.overall_test_set.filename_template' not found.")
                return []
            else:
                logger.error(f"Invalid set number: {set_number}. Available sets based on config: 1 to {num_sets}")
                return []
        else:
            logger.error(f"Invalid test type: {test_type}. Choose 'function' or 'overall'.")
            return []
    except KeyError as e:
         logger.error(f"Missing configuration key needed for loading test cases: {e}")
         return []
    except Exception as e:
         logger.error(f"Error accessing test configuration for loading test cases: {e}")
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


# --- 만족도 평가 함수 ---
async def evaluate_satisfaction_async(
    question: str,
    response: str,
    session: aiohttp.ClientSession
) -> Optional[Dict[str, Any]]:
    """
    GPT (config에서 지정한 모델)를 사용하여 응답 만족도를 평가합니다.

    Args:
        question (str): 사용자 질문.
        response (str): 챗봇의 응답.
        session (aiohttp.ClientSession): API 호출에 사용할 aiohttp 세션.

    Returns:
        Optional[Dict[str, Any]]: 평가 결과 딕셔너리 (점수, 이유 포함). 오류 시 None 또는 에러 정보 포함 딕셔너리.
    """
    global SATISFACTION_MODEL # 전역 설정값 사용

    if not response or not isinstance(response, str) or len(response.strip()) == 0:
        logger.warning("Empty or invalid response received for satisfaction evaluation.")
        # 오류 정보 명확화
        return {"error": "Empty or invalid response provided", "status": "evaluation_skipped"}

    # 평가 프롬프트 (내용은 이전 답변 참조 - config.yaml로 이동 고려 가능)
    # TODO: 이 프롬프트도 config.yaml의 prompts 섹션으로 이동 관리 가능
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
    logger.debug(f"Requesting satisfaction evaluation from {SATISFACTION_MODEL}...")
    try:
        # call_gpt_async 재사용 (gpt_interface 모듈에서 임포트됨)
        eval_response_data = await call_gpt_async(
            messages=messages,
            model=SATISFACTION_MODEL, # config에서 읽은 모델 사용
            temperature=0.1, # 일관된 평가를 위해 낮은 온도 사용
            max_tokens=300, # JSON 결과 받기에 충분한 토큰
            session=session,
            response_format={"type": "json_object"} # JSON 모드 시도
        )

        if eval_response_data and eval_response_data.get("choices"):
            eval_content = eval_response_data["choices"][0].get("message", {}).get("content", "")
            logger.debug(f"Raw satisfaction evaluation response: {eval_content[:150]}...")

            # JSON 파싱 강화 (코드 블록 및 불필요 문자 제거 시도)
            clean_eval_content = eval_content.strip()
            if clean_eval_content.startswith("```json"): clean_eval_content = clean_eval_content[7:-3].strip()
            elif clean_eval_content.startswith("```"): clean_eval_content = clean_eval_content[3:-3].strip()

            try:
                json_start = clean_eval_content.find('{'); json_end = clean_eval_content.rfind('}')
                if json_start != -1 and json_end != -1:
                     json_string = clean_eval_content[json_start:json_end+1]
                else:
                     json_string = clean_eval_content # 중괄호 없으면 그대로 시도

                evaluation_result = json.loads(json_string)

                # 점수 타입 변환 및 검증 강화
                required_scores = ["relevance_score", "accuracy_score", "completeness_score", "conciseness_score", "tone_score", "overall_satisfaction_score"]
                valid_result = {"status": "success"}
                missing_scores = []
                conversion_errors = []

                for key in required_scores:
                    if key not in evaluation_result:
                         missing_scores.append(key)
                         valid_result[key] = None # 키가 없으면 None
                         continue
                    try:
                         # 점수가 숫자가 아니거나 1~5 범위 밖이면 None 처리 또는 경고
                         score = int(evaluation_result[key])
                         if 1 <= score <= 5:
                              valid_result[key] = score
                         else:
                              logger.warning(f"Evaluation score '{key}' ({score}) out of range (1-5). Setting to None.")
                              valid_result[key] = None
                              conversion_errors.append(f"{key}: out_of_range({score})")
                    except (ValueError, TypeError):
                         logger.warning(f"Could not convert score '{key}' to int: {evaluation_result[key]}. Setting to None.")
                         valid_result[key] = None
                         conversion_errors.append(f"{key}: conversion_error({evaluation_result[key]})")

                valid_result["evaluation_reason"] = evaluation_result.get("evaluation_reason", "N/A") # 이유 텍스트

                if missing_scores:
                     logger.warning(f"Evaluation result missing scores: {missing_scores}")
                     valid_result["status"] = "parsing_warning_missing_scores"
                if conversion_errors:
                     logger.warning(f"Evaluation score conversion issues: {conversion_errors}")
                     valid_result["status"] = "parsing_warning_conversion_error"

                logger.info("Satisfaction evaluation successful.")
                return valid_result # 정제된 결과 반환

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from satisfaction evaluation: {e}. Cleaned Content: '{clean_eval_content}'")
                return {"error": f"JSON Decode Error: {e}", "raw_content": eval_content, "status": "evaluation_failed_parsing"}
        else:
            logger.warning(f"Failed to get valid response/choices from evaluation model {SATISFACTION_MODEL}.")
            # API 호출 로그 확인 필요
            return {"error": "No valid choices from evaluation model", "status": "evaluation_failed_api_no_choice"}

    except Exception as e:
        logger.error(f"Error during satisfaction evaluation API call: {e}", exc_info=True)
        return {"error": f"Exception during evaluation: {e}", "status": "evaluation_failed_exception"}

# --- 단일 테스트 실행 함수 ---
async def run_test_async(
    test_case: Dict,
    session: aiohttp.ClientSession # 비동기 호출 위해 세션 필요
) -> Dict:
    """
    단일 테스트 케이스를 비동기적으로 실행하고 결과를 반환합니다 (3 모드 비교).
    config.yaml에서 관련 설정을 읽어 사용합니다.

    Args:
        test_case (Dict): 실행할 테스트 케이스 딕셔너리 ('question' 키 필수).
        session (aiohttp.ClientSession): API 호출에 사용할 aiohttp 세션.

    Returns:
        Dict: 테스트 실행 결과 딕셔너리 (모드별 응답, 레이턴시, 디버그 정보, 만족도 평가 등 포함).
    """
    # 전역 설정값 사용
    global DEFAULT_BASELINE_MODEL, DEFAULT_SYSTEM_PROMPT, CHATBOT_API_URL, TEST_MODE_HEADER_NAME, API_TIMEOUT_SECONDS, gen_config

    question = test_case.get("question")
    if not question:
        logger.warning("Test case missing 'question' field. Skipping.")
        # 입력 데이터 포함하여 반환
        return {"error": "Missing 'question' field", "status": "skipped", **test_case}

    result_data = test_case.copy() # 원본 데이터 복사
    result_data['test_run_id'] = f"test_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}" # 실행 ID 추가
    result_data['test_start_time'] = datetime.now().isoformat() # 테스트 시작 시간
    result_data['results'] = {} # 각 모드별 결과 저장

    # aiohttp 타임아웃 객체
    api_timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
    # 테스트 모드 헤더 딕셔너리
    test_mode_header_dict = {TEST_MODE_HEADER_NAME: 'true'}

    # 공통 파라미터 (Mode 1, 2 용)
    common_params = {
        "session": session,
        # generation 섹션에서 온도/토큰 읽기 (기본값 설정)
        "temperature": gen_config.get('final_response_temperature', 0.7),
        "max_tokens": gen_config.get('final_response_max_tokens', 500)
    }

    logger.debug(f"Running test case ID: {result_data['test_run_id']} for question: '{question[:50]}...'")

    # --- Mode 1: User Prompt Only ---
    mode1_start = time.time()
    result_data['results']['mode1'] = {"success": False, "status": "pending"} # 초기화
    try:
        logger.debug(f"[{result_data['test_run_id']}] Running Mode 1: User Prompt Only (Model: {DEFAULT_BASELINE_MODEL})")
        messages_m1 = [{"role": "user", "content": question}]
        model_m1 = DEFAULT_BASELINE_MODEL # config에서 읽은 값 사용
        response_m1_data = await call_gpt_async(messages=messages_m1, model=model_m1, **common_params)
        mode1_latency = time.time() - mode1_start
        if response_m1_data and response_m1_data.get("choices"):
            response_text = response_m1_data["choices"][0].get("message", {}).get("content", "")
            result_data['results']['mode1'].update({
                "success": True,
                "status": "completed",
                "response": response_text,
                "latency_seconds": round(mode1_latency, 4),
                "model_used": model_m1,
                "token_usage": response_m1_data.get("usage") # 토큰 사용량 추가
            })
        else:
             result_data['results']['mode1'].update({
                 "error": "API call failed or no choices",
                 "status": "failed_api",
                 "latency_seconds": round(mode1_latency, 4),
                 "raw_response": response_m1_data # 실패 시 원본 응답 저장
             })
    except Exception as e:
        mode1_latency = time.time() - mode1_start
        logger.error(f"[{result_data['test_run_id']}] Error in Mode 1 execution: {e}", exc_info=True)
        result_data['results']['mode1'].update({
            "error": f"Exception: {str(e)}",
            "status": "failed_exception",
            "latency_seconds": round(mode1_latency, 4)
        })

    # --- Mode 2: User + System Prompt ---
    mode2_start = time.time()
    result_data['results']['mode2'] = {"success": False, "status": "pending"} # 초기화
    try:
        logger.debug(f"[{result_data['test_run_id']}] Running Mode 2: User + System Prompt (Model: {DEFAULT_BASELINE_MODEL})")
        messages_m2 = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": question}]
        model_m2 = DEFAULT_BASELINE_MODEL # config에서 읽은 값 사용
        response_m2_data = await call_gpt_async(messages=messages_m2, model=model_m2, **common_params)
        mode2_latency = time.time() - mode2_start
        if response_m2_data and response_m2_data.get("choices"):
             response_text = response_m2_data["choices"][0].get("message", {}).get("content", "")
             result_data['results']['mode2'].update({
                 "success": True,
                 "status": "completed",
                 "response": response_text,
                 "latency_seconds": round(mode2_latency, 4),
                 "model_used": model_m2,
                 "token_usage": response_m2_data.get("usage")
             })
        else:
             result_data['results']['mode2'].update({
                 "error": "API call failed or no choices",
                 "status": "failed_api",
                 "latency_seconds": round(mode2_latency, 4),
                 "raw_response": response_m2_data
             })
    except Exception as e:
        mode2_latency = time.time() - mode2_start
        logger.error(f"[{result_data['test_run_id']}] Error in Mode 2 execution: {e}", exc_info=True)
        result_data['results']['mode2'].update({
            "error": f"Exception: {str(e)}",
            "status": "failed_exception",
            "latency_seconds": round(mode2_latency, 4)
        })

    # --- Mode 3: Full Pipeline (via Chatbot API) ---
    mode3_start = time.time()
    result_data['results']['mode3'] = {"success": False, "status": "pending"} # 초기화
    try:
        logger.debug(f"[{result_data['test_run_id']}] Running Mode 3: Full Pipeline via POST to {CHATBOT_API_URL}")
        async with session.post(CHATBOT_API_URL,
                                json={"user_input": question},
                                headers=test_mode_header_dict, # 테스트 모드 헤더 전달
                                timeout=api_timeout) as response:
            mode3_latency = time.time() - mode3_start
            status_code = response.status
            response_text = await response.text() # 텍스트 먼저 읽기

            result_data['results']['mode3'].update({
                "latency_seconds": round(mode3_latency, 4),
                "status_code": status_code
            })

            if response.ok:
                try:
                    api_response = json.loads(response_text)
                    response_m3 = api_response.get("response") # 기본 응답
                    debug_info_m3 = api_response.get("debug_info") # 디버그 정보

                    if response_m3 is None: logger.warning(f"[{result_data['test_run_id']}] Mode 3 response missing 'response' field.")
                    if debug_info_m3 is None: logger.warning(f"[{result_data['test_run_id']}] Mode 3 response missing 'debug_info' field (is test mode header working?).")

                    result_data['results']['mode3'].update({
                        "success": True,
                        "status": "completed",
                        "response": response_m3 or "", # None 대신 빈 문자열
                        "debug_info": debug_info_m3 or {} # None 대신 빈 딕셔너리
                        # TODO: Mode 3의 토큰 사용량 정보는 debug_info에서 추출해야 함 (app.py 수정 필요)
                    })

                    # 성공 시 만족도 평가 호출
                    if response_m3: # 응답이 있을 때만 평가
                        logger.info(f"[{result_data['test_run_id']}] Requesting satisfaction evaluation for Mode 3...")
                        satisfaction_result = await evaluate_satisfaction_async(question, response_m3, session)
                        result_data['satisfaction_evaluation'] = satisfaction_result # 평가 결과 저장
                        logger.info(f"[{result_data['test_run_id']}] Finished satisfaction evaluation.")
                    else:
                        result_data['satisfaction_evaluation'] = {"error": "Mode 3 response was empty, skipped evaluation.", "status": "evaluation_skipped"}

                except json.JSONDecodeError:
                    logger.error(f"[{result_data['test_run_id']}] Mode 3 API call OK (status {status_code}) but failed to decode JSON: {response_text[:200]}...")
                    result_data['results']['mode3'].update({"success": False, "status": "failed_parsing", "error": "JSON Decode Error", "response_text": response_text})
            else:
                logger.warning(f"[{result_data['test_run_id']}] Mode 3 API call failed with status {status_code}: {response_text[:200]}...")
                result_data['results']['mode3'].update({"success": False, "status": f"failed_api_{status_code}", "error": f"API Error: {status_code}", "response_text": response_text})

    except asyncio.TimeoutError:
        mode3_latency = time.time() - mode3_start
        logger.error(f"[{result_data['test_run_id']}] Mode 3 API call timed out after {API_TIMEOUT_SECONDS} seconds.")
        result_data['results']['mode3'].update({"success": False, "status": "failed_timeout", "error": "Request Timeout", "latency_seconds": round(mode3_latency, 4)})
    except aiohttp.ClientConnectorError as e:
        mode3_latency = time.time() - mode3_start
        logger.error(f"[{result_data['test_run_id']}] Mode 3 API call connection error: {e}. Is server running at {CHATBOT_API_URL}?", exc_info=False)
        result_data['results']['mode3'].update({"success": False, "status": "failed_connection", "error": f"Connection Error: {e}", "latency_seconds": round(mode3_latency, 4)})
    except Exception as e:
        mode3_latency = time.time() - mode3_start
        logger.error(f"[{result_data['test_run_id']}] Unexpected error in Mode 3 execution: {e}", exc_info=True)
        result_data['results']['mode3'].update({"success": False, "status": "failed_exception", "error": f"Exception: {str(e)}", "latency_seconds": round(mode3_latency, 4)})

    result_data['test_end_time'] = datetime.now().isoformat()
    logger.info(f"Finished test case ID: {result_data['test_run_id']} (Mode 3 status: {result_data['results']['mode3'].get('status', 'unknown')})")
    return result_data


# --- 메인 실행 로직 ---
async def main():
    """메인 비동기 실행 함수: 인자 파싱, 테스트 로드, 병렬 실행, 결과 저장 및 요약 출력"""
    parser = argparse.ArgumentParser(description="Run chatbot tests and compare modes.")
    parser.add_argument('--test-type', type=str, required=True, choices=['function', 'overall'],
                        help="Type of test to run ('function' or 'overall'). Reads config for filenames.")
    parser.add_argument('--set', type=int,
                        help="Set number for 'overall' test type. Required if --test-type is 'overall'.")
    parser.add_argument('--concurrency', type=int, default=5,
                        help="Number of tests to run concurrently (default: 5).")
    args = parser.parse_args()

    # overall 타입 시 set 번호 검증 (config 기반)
    if args.test_type == 'overall':
        overall_conf = testing_config.get('overall_test_set', {})
        num_sets = overall_conf.get('num_sets', 0)
        if args.set is None:
            parser.error("--set <number> is required when --test-type is 'overall'.")
        if not (1 <= args.set <= num_sets):
             parser.error(f"Invalid set number: {args.set}. Available sets based on config: 1 to {num_sets}")

    logger.info(f"--- Starting Chatbot Test Runner ---")
    logger.info(f"Test Type: {args.test_type}, Set Number: {args.set if args.set else 'N/A'}, Concurrency: {args.concurrency}")
    overall_start_time = time.time()

    # 1. Load Test Cases based on args and config
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

    # aiohttp 세션 생성
    # API 키는 gpt_interface에서 환경변수/dotenv로 로드되므로 여기서 명시적 전달 불필요
    async with aiohttp.ClientSession() as session:
        logger.info(f"Running {len(test_cases)} test cases against {CHATBOT_API_URL} with concurrency limit {args.concurrency}...")

        async def run_with_semaphore(test_case):
            async with semaphore:
                # 각 테스트 사이에 약간의 지연 추가 (API 부하 분산 - 선택 사항)
                await asyncio.sleep(random.uniform(0.05, 0.15)) # 50ms ~ 150ms 랜덤 지연
                test_id = test_case.get('_source_file', 'unknown_file') + "_" + str(test_case.get('id', random.randint(1000,9999)))
                logger.debug(f"Starting test for case: {test_id} - '{test_case['question'][:30]}...'")
                return await run_test_async(test_case, session)

        # tqdm 등 진행률 표시 라이브러리 사용 가능
        # pip install tqdm
        # from tqdm.asyncio import tqdm_asyncio
        # tasks = [run_with_semaphore(tc) for tc in test_cases]
        # test_results = await tqdm_asyncio.gather(*tasks, desc=f"Running {args.test_type} tests")

        # 기본 gather 사용
        tasks = [run_with_semaphore(tc) for tc in test_cases]
        test_results = await asyncio.gather(*tasks)

        logger.info(f"Finished running {len(test_results)} test cases.")

    # 3. Save Results
    save_test_results(test_results, RESULTS_DIR, result_filename)

    # 4. Print Summary (개선)
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    total_cases = len(test_results)
    summary = {"total_cases": total_cases, "total_time_seconds": round(total_time, 2)}
    mode_metrics = {}

    for mode_num in [1, 2, 3]:
        mode_key = f"mode{mode_num}"
        mode_results = [r['results'].get(mode_key) for r in test_results if r.get('results') and r['results'].get(mode_key)]

        success_count = sum(1 for r in mode_results if r and r.get('success'))
        failed_api_count = sum(1 for r in mode_results if r and r.get('status') == 'failed_api')
        failed_exception_count = sum(1 for r in mode_results if r and r.get('status') == 'failed_exception')
        # ... 다른 실패 상태 카운트 추가 가능

        valid_latencies = [r.get('latency_seconds') for r in mode_results if r and r.get('success') and isinstance(r.get('latency_seconds'), (int, float))]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0

        mode_metrics[mode_key] = {
            "success_rate": (success_count / total_cases) * 100 if total_cases > 0 else 0,
            "avg_latency_success": round(avg_latency, 4),
            "failed_api_count": failed_api_count,
            "failed_exception_count": failed_exception_count
        }

    # 만족도 평균 계산 (Mode 3 대상)
    valid_evals = []
    eval_skipped = 0
    eval_failed = 0
    for r in test_results:
        eval_data = r.get('satisfaction_evaluation')
        if isinstance(eval_data, dict):
            if eval_data.get('status') == 'success' and isinstance(eval_data.get('overall_satisfaction_score'), (int, float)):
                 valid_evals.append(eval_data['overall_satisfaction_score'])
            elif eval_data.get('status', '').startswith('evaluation_failed'):
                 eval_failed += 1
            elif eval_data.get('status') == 'evaluation_skipped':
                 eval_skipped +=1
        else: # satisfaction_evaluation 키 자체가 없거나 다른 타입
            eval_skipped +=1 # 또는 다른 카운터 사용

    avg_satisfaction = sum(valid_evals) / len(valid_evals) if valid_evals else 'N/A'
    satisfaction_summary = {
        "average_score": round(avg_satisfaction, 2) if isinstance(avg_satisfaction, (int, float)) else 'N/A',
        "evaluated_count": len(valid_evals),
        "skipped_count": eval_skipped,
        "failed_evaluation_count": eval_failed
    }

    print("\n--- Test Run Summary ---")
    print(f"Test Type: {args.test_type}, Set: {args.set if args.set else 'N/A'}")
    print(f"Total Test Cases Run: {summary['total_cases']}")
    print(f"Concurrency Limit: {args.concurrency}")
    print(f"Total Execution Time: {summary['total_time_seconds']} seconds")
    print("-" * 20)
    print("Performance Metrics per Mode:")
    for mode, metrics in mode_metrics.items():
        print(f"  Mode {mode[-1]}:")
        print(f"    Success Rate: {metrics['success_rate']:.2f}%")
        print(f"    Avg Latency (Successful): {metrics['avg_latency_success']}s")
        print(f"    API Failures: {metrics['failed_api_count']}")
        print(f"    Exceptions: {metrics['failed_exception_count']}")
    print("-" * 20)
    print("Mode 3 Satisfaction Evaluation Summary:")
    print(f"  Average Score (Evaluated): {satisfaction_summary['average_score']}")
    print(f"  Successfully Evaluated Count: {satisfaction_summary['evaluated_count']}")
    print(f"  Skipped Evaluations (No response/Error): {satisfaction_summary['skipped_count']}")
    print(f"  Failed Evaluations (Parsing/API): {satisfaction_summary['failed_evaluation_count']}")
    print("-" * 20)
    print(f"Results saved in: {os.path.join(RESULTS_DIR, result_filename)}")
    print("--- Test Runner Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user.")
    except Exception as e:
        # main 함수 내에서 발생한 예외 처리 (예: 설정 로드 실패)
        logger.critical(f"An critical error occurred during test execution: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}. Please check logs and configuration.")