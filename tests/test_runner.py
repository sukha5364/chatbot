# tests/test_runner.py

import os
import json
import time
import requests # 동기 HTTP 요청 라이브러리
from datetime import datetime

# --- 설정 ---
TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'test_cases')
CHATBOT_API_URL = "http://127.0.0.1:8000/chat" # FastAPI 서버 주소
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')
# 결과 저장 파일명 (타임스탬프 포함)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_FILENAME = f"test_run_{timestamp}.jsonl"

def load_test_cases(directory: str) -> List[Dict]:
    """지정된 디렉토리에서 .jsonl 테스트 케이스 파일들을 로드합니다."""
    all_test_cases = []
    print(f"Loading test cases from: {directory}")
    if not os.path.exists(directory):
        print(f"Error: Test cases directory not found: {directory}")
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            test_case = json.loads(line.strip())
                            # 각 테스트 케이스에 파일명 정보 추가 (구분용)
                            test_case['_source_file'] = filename
                            all_test_cases.append(test_case)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line in {filename}: {line.strip()}")
                print(f" - Loaded {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    print(f"Total test cases loaded: {len(all_test_cases)}")
    return all_test_cases

def run_test(test_case: Dict) -> Dict:
    """단일 테스트 케이스를 실행하고 결과를 반환합니다."""
    question = test_case.get("question")
    if not question:
        return {"error": "Missing 'question' field", **test_case} # 원본 정보 포함

    start_time = time.time()
    result_data = test_case.copy() # 원본 데이터 복사

    try:
        response = requests.post(CHATBOT_API_URL, json={"user_input": question}, timeout=60) # 타임아웃 설정
        end_time = time.time()
        latency = end_time - start_time

        result_data['timestamp'] = datetime.now().isoformat()
        result_data['latency_seconds'] = round(latency, 4)
        result_data['status_code'] = response.status_code

        if response.ok:
            api_response = response.json()
            result_data['response'] = api_response.get("response", "")
            result_data['success'] = True
            # TODO: 자동 평가 로직 추가 영역
            # 예: expected_answer 비교, 키워드 확인, LLM 기반 평가 등
            # result_data['evaluation'] = evaluate_response(result_data['response'], test_case.get('expected_answer'))
        else:
            result_data['response'] = response.text # 오류 메시지 기록
            result_data['success'] = False
            result_data['error'] = f"API Error: {response.status_code}"

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        latency = end_time - start_time
        result_data['timestamp'] = datetime.now().isoformat()
        result_data['latency_seconds'] = round(latency, 4)
        result_data['success'] = False
        result_data['error'] = f"Request Exception: {e}"

    return result_data

def save_test_results(results: List[Dict], output_dir: str, filename: str):
    """테스트 결과를 JSON Lines 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    print(f"\nSaving test results to: {filepath}")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(results)} results.")
    except Exception as e:
        print(f"Error saving test results: {e}")

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("--- Starting Chatbot Test Runner ---")
    overall_start_time = time.time()

    # 1. Load Test Cases
    test_cases = load_test_cases(TEST_CASES_DIR)
    if not test_cases:
        print("No test cases to run. Exiting.")
        exit()

    # 2. Run Tests
    print(f"\nRunning {len(test_cases)} test cases against {CHATBOT_API_URL}...")
    test_results = []
    success_count = 0
    total_latency = 0

    for i, tc in enumerate(test_cases):
        print(f"Running test {i+1}/{len(test_cases)}: {tc.get('question', 'N/A')[:50]}...")
        result = run_test(tc)
        test_results.append(result)
        if result.get('success'):
            success_count += 1
            total_latency += result.get('latency_seconds', 0)
        # 간단한 진행 상황 출력 (옵션)
        # print(f"  -> Success: {result.get('success')}, Latency: {result.get('latency_seconds', 0):.2f}s")
        time.sleep(0.1) # 서버 부하 방지를 위한 약간의 딜레이 (옵션)


    # 3. Save Results
    save_test_results(test_results, RESULTS_DIR, RESULT_FILENAME)

    # 4. Print Summary
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    avg_latency = (total_latency / success_count) if success_count > 0 else 0

    print("\n--- Test Run Summary ---")
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Successful Runs: {success_count}")
    print(f"Failed Runs: {len(test_cases) - success_count}")
    if success_count > 0:
        print(f"Average Latency (successful runs): {avg_latency:.4f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Results saved in: {os.path.join(RESULTS_DIR, RESULT_FILENAME)}")
    print("--- Test Runner Finished ---")