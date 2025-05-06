# tests/visualize_results.py (Mode 3 토큰 보고 기능 추가 최종본)

"""
Chatbot 테스트 실행 결과(.jsonl 파일)를 로드하고,
주요 성능 지표(Latency, 응답 길이, 만족도, 모델 분포, 토큰 사용량 등)를
시각화(그래프 생성)하고 요약 리포트(텍스트 파일)를 생성하는 스크립트입니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
import glob # 파일 패턴 매칭을 위해 추가
import logging
import numpy as np # 요약 계산 위해 추가
from typing import List, Dict, Any, Optional
import platform # 한글 폰트 설정 위해 추가


# --- 로깅 설정 (DEBUG 레벨 고정) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Visualize Results logger initialized with DEBUG level.")

# --- Matplotlib 한글 폰트 설정 ---
# 시스템 환경에 따라 적절한 한글 폰트 설정 시도
try:
    system_os = platform.system()
    logger.debug(f"Operating System detected: {system_os}")

    if system_os == 'Linux':
        # Linux 환경: 나눔고딕 또는 다른 설치된 한글 폰트 시도
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
            logger.info("Set Korean font to NanumGothic for Linux.")
        except Exception:
            logger.warning("NanumGothic not found on Linux. Trying Malgun Gothic (may fail).")
            try: # WSL 등에서 Windows 폰트 접근 가능할 수 있음
                plt.rcParams['font.family'] = 'Malgun Gothic'
                logger.info("Set Korean font to Malgun Gothic (fallback).")
            except Exception:
                logger.warning("Could not set Korean font on Linux. Using default sans-serif.")
                plt.rcParams['font.family'] = 'sans-serif'

    elif system_os == 'Windows':
        # Windows 환경: 맑은 고딕 사용
        plt.rcParams['font.family'] = 'Malgun Gothic'
        logger.info("Set Korean font to Malgun Gothic for Windows.")
    elif system_os == 'Darwin': # MacOS
        # MacOS 환경: AppleGothic 사용
        plt.rcParams['font.family'] = 'AppleGothic'
        logger.info("Set Korean font to AppleGothic for MacOS.")
    else:
        # 기타 OS: 기본 sans-serif 사용
        logger.warning(f"Unsupported OS '{system_os}' for automatic Korean font setting. Using default sans-serif.")
        plt.rcParams['font.family'] = 'sans-serif'

    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    logger.debug("Applied axes.unicode_minus = False setting.")

except Exception as e:
    logger.warning(f"Could not set Korean font automatically due to an error: {e}. Plots might not display Korean characters correctly.", exc_info=True)
    # 오류 발생 시에도 기본 폰트로 계속 진행


# --- 함수 정의 ---

def load_results(filepath_pattern: str) -> Optional[pd.DataFrame]:
    """
    주어진 파일 경로 패턴(glob)에 맞는 모든 .jsonl 파일을 로드하여
    하나의 Pandas DataFrame으로 합칩니다.

    Args:
        filepath_pattern (str): 로드할 결과 파일 경로 패턴 (예: 'test_results/test_run_*.jsonl').

    Returns:
        Optional[pd.DataFrame]: 로드된 결과 데이터프레임. 파일이 없거나 오류 발생 시 None.
    """
    all_results_data = []
    # glob을 사용하여 패턴에 맞는 모든 파일 찾기
    try:
        matching_files = glob.glob(filepath_pattern)
        logger.info(f"Found {len(matching_files)} result file(s) matching pattern: '{filepath_pattern}'")
        if not matching_files:
            logger.error(f"No files found matching the pattern: {filepath_pattern}")
            return None
    except Exception as e:
        logger.error(f"Error during glob file matching for pattern '{filepath_pattern}': {e}", exc_info=True)
        return None

    # 각 파일을 순회하며 데이터 로드
    for filepath in matching_files:
        logger.info(f"Loading results from file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_count_file = 0
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue # 빈 줄 건너뛰기
                    try:
                        data = json.loads(line)
                        # 파일명 정보 추가 (어떤 결과 파일에서 왔는지 추적용)
                        data['_source_results_file'] = os.path.basename(filepath)
                        all_results_data.append(data)
                        loaded_count_file += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {filepath}: {line[:100]}...")
                    except Exception as parse_e: # 기타 파싱 오류
                        logger.warning(f"Error parsing line {i+1} in {filepath}: {parse_e}", exc_info=False)
                logger.info(f"Loaded {loaded_count_file} results from {filepath}")
        except FileNotFoundError:
             logger.error(f"Result file not found during loading (was present during glob?): {filepath}")
             continue # 다음 파일로 진행
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}", exc_info=True)
            # 특정 파일 읽기 실패 시 계속 진행할지 결정 (여기서는 계속 진행)
            # return None # 하나라도 실패하면 중단하려면 주석 해제

    # 최종 데이터 유효성 확인
    if not all_results_data:
        logger.error("No valid data loaded from any result files.")
        return None

    # 리스트의 딕셔너리를 DataFrame으로 변환 (중첩 구조 펼치기)
    try:
        # json_normalize는 중첩된 딕셔너리를 '_'로 연결된 컬럼명으로 펼쳐줌
        df = pd.json_normalize(all_results_data, sep='_')
        logger.info(f"Successfully loaded a total of {len(df)} results into DataFrame.")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error converting loaded data to Pandas DataFrame: {e}", exc_info=True)
        return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    로드된 결과 DataFrame을 분석 및 시각화에 용이하도록 전처리합니다.
    데이터 타입 변환, 결측치 처리, 필요한 파생 컬럼 생성 등을 수행합니다.

    Args:
        df (pd.DataFrame): 원본 결과 데이터프레임.

    Returns:
        pd.DataFrame: 전처리된 데이터프레임.
    """
    logger.info("Preprocessing DataFrame for analysis and visualization...")
    df_processed = df.copy() # 원본 변경 방지를 위해 복사본 사용

    # 1. Latency 데이터 타입 변환 (object -> float, 오류 시 NaN)
    logger.debug("Converting latency columns to numeric...")
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_latency_seconds'
        if col in df_processed.columns:
            original_dtype = df_processed[col].dtype
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            # 변환 후 NaN 개수 로깅 (타입 변환 문제 확인용)
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' (original dtype: {original_dtype}) converted to numeric with {nan_count} NaN values due to conversion errors.")
        else: logger.debug(f"Latency column '{col}' not found, skipping conversion.")

    # 2. Success 데이터 타입 변환 (object -> bool, 결측치는 False로 처리)
    logger.debug("Converting success columns to boolean...")
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_success'
        if col in df_processed.columns:
            original_na = df_processed[col].isna().sum()
            # fillna(False) 로 결측치를 False로 처리 후 bool 타입 변환
            df_processed[col] = df_processed[col].fillna(False).astype(bool)
            if original_na > 0: logger.debug(f"Filled {original_na} NaN values in '{col}' with False.")
        else: logger.debug(f"Success column '{col}' not found, skipping conversion.")

    # 3. Response 길이 컬럼 생성 (성공한 경우만, 문자열 아닌 경우 0)
    logger.debug("Calculating response length columns...")
    for mode in ['mode1', 'mode2', 'mode3']:
        resp_col = f'results_{mode}_response'
        len_col = f'results_{mode}_response_length'
        success_col = f'results_{mode}_success'
        if resp_col in df_processed.columns and success_col in df_processed.columns:
            # 성공(True)하고 응답이 문자열(str)인 경우만 길이 계산, 나머지는 0
            df_processed[len_col] = df_processed.apply(
                lambda row: len(str(row[resp_col])) if pd.notna(row[success_col]) and row[success_col] and isinstance(row[resp_col], str) else 0,
                axis=1
            )
            logger.debug(f"Created response length column '{len_col}'.")
        elif resp_col not in df_processed.columns: logger.debug(f"Response column '{resp_col}' not found, skipping length calculation.")
        # success_col은 위에서 생성/변환 보장

    # 4. 만족도 점수 데이터 타입 변환 (object -> float, 오류 시 NaN)
    logger.debug("Converting satisfaction score columns to numeric...")
    score_cols = [
        'satisfaction_evaluation_relevance_score', 'satisfaction_evaluation_accuracy_score',
        'satisfaction_evaluation_completeness_score', 'satisfaction_evaluation_conciseness_score',
        'satisfaction_evaluation_tone_score', 'satisfaction_evaluation_overall_satisfaction_score'
    ]
    for col in score_cols:
        if col in df_processed.columns:
            original_dtype = df_processed[col].dtype
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' (original dtype: {original_dtype}) converted to numeric with {nan_count} NaN values.")
        else: logger.debug(f"Satisfaction score column '{col}' not found, skipping conversion.")

    # 5. Mode 3 Debug Info에서 필요한 정보 추출 (존재할 경우 안전하게)
    logger.debug("Extracting relevant info from Mode 3 debug data...")
    # 모델 라우팅 정보 (존재한다면)
    if 'results_mode3_debug_info_model_chosen' in df_processed.columns:
        df_processed['mode3_model_chosen'] = df_processed['results_mode3_debug_info_model_chosen'].fillna('Unknown')
        logger.debug("Extracted 'mode3_model_chosen' column.")
    else: logger.debug("Column 'results_mode3_debug_info_model_chosen' not found.")

    # 복잡도 정보 (존재한다면)
    if 'results_mode3_debug_info_complexity_level' in df_processed.columns:
        df_processed['mode3_complexity_level'] = df_processed['results_mode3_debug_info_complexity_level'].fillna('Unknown')
        logger.debug("Extracted 'mode3_complexity_level' column.")
    else: logger.debug("Column 'results_mode3_debug_info_complexity_level' not found.")

    # CoT 사용 여부 (존재한다면)
    if 'results_mode3_debug_info_cot_data_present' in df_processed.columns:
        df_processed['mode3_cot_present'] = df_processed['results_mode3_debug_info_cot_data_present'].fillna(False).astype(bool)
        logger.debug("Extracted 'mode3_cot_present' column.")
    else: logger.debug("Column 'results_mode3_debug_info_cot_data_present' not found.")

    # RAG 결과 수 (존재한다면)
    if 'results_mode3_debug_info_rag_results_count' in df_processed.columns:
        df_processed['mode3_rag_count'] = pd.to_numeric(df_processed['results_mode3_debug_info_rag_results_count'], errors='coerce').fillna(0).astype(int)
        logger.debug("Extracted 'mode3_rag_count' column.")
    else: logger.debug("Column 'results_mode3_debug_info_rag_results_count' not found.")

    # 6. Mode 3 Token Usage 데이터 타입 변환 (추가됨)
    logger.debug("Converting Mode 3 token usage columns to numeric...")
    # --- [주의] 아래 컬럼명들은 test_runner.py에서 저장한 실제 키 이름과 일치해야 합니다 ---
    # --- json_normalize가 생성하는 이름을 기준으로 합니다 ---
    token_cols_mode3 = {
        'prompt': 'results_mode3_token_usage_prompt_tokens',
        'completion': 'results_mode3_token_usage_completion_tokens',
        'total': 'results_mode3_token_usage_total_tokens'
    }
    for token_type, col in token_cols_mode3.items():
        if col in df_processed.columns:
            original_dtype = df_processed[col].dtype
            # 숫자 변환 시도 (오류 시 NaN) 후, NaN은 0으로 채우고 정수 타입으로 변환
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(int)
            # 원본 데이터에서 변환 오류가 발생한 NaN 개수 확인
            original_numeric = pd.to_numeric(df[col], errors='coerce')
            nan_count = original_numeric.isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' (original dtype: {original_dtype}) converted to int, filling {nan_count} NaN/error values with 0.")
            logger.debug(f"Processed Mode 3 token column: {col}")
        else:
            logger.debug(f"Mode 3 token column '{col}' not found, skipping conversion. Will report N/A.")
            # 분석 시 해당 컬럼이 없어도 오류가 나지 않도록 빈 컬럼 생성 (선택적)
            # df_processed[col] = 0 # 필요 시 주석 해제

    logger.info("DataFrame preprocessing finished.")
    return df_processed

# --- 시각화 함수들 ---

def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """
    Mode 1, Mode 2, Mode 3 간의 응답 속도(Latency)를 비교하는 박스 플롯을 생성하고 저장합니다.
    """
    latency_cols = [col for col in ['results_mode1_latency_seconds', 'results_mode2_latency_seconds', 'results_mode3_latency_seconds'] if col in df.columns]
    if not latency_cols:
        logger.warning("No latency columns found in DataFrame. Skipping latency comparison plot.")
        return
    if df[latency_cols].isnull().all().all():
        logger.warning("All latency data is missing. Skipping latency comparison plot.")
        return

    logger.info("Generating latency comparison plot...")
    try:
        df_melted = df.melt(value_vars=latency_cols, var_name='Mode', value_name='Latency (s)')
        df_melted['Mode'] = df_melted['Mode'].str.replace('results_', '').str.replace('_latency_seconds', '').str.upper()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Mode', y='Latency (s)', data=df_melted, order=['MODE 1', 'MODE 2', 'MODE 3'], palette="viridis")
        plt.title('응답 속도 비교 (Mode 1 vs Mode 2 vs Mode 3)', fontsize=14)
        plt.ylabel('Latency (Seconds)', fontsize=12)
        plt.xlabel('Test Mode', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'latency_comparison_boxplot.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved latency comparison plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save latency plot: {e}", exc_info=True)
    finally:
        plt.close()

def plot_response_length_comparison(df: pd.DataFrame, output_dir: str):
    """
    Mode 1, Mode 2, Mode 3 간의 응답 길이(문자 수)를 비교하는 박스 플롯을 생성하고 저장합니다.
    """
    length_cols = [col for col in ['results_mode1_response_length', 'results_mode2_response_length', 'results_mode3_response_length'] if col in df.columns]
    if not length_cols:
        logger.warning("No response length columns found. Skipping response length comparison plot.")
        return
    if df[length_cols].isnull().all().all():
         logger.warning("All response length data is missing. Skipping response length plot.")
         return

    logger.info("Generating response length comparison plot...")
    try:
        df_melted = df.melt(value_vars=length_cols, var_name='Mode', value_name='Response Length')
        df_melted['Mode'] = df_melted['Mode'].str.replace('results_', '').str.replace('_response_length', '').str.upper()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Mode', y='Response Length', data=df_melted, order=['MODE 1', 'MODE 2', 'MODE 3'], palette="magma")
        plt.title('응답 길이 비교 (Mode 1 vs Mode 2 vs Mode 3)', fontsize=14)
        plt.ylabel('Response Length (Characters)', fontsize=12)
        plt.xlabel('Test Mode', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'response_length_comparison_boxplot.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved response length comparison plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save response length plot: {e}", exc_info=True)
    finally:
        plt.close()

def plot_satisfaction_distribution(df: pd.DataFrame, output_dir: str):
    """
    Mode 3 응답 만족도 점수 분포를 히스토그램으로 생성하고 저장합니다.
    """
    score_col = 'satisfaction_evaluation_overall_satisfaction_score'
    if score_col not in df.columns:
        logger.warning(f"Satisfaction score column '{score_col}' not found. Skipping satisfaction distribution plot.")
        return

    valid_scores = df[score_col].dropna()
    if valid_scores.empty:
        logger.warning("No valid satisfaction scores found after dropping NaN. Skipping satisfaction distribution plot.")
        return

    logger.info(f"Generating satisfaction score distribution plot (found {len(valid_scores)} valid scores)...")
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(valid_scores, bins=np.arange(0.5, 6.5, 1), kde=False, stat="count", color="skyblue", edgecolor="black")
        plt.title('Mode 3 응답 만족도 점수 분포', fontsize=14)
        plt.xlabel('Overall Satisfaction Score (1-5)', fontsize=12)
        plt.ylabel('Frequency (Count)', fontsize=12)
        plt.xticks(ticks=range(1, 6), labels=[str(i) for i in range(1, 6)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'mode3_satisfaction_distribution.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved satisfaction distribution plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save satisfaction distribution plot: {e}", exc_info=True)
    finally:
        plt.close()

def plot_model_routing_distribution(df: pd.DataFrame, output_dir: str):
    """
    Mode 3 실행 시 선택된 최종 응답 모델 분포를 파이 차트로 생성하고 저장합니다.
    """
    model_col = 'mode3_model_chosen' # preprocess_data에서 생성
    if model_col not in df.columns:
        logger.warning(f"Model chosen column '{model_col}' not found. Skipping model routing distribution plot.")
        return

    model_counts = df[model_col].value_counts()
    if model_counts.empty or model_counts.sum() == 0:
        logger.warning("No valid model routing data found to plot.")
        return

    logger.info(f"Generating model routing distribution plot (Total cases with model info: {model_counts.sum()})...")
    logger.debug(f"Model counts:\n{model_counts}")

    try:
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            model_counts,
            labels=model_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.85,
            colors=sns.color_palette("pastel")
        )
        plt.title('Mode 3 사용 모델 분포 (Model Routing)', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'mode3_model_routing_piechart.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved model routing distribution plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save model routing plot: {e}", exc_info=True)
    finally:
        plt.close()

def plot_performance_by_difficulty(df: pd.DataFrame, output_dir: str):
    """
    질문 난이도별 Mode 3 응답 속도 및 만족도 점수를 비교하는 박스 플롯 생성.
    """
    difficulty_col = 'test_difficulty'
    latency_col = 'results_mode3_latency_seconds'
    score_col = 'satisfaction_evaluation_overall_satisfaction_score'

    if difficulty_col not in df.columns:
        logger.warning(f"Difficulty column '{difficulty_col}' not found. Skipping performance by difficulty plots.")
        return

    valid_difficulties = df[difficulty_col].dropna().unique()
    logger.info(f"Generating performance plots by difficulty (Found difficulties: {valid_difficulties})...")

    # --- Latency by Difficulty ---
    if latency_col in df.columns:
        plot_df_latency = df[[difficulty_col, latency_col]].dropna()
        if not plot_df_latency.empty:
            logger.debug(f"Plotting latency by difficulty using {len(plot_df_latency)} data points.")
            try:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=difficulty_col, y=latency_col, data=plot_df_latency, order=['basic', 'advanced'], palette="coolwarm")
                plt.title('질문 난이도별 Mode 3 응답 속도', fontsize=14)
                plt.xlabel('Predefined Test Difficulty', fontsize=12)
                plt.ylabel('Latency (Seconds)', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                save_path_latency = os.path.join(output_dir, 'mode3_latency_by_difficulty.png')
                plt.savefig(save_path_latency, dpi=150)
                logger.info(f"Saved latency by difficulty plot to: {save_path_latency}")
            except Exception as e:
                logger.error(f"Failed to generate or save latency by difficulty plot: {e}", exc_info=True)
            finally:
                plt.close()
        else:
            logger.warning(f"No valid latency data found for difficulty comparison ({latency_col}).")
    else:
        logger.warning(f"Latency column '{latency_col}' not found. Skipping latency by difficulty plot.")


    # --- Satisfaction by Difficulty ---
    if score_col in df.columns:
        plot_df_score = df[[difficulty_col, score_col]].dropna()
        if not plot_df_score.empty:
            logger.debug(f"Plotting satisfaction by difficulty using {len(plot_df_score)} data points.")
            try:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=difficulty_col, y=score_col, data=plot_df_score, order=['basic', 'advanced'], palette="YlGnBu")
                plt.title('질문 난이도별 Mode 3 응답 만족도', fontsize=14)
                plt.xlabel('Predefined Test Difficulty', fontsize=12)
                plt.ylabel('Overall Satisfaction Score (1-5)', fontsize=12)
                plt.ylim(0.5, 5.5)
                plt.yticks(ticks=range(1, 6), labels=[str(i) for i in range(1, 6)])
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                save_path_score = os.path.join(output_dir, 'mode3_satisfaction_by_difficulty.png')
                plt.savefig(save_path_score, dpi=150)
                logger.info(f"Saved satisfaction by difficulty plot to: {save_path_score}")
            except Exception as e:
                 logger.error(f"Failed to generate or save satisfaction by difficulty plot: {e}", exc_info=True)
            finally:
                 plt.close()
        else:
            logger.warning(f"No valid satisfaction score data found for difficulty comparison ({score_col}).")
    else:
        logger.warning(f"Satisfaction score column '{score_col}' not found. Skipping satisfaction by difficulty plot.")


def generate_summary_report(df: pd.DataFrame, output_dir: str, input_pattern: str):
    """
    주요 테스트 결과 지표를 요약하여 텍스트 리포트 파일로 생성하고 저장합니다.
    """
    logger.info("Generating summary report...")
    report_lines = []
    report_lines.append("--- Chatbot Test Results Summary Report ---")
    report_lines.append(f"Report Generated On : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input Result Files  : {input_pattern}") # 사용된 입력 파일 패턴 명시
    report_lines.append(f"Total Test Cases    : {len(df)}")

    # Latency Summary
    report_lines.append("\n--- Average Latency (seconds, successful runs) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        lat_col = f'results_{mode}_latency_seconds'
        suc_col = f'results_{mode}_success'
        if lat_col in df.columns and suc_col in df.columns:
            successful_runs = df.loc[df[suc_col] == True, lat_col]
            if not successful_runs.empty:
                 avg_latency = successful_runs.mean()
                 success_count = len(successful_runs)
                 report_lines.append(f"Mode {mode[-1]}: {avg_latency:.4f} (from {success_count} successful runs)")
            else:
                 report_lines.append(f"Mode {mode[-1]}: N/A (No successful runs)")
        else:
            report_lines.append(f"Mode {mode[-1]}: N/A (Data missing)")

    # Success Rate Summary
    report_lines.append("\n--- Success Rate (%) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_success'
        if col in df.columns:
            success_rate = df[col].mean() * 100
            report_lines.append(f"Mode {mode[-1]}: {success_rate:.2f}%")
        else:
            report_lines.append(f"Mode {mode[-1]}: N/A (Data missing)")

    # Response Length Summary
    report_lines.append("\n--- Average Response Length (characters, successful runs) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        len_col = f'results_{mode}_response_length'
        suc_col = f'results_{mode}_success'
        if len_col in df.columns and suc_col in df.columns:
            successful_runs = df.loc[df[suc_col] == True, len_col]
            if not successful_runs.empty:
                avg_len = successful_runs.mean()
                report_lines.append(f"Mode {mode[-1]}: {avg_len:.1f}")
            else:
                report_lines.append(f"Mode {mode[-1]}: N/A (No successful runs)")
        else:
            report_lines.append(f"Mode {mode[-1]}: N/A (Data missing)")

    # --- [수정됨] Mode별 Token Usage Summary ---
    report_lines.append("\n--- Average Token Usage (Successful Runs) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        suc_col = f'results_{mode}_success'
        # 각 토큰 타입별 컬럼 이름 정의 (preprocess_data와 일치 확인)
        # Mode 1/2 과 Mode 3의 토큰 컬럼 이름이 다를 수 있음에 유의 (test_runner.py 저장 방식 확인)
        # 여기서는 Mode 1/2 는 results_modeX_token_usage_*, Mode 3 는 results_mode3_token_usage_* 로 가정
        prompt_col = f'results_{mode}_token_usage_prompt_tokens'
        completion_col = f'results_{mode}_token_usage_completion_tokens'
        total_col = f'results_{mode}_token_usage_total_tokens'

        # 필요한 컬럼들이 모두 존재하는지 확인
        required_cols_exist = all(c in df.columns for c in [suc_col, prompt_col, completion_col, total_col])

        if required_cols_exist:
            # 성공한 실행 건만 필터링
            successful_runs_df = df[df[suc_col] == True]
            num_successful = len(successful_runs_df)

            if num_successful > 0:
                # 각 토큰 타입별 평균 계산 (mean()은 NaN 자동 제외)
                avg_prompt = successful_runs_df[prompt_col].mean()
                avg_completion = successful_runs_df[completion_col].mean()
                avg_total = successful_runs_df[total_col].mean()
                report_lines.append(f"Mode {mode[-1]}: Avg Prompt={avg_prompt:.1f}, Avg Completion={avg_completion:.1f}, Avg Total={avg_total:.1f} (from {num_successful} runs)")
            else:
                report_lines.append(f"Mode {mode[-1]}: N/A (No successful runs with token data)")
        else:
            # 필요한 컬럼 중 하나라도 없으면 N/A 보고
            missing_cols_info = [c for c in [suc_col, prompt_col, completion_col, total_col] if c not in df.columns]
            report_lines.append(f"Mode {mode[-1]}: N/A (Data missing for columns: {missing_cols_info})")
    # --- [토큰 사용량 요약 끝] ---


    # Mode 3 Satisfaction Summary
    score_col = 'satisfaction_evaluation_overall_satisfaction_score'
    report_lines.append("\n--- Mode 3 Satisfaction Evaluation (Overall Score 1-5) ---")
    if score_col in df.columns:
        valid_scores = df[score_col].dropna()
        if not valid_scores.empty:
            avg_score = valid_scores.mean()
            median_score = valid_scores.median()
            std_dev = valid_scores.std()
            min_score = valid_scores.min()
            max_score = valid_scores.max()
            evaluated_count = len(valid_scores)
            total_mode3_runs = df['results_mode3_success'].notna().sum() # Mode 3 실행 횟수 (성공/실패 포함)
            report_lines.append(f"  Average : {avg_score:.2f}")
            report_lines.append(f"  Median  : {median_score:.2f}")
            report_lines.append(f"  Std Dev : {std_dev:.2f}")
            report_lines.append(f"  Min Score: {min_score:.0f}")
            report_lines.append(f"  Max Score: {max_score:.0f}")
            report_lines.append(f"  Evaluated Count: {evaluated_count} / {total_mode3_runs}") # 평가된 케이스 수 / Mode3 실행 케이스 수
        else:
            report_lines.append("  No valid satisfaction scores found.")
    else:
        report_lines.append("  Satisfaction score column not found.")

    # Mode 3 Model Routing Summary
    model_col = 'mode3_model_chosen'
    report_lines.append("\n--- Mode 3 Model Usage Distribution ---")
    if model_col in df.columns:
        # normalize=True 로 비율 계산, dropna=False로 결측치('Unknown' 등)도 포함
        model_counts = df[model_col].value_counts(normalize=True, dropna=False) * 100
        if not model_counts.empty:
            for model, percentage in model_counts.items():
                report_lines.append(f"  - {model}: {percentage:.1f}%")
        else:
            report_lines.append("  No model usage data found.")
    else:
        report_lines.append("  Model chosen column not found.")

    # 기타 분석 결과 추가 가능 (예: CoT 사용 비율, RAG 결과 수 평균 등)
    if 'mode3_cot_present' in df.columns:
         cot_usage_rate = df['mode3_cot_present'].mean() * 100
         report_lines.append(f"\nMode 3 CoT Usage Rate : {cot_usage_rate:.1f}%")
    if 'mode3_rag_count' in df.columns:
         avg_rag_count = df['mode3_rag_count'].mean()
         report_lines.append(f"Mode 3 Average RAG Chunks Found : {avg_rag_count:.1f}")


    # 리포트 파일 저장
    report_path = os.path.join(output_dir, 'summary_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        logger.info(f"Summary report saved successfully to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save summary report to {report_path}: {e}", exc_info=True)


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize chatbot test results from .jsonl files.")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the results .jsonl file or a glob pattern (e.g., 'tests/test_results/*.jsonl'). Ensure the path is correct relative to the execution directory.")
    parser.add_argument('--output-dir', type=str, default='test_plots',
                        help="Directory to save the generated plots and report (default: test_plots).")
    args = parser.parse_args()

    logger.info("--- Starting Test Results Visualization Script ---")
    logger.info(f"Input file/pattern: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")

    # 출력 디렉토리 생성 (이미 있으면 무시)
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {args.output_dir}")
    except Exception as e:
         logger.error(f"Failed to create output directory '{args.output_dir}': {e}. Exiting.")
         exit(1)

    # 1. 데이터 로드
    df_raw = load_results(args.input)

    if df_raw is None or df_raw.empty:
        logger.error("Failed to load or no data found in result file(s). Exiting.")
        exit(1)

    # 2. 데이터 전처리 (Mode 3 토큰 처리 포함)
    df_processed = preprocess_data(df_raw)

    # 3. 그래프 생성 함수 호출
    # 각 함수는 내부적으로 필요한 컬럼 확인 및 오류 처리 수행
    plot_latency_comparison(df_processed, args.output_dir)
    plot_response_length_comparison(df_processed, args.output_dir)
    plot_satisfaction_distribution(df_processed, args.output_dir)
    plot_model_routing_distribution(df_processed, args.output_dir)
    plot_performance_by_difficulty(df_processed, args.output_dir)
    # TODO: 추가적인 분석 및 시각화 함수 호출 가능 (현재는 보류)

    # 4. 요약 리포트 생성 (Mode 3 토큰 보고 포함)
    generate_summary_report(df_processed, args.output_dir, args.input)

    logger.info(f"--- Test Results Visualization Finished. Check outputs in '{args.output_dir}' ---")