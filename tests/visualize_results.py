# tests/visualize_results.py (신규 파일)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
import glob
import logging
from typing import List, Dict, Any, Optional

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Matplotlib 한글 폰트 설정 (시스템에 맞는 폰트 경로 지정 필요)
# 예시: 나눔고딕 (설치 필요: sudo apt-get install fonts-nanum*)
try:
    import platform
    if platform.system() == 'Linux':
        # Linux 환경 예시 (나눔고딕)
        plt.rcParams['font.family'] = 'NanumGothic'
    elif platform.system() == 'Windows':
        # Windows 환경 예시 (맑은 고딕)
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin': # MacOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        # 기타 OS
        plt.rcParams['font.family'] = 'sans-serif' # 기본값 사용

    # 마이너스 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.warning(f"Could not set Korean font for plots: {e}. Using default.")


def load_results(filepath_pattern: str) -> Optional[pd.DataFrame]:
    """
    주어진 파일 경로 패턴(glob)에 맞는 모든 .jsonl 파일을 로드하여
    하나의 Pandas DataFrame으로 합칩니다.
    """
    all_results_data = []
    # glob을 사용하여 패턴에 맞는 모든 파일 찾기
    matching_files = glob.glob(filepath_pattern)

    if not matching_files:
        logger.error(f"No files found matching pattern: {filepath_pattern}")
        return None

    logger.info(f"Found {len(matching_files)} result file(s) matching pattern.")

    for filepath in matching_files:
        logger.info(f"Loading results from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        # 파일명 정보 추가
                        data['_source_results_file'] = os.path.basename(filepath)
                        all_results_data.append(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {filepath}: {line[:100]}...")
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}", exc_info=True)
            # 파일 하나 읽기 실패해도 계속 진행 (선택 사항)
            # return None

    if not all_results_data:
        logger.error("No valid data loaded from result files.")
        return None

    df = pd.json_normalize(all_results_data, sep='_') # 중첩된 JSON 구조를 '_'로 구분하여 펼침
    logger.info(f"Successfully loaded {len(df)} results into DataFrame. Columns: {df.columns.tolist()}")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    로드된 DataFrame을 분석 및 시각화하기 용이하게 전처리합니다.
    (데이터 타입 변환, 결측치 처리, 필요한 컬럼 추출 등)
    """
    logger.info("Preprocessing DataFrame...")

    # 1. Latency 데이터 타입 변환 (object -> float, 오류 시 NaN)
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_latency_seconds'
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Success 데이터 타입 변환 (object -> bool, 오류 시 False)
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_success'
        if col in df.columns:
            # 결측치는 False로 처리
            df[col] = df[col].fillna(False).astype(bool)

    # 3. Response 길이 컬럼 생성 (성공한 경우만)
    for mode in ['mode1', 'mode2', 'mode3']:
        resp_col = f'results_{mode}_response'
        len_col = f'results_{mode}_response_length'
        success_col = f'results_{mode}_success'
        if resp_col in df.columns and success_col in df.columns:
            # 성공하고 응답이 문자열인 경우 길이를, 아니면 0을 할당
            df[len_col] = df.apply(
                lambda row: len(str(row[resp_col])) if row[success_col] and isinstance(row[resp_col], str) else 0,
                axis=1
            )

    # 4. 만족도 점수 데이터 타입 변환 (object -> float, 오류 시 NaN)
    score_cols = [
        'satisfaction_evaluation_relevance_score', 'satisfaction_evaluation_accuracy_score',
        'satisfaction_evaluation_completeness_score', 'satisfaction_evaluation_conciseness_score',
        'satisfaction_evaluation_tone_score', 'satisfaction_evaluation_overall_satisfaction_score'
    ]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Mode 3 Debug Info에서 필요한 정보 추출 (존재할 경우)
    if 'results_mode3_debug_info_model_chosen' in df.columns:
        df['mode3_model_chosen'] = df['results_mode3_debug_info_model_chosen']
    if 'results_mode3_debug_info_complexity_level' in df.columns:
        df['mode3_complexity_level'] = df['results_mode3_debug_info_complexity_level']
    if 'results_mode3_debug_info_cot_data_present' in df.columns:
        # 결측치는 False로 처리
        df['mode3_cot_present'] = df['results_mode3_debug_info_cot_data_present'].fillna(False).astype(bool)
    if 'results_mode3_debug_info_rag_results_count' in df.columns:
        # 결측치는 0으로 처리
        df['mode3_rag_count'] = pd.to_numeric(df['results_mode3_debug_info_rag_results_count'], errors='coerce').fillna(0).astype(int)


    logger.info("DataFrame preprocessing finished.")
    return df

def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """Mode 1/2/3 간 Latency 비교 박스 플롯 생성"""
    latency_cols = [col for col in ['results_mode1_latency_seconds', 'results_mode2_latency_seconds', 'results_mode3_latency_seconds'] if col in df.columns]
    if not latency_cols:
        logger.warning("Latency columns not found. Skipping latency plot.")
        return

    # Wide-form 데이터를 Long-form으로 변환
    df_melted = df.melt(value_vars=latency_cols, var_name='Mode', value_name='Latency (s)')
    # Mode 이름 정리 (예: 'results_mode1_latency_seconds' -> 'Mode 1')
    df_melted['Mode'] = df_melted['Mode'].str.replace('results_', '').str.replace('_latency_seconds', '').str.upper()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Mode', y='Latency (s)', data=df_melted, order=['MODE 1', 'MODE 2', 'MODE 3'])
    plt.title('응답 속도 비교 (Mode 1 vs Mode 2 vs Mode 3)')
    plt.ylabel('Latency (Seconds)')
    plt.xlabel('Test Mode')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'latency_comparison_boxplot.png')
    try:
        plt.savefig(save_path)
        logger.info(f"Saved latency comparison plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save latency plot: {e}")
    plt.close() # 플롯 닫기

def plot_response_length_comparison(df: pd.DataFrame, output_dir: str):
    """Mode 1/2/3 간 응답 길이 비교 박스 플롯 생성"""
    length_cols = [col for col in ['results_mode1_response_length', 'results_mode2_response_length', 'results_mode3_response_length'] if col in df.columns]
    if not length_cols:
        logger.warning("Response length columns not found. Skipping response length plot.")
        return

    df_melted = df.melt(value_vars=length_cols, var_name='Mode', value_name='Response Length')
    df_melted['Mode'] = df_melted['Mode'].str.replace('results_', '').str.replace('_response_length', '').str.upper()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Mode', y='Response Length', data=df_melted, order=['MODE 1', 'MODE 2', 'MODE 3'])
    plt.title('응답 길이 비교 (Mode 1 vs Mode 2 vs Mode 3)')
    plt.ylabel('Response Length (Characters)')
    plt.xlabel('Test Mode')
    # Y축 범위 제한 (너무 긴 응답 때문에 보기 어려울 경우)
    # plt.ylim(0, df_melted['Response Length'].quantile(0.95)) # 예: 상위 5% 제외
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'response_length_comparison_boxplot.png')
    try:
        plt.savefig(save_path)
        logger.info(f"Saved response length comparison plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save response length plot: {e}")
    plt.close()

def plot_satisfaction_distribution(df: pd.DataFrame, output_dir: str):
    """Mode 3의 만족도 점수 분포 히스토그램 생성"""
    score_col = 'satisfaction_evaluation_overall_satisfaction_score'
    if score_col not in df.columns:
        logger.warning(f"'{score_col}' not found. Skipping satisfaction distribution plot.")
        return

    valid_scores = df[score_col].dropna()
    if valid_scores.empty:
         logger.warning("No valid satisfaction scores found to plot.")
         return

    plt.figure(figsize=(8, 5))
    sns.histplot(valid_scores, bins=5, kde=False, discrete=True) # 1~5점 이산값
    plt.title('Mode 3 응답 만족도 점수 분포')
    plt.xlabel('Overall Satisfaction Score (1-5)')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 6)) # X축 눈금 1, 2, 3, 4, 5
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mode3_satisfaction_distribution.png')
    try:
        plt.savefig(save_path)
        logger.info(f"Saved satisfaction distribution plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save satisfaction distribution plot: {e}")
    plt.close()

def plot_model_routing_distribution(df: pd.DataFrame, output_dir: str):
    """Mode 3에서 라우팅된 모델 분포 파이 차트 생성"""
    model_col = 'mode3_model_chosen'
    if model_col not in df.columns:
        logger.warning(f"'{model_col}' column not found. Skipping model routing plot.")
        return

    model_counts = df[model_col].value_counts()
    if model_counts.empty:
         logger.warning("No model routing data found to plot.")
         return

    plt.figure(figsize=(8, 8))
    plt.pie(model_counts, labels=model_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    plt.title('Mode 3 사용 모델 분포 (Model Routing)')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mode3_model_routing_piechart.png')
    try:
        plt.savefig(save_path)
        logger.info(f"Saved model routing distribution plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model routing plot: {e}")
    plt.close()


def plot_performance_by_difficulty(df: pd.DataFrame, output_dir: str):
    """사전 정의된 질문 난이도별 Mode 3 성능(Latency, Satisfaction) 비교"""
    difficulty_col = 'test_difficulty' # test_generator.py 에서 추가된 컬럼
    latency_col = 'results_mode3_latency_seconds'
    score_col = 'satisfaction_evaluation_overall_satisfaction_score'

    if difficulty_col not in df.columns:
        logger.warning(f"'{difficulty_col}' column not found. Skipping performance by difficulty plots.")
        return

    # Latency 비교
    if latency_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=difficulty_col, y=latency_col, data=df, order=['basic', 'advanced'])
        plt.title('질문 난이도별 Mode 3 응답 속도')
        plt.xlabel('Predefined Test Difficulty')
        plt.ylabel('Latency (Seconds)')
        plt.tight_layout()
        save_path_latency = os.path.join(output_dir, 'mode3_latency_by_difficulty.png')
        try:
            plt.savefig(save_path_latency)
            logger.info(f"Saved latency by difficulty plot to: {save_path_latency}")
        except Exception as e:
            logger.error(f"Failed to save latency by difficulty plot: {e}")
        plt.close()
    else:
        logger.warning(f"'{latency_col}' not found. Skipping latency by difficulty plot.")


    # Satisfaction 비교
    if score_col in df.columns:
        # 결측치 제거 후 데이터프레임 생성
        plot_df = df[[difficulty_col, score_col]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=difficulty_col, y=score_col, data=plot_df, order=['basic', 'advanced'])
            plt.title('질문 난이도별 Mode 3 응답 만족도')
            plt.xlabel('Predefined Test Difficulty')
            plt.ylabel('Overall Satisfaction Score (1-5)')
            plt.ylim(0.5, 5.5) # Y축 범위 1~5점으로 고정
            plt.yticks(range(1, 6))
            plt.tight_layout()
            save_path_score = os.path.join(output_dir, 'mode3_satisfaction_by_difficulty.png')
            try:
                plt.savefig(save_path_score)
                logger.info(f"Saved satisfaction by difficulty plot to: {save_path_score}")
            except Exception as e:
                logger.error(f"Failed to save satisfaction by difficulty plot: {e}")
            plt.close()
        else:
             logger.warning("No valid satisfaction scores found for difficulty comparison.")
    else:
        logger.warning(f"'{score_col}' not found. Skipping satisfaction by difficulty plot.")


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """ 주요 비교 지표 요약 리포트 생성 (텍스트 파일) """
    logger.info("Generating summary report...")
    report_lines = []
    report_lines.append("--- Chatbot Test Results Summary ---")
    report_lines.append(f"Report generated on: {pd.Timestamp.now()}")
    report_lines.append(f"Total test cases analyzed: {len(df)}")

    # Latency Summary
    report_lines.append("\n--- Average Latency (seconds) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_latency_seconds'
        if col in df.columns:
            avg_latency = df[col].mean() # NaN은 자동 제외
            report_lines.append(f"Mode {mode[-1]}: {avg_latency:.4f}")
        else:
            report_lines.append(f"Mode {mode[-1]}: N/A")

    # Success Rate Summary
    report_lines.append("\n--- Success Rate (%) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_success'
        if col in df.columns:
            success_rate = df[col].mean() * 100 # True=1, False=0
            report_lines.append(f"Mode {mode[-1]}: {success_rate:.2f}%")
        else:
             report_lines.append(f"Mode {mode[-1]}: N/A")

    # Response Length Summary
    report_lines.append("\n--- Average Response Length (characters) ---")
    for mode in ['mode1', 'mode2', 'mode3']:
        col = f'results_{mode}_response_length'
        if col in df.columns:
            avg_len = df[col].mean()
            report_lines.append(f"Mode {mode[-1]}: {avg_len:.1f}")
        else:
            report_lines.append(f"Mode {mode[-1]}: N/A")

    # Mode 3 Satisfaction Summary
    score_col = 'satisfaction_evaluation_overall_satisfaction_score'
    report_lines.append("\n--- Mode 3 Satisfaction (Overall Score 1-5) ---")
    if score_col in df.columns:
        valid_scores = df[score_col].dropna()
        if not valid_scores.empty:
            avg_score = valid_scores.mean()
            median_score = valid_scores.median()
            std_dev = valid_scores.std()
            report_lines.append(f"Average: {avg_score:.2f}")
            report_lines.append(f"Median: {median_score:.2f}")
            report_lines.append(f"Std Dev: {std_dev:.2f}")
            report_lines.append(f"Evaluated Count: {len(valid_scores)}")
        else:
             report_lines.append("No valid scores found.")
    else:
        report_lines.append("Satisfaction score column not found.")

    # Mode 3 Model Routing Summary
    model_col = 'mode3_model_chosen'
    report_lines.append("\n--- Mode 3 Model Usage ---")
    if model_col in df.columns:
         model_counts = df[model_col].value_counts(normalize=True) * 100
         if not model_counts.empty:
             for model, percentage in model_counts.items():
                 report_lines.append(f"- {model}: {percentage:.1f}%")
         else:
             report_lines.append("No model usage data found.")
    else:
        report_lines.append("Model chosen column not found.")

    # Improvement Ratio (Mode 2 vs Mode 3) - 예시
    report_lines.append("\n--- Improvement Summary (Mode 3 vs Mode 2) ---")
    try:
        avg_latency_m2 = df['results_mode2_latency_seconds'].mean()
        avg_latency_m3 = df['results_mode3_latency_seconds'].mean()
        if pd.notna(avg_latency_m2) and pd.notna(avg_latency_m3) and avg_latency_m2 > 0:
            latency_change = ((avg_latency_m3 - avg_latency_m2) / avg_latency_m2) * 100
            report_lines.append(f"Latency Change: {latency_change:+.1f}%") # + or -
        else: report_lines.append("Latency Change: N/A")

        # 여기에 비용, 만족도 등의 개선율 계산 추가 가능 (데이터 확인 후)

    except KeyError:
         report_lines.append("Could not calculate improvement ratios (missing data).")


    # 파일 저장
    report_path = os.path.join(output_dir, 'summary_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        logger.info(f"Saved summary report to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save summary report: {e}")


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize chatbot test results.")
    # 입력 파일 인자: 특정 파일 또는 패턴(glob) 지원
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the results .jsonl file or a glob pattern (e.g., 'tests/test_results/*.jsonl').")
    # 출력 디렉토리 인자
    parser.add_argument('--output-dir', type=str, default='test_plots',
                        help="Directory to save the generated plots and report.")
    args = parser.parse_args()

    logger.info("--- Starting Test Results Visualization ---")
    logger.info(f"Input file/pattern: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 데이터 로드
    df_raw = load_results(args.input)

    if df_raw is None or df_raw.empty:
        logger.error("Failed to load or no data found in results file(s). Exiting.")
        exit()

    # 2. 데이터 전처리
    df_processed = preprocess_data(df_raw)

    # 3. 그래프 생성
    plot_latency_comparison(df_processed, args.output_dir)
    plot_response_length_comparison(df_processed, args.output_dir)
    plot_satisfaction_distribution(df_processed, args.output_dir)
    plot_model_routing_distribution(df_processed, args.output_dir)
    plot_performance_by_difficulty(df_processed, args.output_dir)
    # 추가 그래프 함수 호출... (예: 비용, 정확성 등은 데이터 확인 후 추가 구현)

    # 4. 요약 리포트 생성
    generate_summary_report(df_processed, args.output_dir)

    logger.info("--- Test Results Visualization Finished ---")