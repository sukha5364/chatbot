# chatbot/model_router.py (기능 제거/축소 버전 - Tool Use로 대체됨)

import json
import logging
from typing import Optional, Dict, Any, List, Union
# import aiohttp # 더 이상 사용 안 함
# import time # 더 이상 사용 안 함

# --- 필요한 모듈 임포트 ---
# try:
#     # 더 이상 GPT 호출이나 설정 로드가 필요하지 않음
#     # from .gpt_interface import call_gpt_async
#     # from .config_loader import get_config
#     logging.info("Dependencies no longer needed for model_router in Tool Use approach.")
# except ImportError as ie:
#     logging.error(f"ERROR (model_router): Failed to import modules: {ie}.", exc_info=True)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

# --- [삭제됨] 복잡도 분류 함수 ---
# async def classify_complexity_level(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     GPT(설정된 모델)를 사용하여 사용자 질문의 복잡도를 분석하고,
#     "easy", "medium", "hard" 중 하나로 분류합니다.
#     """
#     logger.warning("classify_complexity_level function is deprecated and should not be called.")
#     return "easy" # 기본값 또는 에러 반환

# --- [삭제됨] 범용 CoT 생성 함수 ---
# async def generate_general_cot_async(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     주어진 사용자 입력과 이전 대화 컨텍스트(요약, 슬롯)를 바탕으로
#     범용적인 단계별 사고 과정(Chain-of-Thought) 계획을 생성합니다.
#     """
#     logger.warning("generate_general_cot_async function is deprecated and should not be called.")
#     return None # 기본값 또는 에러 반환

# --- [삭제됨] 라우팅 결정 함수 ---
# async def determine_routing_and_reasoning(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     복잡도 분류 및 CoT 생성을 수행하고 최종 응답 모델과 CoT 데이터를 결정합니다.
#     """
#     logger.warning("determine_routing_and_reasoning function is deprecated and should not be called.")
#     # 기본 Fallback 값 반환 (실제 호출되면 안 됨)
#     return {"level": "easy", "model": "gpt-3.5-turbo", "cot_data": None}


# --- (선택적) 유틸리티 함수 ---
# 만약 모델 라우팅과 관련 없는 순수 유틸리티 함수가 있었다면 유지 가능
# 예: def _some_helper_function(...): ...

# --- 예시 사용법 (삭제 또는 주석 처리) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- model_router.py executed as main script ---")
    logger.warning("This module's primary functions (routing, CoT) are deprecated due to Tool Use implementation.")
    print("Model Router module's core functionalities are deprecated.")
    print("Complexity classification, explicit CoT generation, and routing map logic")
    print("are now handled within the Tool Use workflow managed by scheduler.py.")
    # 기존 테스트 코드 제거
    # async def test_model_router_components(): ...
    # try: ... asyncio.run(...) ...