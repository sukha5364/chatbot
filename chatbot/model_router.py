# chatbot/model_router.py (최종 수정 계획 반영 - 기능 제거/축소 버전)

import json
import logging
from typing import Optional, Dict, Any, List, Union
# import aiohttp # 더 이상 사용 안 함
# import time # 더 이상 사용 안 함

# --- 필요한 모듈 임포트 (기존 의존성 제거됨) ---
# try:
#     # from .gpt_interface import call_gpt_async
#     # from .config_loader import get_config
#     logging.info("Dependencies no longer needed for model_router in Tool Use approach.")
# except ImportError as ie:
#     logging.error(f"ERROR (model_router): Failed to import modules: {ie}.", exc_info=True)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)
logger.warning("Module 'model_router.py' is deprecated and its functionalities are no longer used.")
logger.warning("Model routing and reasoning logic are now handled within 'scheduler.py' using the Tool Use approach.")

# --- [삭제됨] 복잡도 분류 함수 ---
# async def classify_complexity_level(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     """
#     # logger.warning("classify_complexity_level function is deprecated and should not be called.")
#     # return "easy" # 호출되지 않아야 함

# --- [삭제됨] 범용 CoT 생성 함수 ---
# async def generate_general_cot_async(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     """
#     # logger.warning("generate_general_cot_async function is deprecated and should not be called.")
#     # return None # 호출되지 않아야 함

# --- [삭제됨] 라우팅 결정 함수 ---
# async def determine_routing_and_reasoning(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     """
#     # logger.warning("determine_routing_and_reasoning function is deprecated and should not be called.")
#     # return {"level": "easy", "model": "gpt-3.5-turbo", "cot_data": None} # 호출되지 않아야 함


# --- (선택적) 유틸리티 함수 ---
# (관련 유틸리티 함수 없음)

# --- 예시 사용법 (삭제 또는 주석 처리) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # 로거 재정의 (테스트용)
    logger.info("--- model_router.py executed as main script (DEPRECATED) ---")
    print("="*50)
    print("WARNING: This module (model_router.py) is DEPRECATED.")
    print("The functionalities for complexity classification, CoT generation,")
    print("and explicit model routing have been replaced by the")
    print("Tool Use workflow managed within scheduler.py.")
    print("This file should ideally be removed or kept only for reference.")
    print("="*50)
    # 기존 테스트 코드 제거됨