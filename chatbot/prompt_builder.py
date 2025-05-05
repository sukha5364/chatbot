# chatbot/prompt_builder.py (최종 수정 계획 반영 - 기능 제거/축소 버전)

import logging
from typing import List, Dict, Optional, Union, Any

# --- 필요한 모듈 임포트 (기존 의존성 제거됨) ---
# try:
#     # from .config_loader import get_config
#     # from .conversation_state import ConversationState # 상태 직접 참조 안 함
#     logging.info("Dependencies potentially no longer needed for prompt_builder in Tool Use approach.")
# except ImportError as ie:
#     logging.error(f"ERROR (prompt_builder): Failed to import modules: {ie}.", exc_info=True)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)
logger.warning("Module 'prompt_builder.py' is deprecated and its functionalities are no longer used.")
logger.warning("Prompt assembly logic is now integrated into the Tool Use workflow within 'scheduler.py'.")


# --- [삭제됨] 최종 프롬프트 생성 함수 ---
# def build_final_prompt(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     주어진 컨텍스트, RAG 결과, CoT 등을 조합하여 최종 프롬프트를 생성하는 로직은
#     이제 scheduler.py 내부에서 Tool Use 워크플로우의 각 단계(1차 호출, 2차 호출 등)에 맞게
#     직접 메시지 리스트를 구성하는 방식으로 처리됩니다.
#     """
#     # logger.warning("build_final_prompt function is deprecated and should not be called.")
#     # return None # 호출되지 않아야 함


# --- (선택적) 유틸리티 함수 ---
# (관련 유틸리티 함수 없음 - 필요 시 scheduler.py 등으로 이동)

# --- 예시 사용법 (삭제 또는 주석 처리) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # 로거 재정의 (테스트용)
    logger.info("--- prompt_builder.py executed as main script (DEPRECATED) ---")
    print("="*50)
    print("WARNING: This module (prompt_builder.py) is DEPRECATED.")
    print("The functionality for assembling the final prompt (build_final_prompt)")
    print("has been integrated into the Tool Use workflow managed within scheduler.py,")
    print("which constructs message lists dynamically for different LLM calls.")
    print("This file should ideally be removed or kept only for reference.")
    print("="*50)
    # 기존 테스트 코드 제거됨