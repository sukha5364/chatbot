# chatbot/prompt_builder.py (기능 제거/축소 버전 - Tool Use로 대체됨)

import logging
from typing import List, Dict, Optional, Union, Any

# --- 필요한 모듈 임포트 ---
# try:
#     # 더 이상 직접적인 설정 로드나 상태 객체 사용이 필요 없을 수 있음
#     # from .config_loader import get_config
#     logging.info("Dependencies potentially no longer needed for prompt_builder in Tool Use approach.")
# except ImportError as ie:
#     logging.error(f"ERROR (prompt_builder): Failed to import modules: {ie}.", exc_info=True)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

# --- [삭제됨] 최종 프롬프트 생성 함수 ---
# def build_final_prompt(...):
#     """
#     [삭제됨] Tool Use 방식으로 대체되었습니다.
#     주어진 사용자 질문, 대화 컨텍스트(요약, 최근 기록, 슬롯), RAG 검색 결과,
#     CoT 데이터를 조합하여 최종적으로 GPT API에 전달할 메시지 리스트를 생성합니다.
#     이 로직은 이제 scheduler.py 내부에서 Tool Use 워크플로우의 각 단계에 맞게 처리됩니다.
#     (1차 LLM 호출 시 Tool 지침 포함 프롬프트 생성, 2차 LLM 호출 시 Tool 결과 포함 메시지 생성 등)
#     """
#     logger.warning("build_final_prompt function is deprecated and should not be called.")
#     return None # 기본값 또는 에러 반환


# --- (선택적) 유틸리티 함수 ---
# 만약 프롬프트 구성과 관련 없는 순수 유틸리티 함수가 있었다면 유지 가능
# 예: def _format_context_for_prompt(...): ...
# 하지만 현재 구조에서는 scheduler.py 에서 직접 처리하는 것이 더 효율적일 수 있습니다.

# --- 예시 사용법 (삭제 또는 주석 처리) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- prompt_builder.py executed as main script ---")
    logger.warning("This module's primary function (build_final_prompt) is deprecated due to Tool Use implementation.")
    print("Prompt Builder module's core functionalities are deprecated.")
    print("Final prompt construction logic is now integrated into the Tool Use workflow")
    print("managed by scheduler.py, handling message assembly for different LLM calls.")
    # 기존 테스트 코드 제거
    # try: ... config = get_config() ... final_messages = build_final_prompt(...) ...