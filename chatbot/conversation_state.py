# chatbot/conversation_state.py (요구사항 반영 최종본: Docstring 및 주석 보강)

import logging
from typing import Dict, Any, Optional, List # List 임포트 추가

# 로거 설정 (기본 설정 상속)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

class ConversationState:
    """
    단일 사용자 세션의 대화 상태를 인메모리에서 관리하는 클래스.
    추출된 Slot 정보, 대화 요약, 전체 대화 기록을 저장합니다.

    Attributes:
        slots (Dict[str, Any]): 추출된 Slot 정보를 저장하는 딕셔너리. (예: {'brand': '나이키', 'size': '270mm'})
        summary (Optional[str]): 현재까지의 대화 요약 문자열. None일 수 있음.
        history (List[Dict[str, str]]): 전체 대화 기록 리스트. 각 항목은 {"role": "user"|"assistant", "content": "..."} 형식.

    Note:
        이 클래스는 단일 사용자 및 단일 세션 환경에 적합합니다.
        실제 다중 사용자 서비스 환경에서는 사용자별 세션 ID와 함께 Redis 같은
        외부 저장소를 사용하여 대화 상태를 관리하는 것이 일반적입니다.
    """
    def __init__(self):
        """ConversationState 인스턴스를 초기화합니다."""
        self.slots: Dict[str, Any] = {}         # Slot 정보를 저장할 딕셔너리 초기화
        self.summary: Optional[str] = None      # 대화 요약을 저장할 변수 초기화 (처음에는 None)
        self.history: List[Dict[str, str]] = [] # 대화 기록을 저장할 리스트 초기화
        logger.debug("ConversationState initialized.")

    def update_slots(self, new_slots: Dict[str, Any]):
        """
        새로 추출된 Slot 정보로 기존 Slot 상태를 업데이트합니다.

        Args:
            new_slots (Dict[str, Any]): 새로 추출된 Slot 정보 딕셔너리.
                                         키는 Slot 이름, 값은 추출된 값.
        """
        if not isinstance(new_slots, dict):
            logger.warning(f"Invalid type for new_slots: {type(new_slots)}. Expected dict. Skipping update.")
            return

        updated_keys = []
        # 새로운 슬롯 정보를 순회하며 업데이트
        for key, value in new_slots.items():
            # 값이 None이 아닌 경우에만 업데이트 (None 값은 무시)
            # 또는 다른 정책 적용 가능 (예: None이면 기존 슬롯 삭제)
            if value is not None:
                # 기존 값과 다를 경우에만 로그 기록 (선택 사항)
                if key not in self.slots or self.slots[key] != value:
                     updated_keys.append(key)
                self.slots[key] = value

        if updated_keys:
             logger.debug(f"Slots updated for keys: {updated_keys}")
             # logger.debug(f"Current Slots: {self.slots}") # 디버깅 시 전체 슬롯 확인용

    def get_slots(self) -> Dict[str, Any]:
        """
        현재 저장된 모든 Slot 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 현재 Slot 상태 딕셔너리.
        """
        return self.slots

    def update_summary(self, summary: str):
        """
        대화 요약 내용을 업데이트합니다.

        Args:
            summary (str): 새로 생성된 대화 요약 문자열.
        """
        if isinstance(summary, str):
            self.summary = summary.strip() # 앞뒤 공백 제거 후 저장
            logger.debug(f"Conversation summary updated. New length: {len(self.summary)} chars.")
            # logger.debug(f"Current Summary: {self.summary[:100]}...") # 디버깅 시 요약 내용 확인용
        else:
            logger.warning(f"Invalid type for summary: {type(summary)}. Expected str. Skipping update.")


    def get_summary(self) -> Optional[str]:
        """
        현재 저장된 대화 요약을 반환합니다. 요약이 없으면 None을 반환합니다.

        Returns:
            Optional[str]: 현재 대화 요약 문자열 또는 None.
        """
        return self.summary

    def add_to_history(self, role: str, content: str):
        """
        대화 내용을 기록(history)에 추가합니다.

        Args:
            role (str): 메시지 발화자 역할 ('user' 또는 'assistant').
            content (str): 메시지 내용.
        """
        if role not in ["user", "assistant"]:
            logger.warning(f"Invalid role '{role}' for history. Using 'unknown'.")
            role = "unknown"
        if not isinstance(content, str):
             logger.warning(f"Invalid content type for history: {type(content)}. Converting to string.")
             content = str(content)

        self.history.append({"role": role, "content": content})
        logger.debug(f"Added '{role}' message to history. History length: {len(self.history)}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        전체 대화 기록 리스트를 반환합니다.

        Returns:
            List[Dict[str, str]]: 전체 대화 기록 리스트.
        """
        return self.history

    def clear(self):
        """모든 대화 상태(slots, summary, history)를 초기화합니다."""
        self.slots = {}
        self.summary = None
        self.history = []
        logger.info("Conversation state (slots, summary, history) has been cleared.")

# --- 예시 사용법 (기존 유지, 로깅 확인용) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running ConversationState Example ---")

    state = ConversationState()

    # Slot 업데이트
    initial_slots = {"brand": "나이키", "size": "270mm", "foot_width": None}
    logger.info(f"Updating slots with: {initial_slots}")
    state.update_slots(initial_slots)
    print(f"Current Slots: {state.get_slots()}")

    # 추가 Slot 정보 업데이트
    new_slots = {"foot_width": "넓은 편", "product_category": "러닝화"}
    logger.info(f"Updating slots with: {new_slots}")
    state.update_slots(new_slots)
    print(f"Current Slots after update: {state.get_slots()}")

    # 대화 요약 업데이트
    summary_text = "고객은 나이키 270mm를 신으며 발볼이 넓은 편이고, 데카트론 러닝화에 관심 있음."
    logger.info(f"Updating summary with: '{summary_text[:50]}...'")
    state.update_summary(summary_text)
    print(f"Current Summary: {state.get_summary()}")

    # 대화 기록 추가
    logger.info("Adding messages to history...")
    state.add_to_history("user", "나이키 270mm 신는데 발볼 넓은 러닝화 추천해주세요.")
    state.add_to_history("assistant", "네, 발볼이 넓으시군요. 데카트론 킵런 시리즈를 추천합니다.")
    print(f"Current History: {state.get_history()}")

    # 상태 초기화
    logger.info("Clearing conversation state...")
    state.clear()
    print(f"Slots after clear: {state.get_slots()}")
    print(f"Summary after clear: {state.get_summary()}")
    print(f"History after clear: {state.get_history()}")
    logger.info("--- ConversationState Example Finished ---")