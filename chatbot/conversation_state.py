# chatbot/conversation_state.py (백그라운드 업데이트 완료 Event 추가)

import logging
import asyncio # asyncio 임포트 추가
from typing import Dict, Any, Optional, List

# 로거 설정 (기본 설정 상속)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

class ConversationState:
    """
    단일 사용자 세션의 대화 상태를 인메모리에서 관리하는 클래스.
    추출된 Slot 정보, 대화 요약, 전체 대화 기록을 저장합니다.
    [신규] 백그라운드 슬롯/요약 업데이트 완료를 위한 Event 객체를 포함합니다.

    Attributes:
        slots (Dict[str, Any]): 추출된 Slot 정보를 저장하는 딕셔너리.
        summary (Optional[str]): 현재까지의 대화 요약 문자열.
        history (List[Dict[str, str]]): 전체 대화 기록 리스트.
        update_complete_event (asyncio.Event): 백그라운드 업데이트 작업 완료 시그널 이벤트.
    """
    def __init__(self):
        """ConversationState 인스턴스를 초기화합니다."""
        self.slots: Dict[str, Any] = {}
        self.summary: Optional[str] = None
        self.history: List[Dict[str, str]] = []
        # [신규] 업데이트 완료 이벤트 객체 생성 및 초기 상태 '완료(set)'로 설정
        self.update_complete_event = asyncio.Event()
        self.update_complete_event.set() # 초기에는 업데이트가 완료된 상태
        logger.debug("ConversationState initialized with update_complete_event set.")

    def update_slots(self, new_slots: Dict[str, Any]):
        """
        새로 추출된 Slot 정보로 기존 Slot 상태를 업데이트합니다.

        Args:
            new_slots (Dict[str, Any]): 새로 추출된 Slot 정보 딕셔너리.
        """
        if not isinstance(new_slots, dict):
            logger.warning(f"Invalid type for new_slots: {type(new_slots)}. Expected dict. Skipping update.")
            return

        updated_keys = []
        for key, value in new_slots.items():
            if value is not None:
                if key not in self.slots or self.slots[key] != value:
                    updated_keys.append(key)
                self.slots[key] = value

        if updated_keys:
            logger.debug(f"Slots updated for keys: {updated_keys}")
            # logger.debug(f"Current Slots: {self.slots}") # 디버깅 시

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
            self.summary = summary.strip()
            logger.debug(f"Conversation summary updated. New length: {len(self.summary)} chars.")
            # logger.debug(f"Current Summary: {self.summary[:100]}...") # 디버깅 시
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
            role (str): 메시지 발화자 역할 ('user' 또는 'assistant' 또는 'tool').
            content (str): 메시지 내용.
        """
        # [수정] 'tool' 역할 추가 허용
        if role not in ["user", "assistant", "tool"]:
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
        """모든 대화 상태(slots, summary, history)를 초기화하고 업데이트 완료 상태로 설정합니다."""
        self.slots = {}
        self.summary = None
        self.history = []
        self.update_complete_event.set() # 초기화 시에도 완료 상태로 설정
        logger.info("Conversation state (slots, summary, history) has been cleared and update_complete_event is set.")

# --- 예시 사용법 (변경 없음) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- Running ConversationState Example ---")

    state = ConversationState()
    print(f"Initial update_complete_event state (should be set): {state.update_complete_event.is_set()}")

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

    # 상태 초기화 및 이벤트 상태 확인
    logger.info("Clearing conversation state...")
    state.clear()
    print(f"Slots after clear: {state.get_slots()}")
    print(f"Summary after clear: {state.get_summary()}")
    print(f"History after clear: {state.get_history()}")
    print(f"update_complete_event state after clear (should be set): {state.update_complete_event.is_set()}")

    # 이벤트 상태 변경 시뮬레이션 (실제로는 app.py에서 제어)
    state.update_complete_event.clear()
    print(f"update_complete_event state after clear() (should be cleared): {state.update_complete_event.is_set()}")
    state.update_complete_event.set()
    print(f"update_complete_event state after set() (should be set): {state.update_complete_event.is_set()}")

    logger.info("--- ConversationState Example Finished ---")