# chatbot/conversation_state.py (백그라운드 업데이트 완료 Event 최종 적용 버전)

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
        history (List[Dict[str, str]]): 전체 대화 기록 리스트 (user, assistant, tool 역할 포함).
        update_complete_event (asyncio.Event): 백그라운드 업데이트 작업 완료 시그널 이벤트.
    """
    def __init__(self):
        """ConversationState 인스턴스를 초기화합니다."""
        self.slots: Dict[str, Any] = {}
        self.summary: Optional[str] = None
        # history는 role과 content 키를 가진 딕셔너리 리스트
        # role은 'user', 'assistant', 'tool'이 될 수 있음
        self.history: List[Dict[str, str]] = []
        # 업데이트 완료 이벤트 객체 생성 및 초기 상태 '완료(set)'로 설정
        self.update_complete_event = asyncio.Event()
        self.update_complete_event.set() # 초기에는 업데이트가 완료된 상태
        logger.debug("ConversationState initialized with update_complete_event set.")

    def update_slots(self, new_slots: Dict[str, Any]):
        """
        새로 추출된 Slot 정보로 기존 Slot 상태를 업데이트합니다.
        new_slots가 None이거나 dict가 아니면 경고만 로깅하고 넘어갑니다.

        Args:
            new_slots (Dict[str, Any]): 새로 추출된 Slot 정보 딕셔너리.
        """
        if not isinstance(new_slots, dict):
            # None 이거나 dict 아닌 경우 경고. Background task 이므로 에러 발생시키지 않음.
            if new_slots is not None:
                 logger.warning(f"Invalid type for new_slots: {type(new_slots)}. Expected dict. Skipping slot update.")
            # else: logger.debug("Received None for new_slots. Skipping slot update.") # None은 정상일 수 있음
            return

        updated_keys = []
        for key, value in new_slots.items():
            # 값이 None이 아닌 경우에만 업데이트 고려 (None으로 덮어쓰지 않음)
            if value is not None:
                 # 기존에 없거나 값이 다른 경우만 업데이트
                if key not in self.slots or self.slots[key] != value:
                    updated_keys.append(key)
                self.slots[key] = value
            # else: # 값이 None이면 기존 값 유지 (또는 삭제 원하면 로직 추가)
            #     if key in self.slots:
            #         logger.debug(f"Slot '{key}' received None, keeping existing value: {self.slots[key]}")

        if updated_keys:
            logger.debug(f"Slots updated/added for keys: {updated_keys}")
            # logger.debug(f"Current Slots state: {self.slots}") # 디버깅 필요 시

    def get_slots(self) -> Dict[str, Any]:
        """
        현재 저장된 모든 Slot 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 현재 Slot 상태 딕셔너리.
        """
        # 방어적으로 복사본 반환 고려 가능 (얕은 복사)
        # return self.slots.copy()
        return self.slots

    def update_summary(self, summary: Optional[str]):
        """
        대화 요약 내용을 업데이트합니다. None이나 빈 문자열은 이전 요약을 유지합니다.

        Args:
            summary (Optional[str]): 새로 생성된 대화 요약 문자열.
        """
        if summary is not None and isinstance(summary, str) and summary.strip():
            self.summary = summary.strip()
            logger.debug(f"Conversation summary updated. New length: {len(self.summary)} chars.")
            # logger.debug(f"Current Summary preview: {self.summary[:100]}...") # 디버깅 필요 시
        elif summary is not None: # None은 아니지만 빈 문자열인 경우
             logger.warning("Received empty string for summary. Skipping update.")
        # else: logger.debug("Received None for summary. Skipping update.")

    def get_summary(self) -> Optional[str]:
        """
        현재 저장된 대화 요약을 반환합니다. 요약이 없으면 None을 반환합니다.

        Returns:
            Optional[str]: 현재 대화 요약 문자열 또는 None.
        """
        return self.summary

    def add_to_history(self, role: str, content: Any, **kwargs):
        """
        대화 내용을 기록(history)에 추가합니다.
        role은 'user', 'assistant', 'tool' 중 하나여야 합니다.
        'assistant' 역할 시 tool_calls 정보 추가 가능.
        'tool' 역할 시 tool_call_id 정보 추가 필수.

        Args:
            role (str): 메시지 발화자 역할 ('user', 'assistant', 'tool').
            content (Any): 메시지 내용 (주로 str). tool 역할 시 JSON 결과 문자열.
            **kwargs: 추가 정보 (예: tool_calls, tool_call_id).
        """
        if role not in ["user", "assistant", "tool"]:
            logger.warning(f"Invalid role '{role}' for history. Using 'unknown'.")
            role = "unknown" # 혹은 에러 발생

        message: Dict[str, Any] = {"role": role}

        # Content 처리
        if not isinstance(content, str):
             logger.warning(f"History content type is not str ({type(content)}). Converting to string.")
             message["content"] = str(content)
        else:
             message["content"] = content

        # 역할별 추가 정보 처리
        if role == "assistant":
            tool_calls = kwargs.get('tool_calls')
            if tool_calls:
                message["tool_calls"] = tool_calls # OpenAI 형식 그대로 저장
                # content가 None이고 tool_calls가 있을 수 있음
                if message["content"] is None: message["content"] = "" # content는 필수 필드 가정
        elif role == "tool":
            tool_call_id = kwargs.get('tool_call_id')
            if not tool_call_id:
                 logger.error("Missing 'tool_call_id' for role 'tool' in history. This might break context.")
                 # return # 또는 ID 없이 추가? API 요구사항 확인 필요
            else:
                 message["tool_call_id"] = tool_call_id
            # tool 역할 메시지의 content는 일반적으로 tool 실행 결과 (JSON 문자열)

        self.history.append(message)
        logger.debug(f"Added '{role}' message to history. History length: {len(self.history)}")
        # logger.debug(f"Last message added: {message}") # 디버깅 필요 시

    def get_history(self) -> List[Dict[str, Any]]:
        """
        전체 대화 기록 리스트를 반환합니다.
        호출자에게 원본 리스트 대신 복사본을 제공하여 외부 변경 방지.

        Returns:
            List[Dict[str, Any]]: 전체 대화 기록 리스트 (복사본).
        """
        # 중요: 상태 객체 내부 리스트를 직접 반환하지 않고 복사본 반환
        return self.history[:]

    def clear(self):
        """모든 대화 상태(slots, summary, history)를 초기화하고 업데이트 완료 상태로 설정합니다."""
        self.slots = {}
        self.summary = None
        self.history = []
        self.update_complete_event.set() # 초기화 시에도 완료 상태로 설정
        logger.info("Conversation state (slots, summary, history) cleared and update_complete_event is set.")

# --- 예시 사용법 (변경 없음) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # logger 재정의 필요 없음
    logger.info("--- Running ConversationState Example ---")

    state = ConversationState()
    print(f"Initial event state (should be set): {state.update_complete_event.is_set()}")

    # Slot 업데이트
    initial_slots = {"brand": "나이키", "size": "270mm", "foot_width": None}
    state.update_slots(initial_slots)
    print(f"Slots after initial update: {state.get_slots()}")
    new_slots = {"foot_width": "넓은 편", "product_category": "러닝화", "brand": None} # brand=None 테스트
    state.update_slots(new_slots)
    print(f"Slots after second update (brand=None ignored): {state.get_slots()}")

    # 요약 업데이트
    summary_text = "고객은 나이키 270mm를 신으며 발볼이 넓은 편이고, 데카트론 러닝화에 관심 있음."
    state.update_summary(summary_text)
    print(f"Current Summary: {state.get_summary()}")
    state.update_summary("") # 빈 문자열 테스트
    print(f"Summary after empty string update (should be unchanged): {state.get_summary()}")
    state.update_summary(None) # None 테스트
    print(f"Summary after None update (should be unchanged): {state.get_summary()}")


    # 대화 기록 추가 (user, assistant, tool)
    state.add_to_history("user", "나이키 페가수스랑 비슷한 데카트론 신발 찾아줘.")
    # Assistant 응답 (Tool 호출 포함 가정)
    assistant_tool_call = {
        "role": "assistant",
        "content": None, # 답변 대신 Tool 호출만 할 경우 content는 None일 수 있음
        "tool_calls": [
            {"id": "call_abc", "type": "function", "function": {"name": "product_search", "arguments": "{...}"}}
        ]
    }
    state.add_to_history(role="assistant", content=assistant_tool_call.get("content"), tool_calls=assistant_tool_call.get("tool_calls"))
    # Tool 결과
    tool_result_content = json.dumps({"results": [{"name": "Kiprun KS900"}], "results_found": True})
    state.add_to_history(role="tool", content=tool_result_content, tool_call_id="call_abc")
    # 최종 Assistant 응답
    state.add_to_history("assistant", "네, 킵런 KS900 모델을 추천합니다.")

    print(f"\nCurrent History (Copy): {state.get_history()}")

    # 상태 초기화 및 이벤트 상태 확인
    state.clear()
    print(f"\nSlots after clear: {state.get_slots()}")
    print(f"Summary after clear: {state.get_summary()}")
    print(f"History after clear: {state.get_history()}")
    print(f"Event state after clear (should be set): {state.update_complete_event.is_set()}")

    # 이벤트 상태 변경 시뮬레이션
    state.update_complete_event.clear()
    print(f"Event state after clear() (should be cleared): {state.update_complete_event.is_set()}")
    state.update_complete_event.set()
    print(f"Event state after set() (should be set): {state.update_complete_event.is_set()}")

    logger.info("--- ConversationState Example Finished ---")