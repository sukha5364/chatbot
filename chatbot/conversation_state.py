# chatbot/conversation_state.py

from typing import Dict, Any, Optional

class ConversationState:
    """
    단일 사용자 세션의 대화 상태를 인메모리에서 관리하는 클래스.
    Slot 정보와 대화 요약을 저장합니다.
    (참고: 실제 서비스에서는 다중 사용자를 위해 Redis 등 외부 저장소 사용 고려)
    """
    def __init__(self):
        self.slots: Dict[str, Any] = {}
        self.summary: Optional[str] = None
        self.history: list[Dict[str, str]] = [] # 전체 대화 기록 (요약 생성 등에 사용 가능)

    def update_slots(self, new_slots: Dict[str, Any]):
        """
        추출된 Slot 정보를 기존 상태에 업데이트합니다.
        None 값은 업데이트하지 않거나, 명시적으로 삭제 로직 추가 가능.
        """
        for key, value in new_slots.items():
            if value is not None: # None 값은 무시 (또는 다른 정책 적용 가능)
                self.slots[key] = value
        # print(f"Slots updated: {self.slots}") # 디버깅용

    def get_slots(self) -> Dict[str, Any]:
        """현재 저장된 Slot 정보를 반환합니다."""
        return self.slots

    def update_summary(self, summary: str):
        """대화 요약 내용을 업데이트합니다."""
        self.summary = summary
        # print(f"Summary updated: {self.summary}") # 디버깅용

    def get_summary(self) -> Optional[str]:
        """현재 저장된 대화 요약을 반환합니다."""
        return self.summary

    def add_to_history(self, role: str, content: str):
        """대화 내용을 기록에 추가합니다."""
        self.history.append({"role": role, "content": content})

    def get_history(self) -> list[Dict[str, str]]:
        """전체 대화 기록을 반환합니다."""
        return self.history

    def clear(self):
        """대화 상태를 초기화합니다."""
        self.slots = {}
        self.summary = None
        self.history = []
        print("Conversation state cleared.")

# --- 예시 사용법 ---
if __name__ == "__main__":
    state = ConversationState()

    # Slot 업데이트
    initial_slots = {"brand": "나이키", "size": "270mm", "foot_width": None}
    state.update_slots(initial_slots)
    print(f"Current Slots: {state.get_slots()}")

    # 추가 Slot 정보 업데이트
    new_slots = {"foot_width": "넓은 편", "category": "러닝화"}
    state.update_slots(new_slots)
    print(f"Current Slots after update: {state.get_slots()}")

    # 대화 요약 업데이트
    state.update_summary("고객은 나이키 270mm를 신으며 발볼이 넓은 편이고, 러닝화에 관심 있음.")
    print(f"Current Summary: {state.get_summary()}")

    # 대화 기록 추가
    state.add_to_history("user", "나이키 270mm 신는데 발볼 넓은 러닝화 추천해주세요.")
    state.add_to_history("assistant", "네, 발볼이 넓으시군요. 데카트론 킵런 시리즈를 추천합니다.")
    print(f"Conversation History: {state.get_history()}")

    # 상태 초기화
    state.clear()
    print(f"Slots after clear: {state.get_slots()}")
    print(f"Summary after clear: {state.get_summary()}")