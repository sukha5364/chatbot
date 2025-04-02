# tests/test_generator.py

import json
import random
import os

# --- 설정 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_cases') # tests/test_cases/ 에 저장
NUM_SETS = 3  # 생성할 테스트셋 파일 개수
QUESTIONS_PER_SET = 50 # 세트 당 질문 개수

# 질문 템플릿
TEMPLATES = [
    "{brand} {size} 신는데 어떤 사이즈 추천해줘?",
    "{brand} 러닝화는 발볼이 넓나요?",
    "{brand} 신발과 {other_brand} 비교하면 어때요?",
    "{brand} 신발 반품 정책 알려줘.",
    "가장 인기 있는 {category} 제품은 뭔가요?",
    "{category} 제품 재고 있나요?",
    "가까운 데카트론 매장 위치 알려줘.",
    "{brand} {category} 신제품 나왔나요?",
    "{size} 사이즈 재고 확인 부탁드려요.",
    "발볼이 {foot_width} 편인데 {brand} 신발 괜찮을까요?",
    "온라인 주문 배송 얼마나 걸려요?",
]

# 키워드 리스트
BRANDS = ["나이키", "아디다스", "호카", "데카트론", "뉴발란스", "아식스"]
SIZES = ["260mm", "265", "270", "275mm", "280", "285", "US 9", "UK 8"]
CATEGORIES = ["러닝화", "트레킹화", "축구화", "운동화", "등산화", "스포츠 양말", "기능성 티셔츠"]
FOOT_WIDTHS = ["좁은", "보통", "넓은", "아주 넓은"]

def generate_test_questions(n=50):
    """랜덤한 테스트 질문 데이터 생성"""
    questions = []
    used_questions = set() # 중복 방지용

    while len(questions) < n:
        template = random.choice(TEMPLATES)
        other_brand = random.choice([b for b in BRANDS if b != "{brand}"]) # 자기 자신 제외

        try:
            # format_map 사용 시 누락된 키 에러 방지
            params = {
                "brand": random.choice(BRANDS),
                "other_brand": other_brand,
                "size": random.choice(SIZES),
                "category": random.choice(CATEGORIES),
                "foot_width": random.choice(FOOT_WIDTHS)
            }
            # 템플릿에 필요한 키만 전달
            required_keys = {key for key in ["brand", "other_brand", "size", "category", "foot_width"] if f"{{{key}}}" in template}
            format_params = {k: params[k] for k in required_keys}

            question = template.format(**format_params)

            if question not in used_questions:
                # TODO: 향후 expected_answer, expected_slots 등 추가 필드 생성 로직 구현 가능
                questions.append({
                    "question": question,
                    # "expected_answer": "...", # 예시: 정답이 명확한 경우
                    # "category": "...",       # 질문 유형 분류
                    # "needs_rag": True/False, # RAG 필요 여부
                    # "expected_slots": {"brand": "...", "size": "..."} # 예상 Slot
                    })
                used_questions.add(question)
        except KeyError as e:
            # 가끔 format 키가 누락될 수 있음 (템플릿과 파라미터 불일치 시)
            # print(f"Skipping due to KeyError: {e} in template: {template}")
            continue
        except Exception as e:
            print(f"Error generating question: {e}")

    return questions

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("--- Starting Test Set Generation ---")
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    for i in range(NUM_SETS):
        set_filename = f"test_set_{i+1}.jsonl"
        set_filepath = os.path.join(OUTPUT_DIR, set_filename)
        print(f"Generating {set_filename} ({QUESTIONS_PER_SET} questions)...")

        generated_questions = generate_test_questions(QUESTIONS_PER_SET)

        try:
            with open(set_filepath, "w", encoding="utf-8") as f:
                for item in generated_questions:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Successfully generated {set_filename}")
        except Exception as e:
            print(f"Error writing file {set_filename}: {e}")

    print(f"\n--- Test Set Generation Finished ({NUM_SETS} sets created) ---")