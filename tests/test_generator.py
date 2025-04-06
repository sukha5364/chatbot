# tests/test_generator.py

import json
import random
import os
import logging
from typing import List, Dict, Tuple, Literal, Optional # Optional 추가

# 로거 설정
logger = logging.getLogger(__name__)
# 기본 로깅 레벨 INFO로 설정 (필요시 DEBUG로 변경)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 설정 ---
# 출력 디렉토리: tests/test_cases/
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_cases')

# 생성할 테스트셋 설정
TEST_SET_CONFIG = {
    "function": {
        "count": 30, # 기능 테스트 질문 수
        "complexity_ratio": None, # 기능 테스트는 난이도 무관하게 생성
        "filename_template": "functional_tests.jsonl"
    },
    "overall": {
        "count": 200, # 전체 테스트 세트당 총 질문 수
        "complexity_ratio": 0.5, # 전체 테스트는 basic 50%, advanced 50% 비율
        "num_sets": 3, # 전체 테스트 세트 개수
        "filename_template": "overall_tests_set{set_num}.jsonl"
    }
}

# [수정] 질문 템플릿 (난이도 마커 추가: 'basic' 또는 'advanced')
# 기준: basic = 단일 의도, 정보 검색형 / advanced = 비교, 추론, 다중 의도, 맥락 의존
TEMPLATES: List[Tuple[Literal['basic', 'advanced'], str]] = [
    # Basic Templates
    ('basic', "{brand} {size} 재고 확인해줘."),
    ('basic', "{brand} 신발 반품 정책 알려줘."),
    ('basic', "가까운 데카트론 매장 위치 알려줘."),
    ('basic', "{size} 사이즈 재고 있어?"),
    ('basic', "온라인 주문 배송 얼마나 걸려요?"),
    ('basic', "{product_category} 제품 중에 제일 싼 거 뭐야?"),
    ('basic', "{brand} 매장 전화번호 알려줘."),
    ('basic', "{product_category} 사용법 알려줘."),
    ('basic', "영업시간이 어떻게 되나요?"),

    # Advanced Templates
    ('advanced', "{brand} {size} 신는데 {other_brand} {size2}와 비교하면 어때?"),
    ('advanced', "{brand} 러닝화는 발볼이 {foot_width} 사람에게도 괜찮을까?"),
    ('advanced', "가장 인기 있는 {category} 제품 추천해줘. {feature} 기능이 있으면 좋겠어."),
    ('advanced', "{activity} 할 때 입기 좋은 {product_category} 추천해줘. 나는 {skill_level}이고 {goal}을 위해 운동해."),
    ('advanced', "발볼이 {foot_width} 편인데 {brand} 신발 괜찮을까요? 주로 {environment}에서 신을 거야."),
    ('advanced', "{brand} {category} 신제품 나왔어? 이전에 쓰던 모델보다 뭐가 좋아졌는지 설명해줘."),
    ('advanced', "나는 {age_group} {gender}인데, {activity}용으로 {budget} 가격대의 {product_category}를 찾고 있어. 어떤 게 좋을까?"),
    ('advanced', "겨울철 {activity}에 적합한 {product_category}를 찾는데, {feature} 기능이랑 {feature2} 기능 둘 다 중요해. 어떤 제품이 좋을지 이유와 함께 설명해줘."),
    ('advanced', "지난 번 {brand} {product_category} 구매했는데 {problem} 문제가 발생했어. 어떻게 해결해야 할까?")
]

# 키워드 리스트 (더 다양하게 추가 가능)
BRANDS = ["나이키", "아디다스", "호카", "데카트론", "뉴발란스", "아식스", "킵런", "퀘차", "포클라즈"]
SIZES = ["260mm", "265", "270", "275mm", "280", "285", "US 9", "UK 8", "M", "L", "XL", "95", "100", "105", "S"]
CATEGORIES = ["러닝화", "트레킹화", "축구화", "운동화", "등산화", "스포츠 양말", "기능성 티셔츠", "텐트", "배낭", "자켓", "침낭", "바지"]
FOOT_WIDTHS = ["좁은", "보통", "넓은", "아주 넓은"]
FEATURES = ["방수되는", "가벼운", "쿠션 좋은", "통기성 좋은", "내구성 강한", "보온성 좋은", "빠르게 마르는", "신축성 있는"]
ACTIVITIES = ["러닝", "등산", "캠핑", "축구", "트레일 러닝", "일상", "여행", "백패킹", "낚시", "자전거"]
SKILL_LEVELS = ["초보", "중급자", "상급자", "입문자", "전문가"]
GOALS = ["다이어트", "대회 준비", "취미", "건강 관리", "기록 단축", "생존", "편안함"]
ENVIRONMENTS = ["로드", "트레일", "우천 시", "겨울철", "실내", "도심", "야간", "해변"]
AGE_GROUPS = ["20대", "30대", "40대", "50대", "청소년", "성인", "아동", "노년"]
GENDERS = ["남자", "여자", "남성", "여성"]
BUDGETS = ["10만원 이하", "10만원대", "20만원 미만", "30만원 이하", "5만원 근처", "상관없음"]
PROBLEMS = ["사이즈가 안맞아", "색상이 달라", "제품에 하자가 있어", "사용법을 모르겠어", "배송이 안 와"]

# 중복 제거 및 리스트화
BRANDS = list(set(BRANDS))
# ... 다른 키워드 리스트들도 필요시 중복 제거 ...
FEATURES2 = list(set(FEATURES)) # feature2 용도


# [수정] 난이도별, 개수별 질문 생성 함수
def generate_test_questions(n: int, target_complexity: Optional[Literal['basic', 'advanced']] = None) -> List[Dict]:
    """
    지정된 난이도와 개수만큼 랜덤한 테스트 질문 데이터를 생성합니다.
    결과 딕셔너리에 'test_difficulty' 키를 포함합니다.
    """
    questions = []
    used_questions = set() # 중복 방지용

    # 해당 난이도의 템플릿만 필터링
    if target_complexity:
        filtered_templates = [tpl for tpl in TEMPLATES if tpl[0] == target_complexity]
        if not filtered_templates:
             logger.warning(f"No templates found for complexity: {target_complexity}")
             return []
    else:
        filtered_templates = TEMPLATES # None이면 모든 템플릿 사용

    logger.info(f"Generating {n} questions with complexity: {target_complexity or 'any'}...")

    attempts = 0
    # 최대 시도 횟수를 늘려 중복으로 인해 목표 개수를 못 채우는 경우 줄이기
    max_attempts = n * 10

    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        if not filtered_templates: # 필터링 결과 템플릿이 없으면 중단
             logger.warning("No templates available to generate questions.")
             break

        complexity_marker, template = random.choice(filtered_templates)

        # 키워드 랜덤 선택
        brand = random.choice(BRANDS)
        other_brand = random.choice([b for b in BRANDS if b != brand])
        size = random.choice(SIZES)
        size2 = random.choice([s for s in SIZES if s != size])
        category = random.choice(CATEGORIES)
        feature = random.choice(FEATURES)
        feature2 = random.choice([f for f in FEATURES2 if f != feature]) # 다른 특징
        problem = random.choice(PROBLEMS)

        try:
            params = {
                "brand": brand, "other_brand": other_brand,
                "size": size, "size2": size2,
                "category": category, "product_category": category, # 둘 다 사용 가능하게
                "foot_width": random.choice(FOOT_WIDTHS),
                "feature": feature, "feature2": feature2,
                "activity": random.choice(ACTIVITIES),
                "skill_level": random.choice(SKILL_LEVELS),
                "goal": random.choice(GOALS),
                "environment": random.choice(ENVIRONMENTS),
                "age_group": random.choice(AGE_GROUPS),
                "gender": random.choice(GENDERS),
                "budget": random.choice(BUDGETS),
                "problem": problem
            }
            # 템플릿에 필요한 키만 찾기
            required_keys = {key for key in params.keys() if f"{{{key}}}" in template}
            format_params = {k: params[k] for k in required_keys}

            question = template.format(**format_params)

            if question not in used_questions:
                item = {
                    "question": question,
                    "test_difficulty": complexity_marker # [신규] 난이도 마커 추가 ('basic' or 'advanced')
                    # TODO: expected_slots 등 추가 필드 생성 로직 구현 가능
                }
                questions.append(item)
                used_questions.add(question)
        except KeyError as e:
            # logger.warning(f"Skipping template due to missing key {e}: {template}")
            continue
        except Exception as e:
            logger.error(f"Error generating question for template '{template}': {e}", exc_info=False) # 스택 트레이스 제외

    if len(questions) < n:
         logger.warning(f"Could only generate {len(questions)} unique questions out of {n} requested (attempts: {attempts}). Increase attempts or add more templates/keywords.")

    return questions


# --- 메인 실행 로직 (수정됨: 기능/전체 테스트 분리 생성) ---
if __name__ == "__main__":
    logger.info("--- Starting Test Set Generation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # 1. 기능 테스트셋 생성
    logger.info("Generating Functional Test Set...")
    func_config = TEST_SET_CONFIG.get('function')
    if func_config:
        func_questions = generate_test_questions(func_config['count'], target_complexity=None) # 모든 난이도 포함
        func_filename = func_config['filename_template']
        func_filepath = os.path.join(OUTPUT_DIR, func_filename)
        try:
            with open(func_filepath, "w", encoding="utf-8") as f:
                for item in func_questions:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Successfully generated functional test set: {func_filename} ({len(func_questions)} questions)")
        except Exception as e:
            logger.error(f"Error writing file {func_filename}: {e}")
    else:
         logger.warning("Functional test configuration not found in TEST_SET_CONFIG.")

    # 2. 전체 테스트셋 생성 (여러 세트)
    overall_config = TEST_SET_CONFIG.get('overall')
    if overall_config:
        num_sets = overall_config.get('num_sets', 1)
        count_per_set = overall_config.get('count', 100)
        ratio = overall_config.get('complexity_ratio', 0.5)
        num_basic = int(count_per_set * ratio)
        num_advanced = count_per_set - num_basic
        filename_tpl = overall_config.get('filename_template')

        if not filename_tpl:
             logger.error("Overall test filename template not found in config.")
        else:
             for i in range(1, num_sets + 1):
                 set_filename = filename_tpl.format(set_num=i)
                 set_filepath = os.path.join(OUTPUT_DIR, set_filename)
                 logger.info(f"Generating Overall Test Set {i} ({set_filename})...")

                 # Basic, Advanced 질문 각각 생성 후 합치고 섞기
                 basic_questions = generate_test_questions(num_basic, target_complexity='basic')
                 advanced_questions = generate_test_questions(num_advanced, target_complexity='advanced')
                 combined_questions = basic_questions + advanced_questions
                 random.shuffle(combined_questions)

                 # 실제 생성된 개수 로깅
                 actual_basic = len(basic_questions)
                 actual_advanced = len(advanced_questions)
                 actual_total = len(combined_questions)

                 if actual_total < count_per_set * 0.9: # 목표치의 90% 미만이면 경고
                      logger.warning(f"Generated significantly fewer questions ({actual_total}) than requested ({count_per_set}) for set {i}.")

                 try:
                     with open(set_filepath, "w", encoding="utf-8") as f:
                         for item in combined_questions:
                             f.write(json.dumps(item, ensure_ascii=False) + '\n')
                     logger.info(f"Successfully generated overall test set: {set_filename} ({actual_total} questions: {actual_basic} basic, {actual_advanced} advanced)")
                 except Exception as e:
                     logger.error(f"Error writing file {set_filename}: {e}")
    else:
         logger.warning("Overall test configuration not found in TEST_SET_CONFIG.")


    logger.info(f"\n--- Test Set Generation Finished ---")


