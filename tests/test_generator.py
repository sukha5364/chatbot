# tests/test_generator.py (수정본: 데카트론 중심 질문 생성 로직 적용)

import json
import random
import os
import logging
from typing import List, Dict, Tuple, Literal, Optional, Any # Any 추가

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 설정 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_cases')

# 테스트셋 생성 설정 (기존 유지)
TEST_SET_CONFIG = {
    "function": {
        "count": 30,
        "complexity_ratio": None,
        "filename_template": "functional_tests.jsonl"
    },
    "overall": {
        "count": 200,
        "complexity_ratio": 0.5,
        "num_sets": 3,
        "filename_template": "overall_tests_set{set_num}.jsonl"
    }
}

# --- 키워드 리스트 (수정: 브랜드 분리) ---
DECATHLON_BRANDS = ["데카트론", "킵런", "퀘차", "포클라즈", "칼렌지", "아르텡고", "돔요스", "오라오", "뉴페일"]
OTHER_BRANDS = ["나이키", "아디다스", "호카", "뉴발란스", "아식스", "언더아머", "푸마", "컬럼비아", "노스페이스", "K2"]
# 모든 브랜드를 합친 리스트도 필요시 사용 가능
ALL_BRANDS = list(set(DECATHLON_BRANDS + OTHER_BRANDS))

SIZES = ["260mm", "265", "270", "275mm", "280", "285", "US 9", "UK 8", "M", "L", "XL", "95", "100", "105", "S", "XS"]
# 데카트론에서 주로 다루는 카테고리 위주로 조정 가능
CATEGORIES = ["러닝화", "트레킹화", "축구화", "운동화", "등산화", "스포츠 양말", "기능성 티셔츠", "텐트", "배낭", "자켓", "침낭", "바지", "수영복", "자전거 의류", "요가 매트"]
FOOT_WIDTHS = ["좁은", "보통", "넓은", "아주 넓은"]
FEATURES = ["방수되는", "가벼운", "쿠션 좋은", "통기성 좋은", "내구성 강한", "보온성 좋은", "빠르게 마르는", "신축성 있는", "미끄럼 방지", "자외선 차단"]
ACTIVITIES = ["러닝", "등산", "캠핑", "축구", "트레일 러닝", "일상", "여행", "백패킹", "낚시", "자전거 타기", "수영", "요가", "헬스"]
SKILL_LEVELS = ["초보", "중급자", "상급자", "입문자", "전문가"]
GOALS = ["다이어트", "대회 준비", "취미", "건강 관리", "기록 단축", "생존", "편안함", "스타일"]
ENVIRONMENTS = ["로드", "트레일", "우천 시", "겨울철", "실내", "도심", "야간", "해변", "강", "수영장"]
AGE_GROUPS = ["20대", "30대", "40대", "50대", "청소년", "성인", "아동", "노년"]
GENDERS = ["남자", "여자", "남성", "여성"]
BUDGETS = ["10만원 이하", "10만원대", "20만원 미만", "30만원 이하", "5만원 근처", "상관없음", "가성비 좋은"]
PROBLEMS = ["사이즈가 안맞아", "색상이 달라", "제품에 하자가 있어", "사용법을 모르겠어", "배송이 안 와", "AS 받고 싶어", "환불하고 싶어"]

FEATURES2 = list(set(FEATURES)) # 중복 제거된 다른 특징 리스트


# --- 질문 템플릿 (수정: 유형 마커 추가 및 데카트론 중심 조정) ---
# 유형 마커: 'decathlon_specific', 'generic_product', 'comparison', 'general_faq'
TEMPLATES: List[Tuple[Literal['basic', 'advanced'], Literal['decathlon_specific', 'generic_product', 'comparison', 'general_faq'], str]] = [
    # --- Basic Templates ---
    # Decathlon Specific (Basic)
    ('basic', 'decathlon_specific', "가까운 데카트론 매장 위치 알려줘."),
    ('basic', 'decathlon_specific', "데카트론 {brand} 매장 전화번호 알려줘."), # {brand} -> DECATHLON_BRANDS
    ('basic', 'decathlon_specific', "데카트론 반품 정책 알려줘."),
    ('basic', 'decathlon_specific', "데카트론 멤버십 혜택 뭐에요?"),

    # General FAQ (Basic)
    ('basic', 'general_faq', "온라인 주문 배송 얼마나 걸려요?"),
    ('basic', 'general_faq', "영업시간이 어떻게 되나요?"), # 데카트론 맥락으로 해석될 가능성 높음

    # Generic Product (Basic) - 데카트론 제품 위주로 생성되도록 조정 필요
    ('basic', 'generic_product', "데카트론 {product_category} {size} 재고 확인해줘."), # 명시적으로 데카트론 추가
    ('basic', 'generic_product', "{brand} {product_category} {size} 재고 있어?"), # {brand} -> DECATHLON_BRANDS 우선
    ('basic', 'generic_product', "데카트론 {product_category} 사용법 알려줘."), # 명시적으로 데카트론 추가
    ('basic', 'generic_product', "{brand} {product_category} 중에 제일 싼 거 뭐야?"), # {brand} -> DECATHLON_BRANDS 우선

    # --- Advanced Templates ---
    # Comparison (Advanced) - 데카트론과 비교하는 내용으로 생성되도록 조정 필요
    ('advanced', 'comparison', "{brand} {size} 신는데 데카트론 {product_category}는 어떤 사이즈 신어야해?"), # 명시적 비교
    ('advanced', 'comparison', "{brand} {product_category}랑 데카트론 {product_category2} 비교해줘."), # 명시적 비교
    ('advanced', 'comparison', "{brand} 신발은 발볼이 {foot_width} 사람에게도 괜찮을까? 데카트론은 어때?"), # 명시적 비교

    # Generic Product (Advanced) - 데카트론 제품/상황 위주로 생성되도록 조정 필요
    ('advanced', 'generic_product', "{brand} 러닝화는 발볼이 {foot_width} 사람에게도 괜찮을까?"), # {brand} -> DECATHLON_BRANDS 우선
    ('advanced', 'generic_product', "가장 인기 있는 데카트론 {category} 제품 추천해줘. {feature} 기능이 있으면 좋겠어."), # 데카트론 명시
    ('advanced', 'generic_product', "데카트론에서 {activity} 할 때 입기 좋은 {product_category} 추천해줘. 나는 {skill_level}이고 {goal}을 위해 운동해."), # 데카트론 명시
    ('advanced', 'generic_product', "발볼이 {foot_width} 편인데 {brand} 신발 괜찮을까요? 주로 {environment}에서 신을 거야."), # {brand} -> DECATHLON_BRANDS 우선
    ('advanced', 'generic_product', "{brand} {category} 신제품 나왔어? 이전에 쓰던 데카트론 모델보다 뭐가 좋아졌는지 설명해줘."), # 데카트론 명시적 비교
    ('advanced', 'generic_product', "나는 {age_group} {gender}인데, 데카트론에서 {activity}용으로 {budget} 가격대의 {product_category}를 찾고 있어. 어떤 게 좋을까?"), # 데카트론 명시
    ('advanced', 'generic_product', "겨울철 {activity}에 적합한 데카트론 {product_category}를 찾는데, {feature} 기능이랑 {feature2} 기능 둘 다 중요해. 어떤 제품이 좋을지 이유와 함께 설명해줘."), # 데카트론 명시

    # General FAQ/Problem (Advanced)
    ('advanced', 'general_faq', "지난 번 데카트론에서 구매한 {product_category}에 {problem} 문제가 발생했어. 어떻게 해결해야 할까?") # 데카트론 명시
]


# 난이도별, 개수별 질문 생성 함수 (수정됨)
def generate_test_questions(n: int, target_complexity: Optional[Literal['basic', 'advanced']] = None) -> List[Dict]:
    """
    지정된 난이도와 개수만큼 랜덤한 테스트 질문 데이터를 생성합니다.
    데카트론 중심으로 질문을 생성하도록 로직을 조정합니다.
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
    max_attempts = n * 15 # 중복 회피 및 로직 복잡성 증가로 최대 시도 횟수 늘림

    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        if not filtered_templates:
            logger.warning("No templates available to generate questions.")
            break

        complexity_marker, template_type, template = random.choice(filtered_templates)

        # --- 키워드 선택 로직 (수정됨) ---
        params: Dict[str, Any] = {} # 최종 포맷팅에 사용할 파라미터 딕셔너리

        # 1. 브랜드 선택 로직 조정
        decathlon_brand_chosen = None
        other_brand_chosen = None
        if '{brand}' in template or '{other_brand}' in template:
            if template_type == 'comparison':
                # 비교 템플릿: 하나는 데카트론, 하나는 외부 브랜드
                decathlon_brand_chosen = random.choice(DECATHLON_BRANDS)
                other_brand_chosen = random.choice(OTHER_BRANDS)
                # 랜덤하게 순서 섞기 (Nike 신는데 Kiprun vs Kiprun 신는데 Nike)
                if random.random() < 0.5:
                    params['brand'] = decathlon_brand_chosen
                    params['other_brand'] = other_brand_chosen
                else:
                    params['brand'] = other_brand_chosen
                    params['other_brand'] = decathlon_brand_chosen
            elif template_type == 'generic_product' or template_type == 'decathlon_specific':
                 # 일반/데카트론 특정 템플릿: 데카트론 브랜드 우선 선택 (예: 90% 확률)
                if random.random() < 0.9 or not OTHER_BRANDS: # 외부 브랜드 없으면 무조건 데카트론
                    params['brand'] = random.choice(DECATHLON_BRANDS)
                else:
                     # 낮은 확률로 외부 브랜드 선택 (이 질문이 부적절할 수 있으나, 다양성 위해 일부 허용 가능성)
                    params['brand'] = random.choice(OTHER_BRANDS)
                params['other_brand'] = random.choice(OTHER_BRANDS) # 비교 아닌 템플릿에도 other_brand 키 존재 시 대비
            else: # general_faq 등
                params['brand'] = random.choice(ALL_BRANDS)
                params['other_brand'] = random.choice(OTHER_BRANDS)

        # 2. 나머지 키워드 랜덤 선택 (기존 방식 유지 또는 카테고리 등도 데카트론 중심으로 조정 가능)
        params['size'] = random.choice(SIZES)
        params['size2'] = random.choice([s for s in SIZES if s != params.get('size')])
        # 카테고리도 데카트론 중심으로? (선택 사항)
        params['category'] = random.choice(CATEGORIES)
        params['product_category'] = params['category']
        params['product_category2'] = random.choice([c for c in CATEGORIES if c != params.get('category')])
        params['foot_width'] = random.choice(FOOT_WIDTHS)
        params['feature'] = random.choice(FEATURES)
        params['feature2'] = random.choice([f for f in FEATURES2 if f != params.get('feature')])
        params['activity'] = random.choice(ACTIVITIES)
        params['skill_level'] = random.choice(SKILL_LEVELS)
        params['goal'] = random.choice(GOALS)
        params['environment'] = random.choice(ENVIRONMENTS)
        params['age_group'] = random.choice(AGE_GROUPS)
        params['gender'] = random.choice(GENDERS)
        params['budget'] = random.choice(BUDGETS)
        params['problem'] = random.choice(PROBLEMS)

        # --- 질문 생성 및 검증 ---
        try:
            # 템플릿에 필요한 키만 사용하여 포맷팅
            required_keys = {key for key in params.keys() if f"{{{key}}}" in template}
            format_params = {k: params[k] for k in required_keys if k in params} # params 딕셔너리에 실제 키가 있는지 확인

            # 모든 required_keys가 format_params에 있는지 최종 확인
            if not all(key in format_params for key in required_keys):
                 # logger.warning(f"Could not fill all required keys for template: {template}. Skipping.")
                 continue # 필요한 키가 없으면 이 템플릿 건너뛰기

            question = template.format(**format_params)

            # 간단한 검증: 외부 브랜드만 있고 데카트론 관련 언급 없는 질문 제외 (선택적 필터링은 안하기로 함)
            # is_other_brand_only = any(ob in question for ob in OTHER_BRANDS) and \
            #                         not any(db in question for db in DECATHLON_BRANDS) and \
            #                         '데카트론' not in question and \
            #                         template_type != 'comparison' # 비교 템플릿은 제외

            # if is_other_brand_only:
            #     logger.debug(f"Skipping potentially irrelevant question: {question}")
            #     continue

            if question not in used_questions:
                item = {
                    "question": question,
                    "test_difficulty": complexity_marker,
                    "template_type": template_type # 생성에 사용된 템플릿 유형 정보 추가
                }
                # 필요 시 예상 슬롯 등 추가 정보 포함 가능
                # item["expected_slots"] = { ... }
                questions.append(item)
                used_questions.add(question)
        except KeyError as e:
            # logger.warning(f"Skipping template due to missing key {e} during formatting: {template}")
            continue
        except Exception as e:
            logger.error(f"Error generating question for template '{template}': {e}", exc_info=False)

    if len(questions) < n:
        logger.warning(f"Could only generate {len(questions)} unique relevant questions out of {n} requested (attempts: {attempts}). Consider adding more Decathlon-focused templates or adjusting keyword probabilities.")

    return questions


# --- 메인 실행 로직 (기존과 동일) ---
if __name__ == "__main__":
    logger.info("--- Starting Test Set Generation (Decathlon Focused) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # 1. 기능 테스트셋 생성
    logger.info("Generating Functional Test Set...")
    func_config = TEST_SET_CONFIG.get('function')
    if func_config:
        func_questions = generate_test_questions(func_config['count'], target_complexity=None)
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

                basic_questions = generate_test_questions(num_basic, target_complexity='basic')
                advanced_questions = generate_test_questions(num_advanced, target_complexity='advanced')
                combined_questions = basic_questions + advanced_questions
                random.shuffle(combined_questions)

                actual_basic = len(basic_questions)
                actual_advanced = len(advanced_questions)
                actual_total = len(combined_questions)

                if actual_total < count_per_set * 0.9:
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