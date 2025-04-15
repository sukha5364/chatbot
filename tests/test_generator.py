# tests/test_generator.py (요구사항 반영 최종본)

import json
import random
import os
import logging
from typing import List, Dict, Tuple, Literal, Optional, Any

# --- 로깅 설정 (DEBUG 레벨 고정) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Test Generator logger initialized with DEBUG level.")

# --- 설정 로더 임포트 및 설정 로드 ---
try:
    # 스크립트 실행 위치에 따라 프로젝트 루트 경로 설정 필요
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # tests 폴더의 상위 -> 프로젝트 루트
    # chatbot 모듈 경로 추가
    import sys
    chatbot_module_path = os.path.join(project_root, 'chatbot')
    if chatbot_module_path not in sys.path:
        sys.path.insert(0, chatbot_module_path)
    from config_loader import get_config
    logger.info("Chatbot config loader imported successfully.")

    config = get_config()
    logger.info("Configuration loaded successfully via get_config().")

    # 필수 설정 섹션 확인
    if 'testing' not in config:
        raise KeyError("Required 'testing' section not found in config.yaml")
    testing_config = config['testing']
    if 'generation_probabilities' not in testing_config:
         raise KeyError("Required 'testing.generation_probabilities' section not found in config.yaml")
    gen_probs = testing_config['generation_probabilities']
    # 기능/전체 테스트 설정은 main 블록에서 로드

except (ImportError, KeyError, FileNotFoundError, Exception) as e:
    logger.error(f"CRITICAL: Failed to load configuration or required sections: {e}", exc_info=True)
    # 설정 로드 실패 시 스크립트 실행 중단
    exit(1)


# --- 설정값 변수화 (config에서 로드) ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_cases')
DECATHLON_BRAND_FOCUS_PROB = gen_probs.get('decathlon_brand_focus_prob', 0.9) # 기본값 0.9
# 템플릿 유형별 가중치 (config에서 로드, 기본값 설정)
TEMPLATE_TYPE_WEIGHTS = gen_probs.get('template_type_weights', {
    "decathlon_specific": 30, "generic_product": 30, "comparison": 20, "general_faq": 20
})

# --- 키워드 리스트 (기존 유지 또는 config에서 관리 고려 가능) ---
# TODO: 키워드 리스트도 config.yaml로 옮겨 관리하는 방안 고려 가능
DECATHLON_BRANDS = ["데카트론", "킵런", "퀘차", "포클라즈", "칼렌지", "아르텡고", "돔요스", "오라오", "뉴페일"]
OTHER_BRANDS = ["나이키", "아디다스", "호카", "뉴발란스", "아식스", "언더아머", "푸마", "컬럼비아", "노스페이스", "K2"]
ALL_BRANDS = list(set(DECATHLON_BRANDS + OTHER_BRANDS))
SIZES = ["260mm", "265", "270", "275mm", "280", "285", "US 9", "UK 8", "M", "L", "XL", "95", "100", "105", "S", "XS"]
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


# --- 질문 템플릿 (기존 유지) ---
# 유형 마커: 'decathlon_specific', 'generic_product', 'comparison', 'general_faq'
TEMPLATES: List[Tuple[Literal['basic', 'advanced'], Literal['decathlon_specific', 'generic_product', 'comparison', 'general_faq'], str]] = [
    # --- Basic Templates ---
    ('basic', 'decathlon_specific', "가까운 데카트론 매장 위치 알려줘."),
    ('basic', 'decathlon_specific', "데카트론 {brand} 매장 전화번호 알려줘."), # {brand} -> DECATHLON_BRANDS
    ('basic', 'decathlon_specific', "데카트론 반품 정책 알려줘."),
    ('basic', 'decathlon_specific', "데카트론 멤버십 혜택 뭐에요?"),
    ('basic', 'general_faq', "온라인 주문 배송 얼마나 걸려요?"),
    ('basic', 'general_faq', "영업시간이 어떻게 되나요?"),
    ('basic', 'generic_product', "데카트론 {product_category} {size} 재고 확인해줘."),
    ('basic', 'generic_product', "{brand} {product_category} {size} 재고 있어?"), # {brand} -> DECATHLON_BRANDS 우선
    ('basic', 'generic_product', "데카트론 {product_category} 사용법 알려줘."),
    ('basic', 'generic_product', "{brand} {product_category} 중에 제일 싼 거 뭐야?"), # {brand} -> DECATHLON_BRANDS 우선

    # --- Advanced Templates ---
    ('advanced', 'comparison', "{brand} {size} 신는데 데카트론 {product_category}는 어떤 사이즈 신어야해?"),
    ('advanced', 'comparison', "{brand} {product_category}랑 데카트론 {product_category2} 비교해줘."),
    ('advanced', 'comparison', "{brand} 신발은 발볼이 {foot_width} 사람에게도 괜찮을까? 데카트론은 어때?"),
    ('advanced', 'generic_product', "{brand} 러닝화는 발볼이 {foot_width} 사람에게도 괜찮을까?"), # {brand} -> DECATHLON_BRANDS 우선
    ('advanced', 'generic_product', "가장 인기 있는 데카트론 {category} 제품 추천해줘. {feature} 기능이 있으면 좋겠어."),
    ('advanced', 'generic_product', "데카트론에서 {activity} 할 때 입기 좋은 {product_category} 추천해줘. 나는 {skill_level}이고 {goal}을 위해 운동해."),
    ('advanced', 'generic_product', "발볼이 {foot_width} 편인데 {brand} 신발 괜찮을까요? 주로 {environment}에서 신을 거야."), # {brand} -> DECATHLON_BRANDS 우선
    ('advanced', 'generic_product', "{brand} {category} 신제품 나왔어? 이전에 쓰던 데카트론 모델보다 뭐가 좋아졌는지 설명해줘."),
    ('advanced', 'generic_product', "나는 {age_group} {gender}인데, 데카트론에서 {activity}용으로 {budget} 가격대의 {product_category}를 찾고 있어. 어떤 게 좋을까?"),
    ('advanced', 'generic_product', "겨울철 {activity}에 적합한 데카트론 {product_category}를 찾는데, {feature} 기능이랑 {feature2} 기능 둘 다 중요해. 어떤 제품이 좋을지 이유와 함께 설명해줘."),
    ('advanced', 'general_faq', "지난 번 데카트론에서 구매한 {product_category}에 {problem} 문제가 발생했어. 어떻게 해결해야 할까?")
]


# --- 함수 정의 ---

def generate_test_questions(n: int, target_complexity: Optional[Literal['basic', 'advanced']] = None) -> List[Dict]:
    """
    지정된 난이도와 개수만큼 랜덤한 테스트 질문 데이터를 생성합니다.
    config.yaml에서 읽어온 확률 설정을 반영하여 질문을 생성합니다.

    Args:
        n (int): 생성할 질문의 개수.
        target_complexity (Optional[Literal['basic', 'advanced']]): 생성할 질문의 난이도 ('basic', 'advanced').
                                                                  None이면 모든 난이도 포함.

    Returns:
        List[Dict]: 생성된 테스트 질문 리스트. 각 질문은 'question', 'test_difficulty', 'template_type' 키를 가짐.
    """
    global TEMPLATE_TYPE_WEIGHTS, DECATHLON_BRAND_FOCUS_PROB # 전역 설정값 사용

    questions = []
    used_questions = set() # 중복 방지용

    # 해당 난이도의 템플릿 필터링
    if target_complexity:
        filtered_templates = [tpl for tpl in TEMPLATES if tpl[0] == target_complexity]
        if not filtered_templates:
            logger.warning(f"No templates found for complexity: {target_complexity}")
            return []
        logger.info(f"Generating {n} questions with target complexity: {target_complexity}")
    else:
        filtered_templates = TEMPLATES # None이면 모든 템플릿 사용
        logger.info(f"Generating {n} questions with any complexity")

    # 템플릿 유형별 가중치 계산 (random.choices 사용 위해)
    template_pool = []
    weights = []
    valid_template_types = list(TEMPLATE_TYPE_WEIGHTS.keys())

    for tpl in filtered_templates:
        tpl_complexity, tpl_type, _ = tpl
        if tpl_type in valid_template_types:
            template_pool.append(tpl)
            # target_complexity가 지정된 경우 해당 복잡도 내에서만 가중치 적용, 아니면 전체에서 적용
            weights.append(TEMPLATE_TYPE_WEIGHTS.get(tpl_type, 1)) # 설정에 없는 유형은 기본 가중치 1 부여
        else:
             logger.warning(f"Template type '{tpl_type}' not found in config template_type_weights. Using default weight 1.")
             template_pool.append(tpl)
             weights.append(1)

    if not template_pool:
        logger.error("No valid templates available to generate questions based on weights.")
        return []

    attempts = 0
    max_attempts = n * 15 # 중복 회피 및 로직 복잡성 증가로 최대 시도 횟수 늘림

    while len(questions) < n and attempts < max_attempts:
        attempts += 1

        # 가중치를 반영하여 템플릿 선택
        try:
            complexity_marker, template_type, template = random.choices(template_pool, weights=weights, k=1)[0]
        except ValueError as e:
            logger.error(f"Error selecting template with weights ({len(template_pool)} templates, {len(weights)} weights): {e}. Check config TEMPLATE_TYPE_WEIGHTS.")
            break # 가중치 문제 발생 시 중단

        # --- 키워드 선택 로직 (DECATHLON_BRAND_FOCUS_PROB 반영) ---
        params: Dict[str, Any] = {} # 최종 포맷팅에 사용할 파라미터 딕셔너리

        # 브랜드 선택
        decathlon_brand_chosen = random.choice(DECATHLON_BRANDS) if DECATHLON_BRANDS else "데카트론" # Fallback
        other_brand_chosen = random.choice(OTHER_BRANDS) if OTHER_BRANDS else "타사 브랜드" # Fallback

        if '{brand}' in template or '{other_brand}' in template:
            if template_type == 'comparison':
                # 비교 템플릿: 하나는 데카트론, 하나는 외부 브랜드 (랜덤 순서)
                if random.random() < 0.5:
                    params['brand'] = decathlon_brand_chosen
                    params['other_brand'] = other_brand_chosen
                else:
                    params['brand'] = other_brand_chosen
                    params['other_brand'] = decathlon_brand_chosen
            elif template_type == 'generic_product' or template_type == 'decathlon_specific':
                # 일반/데카트론 특정 템플릿: 설정된 확률로 데카트론 브랜드 선택
                if random.random() < DECATHLON_BRAND_FOCUS_PROB or not OTHER_BRANDS:
                    params['brand'] = decathlon_brand_chosen
                else:
                    params['brand'] = other_brand_chosen # 낮은 확률로 외부 브랜드
                params['other_brand'] = other_brand_chosen # 비교 아닌 템플릿 대비
            else: # general_faq 등
                params['brand'] = random.choice(ALL_BRANDS) if ALL_BRANDS else "데카트론"
                params['other_brand'] = other_brand_chosen

        # 나머지 키워드 랜덤 선택 (기존 방식 유지)
        # TODO: 카테고리 등 다른 키워드도 데카트론 중심으로 편향 가능
        params['size'] = random.choice(SIZES) if SIZES else "M"
        params['size2'] = random.choice([s for s in SIZES if s != params.get('size')]) if len(SIZES) > 1 else params.get('size', 'L')
        params['category'] = random.choice(CATEGORIES) if CATEGORIES else "러닝화"
        params['product_category'] = params['category']
        params['product_category2'] = random.choice([c for c in CATEGORIES if c != params.get('category')]) if len(CATEGORIES) > 1 else params.get('category', '등산화')
        params['foot_width'] = random.choice(FOOT_WIDTHS) if FOOT_WIDTHS else "보통"
        params['feature'] = random.choice(FEATURES) if FEATURES else "가벼운"
        params['feature2'] = random.choice([f for f in FEATURES2 if f != params.get('feature')]) if len(FEATURES2) > 1 else params.get('feature', '방수되는')
        params['activity'] = random.choice(ACTIVITIES) if ACTIVITIES else "러닝"
        params['skill_level'] = random.choice(SKILL_LEVELS) if SKILL_LEVELS else "초보"
        params['goal'] = random.choice(GOALS) if GOALS else "취미"
        params['environment'] = random.choice(ENVIRONMENTS) if ENVIRONMENTS else "로드"
        params['age_group'] = random.choice(AGE_GROUPS) if AGE_GROUPS else "성인"
        params['gender'] = random.choice(GENDERS) if GENDERS else "남성"
        params['budget'] = random.choice(BUDGETS) if BUDGETS else "10만원대"
        params['problem'] = random.choice(PROBLEMS) if PROBLEMS else "사이즈 문의"

        # --- 질문 생성 및 검증 ---
        try:
            # 템플릿에 필요한 키만 사용하여 포맷팅
            required_keys = {key for key in params.keys() if f"{{{key}}}" in template}
            format_params = {k: params[k] for k in required_keys if k in params}

            if not all(key in format_params for key in required_keys):
                logger.debug(f"Skipping template due to missing keys for formatting: {template}. Required: {required_keys}, Available: {list(format_params.keys())}")
                continue # 필요한 키가 없으면 건너뛰기

            question = template.format(**format_params)

            # 간단한 후처리 (예: 연속 공백 제거)
            question = ' '.join(question.split())

            if question not in used_questions:
                item = {
                    "question": question,
                    "test_difficulty": complexity_marker,
                    "template_type": template_type, # 생성에 사용된 템플릿 유형 정보
                    "generation_params": format_params # 생성에 사용된 파라미터 (디버깅용)
                }
                # TODO: 향후 예상 Slot 값 또는 정답 패턴 등을 추가 가능
                # item["expected_slots"] = { ... }
                questions.append(item)
                used_questions.add(question)
                logger.debug(f"Generated question #{len(questions)}: {question}")
            # else: logger.debug(f"Skipping duplicate question: {question}")

        except KeyError as e:
            logger.warning(f"Skipping template due to missing key {e} during formatting: {template}")
            continue
        except Exception as e:
            logger.error(f"Error generating question for template '{template}': {e}", exc_info=False) # 스택 트레이스 제외 가능

    if len(questions) < n:
        logger.warning(f"Could only generate {len(questions)} unique questions out of {n} requested (attempts: {attempts}). Consider adding more templates or adjusting generation probabilities.")

    logger.info(f"Finished generating {len(questions)} questions for complexity: {target_complexity or 'any'}.")
    return questions


# --- 메인 실행 로직 (config.yaml 설정 사용) ---
if __name__ == "__main__":
    logger.info("--- Starting Test Set Generation (Using config.yaml) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # 설정 파일에서 테스트 설정 로드
    try:
        func_test_config = config.get('testing', {}).get('function_test_set') # config 키 이름 확인 필요 (가정)
        overall_test_config = config.get('testing', {}).get('overall_test_set') # config 키 이름 확인 필요 (가정)

        # --- Fallback 설정 (config에 관련 키가 없을 경우) ---
        # TODO: config.yaml에 function_test_set, overall_test_set 섹션 정의 필요
        if not func_test_config:
             logger.warning("Config section 'testing.function_test_set' not found. Using default values.")
             func_test_config = {"count": 30, "filename_template": "functional_tests.jsonl"}
        if not overall_test_config:
             logger.warning("Config section 'testing.overall_test_set' not found. Using default values.")
             overall_test_config = {"count": 200, "complexity_ratio": 0.5, "num_sets": 3, "filename_template": "overall_tests_set{set_num}.jsonl"}

    except KeyError as e:
        logger.error(f"Missing required key in 'testing' config section: {e}. Cannot proceed.")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading test configuration from config.yaml: {e}. Cannot proceed.")
        exit(1)


    # 1. 기능 테스트셋 생성
    logger.info("Generating Functional Test Set...")
    func_count = func_test_config.get('count', 30)
    func_filename = func_test_config.get('filename_template', 'functional_tests.jsonl')
    if func_count > 0 and func_filename:
        func_questions = generate_test_questions(func_count, target_complexity=None)
        func_filepath = os.path.join(OUTPUT_DIR, func_filename)
        try:
            with open(func_filepath, "w", encoding="utf-8") as f:
                for item in func_questions:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Successfully generated functional test set: {func_filename} ({len(func_questions)} questions)")
        except Exception as e:
            logger.error(f"Error writing file {func_filename}: {e}")
    else:
        logger.warning("Functional test generation skipped due to zero count or missing filename template in config.")


    # 2. 전체 테스트셋 생성 (여러 세트)
    num_sets = overall_test_config.get('num_sets', 1)
    count_per_set = overall_test_config.get('count', 100)
    ratio = overall_test_config.get('complexity_ratio', 0.5)
    filename_tpl = overall_test_config.get('filename_template')

    if num_sets > 0 and count_per_set > 0 and filename_tpl:
        num_basic = int(count_per_set * ratio)
        num_advanced = count_per_set - num_basic

        for i in range(1, num_sets + 1):
            set_filename = filename_tpl.format(set_num=i)
            set_filepath = os.path.join(OUTPUT_DIR, set_filename)
            logger.info(f"Generating Overall Test Set {i} ({set_filename}). Target: {num_basic} basic, {num_advanced} advanced...")

            basic_questions = generate_test_questions(num_basic, target_complexity='basic')
            advanced_questions = generate_test_questions(num_advanced, target_complexity='advanced')
            combined_questions = basic_questions + advanced_questions
            random.shuffle(combined_questions)

            actual_basic = len(basic_questions)
            actual_advanced = len(advanced_questions)
            actual_total = len(combined_questions)

            if actual_total < count_per_set * 0.8: # 요청 개수의 80% 미만이면 경고
                logger.warning(f"Generated significantly fewer questions ({actual_total}) than requested ({count_per_set}) for set {i}.")

            try:
                with open(set_filepath, "w", encoding="utf-8") as f:
                    for item in combined_questions:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                logger.info(f"Successfully generated overall test set: {set_filename} ({actual_total} questions: {actual_basic} basic, {actual_advanced} advanced)")
            except Exception as e:
                logger.error(f"Error writing file {set_filename}: {e}")
    else:
        logger.warning("Overall test set generation skipped due to zero count/sets or missing filename template in config.")

    logger.info(f"--- Test Set Generation Finished ---")