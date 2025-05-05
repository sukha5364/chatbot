# pipeline/rag_generator.py (최종 수정 계획 반영 버전)

import os
import json
import time
import logging
import re # 정규표현식 사용
from typing import List, Dict, Any, Optional, Set
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- 필요한 라이브러리 임포트 ---
try:
    import faiss
    logging.info("FAISS library imported successfully.")
except ImportError:
    logging.error("CRITICAL: faiss library not found. RAG pipeline cannot run. Please install it: pip install faiss-cpu or faiss-gpu")
    faiss = None

try:
    import openai
    from dotenv import load_dotenv
    logging.info("OpenAI and python-dotenv libraries imported successfully.")
except ImportError:
    logging.error("CRITICAL: openai or python-dotenv library not found. RAG pipeline cannot run. Please install them: pip install openai python-dotenv")
    openai = None

# Langchain TextSplitter (참조용 - 현재 로직에서는 직접 사용 안 함)
try:
    # from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
    # import tiktoken
    logging.info("Langchain/tiktoken are available but not used for primary splitting in this version.")
except ImportError:
    logging.warning("langchain or tiktoken library not found (not critical for current logic).")

# --- 설정 로더 임포트 ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    chatbot_module_path = os.path.join(PROJECT_ROOT, 'chatbot')
    if chatbot_module_path not in sys.path:
        sys.path.insert(0, chatbot_module_path)
    from chatbot.config_loader import get_config
    logging.info("Chatbot config loader imported successfully.")
except ImportError as e:
    logging.error(f"CRITICAL: Could not import config_loader from chatbot module: {e}. Ensure chatbot module structure is correct.")
    get_config = None
except Exception as e:
    logging.error(f"CRITICAL: Unexpected error during config_loader import setup: {e}")
    get_config = None

# --- 로깅 설정 (DEBUG 레벨 고정) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("RAG Generator pipeline logger initialized with DEBUG level.")

# --- 설정 로드 및 전역 변수 설정 ---
config: Optional[Dict[str, Any]] = None
openai_client: Optional[openai.OpenAI] = None
EXPECTED_EMBEDDING_DIM = 3072 # 기본값, 설정 로드 후 덮어씀
DECATHLON_BRANDS_LIST = ["Quechua", "Kiprun", "Kalenji", "Forclaz", "Evadict", "Newfeel", "Wedze", "Simond", "Artengo", "Domyos", "Orao", "Nabaiji", "Fouganza", "Itiwit", "Van Rysel", "Rockrider", "Btwin", "Solognac", "Outshock", "Inesis", "Aptonia", "Geologic", "Oxelo", "Tarmak", "Allsix", "Copaya", "Perfly", "Pongori", "Decathlon"] # 데카트론 포함하여 확장
OTHER_KNOWN_BRANDS = ["Nike", "Adidas", "New Balance", "Hoka", "Asics", "Under Armour", "Puma", "Columbia", "The North Face", "K2"] # 파일명 및 키워드 검사용

try:
    if get_config:
        config = get_config()
        logger.info("Configuration loaded successfully via get_config().")

        # OpenAI API 키 설정
        if openai:
            load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
            openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully.")
        else:
            raise ImportError("OpenAI library not imported.")

        # RAG 파이프라인 설정값 추출
        rag_config = config.get('rag', {})
        pipeline_config = rag_config.get('pipeline', {})

        EMBEDDING_MODEL_NAME = rag_config.get('embedding_model', 'text-embedding-3-large')
        EXPECTED_EMBEDDING_DIM = rag_config.get('embedding_dimension', 3072) # 설정값으로 덮어쓰기
        EMBEDDING_BATCH_SIZE = pipeline_config.get('embedding_batch_size', 100)

        logger.info(f"RAG Config: Embedding Model='{EMBEDDING_MODEL_NAME}', Dim={EXPECTED_EMBEDDING_DIM}, BatchSize={EMBEDDING_BATCH_SIZE}")

    else:
        raise ImportError("Config loader (get_config) is not available.")

except (ValueError, ImportError, KeyError, Exception) as e:
    logger.error(f"CRITICAL: Failed to load configuration or initialize components: {e}", exc_info=True)
    config = None # 실패 시 None으로 설정

# --- 경로 설정 ---
BASE_DIR = PROJECT_ROOT
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'original')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'index.faiss')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'doc_meta.jsonl')


# --- 함수 정의 ---

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    원본 데이터 디렉토리에서 .txt 파일들을 로드합니다. (하위 디렉토리 포함)
    파일명을 사용하여 초기 브랜드 정보(Nike, Adidas 등)를 추론합니다.

    Args:
        data_dir (str): 원본 텍스트 파일들이 있는 디렉토리 경로.

    Returns:
        List[Dict[str, str]]: 로드된 문서 리스트. 각 문서는 'source_file', 'content', 'inferred_brand' 키를 가짐.
    """
    documents = []
    logger.info(f"Loading documents from: {data_dir}")
    if not os.path.isdir(data_dir):
        logger.error(f"Original data directory not found: {data_dir}")
        return []
    try:
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, data_dir)
                    inferred_brand = "Unknown" # 기본값

                    # 파일명에서 브랜드 추론 (간단한 방식)
                    fn_lower = filename.lower()
                    if "nike" in fn_lower: inferred_brand = "Nike"
                    elif "adidas" in fn_lower or "new balance" in fn_lower : inferred_brand = "Adidas" if "adidas" in fn_lower else "New Balance"
                    elif "decathlon" in fn_lower or any(db.lower() in fn_lower for db in DECATHLON_BRANDS_LIST) : inferred_brand = "Decathlon" # 데카트론 파일이면 기본값

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) > 10:
                                documents.append({
                                    "source_file": relative_path,
                                    "content": content,
                                    "inferred_brand": inferred_brand # 파일명 기반 브랜드 추가
                                })
                                logger.debug(f" - Loaded: {relative_path} (Length: {len(content)}, Inferred Brand: {inferred_brand})")
                            else:
                                logger.warning(f" - Skipped empty or too short file: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error loading file {relative_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error walking through directory {data_dir}: {e}", exc_info=True)
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def extract_features_from_text(text: str, prefix: str = "") -> List[str]:
    """주어진 텍스트에서 불릿 포인트나 주요 구문을 추출하여 특징 리스트로 반환."""
    features = []
    lines = text.strip().split('\n')
    for line in lines:
        stripped_line = line.strip()
        # 불릿 포인트 (- *) 제거 및 내용 추출
        match = re.match(r'^[-*]\s*(.*)', stripped_line)
        if match:
            feature = match.group(1).strip().lower()
            if feature: features.append(f"{prefix}:{feature}" if prefix else feature)
        # 또는, 불릿 아닌 라인도 특정 길이 이상이면 특징으로 간주 (선택적)
        # elif len(stripped_line) > 5 and len(stripped_line.split()) < 10: # 예: 짧은 구문
        #    features.append(f"{prefix}:{stripped_line.lower()}" if prefix else stripped_line.lower())
    return [f for f in features if f] # 빈 문자열 제외

def parse_product_block(block_text: str, inferred_brand_from_filename: str) -> Dict[str, Any]:
    """
    개별 제품 정보 블록 텍스트를 파싱하여 구조화된 메타데이터를 추출합니다.
    브랜드, 가격, 카테고리, 주요 특징 등을 상세히 추출하고 누락 시 None 처리.

    Args:
        block_text (str): "--- 다음 제품 ---"으로 분리된 개별 제품 정보 텍스트.
        inferred_brand_from_filename (str): 파일명에서 추론된 브랜드명.

    Returns:
        Dict[str, Any]: 추출된 메타데이터 딕셔너리.
    """
    metadata = {
        "product_name": None,
        "brand": None,
        "category": None,
        "price": None,
        "price_numeric": None,
        "target_audience": None,
        "description_summary": None, # 핵심 설명 요약 (임베딩용)
        "features": [], # 통합된 특징 리스트
        "size_fit_keywords": [], # 사이즈/핏 관련 키워드
        "review_keywords": [], # 리뷰 관련 키워드
        "raw_text_preview": block_text[:200].replace('\n', ' ') + "..." # 디버깅용
    }
    lines = block_text.strip().split('\n')
    current_section_key = None
    section_texts = {} # 섹션별 텍스트 저장용

    # 1. 섹션 분리 및 텍스트 저장
    buffer = []
    for line in lines:
        line_stripped = line.strip()
        # 섹션 헤더 탐지 (## 또는 특정 키워드 시작)
        section_match = re.match(r'^##\s*(.+?)\s*##?$', line_stripped) # ## 섹션명 ## 또는 ## 섹션명
        detected_section = None
        if section_match:
            section_name_raw = section_match.group(1).strip().lower()
            # 섹션 이름 정규화 (예: '주요 특징' -> 'features')
            if '제품 정보' in section_name_raw: detected_section = 'info'
            elif '상세 설명' in section_name_raw: detected_section = 'description'
            elif '주요 특징' in section_name_raw: detected_section = 'main_features'
            elif '사이즈' in section_name_raw and '핏' in section_name_raw: detected_section = 'size_fit'
            elif '사용자 리뷰' in section_name_raw: detected_section = 'reviews'
            elif '활용 정보' in section_name_raw: detected_section = 'usage'
            # 기타 섹션 추가...
            else: detected_section = section_name_raw # 정규화 안되면 원본 사용 (혹은 무시)
        # 키워드 기반 섹션 헤더 (## 없을 경우 대비) - 필요시 추가
        # elif line_stripped.startswith("가격:"): detected_section = 'info'
        # ...

        if detected_section:
            if current_section_key and buffer:
                section_texts[current_section_key] = "\n".join(buffer).strip()
            current_section_key = detected_section
            buffer = [] # 새 섹션 시작, 버퍼 초기화
        elif current_section_key and line_stripped: # 현재 섹션에 내용 추가
            buffer.append(line_stripped)
        elif not current_section_key and line_stripped and not metadata['product_name']: # 제품명 추출 (첫 유효 라인)
             # 제품명에서 브랜드 먼저 분리 시도 (# 브랜드명 / 제품명 형식)
             name_brand_match = re.match(r'^\#?\s*([^/\n]+?)\s*/\s*(.+)', line_stripped)
             if name_brand_match:
                 metadata['brand'] = name_brand_match.group(1).strip()
                 metadata['product_name'] = name_brand_match.group(2).strip()
             else:
                 metadata['product_name'] = line_stripped # 형식 안 맞으면 전체를 제품명으로
                 # 브랜드는 나중에 다시 시도
        # elif not current_section_key and line_stripped: # 섹션 시작 전 내용 처리 (예: 제품명 외 첫 줄)
        #    buffer.append(line_stripped)

    # 마지막 섹션 버퍼 처리
    if current_section_key and buffer:
        section_texts[current_section_key] = "\n".join(buffer).strip()

    # 제품명 다시 확인 (첫 줄에서 못 찾았거나 너무 길 경우 대비)
    if not metadata['product_name'] and lines:
        metadata['product_name'] = lines[0].strip() # Fallback: 첫 줄 사용

    # 2. 브랜드 추출 (파일명 -> 제품명 -> 본문 키워드 순)
    if not metadata['brand']: # 제품명 파싱 시 브랜드 못 찾은 경우
        # 제품명에서 브랜드 키워드 찾기
        if metadata['product_name']:
            pn_lower = metadata['product_name'].lower()
            found_brand_in_name = None
            all_known_brands = DECATHLON_BRANDS_LIST + OTHER_KNOWN_BRANDS
            for brand_keyword in all_known_brands:
                if brand_keyword.lower() in pn_lower:
                    found_brand_in_name = brand_keyword
                    break
            if found_brand_in_name:
                metadata['brand'] = found_brand_in_name
            else: # 제품명에도 없으면 파일명 기반 브랜드 사용
                metadata['brand'] = inferred_brand_from_filename

    # 그래도 Unknown이면 'Decathlon'으로 간주 (데카트론 파일일 가능성 높음)
    if metadata['brand'] == "Unknown":
         metadata['brand'] = "Decathlon" # 정책 결정 필요

    # 3. 섹션별 정보 파싱 및 메타데이터 채우기
    all_features_set: Set[str] = set() # 중복 제거용 통합 특징 세트

    # 제품 정보 섹션 (카테고리, 가격, 대상 등)
    info_text = section_texts.get('info', '')
    if info_text:
        for line in info_text.split('\n'):
            line_stripped = line.strip()
            if ':' in line_stripped:
                key, value = [part.strip() for part in line_stripped.split(':', 1)]
                key_lower = key.lower()
                if '카테고리' in key_lower: metadata['category'] = value
                elif '가격' in key_lower:
                    metadata['price'] = value
                    try:
                        price_cleaned = re.sub(r'[^\d]', '', value)
                        if price_cleaned: metadata['price_numeric'] = int(price_cleaned)
                    except ValueError: logger.warning(f"Could not parse price '{value}' to numeric for '{metadata['product_name']}'.")
                elif '주요 대상' in key_lower or '성별' in key_lower: metadata['target_audience'] = value
                # 기타 정보 추출...

    # 상세 설명 요약 및 특징 추출
    desc_text = section_texts.get('description', '')
    if desc_text:
        # 간단 요약 (예: 첫 2문장 또는 특정 길이)
        sentences = re.split(r'(?<=[.!?])\s+', desc_text) # 문장 분리 (간단 방식)
        metadata['description_summary'] = " ".join(sentences[:2])[:150] if sentences else desc_text[:150]
        # 설명에서 특징 키워드 추출 (예: '방수', '가볍', '메쉬 소재') - 필요시 구현
        # all_features_set.update(extract_features_from_text(desc_text, prefix="desc"))

    # 주요 특징 섹션
    main_features_text = section_texts.get('main_features', '')
    if main_features_text:
        all_features_set.update(extract_features_from_text(main_features_text, prefix="feature"))

    # 사이즈 및 핏 분석 섹션
    size_fit_text = section_texts.get('size_fit', '')
    if size_fit_text:
        metadata['size_fit_keywords'] = extract_features_from_text(size_fit_text, prefix="fit")
        all_features_set.update(metadata['size_fit_keywords']) # 핏 정보도 features에 포함

    # 사용자 리뷰 섹션 (장점/단점 키워드화)
    reviews_text = section_texts.get('reviews', '')
    if reviews_text:
        pros_text = ""
        cons_text = ""
        # 리뷰 텍스트에서 장점/단점 영역 찾기 (예: "주요 장점:", "주요 단점:")
        pros_match = re.search(r'(?:주요|핵심)\s*장점\s*[:\n](.*?)(?:(?:주요|핵심)\s*단점|$)', reviews_text, re.DOTALL | re.IGNORECASE)
        cons_match = re.search(r'(?:주요|핵심)\s*단점\s*[:\n](.*?)(?:(?:추천|활용) 정보|$)', reviews_text, re.DOTALL | re.IGNORECASE)
        if pros_match: pros_text = pros_match.group(1)
        if cons_match: cons_text = cons_match.group(1)

        good_keywords = extract_features_from_text(pros_text, prefix="review_good")
        bad_keywords = extract_features_from_text(cons_text, prefix="review_bad")
        metadata['review_keywords'] = good_keywords + bad_keywords
        all_features_set.update(metadata['review_keywords']) # 리뷰 키워드도 features에 포함

    # 활용 정보 섹션 (추천 용도 등)
    usage_text = section_texts.get('usage', '')
    if usage_text:
        # 추천 용도, 관리 팁 등에서 키워드 추출 가능
        all_features_set.update(extract_features_from_text(usage_text, prefix="usage"))


    # 최종 features 리스트 생성 (중복 제거 및 정렬)
    metadata['features'] = sorted(list(all_features_set))

    # 필수 정보 누락 시 로깅
    if not metadata['product_name']: logger.warning(f"Product name could not be parsed for a block in '{inferred_brand_from_filename}'.")
    if not metadata['brand']: logger.warning(f"Brand could not be determined for product '{metadata['product_name']}'.")
    if not metadata['category']: logger.debug(f"Category not found for product '{metadata['product_name']}'.")
    if not metadata['price_numeric']: logger.debug(f"Numeric price not found for product '{metadata['product_name']}'.")

    return metadata


def create_chunks_from_products(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    로드된 문서의 내용을 제품 구분자로 분리하고, 각 제품 블록을
    하나의 청크로 만들며, 임베딩용 텍스트 생성 및 메타데이터를 추가합니다.

    Args:
        documents (List[Dict[str, str]]): 로드된 문서 리스트 ('content', 'source_file', 'inferred_brand' 포함).

    Returns:
        List[Dict[str, Any]]: 생성된 청크 리스트. 각 청크는 'id', 'text'(임베딩용),
                                'raw_block_text'(원본), 및 파싱된 메타데이터 포함.
    """
    all_chunks = []
    product_delimiter_pattern = re.compile(r'\s*---\s*다음\s*제품\s*---\s*', re.IGNORECASE)
    logger.info("Creating single chunk per product block with optimized embedding text...")
    total_blocks_processed = 0

    for doc_index, doc in enumerate(documents):
        source_file = doc.get('source_file', f'unknown_doc_{doc_index}')
        content = doc.get('content', '')
        inferred_brand = doc.get('inferred_brand', 'Unknown') # 파일명 기반 브랜드 사용
        if not content.strip():
            logger.warning(f"Skipping empty content from {source_file}")
            continue

        product_blocks = product_delimiter_pattern.split(content)
        logger.debug(f" - Splitting '{source_file}': Found {len(product_blocks)} potential product blocks.")

        for block_index, block_text in enumerate(product_blocks):
            block_text_stripped = block_text.strip()
            if len(block_text_stripped) < 20: # 너무 짧은 블록 건너뛰기
                logger.debug(f"   - Skipping very short block {block_index} in {source_file}.")
                continue

            try:
                # 메타데이터 파싱 (파일명 기반 브랜드 정보 전달)
                block_metadata = parse_product_block(block_text_stripped, inferred_brand)
            except Exception as e:
                logger.error(f"Failed to parse product block {block_index} in {source_file}: {e}", exc_info=True)
                block_metadata = {"product_name": f"Parse Error in {source_file} Block {block_index}", "brand": inferred_brand} # 최소 정보

            # 임베딩 대상 텍스트 생성 (핵심 정보 조합)
            embedding_text_parts = []
            if block_metadata.get('product_name'): embedding_text_parts.append(f"제품명: {block_metadata['product_name']}")
            if block_metadata.get('brand'): embedding_text_parts.append(f"브랜드: {block_metadata['brand']}")
            if block_metadata.get('category'): embedding_text_parts.append(f"카테고리: {block_metadata['category']}")
            if block_metadata.get('target_audience'): embedding_text_parts.append(f"대상: {block_metadata['target_audience']}")
            if block_metadata.get('description_summary'): embedding_text_parts.append(f"요약설명: {block_metadata['description_summary']}")
            # 특징 리스트 결합 (',' 구분, 최대 길이 제한 가능)
            if block_metadata.get('features'):
                features_str = ", ".join(block_metadata['features'])
                embedding_text_parts.append(f"주요 특징: {features_str[:300]}") # 특징 너무 길면 자르기

            embedding_text = " | ".join(filter(None, embedding_text_parts)) # None이나 빈 문자열 제외하고 결합
            logger.debug(f"   - Generated embedding text (len={len(embedding_text)}): {embedding_text[:150]}...")

            safe_filename = os.path.splitext(source_file.replace(os.sep, '_'))[0]
            chunk_id = f"{safe_filename}-block{block_index}"

            chunk_data = {
                "id": chunk_id,
                "source_file": source_file,
                "block_index": block_index,
                "text": embedding_text, # 임베딩 생성에 사용될 최적화된 텍스트
                "raw_block_text": block_text_stripped, # 원본 블록 텍스트 (메타데이터 저장 시 제외됨)
                **block_metadata # 파싱된 메타데이터 결합
            }
            # 메타데이터에서 임베딩/디버깅용 필드는 제외
            chunk_data.pop("raw_text_preview", None)
            chunk_data.pop("description_summary", None) # 임베딩 텍스트 생성 후 불필요하면 제거 가능

            all_chunks.append(chunk_data)
            total_blocks_processed += 1
            logger.debug(f"   - Created chunk for product: {block_metadata.get('product_name', 'N/A')} (ID: {chunk_id})")

    logger.info(f"Total product blocks (chunks) created: {total_blocks_processed}")
    return all_chunks


# --- OpenAI 임베딩 생성 함수 (재시도 포함) ---
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI API call due to: {retry_state.outcome.exception()}. Attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep:.2f}s..."))
def get_embeddings_with_retry(client: openai.OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    OpenAI 임베딩 API를 호출하고 결과를 반환합니다 (재시도 및 빈 텍스트 처리 포함).
    빈 텍스트 입력 시 빈 리스트를 반환합니다.
    """
    global EXPECTED_EMBEDDING_DIM

    valid_texts = []
    original_indices = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_texts.append(text.replace("\n", " ")) # 임베딩 모델은 개행문자 불필요
            original_indices.append(i)
        else:
            logger.warning(f"Empty or invalid text detected at index {i} in the batch. Skipping embedding.")

    if not valid_texts:
        logger.warning("No valid texts found in the batch to send for embedding.")
        return [[] for _ in texts]

    logger.debug(f"Calling OpenAI Embeddings API for {len(valid_texts)} texts with model {model}")
    response = client.embeddings.create(input=valid_texts, model=model, encoding_format="float")

    if not response.data:
        raise ValueError("OpenAI API response did not contain embedding data.")

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(valid_texts):
        raise ValueError(f"Mismatch between valid texts ({len(valid_texts)}) and returned embeddings ({len(embeddings)})")

    logger.debug(f"Successfully received {len(embeddings)} embeddings.")

    full_embeddings = [[] for _ in texts]
    for i, valid_idx in enumerate(original_indices):
        if i < len(embeddings):
            if len(embeddings[i]) != EXPECTED_EMBEDDING_DIM:
                raise ValueError(f"Incorrect embedding dimension: expected {EXPECTED_EMBEDDING_DIM}, got {len(embeddings[i])}")
            full_embeddings[valid_idx] = embeddings[i]
        else:
             raise ValueError("API returned fewer embeddings than valid texts sent.")

    return full_embeddings

def generate_openai_embeddings(chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    주어진 Chunk 리스트의 'text' 필드에 대해 OpenAI API를 사용하여 임베딩을 생성합니다.
    빈 텍스트 청크는 0 벡터로 대체합니다.

    Args:
        chunks (List[Dict[str, Any]]): 생성된 청크 리스트. 각 청크는 'text' 필드 포함.

    Returns:
        Optional[np.ndarray]: 생성된 임베딩 벡터들의 Numpy 배열 (float32). 실패 시 None.
    """
    global config, openai_client, EXPECTED_EMBEDDING_DIM

    if not config or not openai_client or not chunks:
        logger.error("Configuration, OpenAI client, or chunks not available for embedding.")
        return None

    model_name = EMBEDDING_MODEL_NAME
    batch_size = EMBEDDING_BATCH_SIZE

    logger.info(f"Generating embeddings using OpenAI model: {model_name} (Batch Size: {batch_size})...")
    all_embeddings: List[List[float]] = []
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    start_time_embed = time.time()
    processed_chunks = 0
    chunks_with_zero_vectors = 0

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        # 수정됨: 최적화된 'text' 필드 사용
        batch_texts = [chunk.get('text', '') for chunk in batch_chunks]

        current_batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {current_batch_num}/{total_batches} (size: {len(batch_texts)})")

        try:
            batch_embeddings_raw = get_embeddings_with_retry(openai_client, texts=batch_texts, model=model_name)

            if len(batch_embeddings_raw) != len(batch_texts):
                 logger.error(f"FATAL: Embedding count mismatch in batch {current_batch_num}. Expected {len(batch_texts)}, got {len(batch_embeddings_raw)}. Stopping.")
                 return None

            processed_batch_embeddings = []
            for idx, emb in enumerate(batch_embeddings_raw):
                if not emb:
                    logger.warning(f"Received empty embedding for chunk index {i + idx} (text: '{batch_texts[idx][:50]}...'). Replacing with zero vector.")
                    processed_batch_embeddings.append([0.0] * EXPECTED_EMBEDDING_DIM)
                    chunks_with_zero_vectors += 1
                else:
                    processed_batch_embeddings.append(emb)

            all_embeddings.extend(processed_batch_embeddings)
            processed_chunks += len(batch_texts)
            # time.sleep(0.1) # Rate Limit 방지 - 필요 시 조절

        except Exception as e:
            logger.error(f"FATAL: Error getting embeddings for batch {current_batch_num}: {e}", exc_info=True)
            return None

    end_time_embed = time.time()
    logger.info(f"Embedding generation took {end_time_embed - start_time_embed:.2f} seconds.")
    if chunks_with_zero_vectors > 0:
        logger.warning(f"Replaced {chunks_with_zero_vectors} chunks with zero vectors.")

    if not all_embeddings or len(all_embeddings) != total_chunks:
        logger.error(f"Error: Final embedding count ({len(all_embeddings)}) does not match total chunks ({total_chunks}).")
        return None

    try:
        embeddings_np = np.array(all_embeddings).astype('float32')
    except ValueError as e:
        logger.error(f"Error converting embeddings to NumPy array. Possible inconsistent dimensions? Error: {e}")
        unique_dims = {len(emb) for emb in all_embeddings if emb}
        logger.error(f"Unique non-empty dimensions found: {unique_dims}")
        return None

    logger.info(f"Embeddings generated successfully. Final shape: {embeddings_np.shape}")
    if embeddings_np.shape[1] != EXPECTED_EMBEDDING_DIM:
         logger.error(f"FATAL: Final embedding dimension mismatch! Expected {EXPECTED_EMBEDDING_DIM}, Got {embeddings_np.shape[1]}.")
         return None

    return embeddings_np


# --- FAISS 인덱스 빌드 함수 (변경 없음) ---
def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """
    주어진 임베딩 배열로부터 FAISS 인덱스(IndexFlatIP)를 빌드합니다.
    """
    global EXPECTED_EMBEDDING_DIM

    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        logger.error("Invalid or empty embeddings for building FAISS index.")
        return None
    if not faiss:
        logger.error("FAISS library not available. Cannot build index.")
        return None

    embedding_dim = EXPECTED_EMBEDDING_DIM
    if embeddings.shape[1] != embedding_dim:
        logger.error(f"Cannot build FAISS index: Embedding dim ({embeddings.shape[1]}) != configured dim ({embedding_dim}).")
        return None

    logger.info(f"Building FAISS index (IndexFlatIP) for dimension {embedding_dim}...")
    start_time_faiss = time.time()
    try:
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
        end_time_faiss = time.time()
        logger.info(f"FAISS index built successfully in {end_time_faiss - start_time_faiss:.2f} seconds. Index size: {index.ntotal} vectors.")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        return None


# --- 결과 저장 함수 (수정됨: text, raw_block_text 제외) ---
def save_results(index: faiss.Index, metadata_list: List[Dict[str, Any]], index_path: str, metadata_path: str):
    """
    FAISS 인덱스와 메타데이터(임베딩용 'text' 및 'raw_block_text' 필드 제외)를 저장합니다.
    """
    logger.info(f"Saving results to {index_path} and {metadata_path}...")
    output_dir = os.path.dirname(index_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save FAISS index
    if index and faiss:
        try:
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index saved successfully to: {index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {index_path}: {e}", exc_info=True)
    elif not faiss: logger.error("FAISS library not available, cannot save index.")
    else: logger.error("FAISS index object is None, cannot save index.")

    # 2. Save metadata (JSON Lines, 'text' and 'raw_block_text' 제외)
    if metadata_list:
        saved_count = 0
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in metadata_list:
                    item_to_save = item.copy()
                    # 임베딩용 text 필드와 원본 블록 텍스트 필드 제외
                    item_to_save.pop('text', None)
                    item_to_save.pop('raw_block_text', None)
                    try:
                        f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
                        saved_count += 1
                    except TypeError as te:
                         logger.warning(f"Could not serialize metadata item ID '{item.get('id', 'N/A')}' due to TypeError: {te}. Skipping.")
            logger.info(f"Metadata ({saved_count}/{len(metadata_list)} items) saved successfully to: {metadata_path} (excluding 'text' and 'raw_block_text')")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}", exc_info=True)
    else:
        logger.warning("Metadata list is empty, nothing to save.")


# --- 메인 실행 로직 (변경 없음) ---
if __name__ == "__main__":
    start_pipeline_time = time.time()
    logger.info("--- Starting RAG Offline Pipeline (Product Blocks as Chunks, Enhanced Metadata, Optimized Embedding Text) ---")

    # 필수 요소 확인
    if not config or not openai_client or not faiss:
        logger.error("CRITICAL: Configuration, OpenAI client, or FAISS library not loaded. Pipeline cannot proceed.")
        exit(1)

    # 1. 원본 문서 로드
    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        exit(1)

    # 2. 제품 블록 단위 청크 생성 (메타데이터 파싱 + 임베딩용 텍스트 생성)
    chunks_with_metadata = create_chunks_from_products(docs)
    if not chunks_with_metadata:
        logger.error("No chunks created from product blocks. Exiting.")
        exit(1)

    # 3. OpenAI 임베딩 생성 ('text' 필드 사용)
    embeddings_np = generate_openai_embeddings(chunks_with_metadata)
    if embeddings_np is None:
        logger.error("Failed to generate OpenAI embeddings. Exiting.")
        exit(1)

    # 4. FAISS 인덱스 빌드
    faiss_index = build_faiss_index(embeddings_np)
    if faiss_index is None:
        logger.error("Failed to build FAISS index. Exiting.")
        exit(1)

    # 5. 결과 저장 (메타데이터에서 text, raw_block_text 필드 제외)
    save_results(faiss_index, chunks_with_metadata, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    total_duration = end_pipeline_time - start_pipeline_time
    logger.info(f"--- RAG Offline Pipeline Finished in {total_duration:.2f} seconds ---")
    logger.info(f"--- Processed {len(chunks_with_metadata)} product blocks as individual chunks. ---")
    logger.info(f"--- Used OpenAI model '{EMBEDDING_MODEL_NAME}' for embeddings. ---")
    logger.info(f"--- Results saved to {FAISS_INDEX_PATH} and {METADATA_PATH} ---")