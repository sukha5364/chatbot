# pipeline/rag_generator.py (제품 정보 형식 반영 및 메타데이터 파싱 강화)

import os
import json
import time
import logging
import re # 정규표현식 사용
from typing import List, Dict, Any, Optional
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

# Langchain TextSplitter (사용되지 않지만 참조용 또는 제거 가능)
try:
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
    import tiktoken
    logging.info("Langchain (TextSplitters) and tiktoken libraries imported successfully (though not used for primary splitting).")
except ImportError:
    logging.warning("langchain or tiktoken library not found. These are not critical for the current product-block-as-chunk logic.")
    CharacterTextSplitter = None
    RecursiveCharacterTextSplitter = None
    TokenTextSplitter = None
    tiktoken = None

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
        logger.warning("NOTE: Splitter settings in config are NO LONGER USED for chunking.")

    else:
        raise ImportError("Config loader (get_config) is not available.")

except (ValueError, ImportError, KeyError, Exception) as e:
    logger.error(f"CRITICAL: Failed to load configuration or initialize components: {e}", exc_info=True)
    config = None

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

    Args:
        data_dir (str): 원본 텍스트 파일들이 있는 디렉토리 경로.

    Returns:
        List[Dict[str, str]]: 로드된 문서 리스트. 각 문서는 'source_file'(상대 경로)과 'content' 키를 가짐.
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
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) > 10: # 너무 짧은 파일 제외
                                documents.append({"source_file": relative_path, "content": content})
                                logger.debug(f" - Loaded: {relative_path} (Length: {len(content)})")
                            else:
                                logger.warning(f" - Skipped empty or too short file: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error loading file {relative_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error walking through directory {data_dir}: {e}", exc_info=True)
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def parse_product_block(block_text: str) -> Dict[str, Any]:
    """
    개별 제품 정보 블록 텍스트를 파싱하여 구조화된 메타데이터를 추출합니다.
    가격, 주요 특징 등 필터링에 사용할 필드를 추출하고 정제합니다.

    Args:
        block_text (str): "--- 다음 제품 ---"으로 분리된 개별 제품 정보 텍스트.
    Returns:
        Dict[str, Any]: 추출된 메타데이터 딕셔너리.
    """
    metadata = {
        "product_name": "Unknown",
        "brand": "Unknown", # 브랜드 추가
        "category": None,
        "price": None,
        "price_numeric": None, # 숫자 가격 필드 추가
        "target_audience": None,
        "description": "",
        "features": [], # 주요 특징 리스트
        "size_fit_analysis": "",
        "reviews_summary": "",
        "reviews_overall": None,
        "reviews_pros": [],
        "reviews_cons": [],
        "reviews_languages": [],
        "usage_recommendation": [],
        "care_tips": [],
        "raw_text_preview": block_text[:200].replace('\n', ' ') + "..."
    }
    lines = block_text.strip().split('\n')
    current_section = None
    sub_section = None
    buffer = []

    # 첫 줄은 제품명으로 간주 (브랜드 추출 시도)
    if lines:
        product_name_line = lines[0].strip()
        metadata["product_name"] = product_name_line
        # 브랜드 이름 추출 (괄호 안 내용 또는 특정 키워드 - 예: '데카트론', 'Kiprun' 등)
        match_bracket = re.search(r'\(([^)]+)\)', product_name_line) # 괄호 안 텍스트
        # 데카트론 브랜드 리스트 (필요시 config에서 관리)
        decathlon_brands_list = ["Quechua", "Kiprun", "Kalenji", "Forclaz", "Evadict", "Newfeel", "Wedze", "Simond", "Artengo", "Domyos", "Orao", "Nabaiji", "Fouganza"]
        extracted_brand = "Unknown"
        if match_bracket:
            extracted_brand = match_bracket.group(1).strip()
        else: # 괄호 없으면 브랜드 리스트에서 찾아보기
            for brand_keyword in decathlon_brands_list:
                if brand_keyword.lower() in product_name_line.lower():
                    extracted_brand = brand_keyword
                    break
            # 데카트론 브랜드명 자체가 포함된 경우 (예: "데카트론 500 넥워머")
            if extracted_brand == "Unknown" and "데카트론" in product_name_line:
                extracted_brand = "Decathlon" # 기본값 또는 다른 이름

        metadata["brand"] = extracted_brand if extracted_brand != "Unknown" else "Decathlon" # 기본값 설정 또는 Unknown 유지

    # 나머지 줄 처리
    for line in lines[1:]:
        line_stripped = line.strip()
        normalized_line = re.sub(r'\s+', '', line_stripped.lower()) # 공백 제거 및 소문자 변환

        # 섹션 헤더 식별 강화 (정규식 또는 다양한 키워드 매칭 고려)
        if normalized_line.startswith("제품정보"): current_section = "info"; sub_section = None; buffer = []; continue
        if normalized_line.startswith("상세설명"): current_section = "description"; sub_section = None; buffer = []; continue
        if normalized_line.startswith("주요특징"): current_section = "features"; sub_section = None; buffer = []; continue
        if normalized_line.startswith("사이즈및핏분석"): current_section = "size_fit"; sub_section = None; buffer = []; continue
        if normalized_line.startswith("사용자리뷰요약"): current_section = "reviews"; sub_section = None; buffer = []; continue
        if normalized_line.startswith("활용정보"): current_section = "usage"; sub_section = None; buffer = []; continue
        # (더 많은 섹션 헤더 추가 가능)

        # 하위 섹션 식별 (기존 로직 유지)
        if current_section == "reviews":
            if line_stripped.startswith("전반적 평가:"):
                sub_section = "overall"; metadata["reviews_overall"] = line_stripped.split(":", 1)[1].strip(); buffer = []; continue
            if line_stripped.startswith("주요 장점:"):
                sub_section = "pros"; pros_text = line_stripped.split(":", 1)[1].strip(); metadata["reviews_pros"] = [item.strip() for item in re.split(r'[,\.]', pros_text) if item.strip()]; continue
            if line_stripped.startswith("주요 단점/개선점:"):
                sub_section = "cons"; cons_text = line_stripped.split(":", 1)[1].strip(); metadata["reviews_cons"] = [item.strip() for item in re.split(r'[,\.]', cons_text) if item.strip()]; continue
            if line_stripped.startswith("언어:"):
                sub_section = "languages"; lang_text = line_stripped.split(":", 1)[1].strip(); metadata["reviews_languages"] = [item.strip() for item in lang_text.split(',') if item.strip()]; continue
        if current_section == "usage":
            if line_stripped.startswith("추천 용도:"):
                sub_section = "recommendation"; rec_text = line_stripped.split(":", 1)[1].strip(); metadata["usage_recommendation"] = [item.strip() for item in rec_text.split(',') if item.strip()]; continue
            if line_stripped.startswith("관리 팁:"):
                sub_section = "care"; care_text = line_stripped.split(":", 1)[1].strip(); metadata["care_tips"] = [item.strip() for item in care_text.split(',') if item.strip()]; continue

        # 현재 섹션 내용 파싱
        if current_section == "info":
            if line_stripped.lower().startswith("카테고리:"): metadata["category"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.lower().startswith("가격:"):
                price_str = line_stripped.split(":", 1)[1].strip()
                metadata["price"] = price_str
                # 숫자 가격 추출 시도
                try:
                    price_cleaned = re.sub(r'[^\d]', '', price_str) # 숫자만 남김
                    if price_cleaned: metadata["price_numeric"] = int(price_cleaned)
                except ValueError:
                    logger.warning(f"Could not parse price '{price_str}' to numeric for product '{metadata['product_name']}'.")
            elif line_stripped.lower().startswith("주요 대상:"): metadata["target_audience"] = line_stripped.split(":", 1)[1].strip()
        elif current_section == "description":
            if line_stripped: buffer.append(line_stripped)
        elif current_section == "features":
            # '-' 또는 '*' 같은 리스트 마커 제거 및 정규화
            feature_text = re.sub(r'^[*-]\s*', '', line_stripped).strip()
            if feature_text:
                # 소문자 변환 등 정규화 추가 가능 (필터 매칭 위해)
                metadata["features"].append(feature_text.lower())
        elif current_section == "size_fit":
            if line_stripped: buffer.append(line_stripped)
        elif current_section == "reviews":
            if sub_section is None and line_stripped: buffer.append(line_stripped)
        # 다른 섹션 처리 로직 ...

    # 버퍼 내용 처리
    if current_section == "description": metadata["description"] = " ".join(buffer).strip()
    elif current_section == "size_fit": metadata["size_fit_analysis"] = " ".join(buffer).strip()
    elif current_section == "reviews" and sub_section is None: metadata["reviews_summary"] = " ".join(buffer).strip()

    # 최종 정제 (예: features 중복 제거)
    if metadata["features"]:
        metadata["features"] = sorted(list(set(metadata["features"])))

    return metadata


def create_chunks_from_products(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    로드된 문서의 내용을 제품 구분자로 분리하고, 각 제품 블록을
    하나의 청크로 만들며, 해당 블록에서 파싱된 메타데이터를 추가합니다.

    Args:
        documents (List[Dict[str, str]]): 로드된 문서 리스트.

    Returns:
        List[Dict[str, Any]]: 생성된 청크 리스트. 각 청크는 'id', 'text', 및 파싱된 메타데이터 포함.
    """
    all_chunks = []
    product_delimiter_pattern = re.compile(r'\s*---\s*다음\s*제품\s*---\s*', re.IGNORECASE)
    logger.info("Creating single chunk per product block...")
    total_blocks_processed = 0

    for doc_index, doc in enumerate(documents):
        source_file = doc.get('source_file', f'unknown_doc_{doc_index}')
        content = doc.get('content', '')
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
                block_metadata = parse_product_block(block_text_stripped)
            except Exception as e:
                logger.error(f"Failed to parse product block {block_index} in {source_file}: {e}", exc_info=True)
                block_metadata = {"product_name": f"Parse Error in {source_file} Block {block_index}", "brand": "Unknown"}

            safe_filename = os.path.splitext(source_file.replace(os.sep, '_'))[0]
            chunk_id = f"{safe_filename}-block{block_index}"

            chunk_data = {
                "id": chunk_id,
                "source_file": source_file,
                "block_index": block_index,
                "text": block_text_stripped, # 원본 텍스트 포함
                **block_metadata
            }
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
    빈 텍스트 입력 시 0 벡터 대신 빈 리스트를 반환하도록 수정 (호출 측에서 처리).
    """
    global EXPECTED_EMBEDDING_DIM # 설정된 임베딩 차원 사용

    # 입력 텍스트 유효성 검사 및 빈 텍스트 위치 추적
    valid_texts = []
    original_indices = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_texts.append(text)
            original_indices.append(i)
        else:
            logger.warning(f"Empty or invalid text detected at index {i} in the batch. Skipping embedding for this item.")

    if not valid_texts:
        logger.warning("No valid texts found in the batch to send for embedding.")
        return [[] for _ in texts] # 원본 개수만큼 빈 리스트 반환

    logger.debug(f"Calling OpenAI Embeddings API for {len(valid_texts)} texts (out of {len(texts)}) with model {model}")
    response = client.embeddings.create(input=valid_texts, model=model, encoding_format="float")

    if not response.data:
        raise ValueError("OpenAI API response did not contain embedding data.")

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(valid_texts):
        raise ValueError(f"Mismatch between number of valid input texts ({len(valid_texts)}) and returned embeddings ({len(embeddings)})")

    logger.debug(f"Successfully received {len(embeddings)} embeddings from API.")

    # 원본 리스트 길이에 맞춰 결과 재구성 (빈 텍스트 위치는 빈 리스트로 채움)
    full_embeddings = [[] for _ in texts] # 먼저 원본 길이만큼 빈 리스트로 초기화
    for i, valid_idx in enumerate(original_indices):
        if i < len(embeddings):
            # 차원 검증 추가
            if len(embeddings[i]) != EXPECTED_EMBEDDING_DIM:
                 logger.error(f"FATAL: Incorrect embedding dimension received for text index {valid_idx}. Expected {EXPECTED_EMBEDDING_DIM}, got {len(embeddings[i])}. Stopping.")
                 # 또는 예외 발생
                 raise ValueError(f"Incorrect embedding dimension: expected {EXPECTED_EMBEDDING_DIM}, got {len(embeddings[i])}")
            full_embeddings[valid_idx] = embeddings[i]
        else:
            # API 응답 개수가 예상보다 적은 경우 (오류)
            logger.error(f"Critical error: API returned fewer embeddings ({len(embeddings)}) than valid texts sent ({len(valid_texts)}). Index reconstruction failed.")
            # 예외를 발생시키거나, 빈 리스트로 두거나 결정 필요
            raise ValueError("Failed to reconstruct full embedding list due to API returning fewer results.")

    return full_embeddings

def generate_openai_embeddings(chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    주어진 Chunk 리스트에 대해 OpenAI API를 사용하여 임베딩을 생성합니다. (배치 처리)
    빈 텍스트 청크는 0 벡터로 대체합니다.

    Args:
        chunks (List[Dict[str, Any]]): 생성된 청크 리스트. 각 청크는 'text' 필드 포함.

    Returns:
        Optional[np.ndarray]: 생성된 임베딩 벡터들의 Numpy 배열 (float32). 실패 시 None.
    """
    global config, openai_client, EXPECTED_EMBEDDING_DIM # 전역 설정 사용

    if not config or not openai_client or not chunks:
        logger.error("Configuration, OpenAI client, or chunks are not available for embedding generation.")
        return None

    model_name = config['rag'].get('embedding_model', 'text-embedding-3-large')
    batch_size = config['rag']['pipeline'].get('embedding_batch_size', 100)
    # expected_dim 은 전역 변수 EXPECTED_EMBEDDING_DIM 사용

    logger.info(f"Generating embeddings using OpenAI model: {model_name} (Batch Size: {batch_size})...")
    all_embeddings: List[List[float]] = [] # 최종 임베딩 저장 리스트
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    start_time_embed = time.time()
    processed_chunks = 0
    chunks_with_zero_vectors = 0 # 0 벡터로 대체된 청크 카운트

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk.get('text', '') for chunk in batch_chunks]

        current_batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {current_batch_num}/{total_batches} (size: {len(batch_texts)})")

        try:
            # get_embeddings_with_retry는 빈 텍스트에 대해 빈 리스트를 반환
            batch_embeddings_raw = get_embeddings_with_retry(openai_client, texts=batch_texts, model=model_name)

            if len(batch_embeddings_raw) != len(batch_texts):
                logger.error(f"FATAL: Embedding count mismatch in batch {current_batch_num}. Expected {len(batch_texts)}, got {len(batch_embeddings_raw)}. Stopping.")
                return None

            # 빈 리스트를 0 벡터로 대체
            processed_batch_embeddings = []
            for idx, emb in enumerate(batch_embeddings_raw):
                if not emb: # 빈 리스트인 경우 (원본 텍스트가 비었거나 API 문제)
                    logger.warning(f"Received empty embedding for chunk index {i + idx}. Replacing with zero vector.")
                    processed_batch_embeddings.append([0.0] * EXPECTED_EMBEDDING_DIM) # 0 벡터로 대체
                    chunks_with_zero_vectors += 1
                else:
                    processed_batch_embeddings.append(emb) # 유효한 임베딩 사용

            all_embeddings.extend(processed_batch_embeddings)
            processed_chunks += len(batch_texts)
            time.sleep(0.1) # Rate Limit 방지 (약간 줄임)

        except Exception as e:
            logger.error(f"FATAL: Error getting embeddings for batch {current_batch_num} (starting index {i}): {e}", exc_info=True)
            logger.error("Stopping embedding generation due to API error.")
            return None

    end_time_embed = time.time()
    logger.info(f"Embedding generation took {end_time_embed - start_time_embed:.2f} seconds.")
    if chunks_with_zero_vectors > 0:
        logger.warning(f"Found {chunks_with_zero_vectors} chunks replaced with zero vectors due to empty text or API issues.")

    if not all_embeddings or len(all_embeddings) != total_chunks:
        logger.error(f"Error: Embedding generation resulted in {len(all_embeddings)} embeddings, but expected {total_chunks}.")
        return None

    try:
        embeddings_np = np.array(all_embeddings).astype('float32')
        zero_vector_count = np.sum(np.all(embeddings_np == 0, axis=1))
        if zero_vector_count > 0:
            logger.info(f"Final embeddings array contains {zero_vector_count} zero vectors.")
    except ValueError as e:
        logger.error(f"Error converting embeddings to NumPy array. Possible inconsistent dimensions? Error: {e}")
        unique_dims = {len(emb) for emb in all_embeddings if emb}
        logger.error(f"Unique non-empty dimensions found: {unique_dims}")
        return None

    logger.info(f"Embeddings generated successfully. Final shape: {embeddings_np.shape}")

    if embeddings_np.shape[1] != EXPECTED_EMBEDDING_DIM:
        logger.error(f"FATAL: Final embedding dimension mismatch! Expected {EXPECTED_EMBEDDING_DIM}, Got {embeddings_np.shape[1]}. Check config and model.")
        return None

    return embeddings_np


# --- FAISS 인덱스 빌드 함수 (변경 없음) ---
def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """
    주어진 임베딩 배열로부터 FAISS 인덱스를 빌드합니다.
    """
    global config, EXPECTED_EMBEDDING_DIM # 전역 설정 사용

    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        logger.error("Error: Invalid or empty embeddings provided for building FAISS index.")
        return None
    if not faiss:
        logger.error("FAISS library not available. Cannot build index.")
        return None

    # 설정에서 임베딩 차원 읽기
    embedding_dim = EXPECTED_EMBEDDING_DIM # 전역 변수 사용
    if embeddings.shape[1] != embedding_dim:
        logger.error(f"Cannot build FAISS index: Embedding dimension ({embeddings.shape[1]}) does not match configured dimension ({embedding_dim}).")
        return None

    logger.info(f"Building FAISS index (using IndexFlatIP for dimension {embedding_dim})...")
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


# --- 결과 저장 함수 (변경 없음 - 'text' 필드 제외 확인) ---
def save_results(index: faiss.Index, metadata: List[Dict[str, Any]], index_path: str, metadata_path: str):
    """
    FAISS 인덱스와 메타데이터('text' 필드 제외)를 저장합니다.
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

    # 2. Save metadata (JSON Lines format, 'text' 제외)
    if metadata:
        saved_count = 0
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in metadata:
                    item_to_save = item.copy()
                    item_to_save.pop('text', None) # 'text' 필드 제외 확인
                    try:
                        f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
                        saved_count += 1
                    except TypeError as te:
                        logger.warning(f"Could not serialize metadata item ID '{item.get('id', 'N/A')}' due to TypeError: {te}. Skipping item.")
            logger.info(f"Metadata ({saved_count}/{len(metadata)} items) saved successfully to: {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}", exc_info=True)
    else:
        logger.warning("Metadata list is empty, nothing to save.")


# --- 메인 실행 로직 (변경 없음) ---
if __name__ == "__main__":
    start_pipeline_time = time.time()
    logger.info("--- Starting RAG Offline Pipeline (Product Blocks as Chunks, Enhanced Metadata) ---")

    # 필수 요소 확인
    if not config or not openai_client or not faiss:
        logger.error("CRITICAL: Configuration, OpenAI client, or FAISS library not loaded. Pipeline cannot proceed.")
        exit(1)

    # 1. 원본 문서 로드
    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        exit(1)

    # 2. 제품 블록 단위 청크 생성 및 메타데이터 파싱
    chunks_with_metadata = create_chunks_from_products(docs)
    if not chunks_with_metadata:
        logger.error("No chunks created from product blocks. Exiting.")
        exit(1)

    # 3. OpenAI 임베딩 생성
    embeddings_np = generate_openai_embeddings(chunks_with_metadata)
    if embeddings_np is None:
        logger.error("Failed to generate OpenAI embeddings. Exiting.")
        exit(1)

    # 4. FAISS 인덱스 빌드
    faiss_index = build_faiss_index(embeddings_np)
    if faiss_index is None:
        logger.error("Failed to build FAISS index. Exiting.")
        exit(1)

    # 5. 결과 저장
    save_results(faiss_index, chunks_with_metadata, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    total_duration = end_pipeline_time - start_pipeline_time
    logger.info(f"--- RAG Offline Pipeline Finished in {total_duration:.2f} seconds ---")
    logger.info(f"--- Processed {len(chunks_with_metadata)} product blocks as individual chunks. ---")
    logger.info(f"--- Used OpenAI model '{config.get('rag', {}).get('embedding_model')}' for embeddings. ---")
    logger.info(f"--- Results saved to {FAISS_INDEX_PATH} and {METADATA_PATH} ---")