# pipeline/rag_generator.py (제품 정보 형식 반영 및 단일 청크 수정본 - 전체 코드)

import os
import json
import time
import logging
import re # 정규표현식 사용 위해 추가
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

# Langchain TextSplitter는 이제 사용되지 않지만, 혹시 모를 참조를 위해 임포트 유지
# 또는 필요시 제거 가능
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
    # 프로젝트 루트 경로 설정 (rag_generator.py는 pipeline/ 안에 있으므로 상위 -> 상위)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # chatbot 모듈 경로 추가 (config_loader 임포트 위함)
    import sys
    chatbot_module_path = os.path.join(PROJECT_ROOT, 'chatbot')
    if chatbot_module_path not in sys.path:
        sys.path.insert(0, chatbot_module_path)
    from config_loader import get_config
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

try:
    if get_config:
        config = get_config()
        logger.info("Configuration loaded successfully via get_config().")

        # OpenAI API 키 설정
        if openai:
            load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env')) # 프로젝트 루트 .env 로드
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
            openai_client = openai.OpenAI(api_key=api_key) # 클라이언트 초기화
            logger.info("OpenAI client initialized successfully.")
        else:
            raise ImportError("OpenAI library not imported.")

        # RAG 파이프라인 설정값 추출 (임베딩 관련 설정만 유효)
        rag_config = config.get('rag', {})
        pipeline_config = rag_config.get('pipeline', {}) # 스플리터 설정은 이제 사용 안 함

        EMBEDDING_MODEL_NAME = rag_config.get('embedding_model', 'text-embedding-3-large')
        EMBEDDING_DIM = rag_config.get('embedding_dimension', 3072)
        EMBEDDING_BATCH_SIZE = pipeline_config.get('embedding_batch_size', 100)

        # 스플리터 관련 설정 로깅 (이제 사용 안 함 명시)
        logger.info(f"RAG Config: Embedding Model='{EMBEDDING_MODEL_NAME}', Dim={EMBEDDING_DIM}, BatchSize={EMBEDDING_BATCH_SIZE}")
        logger.warning("NOTE: Splitter settings (splitter_type, chunk_size, overlap) in config are NO LONGER USED for chunking in this modified version.")

    else:
        raise ImportError("Config loader (get_config) is not available.")

except (ValueError, ImportError, KeyError, Exception) as e:
    logger.error(f"CRITICAL: Failed to load configuration or initialize components: {e}", exc_info=True)
    # 필수 설정 로드 실패 시 파이프라인 실행 불가 (main 블록에서 체크)
    config = None

# --- 경로 설정 ---
BASE_DIR = PROJECT_ROOT # 이미 위에서 정의됨
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
                               오류 발생 시 빈 리스트 반환.
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
                    # data_dir 기준 상대 경로 생성
                    relative_path = os.path.relpath(file_path, data_dir)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # 빈 파일이나 너무 짧은 파일은 건너뛰기 (최소 길이 설정 가능)
                            if len(content.strip()) > 10:
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
    Args:
        block_text (str): "--- 다음 제품 ---"으로 분리된 개별 제품 정보 텍스트.
    Returns:
        Dict[str, Any]: 추출된 메타데이터 딕셔너리.
    """
    metadata = {
        "product_name": "Unknown",
        "category": None,
        "price": None,
        "target_audience": None,
        "description": "", # 여러 줄일 수 있으므로 빈 문자열로 초기화
        "features": [],
        "size_fit_analysis": "",
        "reviews_summary": "", # 리뷰 요약 섹션 전체 텍스트 (하위 제외)
        "reviews_overall": None,
        "reviews_pros": [],
        "reviews_cons": [],
        "reviews_languages": [],
        "usage_recommendation": [],
        "care_tips": [],
        "raw_text_preview": block_text[:200].replace('\n', ' ') + "..." # 메타데이터에 미리보기 추가
    }
    lines = block_text.strip().split('\n')
    current_section = None
    sub_section = None # 리뷰 요약 내 하위 섹션 추적
    buffer = [] # 여러 줄 내용 버퍼

    if lines:
        metadata["product_name"] = lines[0].strip()

    for line in lines[1:]: # 첫 줄(제품명) 제외하고 처리
        line_stripped = line.strip()

        # 섹션 헤더 식별 (대소문자, 공백 무시하고 비교)
        normalized_line = line_stripped.lower().replace(" ", "")
        if normalized_line == "제품정보": current_section = "info"; sub_section = None; buffer = []; continue
        if normalized_line == "상세설명": current_section = "description"; sub_section = None; buffer = []; continue
        if normalized_line == "주요특징": current_section = "features"; sub_section = None; buffer = []; continue
        if normalized_line == "사이즈및핏분석(리뷰기반)": current_section = "size_fit"; sub_section = None; buffer = []; continue
        if normalized_line == "사용자리뷰요약": current_section = "reviews"; sub_section = None; buffer = []; continue
        if normalized_line == "활용정보(optional)": current_section = "usage"; sub_section = None; buffer = []; continue

        # 리뷰 요약 내 하위 섹션 식별
        if current_section == "reviews":
            if line_stripped.startswith("전반적 평가:"):
                sub_section = "overall"
                metadata["reviews_overall"] = line_stripped.split(":", 1)[1].strip()
                buffer = [] # 리뷰 요약 섹션 시작 시 버퍼 초기화
                continue
            if line_stripped.startswith("주요 장점:"):
                sub_section = "pros"
                # 쉼표, 마침표 등으로 구분된 리스트 파싱
                pros_text = line_stripped.split(":", 1)[1].strip()
                metadata["reviews_pros"] = [item.strip() for item in re.split(r'[,\.]', pros_text) if item.strip()]
                continue
            if line_stripped.startswith("주요 단점/개선점:"):
                sub_section = "cons"
                cons_text = line_stripped.split(":", 1)[1].strip()
                metadata["reviews_cons"] = [item.strip() for item in re.split(r'[,\.]', cons_text) if item.strip()]
                continue
            if line_stripped.startswith("언어:"):
                sub_section = "languages"
                lang_text = line_stripped.split(":", 1)[1].strip()
                metadata["reviews_languages"] = [item.strip() for item in lang_text.split(',') if item.strip()]
                continue

        # 활용 정보 내 하위 섹션 식별
        if current_section == "usage":
            if line_stripped.startswith("추천 용도:"):
                sub_section = "recommendation"
                rec_text = line_stripped.split(":", 1)[1].strip()
                metadata["usage_recommendation"] = [item.strip() for item in rec_text.split(',') if item.strip()]
                continue
            if line_stripped.startswith("관리 팁:"):
                sub_section = "care"
                care_text = line_stripped.split(":", 1)[1].strip()
                metadata["care_tips"] = [item.strip() for item in care_text.split(',') if item.strip()]
                continue

        # 현재 섹션에 따라 내용 파싱
        if current_section == "info":
            if line_stripped.startswith("카테고리:"): metadata["category"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.startswith("가격:"): metadata["price"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.startswith("주요 대상:"): metadata["target_audience"] = line_stripped.split(":", 1)[1].strip()
        elif current_section == "description":
            if line_stripped: buffer.append(line_stripped) # 상세 설명은 여러 줄 가능
        elif current_section == "features":
            # 주요 특징은 각 줄이 하나의 특징 항목으로 간주
            if line_stripped: metadata["features"].append(line_stripped)
        elif current_section == "size_fit":
             if line_stripped: buffer.append(line_stripped) # 사이즈/핏 분석도 여러 줄 가능
        elif current_section == "reviews":
             # 하위 섹션이 아닌 경우, 일반적인 리뷰 요약 내용으로 간주하여 버퍼에 추가
             if sub_section is None and line_stripped:
                  buffer.append(line_stripped)
        elif current_section == "usage":
             # 이미 하위 섹션에서 처리됨
             pass
        # 다른 섹션 처리 로직 추가 가능...

    # 루프 종료 후 버퍼에 남은 내용 처리
    if current_section == "description": metadata["description"] = " ".join(buffer).strip()
    elif current_section == "size_fit": metadata["size_fit_analysis"] = " ".join(buffer).strip()
    elif current_section == "reviews" and sub_section is None: metadata["reviews_summary"] = " ".join(buffer).strip()

    # 데이터 정제 (예: 가격에서 '원' 제거 후 숫자로 변환 시도)
    if metadata["price"]:
        try:
            # 숫자와 쉼표만 남기고 제거 후 정수 변환
            price_cleaned = re.sub(r'[^\d,]', '', metadata["price"])
            metadata["price_numeric"] = int(price_cleaned.replace(',', ''))
        except ValueError:
            logger.warning(f"Could not parse price '{metadata['price']}' to numeric for product '{metadata['product_name']}'. Keeping original string.")
            metadata["price_numeric"] = None # 변환 실패 시 None

    # 브랜드 이름 추출 (제품명에서 괄호 안 내용)
    match = re.match(r"(.+?)\s*\((.+?)\)", metadata["product_name"])
    if match:
        metadata["brand"] = match.group(2).strip()
        # 제품명에서 브랜드 부분 제외 가능 (선택 사항)
        # metadata["product_name_only"] = match.group(1).strip()
    else:
        metadata["brand"] = "Unknown" # 괄호 없으면 Unknown

    # logger.debug(f"Parsed metadata for '{metadata['product_name']}': {list(metadata.keys())}")
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
    # 제품 구분자 정규식 (앞뒤 공백 허용, 대소문자 무시)
    product_delimiter_pattern = re.compile(r'\s*---\s*다음\s*제품\s*---\s*', re.IGNORECASE)

    logger.info("Creating single chunk per product block...")
    total_blocks_processed = 0

    for doc_index, doc in enumerate(documents):
        source_file = doc.get('source_file', f'unknown_doc_{doc_index}')
        content = doc.get('content', '')
        if not content.strip():
            logger.warning(f"Skipping empty content from {source_file}")
            continue

        # 제품 구분자를 기준으로 블록 분리
        product_blocks = product_delimiter_pattern.split(content)
        logger.debug(f" - Splitting '{source_file}': Found {len(product_blocks)} potential product blocks.")

        for block_index, block_text in enumerate(product_blocks):
            block_text_stripped = block_text.strip()
            # 매우 짧은 블록(예: 구분자만 있거나 빈 줄) 건너뛰기
            if len(block_text_stripped) < 20:
                logger.debug(f"   - Skipping very short block {block_index} in {source_file} (likely empty or delimiter residue).")
                continue

            # 제품 블록 내용 파싱하여 메타데이터 추출
            try:
                block_metadata = parse_product_block(block_text_stripped)
                # 파싱된 제품명이 Unknown이면 건너뛸 수도 있음 (선택 사항)
                # if block_metadata.get("product_name") == "Unknown":
                #     logger.warning(f"   - Skipping block {block_index} in {source_file} due to unknown product name after parsing.")
                #     continue
            except Exception as e:
                logger.error(f"Failed to parse product block {block_index} in {source_file}: {e}", exc_info=True)
                # 파싱 실패 시 기본 메타데이터 사용하고 진행
                block_metadata = {"product_name": f"Parse Error in {source_file} Block {block_index}"}


            # 고유 청크 ID 생성 (파일명과 블록 인덱스 사용)
            # 파일명에서 확장자 제거하고 안전한 ID 생성
            safe_filename = os.path.splitext(source_file.replace(os.sep, '_'))[0] # 경로 구분자 '_'로 변경
            chunk_id = f"{safe_filename}-block{block_index}"

            # 청크 정보 생성 (text는 블록 전체, metadata는 파싱 결과)
            chunk_data = {
                "id": chunk_id,
                "source_file": source_file, # 원본 파일명 메타데이터 유지
                "block_index": block_index, # 파일 내 블록 순서
                "text": block_text_stripped, # 청크 텍스트 = 제품 블록 전체 (strip된 버전)
                **block_metadata           # 파싱된 메타데이터 추가
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
    OpenAI 임베딩 API를 호출하고 결과를 반환합니다 (재시도 포함).
    """
    # 입력 텍스트 유효성 검사 강화
    valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not valid_texts:
        logger.warning("No valid texts found in the batch to send for embedding.")
        # 빈 리스트 반환 또는 호출자에게 알릴 방법 필요
        # 여기서는 빈 리스트 반환 (호출하는 쪽에서 처리 필요)
        return [[] for _ in texts] # 원본 개수만큼 빈 리스트 반환 시도 (차원 문제 발생 가능성 있음)
        # raise ValueError("Input texts for embedding cannot be empty or invalid.") # 또는 에러 발생

    if len(valid_texts) < len(texts):
        logger.warning(f"Found {len(texts) - len(valid_texts)} empty or invalid texts in the batch. Sending only {len(valid_texts)} valid texts.")
        # TODO: 원본 인덱스 매핑 및 결과 재구성 로직 필요 시 추가

    logger.debug(f"Calling OpenAI Embeddings API for {len(valid_texts)} texts with model {model}")
    response = client.embeddings.create(input=valid_texts, model=model, encoding_format="float")

    if not response.data:
        raise ValueError("OpenAI API response did not contain embedding data.")

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(valid_texts):
        raise ValueError(f"Mismatch between number of valid input texts ({len(valid_texts)}) and returned embeddings ({len(embeddings)})")

    logger.debug(f"Successfully received {len(embeddings)} embeddings from API.")

    # TODO: 만약 일부 텍스트만 보냈다면, 원본 `texts` 리스트 길이에 맞춰 결과 재구성 필요
    # 예: 빈 텍스트 위치에는 None 또는 빈 벡터 삽입
    if len(valid_texts) < len(texts):
        full_embeddings = []
        valid_idx = 0
        for text in texts:
            if isinstance(text, str) and text.strip():
                if valid_idx < len(embeddings):
                    full_embeddings.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    # 이 경우는 API 응답 개수가 예상보다 적은 심각한 오류
                    logger.error("Critical error: API returned fewer embeddings than valid texts sent.")
                    full_embeddings.append([]) # 임시로 빈 리스트 추가
            else:
                full_embeddings.append([]) # 빈 텍스트에 해당하는 빈 리스트 추가
        if len(full_embeddings) == len(texts):
             return full_embeddings
        else:
             # 길이 불일치 시 오류 처리
             logger.error("Critical error: Failed to reconstruct full embedding list.")
             raise ValueError("Failed to reconstruct full embedding list.")

    return embeddings

def generate_openai_embeddings(chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    주어진 Chunk 리스트(이제 각 청크는 제품 블록 전체 텍스트)에 대해
    OpenAI API를 사용하여 임베딩을 생성합니다. (배치 처리)

    Args:
        chunks (List[Dict[str, Any]]): 생성된 청크 리스트. 각 청크는 'text' 필드 포함.

    Returns:
        Optional[np.ndarray]: 생성된 임베딩 벡터들의 Numpy 배열 (float32). 실패 시 None.
    """
    global config, openai_client # 전역 설정 및 클라이언트 사용

    if not config or not openai_client or not chunks:
        logger.error("Configuration, OpenAI client, or chunks are not available for embedding generation.")
        return None

    model_name = config['rag'].get('embedding_model', 'text-embedding-3-large')
    batch_size = config['rag']['pipeline'].get('embedding_batch_size', 100)
    expected_dim = config['rag'].get('embedding_dimension', 3072)

    logger.info(f"Generating embeddings using OpenAI model: {model_name} (Batch Size: {batch_size})...")
    all_embeddings: List[List[float]] = []
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    start_time_embed = time.time()
    processed_chunks = 0
    chunks_with_empty_embeddings = 0 # 빈 임베딩 카운트

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        # 각 청크의 'text' 필드 사용 (빈 문자열 가능성 있음)
        batch_texts = [chunk.get('text', '') for chunk in batch_chunks] # text 없으면 빈 문자열

        current_batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {current_batch_num}/{total_batches} (size: {len(batch_texts)})")

        try:
            # API 호출 (재시도 로직 포함된 함수 사용)
            # get_embeddings_with_retry 는 빈 텍스트 입력 시 빈 리스트 반환 가능성 있음
            batch_embeddings = get_embeddings_with_retry(openai_client, texts=batch_texts, model=model_name)

            # 결과 길이 확인
            if len(batch_embeddings) != len(batch_texts):
                 logger.error(f"FATAL: Embedding count mismatch in batch {current_batch_num}. Expected {len(batch_texts)}, got {len(batch_embeddings)}. Stopping.")
                 return None

            # 빈 임베딩 결과 확인 및 처리
            valid_batch_embeddings = []
            for idx, emb in enumerate(batch_embeddings):
                if not emb: # 임베딩이 비어있는 경우 (원본 텍스트가 비었거나 API 오류)
                    logger.warning(f"Received empty embedding for chunk index {i + idx} (text preview: '{batch_texts[idx][:50]}...'). Replacing with zero vector.")
                    # 0 벡터로 대체 (차원 맞춰서)
                    valid_batch_embeddings.append([0.0] * expected_dim)
                    chunks_with_empty_embeddings += 1
                elif len(emb) != expected_dim:
                     logger.error(f"FATAL: Incorrect embedding dimension received for chunk index {i + idx}. Expected {expected_dim}, got {len(emb)}. Stopping.")
                     return None
                else:
                    valid_batch_embeddings.append(emb)

            all_embeddings.extend(valid_batch_embeddings)
            processed_chunks += len(batch_texts)
            # Rate Limit 방지를 위한 약간의 대기 시간 (필요에 따라 조절)
            time.sleep(0.2) # 0.2초 대기

        except Exception as e:
            logger.error(f"FATAL: Error getting embeddings for batch {current_batch_num} (starting index {i}): {e}", exc_info=True)
            logger.error("Stopping embedding generation due to API error.")
            return None # 배치 처리 중 하나라도 실패하면 전체 실패

    end_time_embed = time.time()
    logger.info(f"Embedding generation took {end_time_embed - start_time_embed:.2f} seconds.")
    if chunks_with_empty_embeddings > 0:
        logger.warning(f"Found {chunks_with_empty_embeddings} chunks with empty embeddings (replaced with zero vectors).")

    # 최종 결과 검증
    if not all_embeddings or len(all_embeddings) != total_chunks:
        logger.error(f"Error: Embedding generation resulted in {len(all_embeddings)} embeddings, but expected {total_chunks}. Check logs for batch errors.")
        return None

    # Numpy 배열로 변환 (float32)
    try:
        embeddings_np = np.array(all_embeddings).astype('float32')
        # 0 벡터가 포함되었는지 최종 확인 (선택 사항)
        zero_vector_count = np.sum(np.all(embeddings_np == 0, axis=1))
        if zero_vector_count > 0:
             logger.warning(f"Final embeddings array contains {zero_vector_count} zero vectors (due to empty text or API issues).")

    except ValueError as e:
        logger.error(f"Error converting embeddings to NumPy array. Possible inconsistent dimensions? Error: {e}")
        # 차원 불일치 가능성 진단
        unique_dims = {len(emb) for emb in all_embeddings}
        logger.error(f"Unique dimensions found in embedding list: {unique_dims}")
        return None

    logger.info(f"Embeddings generated successfully. Final shape: {embeddings_np.shape}")

    # 차원 확인 (이미 위에서 했지만 최종 확인)
    if embeddings_np.shape[1] != expected_dim:
        logger.error(f"FATAL: Embedding dimension mismatch! Expected {expected_dim}, Got {embeddings_np.shape[1]}. Check config 'rag.embedding_dimension' and ensure it matches the model '{model_name}'.")
        return None # 차원 불일치는 심각한 문제

    return embeddings_np


# --- FAISS 인덱스 빌드 함수 ---
def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """
    주어진 임베딩 배열로부터 FAISS 인덱스를 빌드합니다.

    Args:
        embeddings (np.ndarray): 임베딩 벡터들의 Numpy 배열 (float32).

    Returns:
        Optional[faiss.Index]: 생성된 FAISS 인덱스 객체. 실패 시 None.
    """
    global config # 전역 설정 사용

    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        logger.error("Error: Invalid or empty embeddings provided for building FAISS index.")
        return None
    if not faiss:
        logger.error("FAISS library not available. Cannot build index.")
        return None

    # 설정에서 임베딩 차원 읽기 (build 시점에도 확인)
    embedding_dim = config['rag'].get('embedding_dimension', 3072)
    if embeddings.shape[1] != embedding_dim:
        logger.error(f"Cannot build FAISS index: Embedding dimension ({embeddings.shape[1]}) does not match configured dimension ({embedding_dim}).")
        return None

    logger.info(f"Building FAISS index (using IndexFlatIP for dimension {embedding_dim})...")
    start_time_faiss = time.time()
    try:
        # OpenAI 임베딩은 정규화되어 있으므로 내적(Inner Product)이 코사인 유사도와 동일/비례하며 빠름
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings) # embeddings는 float32 타입이어야 함
        end_time_faiss = time.time()
        logger.info(f"FAISS index built successfully in {end_time_faiss - start_time_faiss:.2f} seconds. Index size: {index.ntotal} vectors.")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        return None

# --- 결과 저장 함수 ---
def save_results(index: faiss.Index, metadata: List[Dict[str, Any]], index_path: str, metadata_path: str):
    """
    FAISS 인덱스와 메타데이터(이제 각 항목은 제품 블록 정보 포함)를 저장합니다.

    Args:
        index (faiss.Index): 빌드된 FAISS 인덱스 객체.
        metadata (List[Dict[str, Any]]): 각 Chunk(제품 블록)의 메타데이터 리스트.
        index_path (str): FAISS 인덱스를 저장할 파일 경로.
        metadata_path (str): 메타데이터를 저장할 JSON Lines 파일 경로.
    """
    logger.info(f"Saving results to {index_path} and {metadata_path}...")
    output_dir = os.path.dirname(index_path)
    os.makedirs(output_dir, exist_ok=True) # 출력 디렉토리 생성

    # 1. Save FAISS index
    if index and faiss:
        try:
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index saved successfully to: {index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {index_path}: {e}", exc_info=True)
    elif not faiss:
        logger.error("FAISS library not available, cannot save index.")
    else:
        logger.error("FAISS index object is None, cannot save index.")


    # 2. Save metadata (JSON Lines format)
    if metadata:
        saved_count = 0
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in metadata:
                    # 메타데이터 저장 시 'text' 필드는 제외하거나 줄이는 것을 고려
                    # (doc_meta.jsonl 파일 크기 관리 및 로딩 속도 향상 목적)
                    item_to_save = item.copy()
                    item_to_save.pop('text', None) # 'text' 필드 제외

                    # 다른 필드 중 너무 긴 내용도 미리보기만 저장 가능
                    # if 'description' in item_to_save and len(item_to_save['description']) > 500:
                    #     item_to_save['description_preview'] = item_to_save['description'][:500] + "..."
                    #     del item_to_save['description']

                    try:
                         f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
                         saved_count += 1
                    except TypeError as te:
                         logger.warning(f"Could not serialize metadata item ID '{item.get('id', 'N/A')}' due to TypeError: {te}. Skipping item.")
                         # 직렬화 불가능한 타입이 포함된 경우 (예: datetime 객체 등)
                         # default=str 등으로 처리할 수도 있음
                         # f.write(json.dumps(item_to_save, ensure_ascii=False, default=str) + '\n')


            logger.info(f"Metadata ({saved_count}/{len(metadata)} items) saved successfully to: {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}", exc_info=True)
    else:
        logger.warning("Metadata list is empty, nothing to save.")

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    start_pipeline_time = time.time()
    logger.info("--- Starting RAG Offline Pipeline (Modified for Product Blocks as Chunks) ---")

    # 필수 요소 확인
    if not config:
        logger.error("CRITICAL: Configuration not loaded. Pipeline cannot proceed.")
        exit(1)
    if not openai_client:
        logger.error("CRITICAL: OpenAI client not initialized. Pipeline cannot proceed.")
        exit(1)
    if not faiss:
        logger.error("CRITICAL: FAISS library not loaded. Pipeline cannot proceed.")
        exit(1)

    # 1. 원본 문서 로드
    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        exit(1)

    # 2. 제품 블록 단위로 청크 생성 및 메타데이터 파싱 (*** 변경된 함수 사용 ***)
    chunks_with_metadata = create_chunks_from_products(docs)
    if not chunks_with_metadata:
        logger.error("No chunks created from product blocks. Exiting.")
        exit(1)

    # 3. OpenAI 임베딩 생성 (*** 변경된 함수 사용 ***)
    embeddings_np = generate_openai_embeddings(chunks_with_metadata)
    if embeddings_np is None:
        logger.error("Failed to generate OpenAI embeddings. Exiting.")
        exit(1)

    # 4. FAISS 인덱스 빌드
    faiss_index = build_faiss_index(embeddings_np)
    if faiss_index is None:
        logger.error("Failed to build FAISS index. Exiting.")
        exit(1)

    # 5. 결과(인덱스, 메타데이터) 저장
    # 메타데이터 저장 시, chunks_with_metadata 사용 (파싱된 정보 포함됨)
    save_results(faiss_index, chunks_with_metadata, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    total_duration = end_pipeline_time - start_pipeline_time
    logger.info(f"--- RAG Offline Pipeline Finished in {total_duration:.2f} seconds ---")
    logger.info(f"--- Processed {len(chunks_with_metadata)} product blocks as individual chunks. ---")
    logger.info(f"--- Used OpenAI model '{config.get('rag', {}).get('embedding_model')}' which incurs API costs. ---")
    logger.info(f"--- Results saved to {FAISS_INDEX_PATH} and {METADATA_PATH} ---")