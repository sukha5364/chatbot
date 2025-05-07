# pipeline/rag_generator.py (최종 수정 계획 반영 버전 - 단순화된 메타데이터 및 raw_block_text 중심)

import os
import json
import time
import logging
import re # 정규표현식 사용
from typing import List, Dict, Any, Optional, Set # Set은 이제 사용 안 함
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
# DECATHLON_BRANDS_LIST 및 OTHER_KNOWN_BRANDS는 parse_product_block 단순화로 인해 직접 사용 안 함

try:
    if get_config:
        config = get_config()
        logger.info("Configuration loaded successfully via get_config().")

        if openai:
            load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
            openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully.")
        else:
            raise ImportError("OpenAI library not imported.")

        rag_config = config.get('rag', {})
        pipeline_config = rag_config.get('pipeline', {})

        EMBEDDING_MODEL_NAME = rag_config.get('embedding_model', 'text-embedding-3-large')
        EXPECTED_EMBEDDING_DIM = rag_config.get('embedding_dimension', 3072)
        EMBEDDING_BATCH_SIZE = pipeline_config.get('embedding_batch_size', 100)

        logger.info(f"RAG Config: Embedding Model='{EMBEDDING_MODEL_NAME}', Dim={EXPECTED_EMBEDDING_DIM}, BatchSize={EMBEDDING_BATCH_SIZE}")
    else:
        raise ImportError("Config loader (get_config) is not available.")

except (ValueError, ImportError, KeyError, Exception) as e:
    logger.critical(f"CRITICAL: Failed to load configuration or initialize components: {e}", exc_info=True)
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
    파일명에서 초기 브랜드 정보는 이제 parse_product_block에서 사용하지 않습니다.
    Args:
        data_dir (str): 원본 텍스트 파일들이 있는 디렉토리 경로.
    Returns:
        List[Dict[str, str]]: 로드된 문서 리스트. 각 문서는 'source_file', 'content' 키를 가짐.
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
                            if len(content.strip()) > 10: # 너무 짧은 파일 건너뛰기
                                documents.append({
                                    "source_file": relative_path,
                                    "content": content,
                                    # "inferred_brand"는 더 이상 직접 사용하지 않음
                                })
                                logger.debug(f" - Loaded: {relative_path} (Length: {len(content)})")
                            else:
                                logger.warning(f" - Skipped empty or too short file: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error loading file {relative_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error walking through directory {data_dir}: {e}", exc_info=True)
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def parse_product_block(block_text: str, source_file: str, block_idx: int) -> Dict[str, Any]:
    """
    개별 제품 정보 블록 텍스트에서 id, source_file, product_name을 추출합니다.
    Args:
        block_text (str): "--- 다음 제품 ---"으로 분리된 개별 제품 정보 텍스트.
        source_file (str): 이 블록이 속한 원본 파일 상대 경로.
        block_idx (int): 문서 내에서 이 블록의 순서 (ID 생성용).
    Returns:
        Dict[str, Any]: 추출된 메타데이터 딕셔너리 ('id', 'source_file', 'product_name').
    """
    safe_filename_for_id = source_file.replace(os.sep, '_')
    safe_filename_for_id = os.path.splitext(safe_filename_for_id)[0]

    metadata = {
        "id": f"{safe_filename_for_id}-block{block_idx}",
        "source_file": source_file,
        "product_name": None
    }
    lines = block_text.strip().split('\n')

    if lines:
        first_line_stripped = lines[0].strip()
        # 제품명 추출 (예: 첫 줄 사용 또는 '# 브랜드 / 제품명' 형식 유지)
        # 여기서는 간단히 첫 줄을 제품명으로 가정합니다. 필요시 기존 정규식 로직 복원 가능.
        # 복잡한 파싱 없이 첫 줄을 제품명으로 사용
        if first_line_stripped:
             metadata['product_name'] = first_line_stripped
        else: # 첫 줄이 비어있으면 다음 유효한 줄을 찾거나 플레이스홀더 사용
            for line_content in lines[1:]:
                if line_content.strip():
                    metadata['product_name'] = line_content.strip()
                    break
            if not metadata['product_name']: # 모든 줄이 비어있을 경우
                 metadata['product_name'] = f"Unnamed Product in {source_file} Block {block_idx}"

    if not metadata['product_name']: # 최종적으로 제품명이 없으면
        logger.warning(f"Product name could not be parsed for block {block_idx} in '{source_file}'. Using placeholder.")
        metadata['product_name'] = f"Unknown Product in {source_file} Block {block_idx}"

    # 상세 메타데이터(브랜드, 카테고리, 가격, 특징 등) 추출 로직은 삭제되었습니다.
    logger.debug(f"Parsed basic metadata for ID '{metadata['id']}': Name='{metadata['product_name'][:50]}...'")
    return metadata


def create_chunks_from_products(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    로드된 문서의 내용을 제품 구분자로 분리하고, 각 제품 블록을
    하나의 청크로 만들며, 임베딩용 텍스트(원본 블록) 및 기본 메타데이터를 추가합니다.
    Args:
        documents (List[Dict[str, str]]): 로드된 문서 리스트 ('content', 'source_file' 포함).
    Returns:
        List[Dict[str, Any]]: 생성된 청크 리스트. 각 청크는 'id', 'source_file', 'product_name',
                                 'text'(임베딩용 원본 블록), 'raw_block_text'(GPT 전달용 원본 블록) 포함.
    """
    all_chunks = []
    product_delimiter_pattern = re.compile(r'\s*---\s*다음\s*제품\s*---\s*', re.IGNORECASE)
    logger.info("Creating single chunk per product block. Embedding text will be the raw block text.")
    total_blocks_processed = 0

    for doc_index, doc in enumerate(documents):
        source_file = doc.get('source_file', f'unknown_doc_{doc_index}')
        content = doc.get('content', '')
        # inferred_brand는 더 이상 사용하지 않음

        if not content.strip():
            logger.warning(f"Skipping empty content from {source_file}")
            continue

        product_blocks = product_delimiter_pattern.split(content)
        logger.debug(f" - Splitting '{source_file}': Found {len(product_blocks)} potential product blocks.")

        for block_index, block_text in enumerate(product_blocks):
            block_text_stripped = block_text.strip()
            if len(block_text_stripped) < 20: # 너무 짧은 블록 건너뛰기
                logger.debug(f"     - Skipping very short block {block_index} in {source_file}.")
                continue

            try:
                # source_file과 block_index를 parse_product_block에 전달
                parsed_meta = parse_product_block(block_text_stripped, source_file, block_index)
            except Exception as e:
                logger.error(f"Failed to parse product block {block_index} in {source_file}: {e}", exc_info=True)
                # 실패 시 최소 정보로 청크 생성
                safe_filename_for_id = source_file.replace(os.sep, '_')
                safe_filename_for_id = os.path.splitext(safe_filename_for_id)[0]
                parsed_meta = {
                    "id": f"{safe_filename_for_id}-block{block_index}-error",
                    "source_file": source_file,
                    "product_name": f"Parse Error in {source_file} Block {block_index}"
                }

            # 청크 데이터 구성
            chunk_data = {
                "id": parsed_meta.get("id"),
                "source_file": parsed_meta.get("source_file"),
                "product_name": parsed_meta.get("product_name"),
                "text": block_text_stripped,  # 임베딩 생성에 사용될 텍스트 = 원본 블록
                "raw_block_text": block_text_stripped # GPT 전달용 원본 블록 텍스트
            }
            # 불필요한 추가 메타데이터는 parse_product_block에서 이미 생성하지 않음

            all_chunks.append(chunk_data)
            total_blocks_processed += 1
            logger.debug(f"     - Created chunk for product: {chunk_data.get('product_name', 'N/A')[:50]}... (ID: {chunk_data.get('id')})")

    logger.info(f"Total product blocks (chunks) created: {total_blocks_processed}")
    return all_chunks


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI API call due to: {retry_state.outcome.exception()}. Attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep:.2f}s..."))
def get_embeddings_with_retry(client: openai.OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    OpenAI 임베딩 API를 호출하고 결과를 반환합니다 (재시도 및 빈 텍스트 처리 포함).
    빈 텍스트 입력 시 빈 리스트(0벡터 아님, 이후 처리)를 반환합니다.
    """
    global EXPECTED_EMBEDDING_DIM

    valid_texts = []
    original_indices = []
    for i, text_content in enumerate(texts): # 변수명 변경 text -> text_content
        if isinstance(text_content, str) and text_content.strip():
            valid_texts.append(text_content.replace("\n", " ")) # 임베딩 모델은 개행문자 불필요
            original_indices.append(i)
        else:
            logger.warning(f"Empty or invalid text detected at index {i} in the batch. Corresponding embedding will be empty list.")

    if not valid_texts:
        logger.warning("No valid texts found in the batch to send for embedding.")
        # 원본 texts 리스트 길이만큼 빈 리스트 반환
        return [[] for _ in texts]

    logger.debug(f"Calling OpenAI Embeddings API for {len(valid_texts)} texts with model {model}")
    response = client.embeddings.create(input=valid_texts, model=model, encoding_format="float")

    if not response.data:
        raise ValueError("OpenAI API response did not contain embedding data.")

    embeddings_from_api = [item.embedding for item in response.data] # 변수명 변경

    if len(embeddings_from_api) != len(valid_texts):
        raise ValueError(f"Mismatch between valid texts ({len(valid_texts)}) and returned embeddings ({len(embeddings_from_api)})")

    logger.debug(f"Successfully received {len(embeddings_from_api)} embeddings from API.")

    # 원본 texts 리스트 길이에 맞춰 최종 임베딩 리스트 구성
    final_embeddings_batch = [[] for _ in texts] # 변수명 변경
    for i, valid_idx in enumerate(original_indices):
        if i < len(embeddings_from_api):
            current_embedding = embeddings_from_api[i]
            if len(current_embedding) != EXPECTED_EMBEDDING_DIM:
                raise ValueError(f"Incorrect embedding dimension for text at original index {valid_idx}: expected {EXPECTED_EMBEDDING_DIM}, got {len(current_embedding)}")
            final_embeddings_batch[valid_idx] = current_embedding
        else:
             # 이 경우는 발생하면 안 됨 (위의 길이 체크에서 걸러져야 함)
            raise ValueError("API returned fewer embeddings than valid texts sent. This indicates an issue.")

    return final_embeddings_batch

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
    all_processed_embeddings: List[List[float]] = [] # 변수명 명확화
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    start_time_embed = time.time()
    processed_chunks_count = 0 # 변수명 명확화
    chunks_with_zero_vectors = 0

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        # 'text' 필드 사용 (create_chunks_from_products에서 raw_block_text로 채워짐)
        batch_texts_for_embedding = [chunk.get('text', '') for chunk in batch_chunks] # 변수명 명확화

        current_batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {current_batch_num}/{total_batches} (size: {len(batch_texts_for_embedding)})")

        try:
            batch_embeddings_list = get_embeddings_with_retry(openai_client, texts=batch_texts_for_embedding, model=model_name) # 변수명 명확화

            if len(batch_embeddings_list) != len(batch_texts_for_embedding):
                logger.error(f"FATAL: Embedding count mismatch in batch {current_batch_num}. Expected {len(batch_texts_for_embedding)}, got {len(batch_embeddings_list)}. Stopping.")
                return None

            # 빈 임베딩을 0-벡터로 처리
            current_batch_final_embeddings = []
            for idx, emb_vector in enumerate(batch_embeddings_list):
                if not emb_vector: # get_embeddings_with_retry가 빈 리스트를 반환한 경우
                    logger.warning(f"Received empty embedding for chunk index {i + idx} (text: '{batch_texts_for_embedding[idx][:50]}...'). Replacing with zero vector.")
                    current_batch_final_embeddings.append([0.0] * EXPECTED_EMBEDDING_DIM)
                    chunks_with_zero_vectors += 1
                else:
                    current_batch_final_embeddings.append(emb_vector)
            
            all_processed_embeddings.extend(current_batch_final_embeddings)
            processed_chunks_count += len(batch_texts_for_embedding)

        except Exception as e:
            logger.error(f"FATAL: Error getting embeddings for batch {current_batch_num}: {e}", exc_info=True)
            return None

    end_time_embed = time.time()
    logger.info(f"Embedding generation took {end_time_embed - start_time_embed:.2f} seconds.")
    if chunks_with_zero_vectors > 0:
        logger.warning(f"Replaced {chunks_with_zero_vectors} chunks with zero vectors due to empty input text or API error.")

    if not all_processed_embeddings or len(all_processed_embeddings) != total_chunks:
        logger.error(f"Error: Final embedding count ({len(all_processed_embeddings)}) does not match total chunks ({total_chunks}).")
        return None

    try:
        embeddings_np = np.array(all_processed_embeddings).astype('float32')
    except ValueError as e:
        logger.error(f"Error converting embeddings to NumPy array. Possible inconsistent dimensions? Error: {e}")
        unique_dims = {len(emb) for emb in all_processed_embeddings if emb} # 수정: emb가 비어있지 않은 경우에만 len 계산
        logger.error(f"Unique non-empty dimensions found: {unique_dims}")
        return None

    logger.info(f"Embeddings generated successfully. Final shape: {embeddings_np.shape}")
    if embeddings_np.shape[1] != EXPECTED_EMBEDDING_DIM:
        logger.error(f"FATAL: Final embedding dimension mismatch! Expected {EXPECTED_EMBEDDING_DIM}, Got {embeddings_np.shape[1]}.")
        return None

    return embeddings_np


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


def save_results(index: faiss.Index, chunk_data_list: List[Dict[str, Any]], index_path: str, metadata_path: str):
    """
    FAISS 인덱스와 메타데이터를 저장합니다.
    메타데이터에는 'id', 'source_file', 'product_name', 'raw_block_text'가 포함됩니다.
    'text' 필드(임베딩용)는 이제 'raw_block_text'와 동일하므로, 중복 저장을 피하기 위해 'text'는 저장하지 않을 수 있습니다.
    여기서는 'text'를 제외하고 'raw_block_text'를 저장합니다.
    """
    logger.info(f"Saving results to {index_path} and {metadata_path}...")
    output_dir = os.path.dirname(index_path)
    os.makedirs(output_dir, exist_ok=True)

    if index and faiss:
        try:
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index saved successfully to: {index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {index_path}: {e}", exc_info=True)
    elif not faiss: logger.error("FAISS library not available, cannot save index.")
    else: logger.error("FAISS index object is None, cannot save index.")

    if chunk_data_list:
        saved_count = 0
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in chunk_data_list:
                    item_to_save = {
                        "id": item.get("id"),
                        "source_file": item.get("source_file"),
                        "product_name": item.get("product_name"),
                        "raw_block_text": item.get("raw_block_text") # 원본 텍스트 저장
                        # 'text' 필드는 raw_block_text와 동일하므로 여기서는 저장하지 않음 (선택사항)
                    }
                    # 누락된 필수 필드가 있는지 확인 (선택적 검증)
                    if not all(item_to_save.get(k) for k in ["id", "source_file", "product_name", "raw_block_text"]):
                        logger.warning(f"Skipping metadata item due to missing essential fields: ID '{item_to_save.get('id', 'N/A')}'")
                        continue
                    try:
                        f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
                        saved_count += 1
                    except TypeError as te:
                        logger.warning(f"Could not serialize metadata item ID '{item_to_save.get('id', 'N/A')}' due to TypeError: {te}. Skipping.")
            logger.info(f"Metadata ({saved_count}/{len(chunk_data_list)} items) saved successfully to: {metadata_path} (including 'raw_block_text')")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}", exc_info=True)
    else:
        logger.warning("Metadata list (chunk_data_list) is empty, nothing to save.")


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    start_pipeline_time = time.time()
    logger.info("--- Starting RAG Offline Pipeline (Simplified Metadata, Raw Text for Embedding/Context) ---")

    if not config or not openai_client or not faiss:
        logger.critical("CRITICAL: Configuration, OpenAI client, or FAISS library not loaded. Pipeline cannot proceed.")
        exit(1)

    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        exit(1)

    chunks_with_metadata_and_raw_text = create_chunks_from_products(docs)
    if not chunks_with_metadata_and_raw_text:
        logger.error("No chunks created from product blocks. Exiting.")
        exit(1)

    # 'text' 필드 (실제로는 raw_block_text)를 사용하여 임베딩 생성
    embeddings_np = generate_openai_embeddings(chunks_with_metadata_and_raw_text)
    if embeddings_np is None:
        logger.error("Failed to generate OpenAI embeddings. Exiting.")
        exit(1)

    faiss_index = build_faiss_index(embeddings_np)
    if faiss_index is None:
        logger.error("Failed to build FAISS index. Exiting.")
        exit(1)

    save_results(faiss_index, chunks_with_metadata_and_raw_text, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    total_duration = end_pipeline_time - start_pipeline_time
    logger.info(f"--- RAG Offline Pipeline Finished in {total_duration:.2f} seconds ---")
    logger.info(f"--- Processed {len(chunks_with_metadata_and_raw_text)} product blocks as individual chunks. ---")
    logger.info(f"--- Used OpenAI model '{EMBEDDING_MODEL_NAME}' for embeddings. ---")
    logger.info(f"--- Results saved to {FAISS_INDEX_PATH} and {METADATA_PATH} ---")