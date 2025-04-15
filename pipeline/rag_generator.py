# pipeline/rag_generator.py (요구사항 반영 최종본)

import os
import json
import time
import logging # 로깅 임포트 추가
from typing import List, Dict, Any, Optional
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- 필요한 라이브러리 임포트 ---
try:
    import faiss
    logging.info("FAISS library imported successfully.")
except ImportError:
    logging.error("CRITICAL: faiss library not found. RAG pipeline cannot run. Please install it: pip install faiss-cpu or faiss-gpu")
    faiss = None # 실행 중단 대신 None 할당하여 이후 로직에서 체크
    # exit() # 스크립트 실행 즉시 중단 대신, main 로직에서 체크

try:
    import openai
    from dotenv import load_dotenv
    logging.info("OpenAI and python-dotenv libraries imported successfully.")
except ImportError:
    logging.error("CRITICAL: openai or python-dotenv library not found. RAG pipeline cannot run. Please install them: pip install openai python-dotenv")
    openai = None
    # exit()

try:
    from langchain.text_splitter import CharacterTextSplitter
    import tiktoken # tiktoken도 직접 임포트하여 사용 가능성 확인
    logging.info("Langchain (CharacterTextSplitter) and tiktoken libraries imported successfully.")
except ImportError:
    logging.error("CRITICAL: langchain or tiktoken library not found. RAG pipeline cannot run. Please install them: pip install langchain tiktoken")
    CharacterTextSplitter = None
    tiktoken = None
    # exit()

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
    # exit()
except Exception as e:
    logging.error(f"CRITICAL: Unexpected error during config_loader import setup: {e}")
    get_config = None
    # exit()


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
            # openai.api_key = api_key # Legacy 방식, OpenAI() 클라이언트 사용 권장
            openai_client = openai.OpenAI(api_key=api_key) # 클라이언트 초기화
            logger.info("OpenAI client initialized successfully.")
        else:
             raise ImportError("OpenAI library not imported.")

        # RAG 파이프라인 설정값 추출 (설정 없을 경우 대비 기본값 설정)
        rag_config = config.get('rag', {})
        pipeline_config = rag_config.get('pipeline', {})

        EMBEDDING_MODEL_NAME = rag_config.get('embedding_model', 'text-embedding-3-large')
        EMBEDDING_DIM = rag_config.get('embedding_dimension', 3072)
        EMBEDDING_BATCH_SIZE = pipeline_config.get('embedding_batch_size', 100)

        DOCUMENT_DELIMITER = pipeline_config.get('document_delimiter', '\n\n\n')
        SPLITTER_TYPE = pipeline_config.get('splitter_type', 'character') # 기본값 character
        SPLITTER_MODEL_NAME = pipeline_config.get('splitter_model_name', 'gpt-4') # tiktoken용
        SPLITTER_CHUNK_SIZE = pipeline_config.get('splitter_chunk_size', 1000)
        SPLITTER_CHUNK_OVERLAP = pipeline_config.get('splitter_chunk_overlap', 100)

        logger.info(f"RAG Config: Embedding Model='{EMBEDDING_MODEL_NAME}', Dim={EMBEDDING_DIM}, BatchSize={EMBEDDING_BATCH_SIZE}")
        logger.info(f"Pipeline Config: Delimiter='{repr(DOCUMENT_DELIMITER)}', Splitter='{SPLITTER_TYPE}', ChunkSize={SPLITTER_CHUNK_SIZE}, Overlap={SPLITTER_CHUNK_OVERLAP}")

    else:
        raise ImportError("Config loader (get_config) is not available.")

except (ValueError, ImportError, KeyError, Exception) as e:
    logger.error(f"CRITICAL: Failed to load configuration or initialize components: {e}", exc_info=True)
    # 필수 설정 로드 실패 시 파이프라인 실행 불가
    config = None # 명시적으로 None 처리

# --- 경로 설정 ---
BASE_DIR = PROJECT_ROOT # 이미 위에서 정의됨
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'original')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'index.faiss')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'doc_meta.jsonl')

# --- 함수 정의 ---

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    원본 데이터 디렉토리에서 .txt 파일들을 로드합니다.

    Args:
        data_dir (str): 원본 텍스트 파일들이 있는 디렉토리 경로.

    Returns:
        List[Dict[str, str]]: 로드된 문서 리스트. 각 문서는 'source_file'과 'content' 키를 가짐.
                                오류 발생 시 빈 리스트 반환.
    """
    documents = []
    logger.info(f"Loading documents from: {data_dir}")
    if not os.path.isdir(data_dir):
        logger.error(f"Original data directory not found: {data_dir}")
        return []
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 빈 파일이나 너무 짧은 파일은 건너뛰기 (최소 길이 설정 가능)
                        if len(content.strip()) > 10:
                            documents.append({"source_file": filename, "content": content})
                            logger.debug(f" - Loaded: {filename} (Length: {len(content)})")
                        else:
                            logger.warning(f" - Skipped empty or too short file: {filename}")
                except Exception as e:
                    logger.error(f"Error loading file {filename}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error listing files in directory {data_dir}: {e}", exc_info=True)
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    로드된 문서를 설정된 구분자(delimiter)로 1차 분할 후,
    설정된 Text Splitter를 사용하여 Chunk로 2차 분할합니다.

    Args:
        documents (List[Dict[str, str]]): 로드된 문서 리스트.

    Returns:
        List[Dict[str, Any]]: 분할된 모든 Chunk 리스트. 각 Chunk는 메타데이터와 텍스트 포함.
                               오류 발생 시 빈 리스트 반환.
    """
    global config # 전역 config 사용 명시

    if not config or not CharacterTextSplitter: # 필수 설정 및 라이브러리 확인
        logger.error("Configuration or Langchain/tiktoken library not loaded. Cannot split documents.")
        return []

    # 설정 값 로드 (함수 호출 시점에 다시 로드할 수도 있음)
    delimiter = config['rag']['pipeline'].get('document_delimiter', '\n\n\n')
    splitter_type = config['rag']['pipeline'].get('splitter_type', 'character')
    chunk_size = config['rag']['pipeline'].get('splitter_chunk_size', 1000)
    chunk_overlap = config['rag']['pipeline'].get('splitter_chunk_overlap', 100)
    model_name = config['rag']['pipeline'].get('splitter_model_name', 'gpt-4') # tiktoken용

    all_chunks = []
    logger.info(f"Splitting documents using delimiter '{repr(delimiter)}' and splitter type '{splitter_type}'...")

    for doc_index, doc in enumerate(documents):
        source_file = doc.get('source_file', f'unknown_doc_{doc_index}')
        content = doc.get('content', '')
        if not content.strip():
            logger.warning(f"Skipping empty content from {source_file}")
            continue

        # 1단계: 설정된 구분자로 1차 분할
        primary_sections = content.split(delimiter)
        logger.debug(f" - Splitting '{source_file}': Found {len(primary_sections)} primary sections using delimiter.")

        for section_index, section_text in enumerate(primary_sections):
            section_text = section_text.strip()
            if not section_text: # 빈 섹션 건너뛰기
                continue

            # 2단계: 설정된 Text Splitter로 2차 분할
            try:
                text_splitter: Optional[CharacterTextSplitter] = None
                if splitter_type == 'tiktoken' and tiktoken:
                    # tiktoken 기반 스플리터 초기화
                    try:
                        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                            model_name=model_name, # 토큰화 기준 모델
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separator="\n\n" # 청크 내에서도 문단 구분을 시도
                        )
                        logger.debug(f"   - Using tiktoken splitter for section {section_index} (chunk_size={chunk_size}, overlap={chunk_overlap})")
                    except Exception as e:
                        logger.error(f"   - Failed to initialize tiktoken splitter for section {section_index}: {e}. Falling back to character splitter.")
                        # Fallback 처리
                        splitter_type = 'character' # 강제로 변경하여 아래 로직 타도록

                # Character 스플리터 (기본 또는 Fallback)
                if splitter_type == 'character' or text_splitter is None:
                    # CharacterTextSplitter 직접 사용
                    text_splitter = CharacterTextSplitter(
                        separator="\n\n", # 문단 기준으로 우선 분리 시도
                        chunk_size=chunk_size, # 토큰 기준보다 덜 정확하므로 값 조정 필요할 수 있음
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        is_separator_regex=False,
                    )
                    logger.debug(f"   - Using character splitter for section {section_index} (chunk_size={chunk_size}, overlap={chunk_overlap})")

                # 텍스트 분할 실행
                chunks_text = text_splitter.split_text(section_text)
                logger.debug(f"   - Section {section_index} split into {len(chunks_text)} chunks.")

                # 각 Chunk에 메타데이터 추가
                for i, chunk_text in enumerate(chunks_text):
                    if not chunk_text.strip(): # 빈 청크 건너뛰기
                        continue

                    # 고유 ID 형식: {source_file}-section{section_index}-chunk{i}
                    chunk_id = f"{source_file}-s{section_index}-c{i}"
                    chunk_metadata = {
                        "id": chunk_id,
                        "source_file": source_file,
                        "primary_section_index": section_index, # 1차 분할 인덱스
                        "chunk_index_in_section": i,         # 2차 분할 인덱스
                        "text": chunk_text,
                        # --- 메타데이터 추론 (선택 사항, 향후 고도화 가능) ---
                        # "brand": "Unknown", # 파일명 기반 추론 등은 아래에서 처리
                        # "category": "General",
                        # "language": "ko"
                        # TODO: 향후 Few-shot 예제 문서 처리를 위한 필드 추가 고려 (예: 'is_few_shot_example': boolean)
                    }
                    # 파일명 기반 기본 메타데이터 추론 (단순 예시)
                    # TODO: 보다 정교한 메타데이터 추출 로직 적용 가능
                    l_source_file = source_file.lower()
                    if "kiprun" in l_source_file or "kalenji" in l_source_file: chunk_metadata["brand"] = "Kiprun/Kalenji"
                    elif "quechua" in l_source_file or "forclaz" in l_source_file: chunk_metadata["brand"] = "Quechua/Forclaz"
                    elif "decathlon" in l_source_file: chunk_metadata["brand"] = "Decathlon General"
                    else: chunk_metadata["brand"] = "Unknown"

                    if "size" in l_source_file: chunk_metadata["category"] = "Size Guide"
                    elif "faq" in l_source_file or "policy" in l_source_file: chunk_metadata["category"] = "FAQ/Policy"
                    elif "tent" in l_source_file: chunk_metadata["category"] = "Tent"
                    elif "shoes" in l_source_file: chunk_metadata["category"] = "Shoes"
                    else: chunk_metadata["category"] = "Product Info"

                    all_chunks.append(chunk_metadata)

            except Exception as e:
                logger.error(f"   - Error splitting section {section_index} of '{source_file}': {e}", exc_info=True)
                continue # 해당 섹션 처리 실패 시 다음 섹션으로

    logger.info(f"Total chunks created after splitting: {len(all_chunks)}")
    return all_chunks

# --- OpenAI 임베딩 생성 함수 (배치 처리 및 재시도 포함) ---
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
       before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI API call due to: {retry_state.outcome.exception()}. Attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep:.2f}s..."))
def get_embeddings_with_retry(client: openai.OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    OpenAI 임베딩 API를 호출하고 결과를 반환합니다 (재시도 포함).

    Args:
        client (openai.OpenAI): 초기화된 OpenAI 클라이언트.
        texts (List[str]): 임베딩을 생성할 텍스트 리스트.
        model (str): 사용할 OpenAI 임베딩 모델 이름.

    Returns:
        List[List[float]]: 생성된 임베딩 벡터 리스트.

    Raises:
        Exception: API 호출 실패 또는 재시도 후에도 실패 시.
    """
    logger.debug(f"Calling OpenAI Embeddings API for {len(texts)} texts with model {model}")
    response = client.embeddings.create(input=texts, model=model, encoding_format="float")
    # 응답 구조 확인 및 임베딩 추출
    if not response.data:
        raise ValueError("OpenAI API response did not contain embedding data.")
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(texts):
        raise ValueError(f"Mismatch between number of input texts ({len(texts)}) and returned embeddings ({len(embeddings)})")
    logger.debug(f"Successfully received {len(embeddings)} embeddings from API.")
    return embeddings

def generate_openai_embeddings(chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    주어진 Chunk 리스트에 대해 OpenAI API를 사용하여 임베딩을 생성합니다. (배치 처리)

    Args:
        chunks (List[Dict[str, Any]]): 분할된 텍스트 Chunk와 메타데이터 리스트.

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

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk['text'] for chunk in batch_chunks if chunk.get('text')] # 빈 텍스트 제외

        if not batch_texts:
            logger.warning(f"Skipping empty batch starting at index {i}.")
            continue

        current_batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {current_batch_num}/{total_batches} (size: {len(batch_texts)})")

        try:
            # API 호출 (재시도 로직 포함된 함수 사용)
            batch_embeddings = get_embeddings_with_retry(openai_client, texts=batch_texts, model=model_name)
            all_embeddings.extend(batch_embeddings)
            processed_chunks += len(batch_texts)
            # Rate Limit 방지를 위한 약간의 대기 시간 (필요에 따라 조절)
            time.sleep(0.2) # 0.2초 대기

        except Exception as e:
            logger.error(f"FATAL: Error getting embeddings for batch {current_batch_num} (starting index {i}): {e}", exc_info=True)
            logger.error("Stopping embedding generation due to API error.")
            return None # 배치 처리 중 하나라도 실패하면 전체 실패

    end_time_embed = time.time()
    logger.info(f"Embedding generation took {end_time_embed - start_time_embed:.2f} seconds.")

    # 최종 결과 검증
    if not all_embeddings or len(all_embeddings) != total_chunks:
        logger.error(f"Error: Embedding generation resulted in {len(all_embeddings)} embeddings, but expected {total_chunks}. Check logs for batch errors.")
        return None

    # Numpy 배열로 변환 (float32)
    try:
        embeddings_np = np.array(all_embeddings).astype('float32')
    except ValueError as e:
         logger.error(f"Error converting embeddings to NumPy array. Possible inconsistent dimensions? Error: {e}")
         return None

    logger.info(f"Embeddings generated successfully. Final shape: {embeddings_np.shape}")

    # 차원 확인
    if embeddings_np.shape[1] != expected_dim:
        logger.error(f"FATAL: Embedding dimension mismatch! Expected {expected_dim}, Got {embeddings_np.shape[1]}. Check config 'rag.embedding_dimension' and ensure it matches the model '{model_name}'.")
        return None # 차원 불일치는 심각한 문제

    return embeddings_np

# --- FAISS 인덱스 빌드 및 결과 저장 함수 ---

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

    embedding_dim = config['rag'].get('embedding_dimension', 3072)
    if embeddings.shape[1] != embedding_dim:
         logger.error(f"Cannot build FAISS index: Embedding dimension ({embeddings.shape[1]}) does not match configured dimension ({embedding_dim}).")
         return None

    logger.info(f"Building FAISS index (using IndexFlatIP for dimension {embedding_dim})...")
    start_time_faiss = time.time()
    try:
        # OpenAI 임베딩은 정규화되어 있으므로 내적(Inner Product)이 코사인 유사도와 동일/비례하며 빠름
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings) # embeddings는 이미 float32로 가정
        end_time_faiss = time.time()
        logger.info(f"FAISS index built successfully in {end_time_faiss - start_time_faiss:.2f} seconds. Index size: {index.ntotal} vectors.")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        return None

def save_results(index: faiss.Index, metadata: List[Dict[str, Any]], index_path: str, metadata_path: str):
    """
    FAISS 인덱스와 메타데이터를 지정된 경로에 파일로 저장합니다.

    Args:
        index (faiss.Index): 빌드된 FAISS 인덱스 객체.
        metadata (List[Dict[str, Any]]): 각 Chunk의 메타데이터 리스트.
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
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in metadata:
                    # 메타데이터에 text 필드가 너무 길 경우 자르거나 제외하는 옵션 추가 가능
                    # item_to_save = item.copy()
                    # if 'text' in item_to_save and len(item_to_save['text']) > 500: # 예시: 500자 초과 시 자르기
                    #     item_to_save['text_preview'] = item_to_save['text'][:500] + "..."
                    #     del item_to_save['text']
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Metadata ({len(metadata)} items) saved successfully to: {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}", exc_info=True)
    else:
         logger.warning("Metadata list is empty, nothing to save.")

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    start_pipeline_time = time.time()
    logger.info("--- Starting RAG Offline Pipeline (Using Configured OpenAI Embeddings) ---")

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
    if not CharacterTextSplitter or not tiktoken:
         logger.error("CRITICAL: Langchain/tiktoken not loaded. Pipeline cannot proceed.")
         exit(1)

    # 1. 원본 문서 로드
    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        exit(1)

    # 2. 문서 분할 (수정된 로직 사용)
    chunks_with_metadata = split_documents(docs)
    if not chunks_with_metadata:
        logger.error("No chunks created after splitting. Exiting.")
        exit(1)

    # 3. OpenAI 임베딩 생성 (수정된 함수 사용)
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
    save_results(faiss_index, chunks_with_metadata, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    total_duration = end_pipeline_time - start_pipeline_time
    logger.info(f"--- RAG Offline Pipeline Finished in {total_duration:.2f} seconds ---")
    logger.info(f"--- Used OpenAI model '{config.get('rag', {}).get('embedding_model')}' which incurs API costs. ---")