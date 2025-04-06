# pipeline/rag_generator.py

import os
import json
import time
from typing import List, Dict, Any, Optional # Optional 추가
import numpy as np
import asyncio # 비동기 처리 (API 호출 시 유용할 수 있으나, 여기서는 동기 배치 처리)
from tenacity import retry, wait_random_exponential, stop_after_attempt # API 재시도

# --- 필요한 라이브러리 임포트 ---
try:
    import faiss
except ImportError:
    print("Error: faiss library not found. Please install it: pip install faiss-cpu")
    exit()

# OpenAI 라이브러리 임포트 (SentenceTransformer 대신 사용)
try:
    import openai
    from dotenv import load_dotenv
except ImportError:
    print("Error: openai or python-dotenv library not found. Please install them: pip install openai python-dotenv")
    exit()

try:
    # Langchain의 Text Splitter 사용 (문서 분할 편의성)
    # CharacterTextSplitter는 tiktoken 인코더와 함께 사용할 수 있음
    from langchain.text_splitter import CharacterTextSplitter
    import tiktoken
except ImportError:
    print("Error: langchain or tiktoken library not found. Please install it: pip install langchain tiktoken")
    exit()

# --- 설정 ---
# .env 파일 로드 (OPENAI_API_KEY 로드 위함)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")

# 프로젝트 루트 디렉토리 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'original')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'index.faiss')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'doc_meta.jsonl')

# --- OpenAI 임베딩 모델 설정 ---
# 사용할 OpenAI 임베딩 모델 이름
# EMBEDDING_MODEL_NAME = "text-embedding-ada-002" # 1536차원
EMBEDDING_MODEL_NAME = "text-embedding-3-large" # 3072차원 (요청하신 모델)
# FAISS 인덱스 차원 (위 모델의 임베딩 차원과 일치해야 함)
# EMBEDDING_DIM = 1536 # for text-embedding-ada-002
EMBEDDING_DIM = 3072 # for text-embedding-3-large

# --- 텍스트 분할 설정 (OpenAI 모델 토큰 기준) ---
# OpenAI 모델은 토큰 단위로 입력을 처리하므로, 토큰 기반 Splitter 사용 권장
# text-embedding-3-large의 최대 입력 토큰은 8191개
# 청크 크기는 모델 최대치보다 훨씬 작게 설정 (API 호출 효율 및 검색 정확도 고려)
# 예: 1000 토큰 크기, 100 토큰 중첩
try:
    # tiktoken 인코더를 사용하여 토큰 기반으로 텍스트 분할
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", # 임베딩 모델과 직접 관련 없지만 토큰화 기준 모델 지정 (cl100k_base 인코딩 사용)
        chunk_size=1000,    # 청크 당 최대 토큰 수
        chunk_overlap=100     # 청크 간 중첩 토큰 수
    )
except Exception as e:
    print(f"Could not load tiktoken encoder for splitter: {e}")
    print("Using basic character splitter as fallback.")
    # Fallback to character splitter if tiktoken fails
    text_splitter = CharacterTextSplitter(
        separator="\n\n", # 문단 기준으로 우선 분리 시도
        chunk_size=2000,  # 글자 수 기준 (토큰 기준보다 덜 정확)
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
)


# --- OpenAI API 호출 설정 ---
# 배치 크기: 한 번의 API 호출에 포함할 텍스트 Chunk 수
# OpenAI API는 배치 처리를 지원하며, 한 번에 여러 텍스트를 보내는 것이 효율적
# Rate Limit(분당 요청/토큰 제한)에 따라 적절히 조절 필요
# text-embedding-3-large는 분당 10,000K 토큰, 분당 5,000 요청 (Free tier는 더 낮음)
EMBEDDING_BATCH_SIZE = 100 # 예시 배치 크기 (필요시 조절)
# API 재시도 설정 (네트워크 오류나 Rate Limit 발생 시)
openai_client = openai.OpenAI() # OpenAI 클라이언트 초기화

# --- 함수 정의 ---

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """원본 데이터 디렉토리에서 .txt 파일들을 로드합니다."""
    # 이 함수는 이전과 동일하게 사용 가능
    documents = []
    print(f"Loading documents from: {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 빈 파일이나 너무 짧은 파일은 건너뛸 수 있음 (선택 사항)
                    if len(content.strip()) > 10:
                        documents.append({"source_file": filename, "content": content})
                        print(f" - Loaded: {filename}")
                    else:
                        print(f" - Skipped empty or too short file: {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    return documents

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Langchain Text Splitter를 사용하여 문서를 Chunk로 분할합니다."""
    # 이 함수는 text_splitter 설정만 변경되었으므로 로직은 거의 동일
    all_chunks = []
    print("\nSplitting documents into chunks...")
    for doc in documents:
        # CharacterTextSplitter.split_text 사용
        chunks_text = text_splitter.split_text(doc['content'])
        print(f" - Splitting '{doc['source_file']}': found {len(chunks_text)} chunks.")
        for i, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip(): # 빈 청크 건너뛰기
                continue

            chunk_metadata = {
                "id": f"{doc['source_file']}-chunk{i}", # 고유 ID 생성
                "source_file": doc['source_file'],
                "chunk_index": i,
                "text": chunk_text,
                "brand": "Unknown",
                "category": "General",
                "language": "ko"
            }
            # 파일명 기반 메타데이터 추론 (이전과 동일)
            if "nike" in doc['source_file'].lower(): chunk_metadata["brand"] = "Nike"
            elif "hoka" in doc['source_file'].lower(): chunk_metadata["brand"] = "Hoka"
            elif "decathlon" in doc['source_file'].lower(): chunk_metadata["brand"] = "Decathlon"
            if "size" in doc['source_file'].lower(): chunk_metadata["category"] = "Size Guide"
            elif "faq" in doc['source_file'].lower(): chunk_metadata["category"] = "FAQ"

            all_chunks.append(chunk_metadata)
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

# --- OpenAI 임베딩 생성 함수 (배치 처리 및 재시도 포함) ---
# 지수 백오프를 사용한 재시도 데코레이터
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings_with_retry(client: openai.OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """OpenAI 임베딩 API를 호출하고 결과를 반환 (재시도 포함)"""
    response = client.embeddings.create(input=texts, model=model, encoding_format="float")
    return [item.embedding for item in response.data]

def generate_openai_embeddings(chunks: List[Dict[str, Any]], client: openai.OpenAI, model_name: str, batch_size: int = EMBEDDING_BATCH_SIZE) -> Optional[np.ndarray]:
    """
    주어진 Chunk 리스트에 대해 OpenAI API를 사용하여 임베딩을 생성합니다. (배치 처리)

    Args:
        chunks (List[Dict[str, Any]]): 분할된 텍스트 Chunk와 메타데이터 리스트.
        client (openai.OpenAI): 초기화된 OpenAI 클라이언트.
        model_name (str): 사용할 OpenAI 임베딩 모델 이름.
        batch_size (int): 한 번의 API 호출에 보낼 텍스트 수.

    Returns:
        Optional[np.ndarray]: 생성된 임베딩 벡터들의 Numpy 배열. 실패 시 None.
    """
    print(f"\nGenerating embeddings using OpenAI model: {model_name}...")
    all_embeddings: List[List[float]] = []
    total_chunks = len(chunks)

    # 배치 단위로 처리
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk['text'] for chunk in batch_chunks]

        if not batch_texts:
            continue

        print(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} (size: {len(batch_texts)})")

        try:
            # API 호출 (재시도 로직 포함)
            batch_embeddings = get_embeddings_with_retry(client, texts=batch_texts, model=model_name)
            all_embeddings.extend(batch_embeddings)
            # Rate Limit 방지를 위한 약간의 대기 시간 (선택 사항)
            time.sleep(0.5)

        except Exception as e:
            print(f"Error getting embeddings for batch starting at index {i}: {e}")
            # 배치 처리 중 하나라도 실패하면 전체를 실패로 간주할 수도 있고,
            # 실패한 배치를 건너뛰고 나머지만 처리할 수도 있음 (여기서는 전체 실패 처리)
            return None

    if not all_embeddings or len(all_embeddings) != total_chunks:
         print(f"Error: Embedding generation resulted in {len(all_embeddings)} embeddings, expected {total_chunks}.")
         return None

    embeddings_np = np.array(all_embeddings).astype('float32')
    print(f"Embeddings generated successfully. Shape: {embeddings_np.shape}")

    # 차원 확인
    if embeddings_np.shape[1] != EMBEDDING_DIM:
         print(f"Error: Embedding dimension mismatch! Expected {EMBEDDING_DIM}, Got {embeddings_np.shape[1]}. Check EMBEDDING_DIM setting.")
         return None
    return embeddings_np

# --- FAISS 인덱스 빌드 및 결과 저장 함수 (이전과 동일) ---

def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """FAISS 인덱스를 빌드합니다."""
    if embeddings is None or embeddings.shape[0] == 0:
        print("Error: No embeddings to build index.")
        return None
    print("\nBuilding FAISS index...")
    try:
        # IndexFlatL2는 정확도가 높지만 메모리를 많이 사용하고 검색 속도가 느릴 수 있음.
        # 벡터 수가 매우 많다면 IndexIVFFlat, IndexHNSWFlat 등을 고려.
        # text-embedding-3-large는 normalize된 임베딩을 반환하므로 IndexFlatIP (내적) 사용이 더 적합하고 빠를 수 있음.
        # index = faiss.IndexFlatL2(EMBEDDING_DIM) # L2 거리 기반
        index = faiss.IndexFlatIP(EMBEDDING_DIM)   # 내적(Inner Product) 기반 - 정규화된 벡터에 더 적합
        index.add(embeddings.astype('float32')) # FAISS는 float32 필요
        print(f"FAISS index built successfully (using Inner Product). Index size: {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None

def save_results(index: faiss.Index, metadata: List[Dict[str, Any]], index_path: str, metadata_path: str):
    """FAISS 인덱스와 메타데이터를 파일로 저장합니다."""
    print("\nSaving results...")
    # Save FAISS index
    if index:
        try:
            faiss.write_index(index, index_path)
            print(f"FAISS index saved to: {index_path}")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    # Save metadata (JSON Lines format)
    if metadata:
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    start_pipeline_time = time.time()
    print("--- Starting RAG Offline Pipeline (Using OpenAI Embeddings) ---")

    # 1. OpenAI 클라이언트 초기화 (API 키는 환경 변수에서 로드됨)
    client = openai_client

    # 2. 원본 문서 로드
    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        print("No documents found in 'data/original/'. Exiting.")
        exit()

    # 3. 문서 분할 및 메타데이터 생성
    chunks_with_metadata = split_documents(docs)
    if not chunks_with_metadata:
        print("No chunks created. Exiting.")
        exit()

    # 4. OpenAI 임베딩 생성 (배치 처리 함수 사용)
    embeddings_np = generate_openai_embeddings(chunks_with_metadata, client, EMBEDDING_MODEL_NAME)
    if embeddings_np is None:
        print("Failed to generate OpenAI embeddings. Exiting.")
        exit()

    # 5. FAISS 인덱스 빌드
    faiss_index = build_faiss_index(embeddings_np)
    if faiss_index is None:
        print("Failed to build FAISS index. Exiting.")
        exit()

    # 6. 결과(인덱스, 메타데이터) 저장
    save_results(faiss_index, chunks_with_metadata, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    print(f"\n--- RAG Offline Pipeline Finished in {end_pipeline_time - start_pipeline_time:.2f} seconds ---")
    print(f"--- NOTE: Using OpenAI model '{EMBEDDING_MODEL_NAME}' incurs API costs. ---")