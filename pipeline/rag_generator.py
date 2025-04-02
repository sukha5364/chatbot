# pipeline/rag_generator.py

import os
import json
import time
from typing import List, Dict, Any
import numpy as np

# --- 필요한 라이브러리 임포트 ---
try:
    import faiss
except ImportError:
    print("Error: faiss library not found. Please install it: pip install faiss-cpu")
    exit()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers library not found. Please install it: pip install sentence-transformers")
    exit()

try:
    # Langchain의 Text Splitter 사용 (문서 분할 편의성)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Tiktoken은 Langchain Splitter가 내부적으로 사용하거나, 직접 토큰 수 계산 시 필요
    import tiktoken
except ImportError:
    print("Error: langchain or tiktoken library not found. Please install it: pip install langchain tiktoken")
    exit()

# --- 설정 ---
# 프로젝트 루트 디렉토리 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'original')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'index.faiss')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'doc_meta.jsonl')

# 사용할 임베딩 모델 (chatbot/searcher.py 와 동일해야 함)
EMBEDDING_MODEL_NAME = 'jhgan/ko-sroberta-multitask' # 한국어 모델 예시
# FAISS 인덱스 차원 (위 모델의 임베딩 차원과 일치해야 함)
# 'jhgan/ko-sroberta-multitask'는 768차원, OpenAI 'text-embedding-ada-002'는 1536차원
EMBEDDING_DIM = 768 # 모델에 맞게 수정 필요

# Langchain Text Splitter 설정
# 청크 크기와 중첩은 문서 특성에 맞게 조절 필요
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # 청크 당 글자 수 (토큰 수 아님)
    chunk_overlap=50,     # 청크 간 중첩 글자 수
    length_function=len,
    is_separator_regex=False,
)

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """원본 데이터 디렉토리에서 .txt 파일들을 로드합니다."""
    documents = []
    print(f"Loading documents from: {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({"source_file": filename, "content": content})
                    print(f" - Loaded: {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    return documents

def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Langchain Text Splitter를 사용하여 문서를 Chunk로 분할합니다."""
    all_chunks = []
    print("\nSplitting documents into chunks...")
    for doc in documents:
        chunks = text_splitter.split_text(doc['content'])
        for i, chunk_text in enumerate(chunks):
            # 간단한 메타데이터 생성 (추후 브랜드, 카테고리 등 자동 추출 로직 추가 가능)
            chunk_metadata = {
                "id": f"{doc['source_file']}-chunk{i}", # 고유 ID 생성
                "source_file": doc['source_file'],
                "chunk_index": i,
                "text": chunk_text,
                # 초기 메타데이터: 필요시 파일명 기반으로 브랜드/카테고리 추론 가능
                "brand": "Unknown",
                "category": "General",
                "language": "ko" # 기본 한국어 가정
            }
            # 예: 파일명으로 브랜드 추론
            if "nike" in doc['source_file'].lower():
                 chunk_metadata["brand"] = "Nike"
            elif "hoka" in doc['source_file'].lower():
                 chunk_metadata["brand"] = "Hoka"
            elif "decathlon" in doc['source_file'].lower():
                 chunk_metadata["brand"] = "Decathlon"
            # 예: 파일명으로 카테고리 추론
            if "size" in doc['source_file'].lower():
                 chunk_metadata["category"] = "Size Guide"
            elif "faq" in doc['source_file'].lower():
                 chunk_metadata["category"] = "FAQ"

            all_chunks.append(chunk_metadata)
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def generate_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> Optional[np.ndarray]:
    """주어진 Chunk 리스트에 대해 임베딩을 생성합니다."""
    print("\nGenerating embeddings for chunks...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    try:
        # show_progress_bar=True : 진행 상태 표시
        embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)
        print(f"Embeddings generated successfully. Shape: {embeddings.shape}")
        # 차원 확인
        if embeddings.shape[1] != EMBEDDING_DIM:
             print(f"Error: Embedding dimension mismatch! Expected {EMBEDDING_DIM}, Got {embeddings.shape[1]}. Check EMBEDDING_DIM setting.")
             return None
        return embeddings
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None

def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """FAISS 인덱스를 빌드합니다."""
    if embeddings is None or embeddings.shape[0] == 0:
        print("Error: No embeddings to build index.")
        return None
    print("\nBuilding FAISS index...")
    try:
        # 가장 기본적인 Flat L2 인덱스 사용 (정확도 높음, 메모리 사용량/검색 속도 보통)
        # 데이터가 매우 클 경우 IVF, HNSW 등 다른 인덱스 타입 고려
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(embeddings.astype('float32')) # FAISS는 float32 필요
        print(f"FAISS index built successfully. Index size: {index.ntotal} vectors.")
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
    print("--- Starting RAG Offline Pipeline ---")

    # 1. Load Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        # GPU 사용 가능 시 device='cuda' 설정
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    except Exception as e:
        print(f"Failed to load embedding model: {e}")
        exit()

    # 2. Load Documents
    docs = load_documents(ORIGINAL_DATA_DIR)
    if not docs:
        print("No documents found in 'data/original/'. Exiting.")
        exit()

    # 3. Split Documents
    chunks_with_metadata = split_documents(docs)
    if not chunks_with_metadata:
        print("No chunks created. Exiting.")
        exit()

    # 4. Generate Embeddings
    embeddings_np = generate_embeddings(chunks_with_metadata, embedding_model)
    if embeddings_np is None:
        print("Failed to generate embeddings. Exiting.")
        exit()

    # 5. Build FAISS Index
    faiss_index = build_faiss_index(embeddings_np)
    if faiss_index is None:
        print("Failed to build FAISS index. Exiting.")
        exit()

    # 6. Save Index and Metadata
    save_results(faiss_index, chunks_with_metadata, FAISS_INDEX_PATH, METADATA_PATH)

    end_pipeline_time = time.time()
    print(f"\n--- RAG Offline Pipeline Finished in {end_pipeline_time - start_pipeline_time:.2f} seconds ---")