# chatbot/searcher.py

import os
import json
from typing import List, Dict, Tuple, Optional
import numpy as np

# --- 필요한 라이브러리 임포트 (설치 필요) ---
try:
    import faiss
except ImportError:
    print("Warning: faiss library not found. Please install it: pip install faiss-cpu")
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers library not found. Please install it: pip install sentence-transformers")
    SentenceTransformer = None

# --- 설정 ---
# 프로젝트 루트 디렉토리 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'data', 'index.faiss')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'doc_meta.jsonl')
# 사용할 임베딩 모델 (rag_generator.py 에서 사용한 모델과 동일해야 함)
# 예: 'jhgan/ko-sroberta-multitask' 또는 OpenAI 모델 사용 시 다른 방식 필요
EMBEDDING_MODEL_NAME = 'jhgan/ko-sroberta-multitask' # 한국어 모델 예시

# --- RAG 검색 클래스 ---
class RagSearcher:
    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name

        self.index = None
        self.metadata: List[Dict] = []
        self.embedding_model = None

        self._load_resources()

    def _load_resources(self):
        """FAISS 인덱스, 메타데이터, 임베딩 모델을 로드합니다."""
        # 1. Load FAISS index
        if faiss and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"FAISS index loaded successfully from {self.index_path}")
                print(f"Index size: {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.index = None
        else:
            print(f"FAISS index file not found at {self.index_path} or faiss library not installed.")

        # 2. Load Metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.metadata.append(json.loads(line.strip()))
                print(f"Metadata loaded successfully from {self.metadata_path}. Total chunks: {len(self.metadata)}")
                # 메타데이터 개수와 인덱스 벡터 개수 일치 확인
                if self.index and self.index.ntotal != len(self.metadata):
                    print(f"Warning: FAISS index size ({self.index.ntotal}) != Metadata size ({len(self.metadata)}). Check rag_generator.py")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata = []
        else:
            print(f"Metadata file not found at {self.metadata_path}")

        # 3. Load Embedding Model (SentenceTransformer 예시)
        # OpenAI 임베딩을 사용한다면, API 호출 로직으로 대체해야 함
        if SentenceTransformer:
            try:
                # GPU 사용 가능 시 'cuda' 지정 가능
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cpu')
                print(f"Embedding model '{self.embedding_model_name}' loaded successfully.")
            except Exception as e:
                print(f"Error loading embedding model '{self.embedding_model_name}': {e}")
                self.embedding_model = None
        else:
            print("SentenceTransformer library not found.")


    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """주어진 텍스트의 임베딩 벡터를 반환합니다."""
        if self.embedding_model:
            try:
                # 모델에 따라 encode 방식이 다를 수 있음
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.reshape(1, -1) # FAISS 검색을 위해 2D 배열 형태로 변환
            except Exception as e:
                print(f"Error getting embedding for text '{text[:50]}...': {e}")
                return None
        else:
            # OpenAI 임베딩 API 호출 로직 추가 가능
            print("Embedding model not loaded.")
            return None

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        주어진 쿼리와 가장 유사한 문서 Chunk K개를 검색합니다.

        Args:
            query (str): 사용자 검색어.
            k (int): 반환할 결과 개수 (기본값: 5).

        Returns:
            List[Dict]: 검색된 Chunk 리스트. 각 Chunk는 메타데이터 포함.
                       실패 시 빈 리스트 반환.
        """
        if not self.index or not self.metadata or not self.embedding_model:
            print("Error: RAG resources (index, metadata, or model) not loaded properly.")
            return []

        # 1. Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []

        # 2. Search FAISS index
        try:
            # D: distances, I: indices
            distances, indices = self.index.search(query_embedding, k)
            print(f"FAISS search completed. Found indices: {indices}")

            # 3. Retrieve metadata for found indices
            results = []
            if len(indices[0]) > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.metadata):
                        result_item = self.metadata[idx].copy() # 원본 메타데이터 변경 방지 위해 복사
                        result_item['similarity_score'] = float(1 - distances[0][i]) # 코사인 유사도로 변환 (FAISS가 L2 거리 반환 시) 또는 그대로 사용
                        # result_item['distance'] = float(distances[0][i]) # 필요시 거리 값 추가
                        results.append(result_item)
                    else:
                        print(f"Warning: Found index {idx} is out of bounds for metadata (size: {len(self.metadata)}).")
            return results

        except Exception as e:
            print(f"Error during FAISS search for query '{query}': {e}")
            return []

# --- 전역 검색기 인스턴스 (싱글톤 패턴 비슷하게 사용) ---
# 앱 실행 시 한번만 로드되도록 함
rag_searcher_instance = RagSearcher()

# --- 예시 사용법 ---
if __name__ == "__main__":
    # 주의: 이 테스트를 실행하려면 FAISS 인덱스와 메타데이터 파일이 생성되어 있어야 함
    # 또한 필요한 라이브러리(faiss-cpu, sentence-transformers) 설치 필요

    print("\n--- RAG Searcher Test ---")
    searcher = RagSearcher() # 인스턴스 생성 (리소스 로드)

    if searcher.index and searcher.metadata and searcher.embedding_model:
        test_query = "발볼 넓은 러닝화 추천"
        print(f"\nSearching for: '{test_query}' (k=3)")
        search_results = searcher.search(test_query, k=3)

        if search_results:
            print("\nSearch Results:")
            for i, result in enumerate(search_results):
                print(f"\n--- Result {i+1} ---")
                print(f"Score: {result.get('similarity_score', 'N/A'):.4f}")
                print(f"ID: {result.get('id', 'N/A')}")
                print(f"Brand: {result.get('brand', 'N/A')}")
                print(f"Category: {result.get('category', 'N/A')}")
                print(f"Text: {result.get('text', 'N/A')[:100]}...") # 텍스트 일부 출력
        else:
            print("No results found or error occurred.")
    else:
        print("\nRAG Searcher could not be initialized properly. Skipping search test.")
        print("Please ensure 'faiss-cpu', 'sentence-transformers' are installed and run 'pipeline/rag_generator.py' first.")