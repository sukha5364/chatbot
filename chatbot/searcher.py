# chatbot/searcher.py

import os
import json
from typing import List, Dict, Optional
import numpy as np
import logging # 로깅 추가

# --- 필요한 라이브러리 임포트 (SentenceTransformer 제거) ---
try:
    import faiss
except ImportError:
    # FAISS가 없으면 RAG 기능 사용 불가
    logging.error("FAISS library not found. RAG search functionality will be disabled. Please install it: pip install faiss-cpu or faiss-gpu")
    faiss = None

# 설정 로더 임포트
from .config_loader import get_config

# 로거 설정
logger = logging.getLogger(__name__)
config = get_config()

# --- 설정 (경로만 사용, 모델명 및 임베딩 로직 제거) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 설정 파일에서 경로 읽어오도록 수정 (선택 사항)
data_config = config.get('rag', {})
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'data', 'index.faiss') # 경로는 유지하거나 config에서 관리
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'doc_meta.jsonl') # 경로는 유지하거나 config에서 관리
EXPECTED_EMBEDDING_DIM = data_config.get('embedding_dimension') # 설정에서 차원 수 읽기

# --- RAG 검색 클래스 ---
class RagSearcher:
    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        # embedding_model_name 제거
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        # self.embedding_model_name 제거

        self.index = None
        self.metadata: List[Dict] = []
        # self.embedding_model 제거

        self._load_resources()

    def _load_resources(self):
        """FAISS 인덱스와 메타데이터를 로드합니다. (임베딩 모델 로딩 제거)"""
        logger.info("Loading RAG resources...")
        # 1. Load FAISS index
        if faiss and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded successfully from {self.index_path}")
                logger.info(f"Index size: {self.index.ntotal} vectors, Dimension: {self.index.d}")
                # 로드된 인덱스 차원과 설정값 비교
                if EXPECTED_EMBEDDING_DIM and self.index.d != EXPECTED_EMBEDDING_DIM:
                     logger.warning(f"FAISS index dimension ({self.index.d}) does not match expected dimension ({EXPECTED_EMBEDDING_DIM}) in config!")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}", exc_info=True)
                self.index = None
        else:
            logger.error(f"FAISS index file not found at {self.index_path} or FAISS library not installed.")
            self.index = None # 명시적으로 None 설정

        # 2. Load Metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = [json.loads(line.strip()) for line in f if line.strip()]
                logger.info(f"Metadata loaded successfully from {self.metadata_path}. Total chunks: {len(self.metadata)}")
                # 메타데이터 개수와 인덱스 벡터 개수 일치 확인
                if self.index and self.index.ntotal != len(self.metadata):
                    logger.warning(f"FAISS index size ({self.index.ntotal}) != Metadata size ({len(self.metadata)}). Index might be outdated.")
            except Exception as e:
                logger.error(f"Error loading or parsing metadata: {e}", exc_info=True)
                self.metadata = []
        else:
            logger.error(f"Metadata file not found at {self.metadata_path}")
            self.metadata = []

        # 3. 임베딩 모델 로딩 로직 제거

        if self.index is None or not self.metadata:
             logger.error("RAG searcher initialization failed due to missing index or metadata.")
        else:
             logger.info("RAG searcher initialized successfully.")

    # _get_embedding 메서드 제거

    def search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """
        주어진 쿼리 임베딩 벡터와 가장 유사한 문서 Chunk K개를 검색합니다.

        Args:
            query_embedding (np.ndarray): 미리 계산된 쿼리 텍스트의 임베딩 벡터 (Numpy 배열, float32, shape (1, D)).
            k (int): 반환할 결과 개수.

        Returns:
            List[Dict]: 검색된 Chunk 리스트. 각 Chunk는 메타데이터와 유사도 점수 포함.
                       실패 시 빈 리스트 반환.
        """
        # 리소스 로드 확인 (embedding_model 확인 제거)
        if not self.index or not self.metadata:
            logger.error("Cannot perform search: RAG index or metadata not loaded properly.")
            return []

        # 입력된 임베딩 벡터 유효성 검사 (형태 및 차원)
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
             logger.error(f"Invalid query embedding shape: {query_embedding.shape}. Expected (1, D).")
             return []
        if self.index.d != query_embedding.shape[1]:
             logger.error(f"Query embedding dimension ({query_embedding.shape[1]}) does not match index dimension ({self.index.d}).")
             return []

        logger.debug(f"Performing FAISS search with k={k}")
        try:
            # 1. Search FAISS index (쿼리 임베딩 직접 사용)
            # index.search는 float32 NumPy 배열을 기대함
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            logger.debug(f"FAISS search completed. Found indices: {indices[0]}, Distances: {distances[0]}")

            # 2. Retrieve metadata and format results
            results = []
            if len(indices[0]) > 0 and indices[0][0] != -1: # -1은 결과 없을 때 반환될 수 있음
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.metadata):
                        result_item = self.metadata[idx].copy() # 원본 메타데이터 변경 방지 위해 복사

                        # [수정] 유사도 점수 처리: IndexFlatIP는 내적값(거리) 반환.
                        # OpenAI 임베딩은 정규화되어 있으므로 내적값은 코사인 유사도와 비례.
                        # 값의 범위는 보통 -1 ~ 1 사이 (완전히 같으면 1). 그대로 사용하거나 필요시 조정.
                        inner_product_score = float(distances[0][i])
                        result_item['similarity_score'] = inner_product_score # 점수 기록
                        # result_item['distance'] = inner_product_score # 필요시 distance 이름으로도 저장

                        # 너무 낮은 유사도 결과는 제외 (선택 사항, 임계값은 실험 필요)
                        # similarity_threshold = 0.7 # 예시 임계값
                        # if inner_product_score < similarity_threshold:
                        #    logger.debug(f"Skipping result index {idx} due to low similarity score: {inner_product_score:.4f}")
                        #    continue

                        results.append(result_item)
                    else:
                        logger.warning(f"Found index {idx} is out of bounds for metadata (size: {len(self.metadata)}).")
            else:
                 logger.info("FAISS search returned no results.")

            # 유사도 점수 기준으로 내림차순 정렬 (이미 FAISS가 정렬해서 반환하지만 확인차)
            results.sort(key=lambda x: x.get('similarity_score', -1.0), reverse=True)

            return results

        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

# --- 전역 검색기 인스턴스 ---
# 애플리케이션 시작 시 RagSearcher 인스턴스를 생성하고 리소스를 로드
# rag_searcher_instance = RagSearcher()

# --- 예시 사용법 (수정됨: 직접 실행 어려움) ---
if __name__ == "__main__":
    # 이 파일을 직접 실행하여 테스트하는 것은 이제 더 복잡합니다.
    # query_embedding을 먼저 생성해야 하기 때문입니다.
    # 테스트는 chatbot/scheduler.py 또는 통합 테스트(test_runner.py)를 통해 수행하는 것이 좋습니다.

    print("\n--- RAG Searcher Module Loaded ---")
    print("This module requires pre-generated FAISS index and metadata.")
    print("To test search functionality:")
    print("1. Ensure the chatbot server (app.py) is running.")
    print("2. Send a query to the /chat endpoint.")
    print("3. Check the logs from scheduler.py and searcher.py for search results.")

    # 아래는 직접 테스트를 위한 의사 코드 (실행 불가)
    # logger.basicConfig(level=logging.DEBUG)
    # test_query = "발볼 넓은 러닝화 추천"
    # print(f"\n[Manual Test Example - Requires Pre-calculated Embedding]")
    # print(f"Simulating search for: '{test_query}' with k=3")
    #
    # # 1. 먼저 test_query에 대한 OpenAI 임베딩 벡터(numpy 배열)를 얻어야 함
    # # embedding_vector = get_openai_embedding_for_query(test_query) # 이 함수는 별도 구현 필요
    # embedding_vector = None # Placeholder
    #
    # if embedding_vector is not None and rag_searcher_instance.index:
    #     search_results = rag_searcher_instance.search(embedding_vector, k=3)
    #     if search_results:
    #         print("\nSearch Results (Simulated):")
    #         for i, result in enumerate(search_results):
    #             print(f"\n--- Result {i+1} ---")
    #             print(f"Score: {result.get('similarity_score', 'N/A'):.4f}")
    #             # ... (다른 메타데이터 출력) ...
    #             print(f"Text: {result.get('text', 'N/A')[:100]}...")
    #     else:
    #         print("No results found or error occurred.")
    # else:
    #     print("Could not perform manual test: Embedding vector missing or RAG resources not loaded.")