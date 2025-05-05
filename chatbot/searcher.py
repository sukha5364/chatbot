# chatbot/searcher.py (최종 수정 계획 확인 - 로직 변경 없음)

import os
import json
from typing import List, Dict, Optional
import numpy as np
import logging
import time

# --- 필요한 라이브러리 임포트 ---
try:
    import faiss
    logging.info("FAISS library imported successfully.")
except ImportError:
    # FAISS가 없으면 RAG 기능 사용 불가
    logging.error("CRITICAL: faiss library not found. RAG search functionality will be disabled. Please install it: pip install faiss-cpu or faiss-gpu")
    faiss = None # None으로 설정하여 이후 로직에서 체크

# --- 설정 로더 임포트 ---
try:
    # searcher.py는 chatbot/chatbot/ 안에 있으므로 상대 경로 사용
    from .config_loader import get_config
    logging.info("config_loader imported successfully in searcher.")
    config = get_config() # 설정 로드
    if not config: raise ValueError("Configuration could not be loaded.")
except ImportError as ie:
    logging.error(f"ERROR (searcher): Failed to import config_loader: {ie}. Check relative paths.", exc_info=True)
    config = {} # 빈 dict로 설정하여 이후 로직에서 오류 방지
except Exception as conf_e:
    logging.error(f"ERROR (searcher): Failed to load configuration: {conf_e}", exc_info=True)
    config = {}

# --- 로거 설정 (기본 설정 상속) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 명시적 설정

# --- 경로 및 설정값 ---
# 프로젝트 루트 디렉토리 기준 경로 설정
# config_loader에서 PROJECT_ROOT_FOUND를 사용할 수 있지만, 여기서는 다시 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # chatbot/chatbot -> chatbot/

# 설정 파일에서 경로 및 차원 읽기 (없으면 기본값 사용)
data_dir_config = config.get('data', {}).get('path', 'data') # config에 data.path 설정 추가 가능 (예: 'data')
FAISS_INDEX_PATH = os.path.join(BASE_DIR, data_dir_config, 'index.faiss')
METADATA_PATH = os.path.join(BASE_DIR, data_dir_config, 'doc_meta.jsonl')
# 예상 임베딩 차원 (config에서 읽기, rag_generator와 일치해야 함)
EXPECTED_EMBEDDING_DIM = config.get('rag', {}).get('embedding_dimension', 3072) # 기본값 설정

# --- RAG 검색 클래스 ---
class RagSearcher:
    """
    미리 빌드된 FAISS 인덱스와 메타데이터를 로드하여,
    주어진 쿼리 임베딩 벡터와 유사한 문서 Chunk를 검색하는 클래스.
    (메타데이터 필터링은 이 클래스 외부에서 수행됨)

    Attributes:
        index_path (str): 로드할 FAISS 인덱스 파일 경로.
        metadata_path (str): 로드할 메타데이터 파일 경로 (JSON Lines 형식).
        index (Optional[faiss.Index]): 로드된 FAISS 인덱스 객체. 로드 실패 시 None.
        metadata (List[Dict]): 로드된 메타데이터 리스트. 로드 실패 시 빈 리스트.
    """
    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
    ):
        """
        RagSearcher 인스턴스를 초기화하고 관련 리소스를 로드합니다.

        Args:
            index_path (str, optional): FAISS 인덱스 파일 경로. Defaults to FAISS_INDEX_PATH.
            metadata_path (str, optional): 메타데이터 파일 경로. Defaults to METADATA_PATH.
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

        logger.info("Initializing RagSearcher...")
        self._load_resources() # 인스턴스 생성 시 리소스 로드 실행

    def _load_resources(self):
        """
        FAISS 인덱스와 메타데이터 파일을 로드합니다.
        성공적으로 로드되면 self.index와 self.metadata 속성에 저장됩니다.
        """
        logger.info("Loading RAG resources (FAISS index and metadata)...")
        load_success = True # 로드 성공 플래그

        # 1. Load FAISS index
        if not faiss:
            logger.error("FAISS library is not installed. Cannot load FAISS index.")
            load_success = False
        elif not os.path.exists(self.index_path):
            logger.error(f"FAISS index file not found at: {self.index_path}")
            load_success = False
        else:
            try:
                start_time = time.time()
                self.index = faiss.read_index(self.index_path)
                load_time = time.time() - start_time
                logger.info(f"FAISS index loaded successfully from {self.index_path} (took {load_time:.2f}s)")
                # 로드된 인덱스 정보 로깅
                index_size = getattr(self.index, 'ntotal', 'N/A')
                index_dim = getattr(self.index, 'd', 'N/A')
                logger.info(f"Loaded index size: {index_size} vectors, Dimension: {index_dim}")

                # 로드된 인덱스 차원과 설정값 비교 검증
                if EXPECTED_EMBEDDING_DIM and index_dim != EXPECTED_EMBEDDING_DIM:
                    logger.warning(f"FAISS index dimension ({index_dim}) does not match expected dimension ({EXPECTED_EMBEDDING_DIM}) in config! Ensure consistency.")
                    # 차원 불일치 경고, 로드 자체는 성공으로 간주할 수 있음

            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}", exc_info=True)
                self.index = None # 로드 실패 시 None으로 설정
                load_success = False

        # 2. Load Metadata
        if not os.path.exists(self.metadata_path):
            logger.error(f"Metadata file not found at: {self.metadata_path}")
            load_success = False
        else:
            try:
                start_time = time.time()
                loaded_meta = []
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line: continue
                        try:
                            # 메타데이터에 'id' 필드가 있는지 확인 (선택적)
                            meta_item = json.loads(line)
                            if 'id' not in meta_item:
                                 logger.warning(f"Metadata item in line {i+1} is missing 'id' field. Adding anyway.")
                            loaded_meta.append(meta_item)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line {i+1} in metadata file: {line[:100]}...")
                self.metadata = loaded_meta
                load_time = time.time() - start_time
                logger.info(f"Metadata loaded successfully from {self.metadata_path} (took {load_time:.2f}s). Total items: {len(self.metadata)}")

                # 메타데이터 개수와 인덱스 벡터 개수 일치 확인 (인덱스 로드 성공 시)
                if self.index and self.index.ntotal != len(self.metadata):
                    logger.warning(f"FAISS index size ({self.index.ntotal}) does not match metadata size ({len(self.metadata)}). Search results might be incorrect if index/metadata are out of sync.")
                    # load_success = False # 엄격하게 처리하려면 주석 해제

            except Exception as e:
                logger.error(f"Error loading or parsing metadata from {self.metadata_path}: {e}", exc_info=True)
                self.metadata = [] # 로드 실패 시 빈 리스트로 설정
                load_success = False

        # 최종 로드 결과 로깅
        if load_success and self.index and self.metadata:
            logger.info("RAG searcher resources initialized successfully.")
        else:
            logger.error("RAG searcher initialization failed due to missing or erroneous resources. RAG search will not function properly.")

    def search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """
        주어진 쿼리 임베딩 벡터와 가장 유사한 문서 Chunk K개를 FAISS 인덱스에서 검색합니다.
        결과에는 유사도 점수('similarity_score')가 포함됩니다.

        Args:
            query_embedding (np.ndarray): Numpy 배열 형태의 쿼리 임베딩 벡터 (float32, shape=(1, D)).
            k (int): 검색할 상위 결과 개수.

        Returns:
            List[Dict]: 검색된 상위 K개의 Chunk 메타데이터 리스트. 'similarity_score' 포함.
                         검색 실패 시 빈 리스트 반환.
        """
        # 1. 리소스 로드 상태 및 입력 유효성 검사
        if not self.index or not self.metadata:
            logger.error("Cannot perform search: RAG index or metadata not loaded.")
            return []
        if not isinstance(k, int) or k <= 0:
            logger.error(f"Invalid value for k: {k}. Must be positive integer.")
            return []
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            logger.error(f"Invalid query embedding shape: {query_embedding.shape if isinstance(query_embedding, np.ndarray) else type(query_embedding)}. Expected (1, D).")
            return []
        if query_embedding.dtype != np.float32:
            logger.warning(f"Query embedding dtype {query_embedding.dtype} != float32. Converting...")
            query_embedding = query_embedding.astype('float32')

        index_dim = getattr(self.index, 'd', None)
        if index_dim is None or index_dim != query_embedding.shape[1]:
            logger.error(f"Query embedding dimension ({query_embedding.shape[1]}) mismatch with index dimension ({index_dim}).")
            return []

        # 검색할 개수(k)가 인덱스 크기보다 크면 인덱스 크기로 조정
        actual_k = min(k, self.index.ntotal)
        if actual_k != k:
            logger.warning(f"Requested k ({k}) is larger than index size ({self.index.ntotal}). Searching for {actual_k} results instead.")
        if actual_k <= 0:
            logger.warning("Index size is 0 or k is non-positive after adjustment. Cannot search.")
            return []

        logger.debug(f"Performing FAISS search with k={actual_k} for query embedding shape {query_embedding.shape}")

        # 2. FAISS 인덱스 검색 실행
        try:
            start_time = time.time()
            # IndexFlatIP는 내적(Inner Product) 거리 사용. 값이 클수록 유사.
            distances, indices = self.index.search(query_embedding, actual_k)
            search_duration = time.time() - start_time
            logger.debug(f"FAISS search completed in {search_duration:.4f} seconds.")
            # logger.debug(f"Found indices: {indices[0]}, Scores: {distances[0]}") # 로그 양 줄이기 위해 주석 처리 가능

            # 3. 결과 처리 및 메타데이터 결합
            results = []
            # indices[0]은 shape (1, k) 중 첫 번째 행 (실제 인덱스 배열)
            # distances[0]은 shape (1, k) 중 첫 번째 행 (실제 거리/점수 배열)
            if len(indices[0]) > 0:
                for i, idx in enumerate(indices[0]):
                    # FAISS는 결과가 k개 미만일 때 -1을 반환할 수 있음
                    if idx == -1:
                         logger.debug(f"Found index -1 at position {i}, stopping result processing for this query.")
                         break # -1 인덱스 이후는 유효하지 않음
                    if 0 <= idx < len(self.metadata):
                        # 메타데이터 복사 후 점수 추가 (원본 메타데이터 수정 방지)
                        result_item = self.metadata[idx].copy()
                        # IndexFlatIP의 결과는 내적 값이므로 similarity_score로 사용
                        similarity_score = float(distances[0][i])
                        result_item['similarity_score'] = similarity_score
                        results.append(result_item)
                    else:
                        logger.warning(f"FAISS index {idx} is out of bounds for metadata (size: {len(self.metadata)}). Skipping.")
            else:
                 logger.info("FAISS search returned no indices.")

            # 유사도 점수 기준 내림차순 정렬 (IndexFlatIP는 점수가 높을수록 유사)
            results.sort(key=lambda x: x.get('similarity_score', -float('inf')), reverse=True)

            logger.debug(f"Returning {len(results)} raw RAG results (before filtering in scheduler).")
            return results

        except AttributeError as ae: # index 객체가 None일 경우 등
             logger.error(f"AttributeError during search (index not loaded?): {ae}", exc_info=True)
             return []
        except Exception as e:
             logger.error(f"Error during FAISS search execution: {e}", exc_info=True)
             return []

# --- 예시 사용법 (직접 실행 어려움) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__) # logger 재정의 필요 없음
    print("\n--- RAG Searcher Module Loaded ---")
    print("Handles FAISS index loading and raw vector search.")
    print("Metadata post-filtering is handled externally (e.g., in scheduler.py).")
    print("No code changes required for this module in the final plan.")

    # 로드 테스트
    print("\n--- Attempting to initialize RagSearcher (will load resources) ---")
    try:
        # 로드 시간 측정
        init_start_time = time.time()
        searcher_instance = RagSearcher()
        init_duration = time.time() - init_start_time
        print(f"(Initialization took {init_duration:.2f} seconds)")

        if searcher_instance.index and searcher_instance.metadata:
            print("\n--- RagSearcher initialized successfully. ---")
            print(f"Index size: {searcher_instance.index.ntotal}, Metadata size: {len(searcher_instance.metadata)}, Index dimension: {searcher_instance.index.d}")

            # 간단한 검색 테스트 (임의 벡터 사용)
            print("\n--- Performing a dummy search test ---")
            if searcher_instance.index.d > 0:
                 dummy_vector = np.random.rand(1, searcher_instance.index.d).astype('float32')
                 search_results = searcher_instance.search(dummy_vector, k=3)
                 print(f"Dummy search returned {len(search_results)} results.")
                 if search_results:
                     print("Top result preview:")
                     # JSON 직렬화 불가능한 객체(numpy 등) 제외하고 출력
                     preview = {k: v for k, v in search_results[0].items() if isinstance(v, (str, int, float, list, dict, bool))}
                     print(json.dumps(preview, indent=2, ensure_ascii=False))
            else:
                 print("Index dimension is 0, cannot perform dummy search.")

        else:
            print("\n--- RagSearcher initialization failed. Check logs. ---")
    except Exception as e:
        print(f"\nAn error occurred during RagSearcher initialization test: {e}")