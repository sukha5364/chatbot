# chatbot/searcher.py (요구사항 반영 최종본: Docstring 및 주석 보강, 안정성 강화)

import os
import json
from typing import List, Dict, Optional
import numpy as np
import logging

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # chatbot/chatbot -> chatbot/

# 설정 파일에서 경로 및 차원 읽기 (없으면 기본값 사용)
# TODO: 경로도 config.yaml에서 관리하도록 변경 고려 가능
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
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded successfully from {self.index_path}")
                # 로드된 인덱스 정보 로깅
                index_size = getattr(self.index, 'ntotal', 'N/A')
                index_dim = getattr(self.index, 'd', 'N/A')
                logger.info(f"Loaded index size: {index_size} vectors, Dimension: {index_dim}")

                # 로드된 인덱스 차원과 설정값 비교 검증
                if EXPECTED_EMBEDDING_DIM and index_dim != EXPECTED_EMBEDDING_DIM:
                    logger.warning(f"FAISS index dimension ({index_dim}) does not match expected dimension ({EXPECTED_EMBEDDING_DIM}) in config! Ensure consistency with rag_generator.py.")
                    # 차원 불일치 시 검색이 실패하거나 잘못될 수 있음
                    # load_success = False # 엄격하게 처리하려면 로드 실패로 간주 가능

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
                loaded_meta = []
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line: continue
                        try:
                             loaded_meta.append(json.loads(line))
                        except json.JSONDecodeError:
                             logger.warning(f"Skipping invalid JSON line {i+1} in metadata file: {line[:100]}...")
                self.metadata = loaded_meta
                logger.info(f"Metadata loaded successfully from {self.metadata_path}. Total chunks: {len(self.metadata)}")

                # 메타데이터 개수와 인덱스 벡터 개수 일치 확인 (인덱스 로드 성공 시)
                if self.index and self.index.ntotal != len(self.metadata):
                    logger.warning(f"FAISS index size ({self.index.ntotal}) does not match metadata size ({len(self.metadata)}). Search results might be incorrect if index is outdated.")
                    # load_success = False # 엄격하게 처리하려면 로드 실패로 간주 가능

            except Exception as e:
                logger.error(f"Error loading or parsing metadata from {self.metadata_path}: {e}", exc_info=True)
                self.metadata = [] # 로드 실패 시 빈 리스트로 설정
                load_success = False

        # 최종 로드 결과 로깅
        if load_success and self.index and self.metadata:
             logger.info("RAG searcher resources loaded successfully.")
        else:
             logger.error("RAG searcher initialization failed due to missing or erroneous resources. RAG search will not function.")
             # 실패 시 self.index 또는 self.metadata가 None/[] 상태 유지

    def search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """
        주어진 쿼리 임베딩 벡터와 가장 유사한 문서 Chunk K개를 FAISS 인덱스에서 검색합니다.

        Args:
            query_embedding (np.ndarray): 미리 계산된 쿼리 텍스트의 임베딩 벡터.
                                         Numpy 배열 형태이며, float32 타입, shape는 (1, D)여야 함 (D는 임베딩 차원).
            k (int): 검색하여 반환할 상위 결과의 개수.

        Returns:
            List[Dict]: 검색된 상위 K개의 Chunk 메타데이터 리스트. 각 딕셔너리에는
                        원본 메타데이터와 함께 'similarity_score'(유사도 점수)가 추가됨.
                        유사도 점수가 높은 순서로 정렬됨.
                        검색 실패 또는 결과 없음 시 빈 리스트 반환.

        Note:
            - 이 메서드는 임베딩 생성을 수행하지 않으며, 외부에서 생성된 임베딩 벡터를 입력받습니다.
            - 현재 IndexFlatIP를 사용하므로, 반환되는 score는 내적(Inner Product) 값입니다.
              OpenAI 임베딩 등 정규화된 벡터에서는 이 값이 코사인 유사도와 비례합니다.
        """
        # 1. 리소스 로드 상태 및 입력 유효성 검사
        if not self.index or not self.metadata:
            logger.error("Cannot perform search: RAG index or metadata not loaded properly.")
            return []
        if not isinstance(k, int) or k <= 0:
             logger.error(f"Invalid value for k (number of results): {k}. Must be a positive integer.")
             return []

        # 입력 임베딩 벡터 형태 및 타입 검증
        if not isinstance(query_embedding, np.ndarray):
             logger.error(f"Invalid query embedding type: {type(query_embedding)}. Expected NumPy ndarray.")
             return []
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            logger.error(f"Invalid query embedding shape: {query_embedding.shape}. Expected (1, D).")
            return []
        if query_embedding.dtype != np.float32:
             logger.warning(f"Query embedding dtype is {query_embedding.dtype}. Converting to float32 for FAISS.")
             query_embedding = query_embedding.astype('float32') # FAISS는 float32 필요

        # 입력 임베딩 차원과 인덱스 차원 일치 검증
        index_dim = getattr(self.index, 'd', None)
        if index_dim is None or index_dim != query_embedding.shape[1]:
            logger.error(f"Query embedding dimension ({query_embedding.shape[1]}) does not match loaded FAISS index dimension ({index_dim}). Cannot perform search.")
            return []

        logger.debug(f"Performing FAISS search with k={k} for query embedding shape {query_embedding.shape}")

        # 2. FAISS 인덱스 검색 실행
        try:
            start_time = time.time()
            # index.search()는 (distances, indices) 튜플 반환
            # distances: 각 결과까지의 거리(또는 유사도 점수) 배열 (shape: (1, k))
            # indices: 각 결과의 인덱스 배열 (shape: (1, k))
            distances, indices = self.index.search(query_embedding, k)
            search_duration = time.time() - start_time
            logger.debug(f"FAISS search completed in {search_duration:.4f} seconds.")
            logger.debug(f"Found indices: {indices[0]}, Distances/Scores: {distances[0]}")

            # 3. 결과 처리 및 메타데이터 결합
            results = []
            # indices[0] 에 검색된 인덱스들이 들어있음
            if len(indices[0]) > 0 and indices[0][0] != -1: # -1은 결과가 없을 때 FAISS가 반환할 수 있음
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.metadata): # 유효한 인덱스 범위 확인
                        # 원본 메타데이터 복사하여 수정 방지
                        result_item = self.metadata[idx].copy()

                        # 유사도 점수 추가 (IndexFlatIP는 내적값을 반환)
                        similarity_score = float(distances[0][i])
                        result_item['similarity_score'] = similarity_score

                        # TODO: 향후 특정 메타데이터 필터링 로직 추가 가능
                        # if result_item.get('brand') != 'Decathlon': continue

                        # 너무 낮은 유사도 결과는 제외 (선택 사항, 임계값 설정 필요)
                        # similarity_threshold = 0.7 # 예시 임계값
                        # if similarity_score < similarity_threshold:
                        #     logger.debug(f"Skipping result index {idx} due to low similarity score: {similarity_score:.4f}")
                        #     continue

                        results.append(result_item)
                    else:
                        # 검색된 인덱스가 메타데이터 범위를 벗어나는 경우 (데이터 불일치 가능성)
                        logger.warning(f"Found index {idx} from FAISS search is out of bounds for loaded metadata (size: {len(self.metadata)}). Skipping this result.")
            else:
                logger.info("FAISS search returned no valid results (indices might be empty or -1).")

            # 유사도 점수 기준으로 내림차순 정렬 (FAISS가 보통 정렬해서 주지만 확인차)
            # IndexFlatIP는 값이 클수록 유사도가 높음
            results.sort(key=lambda x: x.get('similarity_score', -float('inf')), reverse=True)

            logger.info(f"Returning {len(results)} RAG results after processing.")
            return results

        except AttributeError as ae:
             # self.index가 None일 경우 등
             logger.error(f"AttributeError during search, likely index not loaded: {ae}", exc_info=True)
             return []
        except Exception as e:
            # FAISS 검색 중 발생할 수 있는 기타 예외 처리
            logger.error(f"Error during FAISS search execution: {e}", exc_info=True)
            return []

# --- 예시 사용법 (직접 실행 어려움) ---
if __name__ == "__main__":
    # searcher.py를 직접 실행하여 테스트하는 것은 임베딩 벡터가 필요하므로 어려움.
    # 통합 테스트(test_runner.py) 또는 chatbot/app.py를 통해 테스트 권장.
    logging.basicConfig(level=logging.DEBUG) # 테스트 시 DEBUG 레벨
    logger = logging.getLogger(__name__)

    print("\n--- RAG Searcher Module Loaded ---")
    print("This module encapsulates FAISS index search logic.")
    print("It requires pre-calculated query embeddings to perform search.")
    print("To test search functionality, run the main chatbot application (app.py) or integration tests.")

    # 로드 테스트 (인스턴스 생성 시 자동 로드)
    print("\n--- Attempting to initialize RagSearcher (will load resources) ---")
    try:
        # 인스턴스 생성 시 _load_resources 자동 호출
        searcher_instance = RagSearcher()
        if searcher_instance.index and searcher_instance.metadata:
            print("\n--- RagSearcher initialized successfully. Ready for search (if embedding provided). ---")
            print(f"Index size: {searcher_instance.index.ntotal}")
            print(f"Metadata size: {len(searcher_instance.metadata)}")
            print(f"Index dimension: {searcher_instance.index.d}")
        else:
            print("\n--- RagSearcher initialization failed. Check logs for errors. ---")

        # 가상 검색 예시 (실제 실행은 어려움)
        if searcher_instance.index:
             print("\n--- Simulating a search (requires a valid query embedding) ---")
             # 가상의 3072차원 임베딩 벡터 생성 (실제로는 OpenAI API 등으로 생성해야 함)
             dummy_embedding = np.random.rand(1, EXPECTED_EMBEDDING_DIM).astype('float32')
             print(f"Using a dummy query embedding with shape: {dummy_embedding.shape}")
             k_results = 3
             print(f"Searching for top {k_results} results...")
             search_results = searcher_instance.search(dummy_embedding, k=k_results)
             if search_results:
                 print(f"\nFound {len(search_results)} results (showing top {k_results}):")
                 for i, result in enumerate(search_results):
                     print(f"\n--- Result {i+1} ---")
                     print(f"  ID: {result.get('id', 'N/A')}")
                     print(f"  Source: {result.get('source_file', 'N/A')}")
                     print(f"  Score: {result.get('similarity_score', 'N/A'):.4f}")
                     print(f"  Text Preview: {result.get('text', 'N/A')[:100]}...")
             else:
                  print("Search returned no results or failed.")

    except Exception as e:
        print(f"\nAn error occurred during RagSearcher initialization test: {e}")