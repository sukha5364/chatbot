# chatbot/config_loader.py (요구사항 반영 최종본: DEBUG 로깅 고정 및 주석 보강)

"""
YAML 설정 파일을 로드하고 관리하는 모듈입니다.
설정 파일(config.yaml)을 로드하여 애플리케이션 전체에서 공유할 수 있는
설정 객체(딕셔너리)를 제공합니다. 설정은 처음 접근 시 한 번만 로드됩니다 (캐싱).
"""

import yaml
import os
from typing import Dict, Any, Optional # Optional 추가
import logging

# --- 로깅 설정 (DEBUG 레벨 고정) ---
# 이 모듈 자체의 로그 레벨을 DEBUG로 설정합니다.
# 다른 모듈에서 이 모듈을 임포트하여 사용할 때, 해당 모듈의 로거 설정이 우선될 수 있습니다.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.debug("Config Loader logger initialized with DEBUG level.")

# --- 설정 파일 경로 ---
# config.yaml 파일이 프로젝트 루트 디렉토리에 있다고 가정합니다.
# config_loader.py 파일 위치 기준: chatbot/chatbot/ -> 상위 -> 상위 = 프로젝트 루트
try:
    CONFIG_FILENAME = "config.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    CONFIG_PATH = os.path.join(project_root, CONFIG_FILENAME)
    logger.debug(f"Configuration file path set to: {CONFIG_PATH}")
except Exception as path_e:
     logger.error(f"CRITICAL: Failed to determine configuration file path: {path_e}", exc_info=True)
     CONFIG_PATH = None # 경로 설정 실패 시 None

# --- 전역 설정 객체 (캐싱용) ---
# _config 변수는 모듈 레벨에서 한 번 로드된 설정을 저장하여 반복적인 파일 로드를 방지합니다.
_config: Optional[Dict[str, Any]] = None

def load_config() -> Dict[str, Any]:
    """
    YAML 설정 파일 (config.yaml)을 로드하여 딕셔너리로 반환합니다.
    파일이 존재하지 않거나 파싱 오류 발생 시 예외를 발생시킵니다.
    성공적으로 로드되면 내부 _config 변수에 캐싱됩니다.

    Returns:
        Dict[str, Any]: 로드된 설정 딕셔너리.

    Raises:
        FileNotFoundError: 설정 파일 경로가 잘못되었거나 파일이 존재하지 않을 경우.
        ValueError: 설정 파일이 비어있거나 유효한 YAML 형식이 아닐 경우, 또는 파싱 중 오류 발생 시.
        Exception: 파일 읽기 등 기타 예외 발생 시.
    """
    global _config # 전역 변수 참조

    if not CONFIG_PATH: # 경로 설정 실패 시 로드 불가
         raise FileNotFoundError("Configuration file path could not be determined.")

    logger.info(f"Attempting to load configuration from: {CONFIG_PATH}")
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Configuration file not found at the expected path: {CONFIG_PATH}")
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

    try:
        # 설정 파일 열기 (UTF-8 인코딩 명시)
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            # YAML 파싱 (safe_load 권장)
            loaded_yaml = yaml.safe_load(f)

        # 파싱 결과 유효성 검사
        if loaded_yaml is None: # 파일은 있지만 내용이 비어있는 경우
             logger.error("Configuration file is empty.")
             raise ValueError("Configuration file is empty.")
        if not isinstance(loaded_yaml, dict): # 최상위 구조가 딕셔너리가 아닌 경우
            logger.error(f"Configuration file is not a valid YAML dictionary structure. Type found: {type(loaded_yaml)}")
            raise ValueError("Configuration file does not represent a dictionary.")

        # 성공적으로 로드 및 검증 완료 시 전역 변수에 저장
        _config = loaded_yaml
        logger.info("Configuration loaded and validated successfully.")
        logger.debug(f"Loaded config keys: {list(_config.keys())}") # DEBUG 레벨에서 최상위 키 로깅
        return _config

    except yaml.YAMLError as e:
        # YAML 파싱 중 오류 발생
        logger.error(f"Error parsing YAML configuration file ({CONFIG_PATH}): {e}", exc_info=True)
        _config = None # 파싱 오류 시 캐시 초기화
        raise ValueError(f"Error parsing YAML configuration: {e}")
    except FileNotFoundError: # 위에서 이미 체크했지만, 혹시 모를 경쟁 상태 대비
         logger.error(f"Configuration file disappeared before reading: {CONFIG_PATH}")
         _config = None
         raise
    except Exception as e:
        # 파일 읽기 오류 등 기타 예외
        logger.error(f"Unexpected error loading configuration file ({CONFIG_PATH}): {e}", exc_info=True)
        _config = None # 오류 시 캐시 초기화
        raise # 원본 예외 다시 발생

def get_config() -> Dict[str, Any]:
    """
    로드된 설정 객체(딕셔너리)를 반환합니다.
    만약 설정이 아직 로드되지 않았다면 내부적으로 load_config()를 호출하여 로드합니다.

    Returns:
        Dict[str, Any]: 로드된 설정 딕셔너리.

    Raises:
        Exception: load_config() 실패 시 해당 예외를 그대로 전달합니다.
    """
    global _config
    # 설정이 아직 로드되지 않았다면 로드 시도
    if _config is None:
        logger.debug("Configuration cache is empty. Calling load_config().")
        # load_config() 호출, 실패 시 예외 발생 가능
        return load_config()
    else:
        logger.debug("Returning cached configuration.")
        # 캐시된 설정 반환
        return _config

# --- 모듈 임포트 시점에 로드 시도 (선택 사항) ---
# 애플리케이션 시작 시점에 설정을 미리 로드하고 싶을 경우 아래 주석 해제
# 단, 로드 실패 시 애플리케이션 시작 자체가 실패할 수 있음
# try:
#     load_config()
# except Exception as preload_e:
#     logger.warning(f"Failed to preload configuration during module import: {preload_e}. Will attempt loading on first access via get_config().")

# --- 예시 사용법 (기존 유지, 로깅 레벨 확인용) ---
if __name__ == "__main__":
    # 메인 스크립트로 실행 시 로깅 레벨 DEBUG 설정 (이미 위에서 basicConfig 호출됨)
    logger.info("--- Config Loader Test ---")
    try:
        config_data = get_config() # 설정 가져오기 (최초 호출 시 로드됨)
        print("\n--- Accessing Configuration Values ---")
        # .get()을 사용하여 안전하게 값 접근 (키가 없어도 오류 발생 안 함)
        print(f"RAG K value: {config_data.get('rag', {}).get('retrieval_k', 'Not Found')}")
        print(f"Default System Prompt Preview: {config_data.get('prompts', {}).get('default_system_prompt', 'Not Found')[:80]}...")
        print(f"Complexity Classification Model: {config_data.get('model_router', {}).get('complexity_classification', {}).get('model', 'Not Found')}")
        print(f"Logging Level (from config, may be overridden): {config_data.get('logging', {}).get('log_level', 'Not Set in Config')}")
        print(f"Medium routing model: {config_data.get('model_router', {}).get('routing_map', {}).get('medium', 'Not Found')}")

        # 캐싱 확인 (두 번째 호출 시 로드 메시지 없어야 함)
        print("\n--- Accessing Configuration Again (Testing Cache) ---")
        config_data_cached = get_config()
        print(f"Are config objects the same? {'Yes' if config_data is config_data_cached else 'No'}")

    except FileNotFoundError as fnf_e:
         print(f"\nERROR: Configuration file not found. {fnf_e}")
    except ValueError as val_e:
         print(f"\nERROR: Configuration file is invalid or empty. {val_e}")
    except Exception as e:
        print(f"\nERROR: Could not load or access config: {e}")

    logger.info("--- Config Loader Test Finished ---")