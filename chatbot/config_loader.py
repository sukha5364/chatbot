# chatbot/config_loader.py

import yaml
import os
from typing import Dict, Any
import logging # 로깅 추가

# 로거 설정 (간단하게)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # 기본 로깅 레벨 설정

CONFIG_FILENAME = "config.yaml"
# config.yaml 파일이 프로젝트 루트에 있다고 가정
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_FILENAME)

_config: Dict[str, Any] = None

def load_config() -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드하여 딕셔너리로 반환합니다.
    이미 로드된 경우 캐시된 설정을 반환합니다.
    """
    global _config
    if _config is None:
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
            logger.error(f"Configuration file not found at {CONFIG_PATH}")
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                _config = yaml.safe_load(f)
                if not isinstance(_config, dict):
                    # 파일은 존재하나 내용이 비어있거나 형식이 잘못된 경우
                    logger.error("Configuration file is empty or not valid YAML format.")
                    raise ValueError("Configuration file is empty or not valid YAML format.")
            logger.info("Configuration loaded successfully.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    return _config

def get_config() -> Dict[str, Any]:
    """
    로드된 설정 객체를 반환합니다. 아직 로드되지 않았으면 로드합니다.
    """
    if _config is None:
        return load_config()
    return _config

# 모듈이 임포트될 때 한 번 로드 시도 (선택 사항, get_config() 호출 시 로드되도록 해도 됨)
# try:
#     load_config()
# except Exception as e:
#     logger.warning(f"Failed to preload configuration: {e}. Will attempt loading on first access.")

# --- 예시 사용법 ---
if __name__ == "__main__":
    print("--- Config Loader Test ---")
    try:
        config = get_config() # 설정 가져오기
        print(f"RAG K value: {config.get('rag', {}).get('retrieval_k', 'Not Found')}")
        print(f"Default System Prompt: {config.get('prompts', {}).get('default_system_prompt', 'Not Found')}")
        print(f"Classification Model: {config.get('model_router', {}).get('classification_model', 'Not Found')}")
        print(f"Log Level: {config.get('logging', {}).get('log_level', 'INFO')}")
        # 깊은 구조 접근 테스트
        print(f"Medium routing model: {config.get('model_router', {}).get('routing_map', {}).get('medium', 'Not Found')}")
    except Exception as e:
        print(f"Could not load or access config: {e}")