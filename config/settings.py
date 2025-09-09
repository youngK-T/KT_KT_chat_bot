import os
from pathlib import Path

def load_api_keys(filepath="api_key.txt"):
    """API 키 파일에서 환경변수 로드"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# API 키 로드
api_key_path = Path(__file__).parent / "api_key.txt"  # config/api_key.txt
if api_key_path.exists():
    load_api_keys(str(api_key_path))

# Azure OpenAI 설정
AZURE_OPENAI_CONFIG = {
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
    "deployment_name": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
}

# 임베딩 전용 설정
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# API 설정
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Meeting QA API"
API_DESCRIPTION = "회의록 질의응답 시스템 API"
API_VERSION = "1.0.0"

# 외부 서비스 URL 설정 (Azure 환경)
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag-service.example.com:8080")
MEETING_API_URL = os.environ.get("MEETING_API_URL", "http://meeting-api.example.com:8080")

# 기본 설정값
DEFAULT_RAG_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
