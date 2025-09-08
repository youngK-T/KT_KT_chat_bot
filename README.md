# Meeting QA API

회의록을 기반으로 한 질의응답 시스템 API입니다. MSA 환경에서 RAG와 Azure 서비스들을 활용하여 정확한 답변을 제공합니다.

## 🏗️ 아키텍처

```
사용자 질문 → RAG 검색 → PostgreSQL → Azure Blob → 답변 생성
             (요약본)      (메타데이터)   (원본 텍스트)
```

## 📁 프로젝트 구조

```
KT_KT_chat_bot/
├── 📁 config/              # 설정 관리
│   ├── settings.py         # 환경설정
│   └── api_key.txt        # API 키 (gitignore)
├── 📁 models/              # 데이터 모델
│   ├── state.py           # Agent State 정의
│   └── schemas.py         # Pydantic 스키마
├── 📁 agents/              # Agent 로직
│   └── meeting_qa_agent.py # 메인 QA Agent
├── 📁 services/            # 외부 서비스 클라이언트
│   ├── rag_client.py      # RAG 서비스
│   ├── postgres_client.py # PostgreSQL
│   └── blob_client.py     # Azure Blob Storage
├── 📁 utils/               # 유틸리티
│   ├── text_processing.py # 텍스트 처리
│   └── embeddings.py      # 임베딩 관리
├── 📁 api/                 # FastAPI 서버
│   ├── main.py            # FastAPI 앱
│   └── routes.py          # API 라우트
├── 📁 tests/               # 테스트 코드
├── 📄 main.py              # 실행 엔트리포인트
├── 📄 requirements.txt     # 의존성
├── 📄 test_ui.py          # Gradio 테스트 UI
└── 📄 README.md           # 이 파일
```

## 🚀 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. API 키 설정
`config/api_key.txt` 파일을 생성하고 Azure OpenAI 설정을 입력:
```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

### 3. 서버 실행
```bash
python main.py
```

서버가 시작되면:
- API: http://localhost:8000
- 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/api/v1/health

## 📋 API 사용법

### 회의록 질의응답
```bash
POST /api/v1/meeting-qa
Content-Type: application/json

{
    "question": "지난 주 회의에서 결정된 마케팅 전략은?",
    "rag_service_url": "http://rag-service:8080",
    "postgresql_config": {
        "host": "localhost",
        "database": "meeting_db",
        "user": "postgres",
        "password": "password",
        "port": 5432
    }
}
```

### 응답 예시
```json
{
    "answer": "지난 주 회의에서는 소셜미디어 중심의 마케팅 전략이 결정되었습니다...",
    "sources": [
        {
            "meeting_id": "MEETING_2024_0101_001",
            "meeting_title": "마케팅 전략 회의",
            "meeting_date": "2024-01-01",
            "chunk_index": 3,
            "relevance_score": 0.95
        }
    ],
    "confidence_score": 0.88,
    "processing_steps": [
        "질문 전처리 완료",
        "RAG 검색 완료: 3개 관련 요약본 발견",
        "DB 조회 완료: 2개 원본 스크립트 획득",
        "청킹 및 임베딩 완료",
        "관련 청크 선별 완료",
        "최종 답변 생성 완료"
    ]
}
```

## 🔧 Agent 처리 흐름

1. **질문 전처리**: 검색 최적화된 형태로 변환, 키워드 추출
2. **RAG 검색**: 외부 RAG 서비스에서 관련 요약본 검색
3. **메타데이터 조회**: PostgreSQL에서 Blob Storage 정보 획득
4. **원본 다운로드**: Azure Blob Storage에서 원본 텍스트 다운로드
5. **텍스트 처리**: 청킹, 임베딩 생성
6. **관련 청크 선별**: 질문과 유사도 높은 청크 선택
7. **답변 생성**: LLM을 통한 최종 답변 생성

## 🧪 테스트

### Gradio 테스트 UI
```bash
python test_ui.py
```

### 개발 환경 실행
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 환경 설정

### PostgreSQL 테이블 구조
```sql
CREATE TABLE meeting_files (
    meeting_id VARCHAR(50) PRIMARY KEY,
    meeting_title VARCHAR(200),
    meeting_date DATE,
    blob_url TEXT,
    blob_key TEXT,
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 외부 서비스 요구사항
- **RAG 서비스**: `/search` 엔드포인트 필요
- **PostgreSQL**: 회의 메타데이터 저장
- **Azure Blob Storage**: 원본 회의록 텍스트 파일

## 📝 개발 노트

- MSA 환경을 위한 분산 아키텍처
- Azure OpenAI 기반 임베딩 및 답변 생성
- 비동기 처리 지원
- 확장 가능한 모듈 구조
