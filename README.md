# Meeting QA API

회의록을 기반으로 한 질의응답 시스템 API입니다. MSA 환경에서 RAG와 Azure 서비스들을 활용하여 정확한 답변을 제공합니다.

## 🏗️ 아키텍처

```
사용자 질문 → RAG 검색 → 원본 스크립트 조회(API) → 청킹/임베딩 → 답변 생성/평가
            (요약 임베딩)   (GET /api/scripts)     (LangChain Splitter)
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
│   ├── meeting_qa_agent_refactored.py # 메인 QA Agent (그래프 오케스트레이션)
│   └── steps/                       # 단계별 모듈
│       ├── question_processing.py   # 질문 전처리
│       ├── rag_search.py            # RAG 검색(전체/다중)
│       ├── script_fetch.py          # 원본 스크립트 조회(다중 GET)
│       ├── text_processing.py       # 청킹/임베딩/유사도
│       ├── answer_generation.py     # 답변 생성/개선
│       └── quality_evaluation.py    # 품질 평가 및 라우팅
├── 📁 services/            # 외부 서비스 클라이언트
│   └── rag_client.py      # RAG 서비스 클라이언트
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
python -m api.main
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
  "user_selected_script_ids": []
}
```

### 응답 예시
```json
{
    "answer": "지난 주 회의에서는 소셜미디어 중심의 마케팅 전략이 결정되었습니다...",
    "sources": [
        {
            "script_id": "MEETING_2024_0101_001",
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
        "원본 스크립트 조회 완료: 2개",
        "청킹 및 임베딩 완료",
        "관련 청크 선별 완료",
        "최종 답변 생성 완료"
    ]
}
```

## 🔧 Agent 처리 흐름

1. **질문 전처리**: 검색 최적화/키워드 추출
2. **RAG 검색**: 
   - 기본 챗봇: GET `/api/rag/script-summaries`(전체) → 유사도 선별
   - 상세 챗봇: GET `/api/rag/script-summaries?scriptIds=a,b,c`(다중) → 유사도 선별
3. **원본 스크립트 조회**: GET `/api/scripts?scriptIds=a,b,c` → `scriptText` 수신
4. **텍스트 처리**: LangChain `RecursiveCharacterTextSplitter`로 청킹, 임베딩 생성
5. **관련 청크 선별**: 코사인 유사도 기반 Top-K
6. **답변 생성/평가/개선**: LLM 생성 → 품질 평가 → 1회 개선 시도

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

### 외부 서비스 요구사항
- **RAG 서비스**:
  - 전체: `GET /api/rag/script-summaries`
  - 다중: `GET /api/rag/script-summaries?scriptIds=a,b,c`
  - 응답: `{ scriptId, embedding[] }` 배열 또는 `{ scriptId: { embedding } }` 매핑
- **회의 스크립트 서비스**:
  - 전체: `GET /api/scripts`
  - 단일: `GET /api/scripts?scriptIds=abc123`
  - 다중: `GET /api/scripts?scriptIds=a,b,c`
  - 응답: `{ scriptId, storageUrl, scriptText }`

## 📝 개발 노트

- MSA 환경을 위한 분산 아키텍처
- Azure OpenAI 기반 임베딩 및 답변 생성
- 비동기 처리 지원
- 확장 가능한 모듈 구조
