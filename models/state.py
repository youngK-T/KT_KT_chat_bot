from typing import TypedDict, List, Dict, Optional

class MeetingQAState(TypedDict):
    # 사용자 입력
    user_question: str
    processed_question: str  # 전처리된 질문
    
    # RAG 서비스 호출 (요약본 검색)
    search_keywords: List[str]  # 추출된 검색 키워드
    relevant_summaries: List[Dict]  # RAG에서 찾은 관련 요약본들
    # relevant_summaries 구조:
    # [{"summary_text": "...", "meeting_id": "...", "meeting_title": "...", 
    #   "meeting_date": "...", "similarity_score": 0.85}, ...]
    
    # PostgreSQL 조회 단계 (1차 호출)
    selected_script_ids: List[str]  # 선택된 스크립트 ID들
    meeting_metadata: List[Dict]  # PostgreSQL에서 조회한 메타데이터
    # meeting_metadata 구조:
    # [{"meeting_id": "...", "blob_url": "https://...", "blob_key": "...", 
    #   "meeting_title": "...", "meeting_date": "..."}, ...]
    
    # Azure Blob Storage 조회 단계 (2차 호출)
    original_scripts: List[Dict]  # Blob에서 받은 원본 txt들
    # original_scripts 구조:
    # [{"meeting_id": "...", "full_content": "긴 원본 텍스트...", 
    #   "blob_url": "...", "file_size": 1024}, ...]
    
    # 원본 스크립트 처리 단계
    chunked_scripts: List[Dict]  # 청킹된 원본들
    # [{"meeting_id": "...", "chunk_text": "...", "chunk_index": 0, "chunk_embedding": [...]}]
    
    relevant_chunks: List[Dict]  # 질문과 관련된 청크들만 선별
    # [{"meeting_id": "...", "chunk_text": "...", "relevance_score": 0.9, "chunk_index": 0}]
    
    # 답변 생성 단계
    context_chunks: List[str]  # 요약본 + 관련 원본 청크 조합
    final_answer: str  # 최종 답변
    sources: List[Dict]  # 출처 정보
    confidence_score: float  # 답변 신뢰도
    
    # API 연결 정보
    rag_service_url: str  # RAG 서비스 엔드포인트
    meeting_api_url: str  # 회의록 API 서비스 엔드포인트
    
    # 시스템 상태
    current_step: str  # 현재 처리 단계
    error_message: str  # 오류 메시지

    # 대화 메모리 관리
    conversation_memory: str  # 이전 대화 요약
    conversation_count: int   # 대화 횟수
    
     # 답변 품질 관리
    answer_quality_score: int  # 1-5점 품질 점수
    improvement_attempts: int  # 개선 시도 횟수
    
    # 분기 상태 관리
    conversation_mode: str  # "general" 또는 "specific_script"
    user_selected_script_ids: List[str]  # 사용자가 선택한 스크립트 ID 목록
    selected_script_ids: List[str]  # RAG 유사도 검색으로 선별된 스크립트 ID 목록