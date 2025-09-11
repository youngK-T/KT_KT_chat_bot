from typing import TypedDict, List, Dict, Optional

class MeetingQAState(TypedDict):
    # 사용자 입력
    user_question: str
    processed_question: str  # 전처리된 질문
    
    # RAG 서비스 호출 (요약본 검색)
    relevant_summaries: List[Dict]  # RAG에서 찾은 관련 요약본들
    # relevant_summaries 구조:
    # [{"summary_text": "...", "script_id": "...", "meeting_title": "...", 
    #   "meeting_date": "...", "similarity_score": 0.85}, ...]
    
    
    original_scripts: List[Dict]  # 외부 API에서 받은 원본 스크립트들
    # [{"script_id": "...", "content": "...", "title": "...", "timestamp": "...", "filename": "..."}]
    
    script_metadata: Dict[str, Dict]  # 프론트엔드용 제목 정보 (script_id -> metadata)
    # {"script_id": {"title": "회의 제목", "timestamp": "2025-09-10T10:42:47.385515099"}}
    # 원본 스크립트 처리 단계
    chunked_scripts: List[Dict]  # 청킹된 원본들
    # [{"script_id": "...", "chunk_text": "...", "chunk_index": 0, "chunk_embedding": [...]}]
    
    relevant_chunks: List[Dict]  # 질문과 관련된 청크들만 선별
    # [{"script_id": "...", "chunk_text": "...", "relevance_score": 0.9, "chunk_index": 0}]
    
    # 답변 생성 단계
    context_chunks: List[str]  # 요약본 + 관련 원본 청크 조합
    final_answer: str  # 최종 답변
    sources: List[Dict]  # 출처 정보 (청킹 관련 정보만)
    # [{"script_id": "...", "chunk_index": 0, "relevance_score": 0.9}]
    confidence_score: float  # 답변 신뢰도
    used_script_ids: List[str]  # 최종 답변에 실제로 사용된 문서 ID 목록
    
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
    user_selected_script_ids: List[str]  # 사용자가 선택한 스크립트 ID 목록
    selected_script_ids: List[str]  # RAG 유사도 검색으로 선별된 스크립트 ID 목록
    
    # 콘텐츠 필터 관리
    content_filter_triggered: bool  # Azure 콘텐츠 필터 감지 여부