from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class MeetingQARequest(BaseModel):
    """회의록 QA 요청 모델"""
    question: str = Field(..., description="사용자 질문", min_length=1)
    user_selected_script_ids: List[str] = Field(default=[], description="사용자가 선택한 스크립트 ID 목록")

class SourceInfo(BaseModel):
    """출처 정보 모델"""
    meeting_id: str = Field(..., description="회의 ID")
    meeting_title: str = Field(..., description="회의 제목")
    meeting_date: str = Field(..., description="회의 날짜")
    chunk_index: Optional[int] = Field(None, description="청크 인덱스")
    relevance_score: float = Field(..., description="관련성 점수", ge=0.0, le=1.0)

class MeetingQAResponse(BaseModel):
    """회의록 QA 응답 모델"""
    answer: str = Field(..., description="최종 답변")
    sources: List[SourceInfo] = Field(..., description="출처 정보 목록")
    confidence_score: float = Field(..., description="답변 신뢰도", ge=0.0, le=1.0)
    processing_steps: List[str] = Field(..., description="처리 단계 로그")

class ErrorResponse(BaseModel):
    """오류 응답 모델"""
    detail: str = Field(..., description="오류 상세 내용")
    error_code: Optional[str] = Field(None, description="오류 코드")
    
class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    timestamp: str = Field(..., description="확인 시각")
    version: str = Field(..., description="API 버전")



'''RAG 서비스 모델'''

# 요청
class RAGSummariesRequest(BaseModel):
    selected_script_ids: List[str] = Field(default=[], description="선택된 스크립트 ID 목록")

# 전체 요약본 응답
class RAGAllSummariesResponse(BaseModel):
    all_summaries: Dict[str, Dict[str, List[float]]]  # {script_id: {embedding: List[float]}}

# 선택된 요약본 응답  
class RAGSelectedSummaryResponse(BaseModel):
    selected_summary: Dict[str, Dict[str, List[float]]]  # {script_id: {embedding: List[float]}}
