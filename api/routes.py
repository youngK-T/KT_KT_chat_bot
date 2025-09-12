from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging
from typing import Optional

from models.schemas import MeetingQARequest, MeetingQAResponse, HealthResponse, ErrorResponse
from models.state import MeetingQAState
from agents import MeetingQAAgent
from config.settings import API_VERSION

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter()

# Agent 인스턴스 (싱글톤)
_agent_instance = None

def get_agent() -> MeetingQAAgent:
    """Agent 인스턴스 의존성"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MeetingQAAgent()
    return _agent_instance

@router.post("/query", response_model=MeetingQAResponse)
async def process_meeting_question(
    request: MeetingQARequest,
    agent: MeetingQAAgent = Depends(get_agent)
):
    """회의록 질의응답 처리"""
    try:
        logger.info(f"새로운 질문 처리 시작: {request.question[:50]}...")
        
        # 🔍 요청 데이터 상세 로깅 추가
        logger.info(f"🔍 [DEBUG] 요청 데이터 분석:")
        logger.info(f"🔍 [DEBUG] - question: {request.question}")
        logger.info(f"🔍 [DEBUG] - user_selected_script_ids: {request.user_selected_script_ids}")
        logger.info(f"🔍 [DEBUG] - user_selected_script_ids type: {type(request.user_selected_script_ids)}")
        logger.info(f"🔍 [DEBUG] - user_selected_script_ids length: {len(request.user_selected_script_ids) if request.user_selected_script_ids else 0}")
        
        if request.user_selected_script_ids:
            for i, script_id in enumerate(request.user_selected_script_ids):
                logger.info(f"🔍 [DEBUG] - script_id[{i}]: '{script_id}' (type: {type(script_id)})")
        else:
            logger.info(f"🔍 [DEBUG] - user_selected_script_ids is empty or None")
        
        # 초기 상태 설정
        initial_state: MeetingQAState = {
            "user_question": request.question,
            "processed_question": "",
            "user_selected_script_ids": request.user_selected_script_ids,
            "relevant_summaries": [],
            "selected_script_ids": [],
            "original_scripts": [],
            "chunked_scripts": [],
            "relevant_chunks": [],
            "context_chunks": [],
            "final_answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "current_step": "initialized",
            "error_message": "",
            "conversation_count": 0,        # 추가
            "answer_quality_score": 0,      # 추가
            "improvement_attempts": 0       # 추가
        }
        
        # Agent 실행
        final_state = await agent.run(initial_state)
        
        # 오류 체크
        if final_state.get("error_message"):
            raise HTTPException(
                status_code=500,
                detail=final_state["error_message"]
            )
        
        # 처리 단계 로그 생성 (None 안전 처리)
        def _count(v):
            try:
                return len(v) if v is not None else 0
            except Exception:
                return 0

        # 콘텐츠 필터 감지 시 특별 처리
        if final_state.get("content_filter_triggered", False):
            logger.warning("콘텐츠 필터가 감지되어 안전 응답을 반환합니다.")
            processing_steps = [
                "질문 접수",
                "콘텐츠 필터 감지",
                "안전 응답 생성"
            ]
            
            response = MeetingQAResponse(
                final_answer="Azure 콘텐츠 필터에 따라 해당 내용의 답을 할 수 없습니다.",
                evidence_quotes=[],
                sources=[],
                confidence_score=0.0,
                processing_steps=processing_steps,
                used_script_ids=[]
            )
        else:
            # 일반적인 처리 단계 정보 구성
            processing_steps = [
                "질문 전처리 완료",
                f"RAG 검색 완료: {_count(final_state.get('relevant_summaries'))}개 관련 요약본 발견",
                f"원본 스크립트 조회 완료: {_count(final_state.get('original_scripts'))}개",
                f"청킹 및 임베딩 완료: {_count(final_state.get('chunked_scripts'))}개 청크 생성",
                f"관련 청크 선별 완료: {_count(final_state.get('relevant_chunks'))}개 청크 선택",
                "최종 답변 생성 완료"
            ]
            
            # 응답 생성
            response = MeetingQAResponse(
                final_answer=str(final_state.get("final_answer") or "답변을 생성할 수 없습니다."),
                evidence_quotes=final_state.get("evidence_quotes") or [],
                sources=final_state.get("sources") or [],
                confidence_score=float(final_state.get("confidence_score") or 0.0),
                processing_steps=processing_steps,
                used_script_ids=final_state.get("used_script_ids") or []
            )
        
        logger.info(f"질문 처리 완료: 신뢰도 {response.confidence_score:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질문 처리 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"내부 서버 오류: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version=API_VERSION
        )
    except Exception as e:
        logger.error(f"헬스체크 실패: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="서비스를 사용할 수 없습니다"
        )

@router.get("/status")
async def get_status():
    """상세 상태 확인"""
    try:
        # TODO: 외부 서비스들 헬스체크
        status = {
            "api_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "services": {
                "rag_service": "unknown",     # RAG 서비스 상태
                "meeting_api": "unknown"      # 회의록 API 서비스 상태
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"상태 확인 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"상태 확인 실패: {str(e)}"
        )
