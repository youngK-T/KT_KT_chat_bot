"""
7단계: 품질 평가 로직
"""

import logging
from typing import Dict, Literal
from models.state import MeetingQAState
from utils.content_filter import detect_content_filter, create_safe_response

logger = logging.getLogger(__name__)

class QualityEvaluator:
    """품질 평가 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_answer_quality(self, state: MeetingQAState) -> MeetingQAState:
        """답변 품질 평가"""
        try:
            answer = (state.get("final_answer", "") or "")
            context_chunks = state.get("context_chunks", []) or []

            # 규칙 기반 강등
            apology_patterns = ["제공해주신", "포함되어 있지", "찾을 수 없", "없습니다", "죄송", "어렵습니다", "정보가 부족"]
            if not context_chunks:
                return {**state, "answer_quality_score": 1, "current_step": "quality_evaluated"}
            if any(p in answer for p in apology_patterns):
                return {**state, "answer_quality_score": 1, "current_step": "quality_evaluated"}
            if len(answer) < 30:
                return {**state, "answer_quality_score": 2, "current_step": "quality_evaluated"}

            # LLM을 사용한 품질 평가
            evaluation_prompt = f"""
            다음 답변의 품질을 1-5점으로 평가해주세요.
            
            질문: {state.get("processed_question", "")}
            답변: {answer}
            
            평가 기준:
            1점: 전혀 관련 없는 답변
            2점: 관련은 있지만 부정확한 답변
            3점: 부분적으로 정확한 답변
            4점: 대부분 정확하고 유용한 답변
            5점: 완벽하고 매우 유용한 답변
            
            점수만 숫자로 답변해주세요 (예: 4)
            """
            
            response = self.llm.invoke(evaluation_prompt)
            quality_score = int(response.content.strip())
            
            # 4점(애매)도 개선 대상으로 내림
            if quality_score == 4:
                quality_score = 3
            
            improvement_attempts = state.get("improvement_attempts", 0)
            
            return {
                **state,
                "answer_quality_score": quality_score,
                "improvement_attempts": improvement_attempts,
                "current_step": "quality_evaluated"
            }
            
        except Exception as e:
            logger.error(f"답변 품질 평가 실패: {str(e)}")
            
            # Azure 콘텐츠 필터 감지
            filter_info = detect_content_filter(e)
            if filter_info['is_filtered']:
                logger.warning(f"품질 평가 중 콘텐츠 필터 감지: {filter_info}")
                return create_safe_response(state, 'quality_evaluation', filter_info)
            
            # 일반적인 오류 처리
            return {
                **state,
                "answer_quality_score": 3,  # 기본값
                "current_step": "quality_evaluation_failed"
            }
    
    def should_improve_answer(self, state: MeetingQAState) -> Literal["improve", "finish"]:
        """답변 개선 여부 결정"""
        score = int(state.get("answer_quality_score") or 5)
        tries = int(state.get("improvement_attempts") or 0)
        
        # 한 번만 개선 시도
        if tries >= 1:
            return "finish"
        
        # 4 이하이면 개선 진행
        return "improve" if score <= 4 else "finish"
