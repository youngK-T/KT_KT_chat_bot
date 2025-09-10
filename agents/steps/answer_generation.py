"""
6단계: 답변 생성 로직
"""

import logging
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from models.state import MeetingQAState

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """답변 생성 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_final_answer(self, state: MeetingQAState) -> MeetingQAState:
        """7단계: 최종 답변 생성"""
        try:
            user_question = state.get("user_question", "")
            relevant_summaries = state.get("relevant_summaries", [])
            relevant_chunks = state.get("relevant_chunks", [])
            conversation_memory = state.get("conversation_memory", "")
            
            if not user_question:
                raise ValueError("사용자 질문이 없습니다.")
            
            # 컨텍스트 조합
            context_parts = []
            
            # 요약본 추가
            for summary in relevant_summaries[:3]:
                context_parts.append(f"[요약본] {summary.get('summary_text', '')}")
            
            # 관련 청크 추가
            for chunk in relevant_chunks[:5]:
                context_parts.append(f"[원본] {chunk.get('chunk_text', '')}")
            
            context = "\n\n".join(context_parts)
            
            # 대화 메모리 포함한 답변 생성 프롬프트
            memory_context = f"\n\n이전 대화 맥락: {conversation_memory}" if conversation_memory else ""
            
            answer_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 질의응답 전문 AI입니다. 
                주어진 회의록 내용을 바탕으로 사용자의 질문에 정확하고 구체적으로 답변해주세요.
                
                질문: {question}
                {memory_context}
                
                관련 회의록 내용:
                {context}
                
                답변 시 주의사항:
                1. 제공된 회의록 내용만을 근거로 답변하세요
                2. 추측이나 가정은 하지 마세요
                3. 관련 정보가 없다면 "제공된 회의록에서 관련 정보를 찾을 수 없습니다"라고 답변하세요
                4. 구체적인 근거와 함께 답변하세요
                5. 이전 대화 맥락을 고려하여 일관성 있는 답변을 제공하세요
                
                답변:'''
            )
            
            formatted_prompt = answer_prompt.format(
                question=user_question,
                memory_context=memory_context,
                context=context
            )
            
            response = self.llm.invoke(formatted_prompt)
            final_answer = response.content.strip()
            
            # 출처 정보 생성
            sources = []
            for chunk in relevant_chunks[:5]:
                meeting_metadata = None
                for script in state.get("original_scripts", []):
                    if script["meeting_id"] == chunk["meeting_id"]:
                        meeting_metadata = script.get("meeting_metadata", {})
                        break
                
                source = {
                    "meeting_id": chunk["meeting_id"],
                    "meeting_title": meeting_metadata.get("meeting_title", ""),
                    "meeting_date": str(meeting_metadata.get("meeting_date", "")),
                    "chunk_index": chunk["chunk_index"],
                    "relevance_score": chunk["relevance_score"]
                }
                sources.append(source)
            
            # 신뢰도 계산 (간단한 버전)
            confidence_score = min(0.9, max(0.1, 
                sum(chunk["relevance_score"] for chunk in relevant_chunks[:3]) / 3
            )) if relevant_chunks else 0.1
            
            logger.info(f"답변 생성 완료: 신뢰도 {confidence_score:.2f}")
            
            return {
                **state,
                "context_chunks": context_parts,
                "final_answer": final_answer,
                "sources": sources,
                "confidence_score": confidence_score,
                "current_step": "completed"
            }
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {str(e)}")
            return {
                **state,
                "error_message": f"답변 생성 실패: {str(e)}",
                "current_step": "generate_answer_failed"
            }
    
    def improve_answer(self, state: MeetingQAState) -> MeetingQAState:
        """답변 개선"""
        try:
            question = state.get("processed_question", "")
            current_answer = state.get("final_answer", "")
            quality_score = state.get("answer_quality_score", 0)
            relevant_summaries = state.get("relevant_summaries", [])
            relevant_chunks = state.get("relevant_chunks", [])
            improvement_attempts = int(state.get("improvement_attempts") or 0) + 1
            
            # 컨텍스트 조합
            context_parts = []
            
            # 요약본 추가
            for summary in relevant_summaries[:3]:
                context_parts.append(f"[요약본] {summary.get('summary_text', '')}")
            
            # 관련 청크 추가
            for chunk in relevant_chunks[:5]:
                context_parts.append(f"[원본] {chunk.get('chunk_text', '')}")
            
            context = "\n\n".join(context_parts)
            
            # 개선된 답변 생성
            improvement_prompt = f"""
            이전 답변의 품질이 낮았습니다 (점수: {quality_score}/5).
            더 정확하고 유용한 답변으로 개선해주세요.
            
            질문: {question}
            이전 답변: {current_answer}
            
            참고 자료:
            {context}
            
            개선된 답변을 생성해주세요:
            1. 더 구체적이고 정확한 정보 제공
            2. 출처 명시
            3. 사용자에게 도움이 되는 내용
            4. 한국어로 명확하게 작성
            """
            
            response = self.llm.invoke(improvement_prompt)
            improved_answer = response.content.strip()
            
            return {
                **state,
                "final_answer": improved_answer,
                "improvement_attempts": improvement_attempts,
                "current_step": "answer_improved"
            }
            
        except Exception as e:
            logger.error(f"답변 개선 실패: {str(e)}")
            return {
                **state,
                "improvement_attempts": int(state.get("improvement_attempts") or 0) + 1,
                "current_step": "improvement_failed"
            }
