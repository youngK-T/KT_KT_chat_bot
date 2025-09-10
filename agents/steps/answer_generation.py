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
                '''당신은 회의록 기반 “추출형” QA 시스템입니다.
                다음 규칙을 엄격하게 지키세요.
                - 제공된 회의록 내용에서 직접 인용하거나 요약만 허용하며, 추측이나 가정은 절대 금지합니다.
                - 관련 정보가 없을 경우, 정확히 "제공된 회의록에서 관련 정보를 찾을 수 없습니다."라고 답변합니다.
                - 최종 답변은 5문장 이내로 간결하게 구성하고, 쉼표를 사용하여 문장을 부드럽게 연결하세요.
                - 수치, 일정, 담당자는 원문에 있는 표현만 사용하고, 불확실한 표현(예: 아마, 추정, 가능성)은 사용하지 마세요.
                - 답변은 "~입니다.", "~습니다."와 같은 존댓말을 사용해 주세요.
                - 답변 시작 시 "제공된 회의록에 따르면," 같은 문장으로 출처를 밝혀주면 좋습니다.
                - **당신의 답변에 이 규칙들을 반복해서 언급하지 마세요.**

                질문: {question}
                {memory_context}

                관련 회의록 내용:
                {context}

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

            # 실제 사용된 문서 ID 계산
            used_script_ids = sorted({s["meeting_id"] for s in sources})
            
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
                "used_script_ids": used_script_ids,
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
            이전 답변의 품질이 낮습니다 (점수: {quality_score}/5).
            다음 규칙에 따라 더 정확하고 유용한 답변으로 개선해주세요.

            **엄격한 개선 규칙:**
            1. **정확성**: 제공된 '참고 자료'에 기반하여, 이전 답변의 부정확한 부분을 모두 수정하세요.
            2. **명확성**: 불분명했던 내용이나 모호한 표현을 제거하고, 핵심 정보를 명확하게 전달하세요.
            3. **간결성**: 최종 답변은 5문장 이내로 간결하게 요약하되, 필요한 정보는 모두 포함하세요.
            4. **출처 명시**: 답변 내에 근거가 되는 '참고 자료'의 핵심 문구를 직접 인용하여 신뢰도를 높이세요.
            5. **추측 금지**: '참고 자료'에 없는 내용은 절대 추가하거나 추측하지 마세요.
            6. **규칙 언급 금지**: 답변 내용에 이 규칙들을 다시 언급하지 마세요.

            질문: {question}
            이전 답변: {current_answer}
            참고 자료:
            {context}
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
