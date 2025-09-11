"""
6단계: 답변 생성 로직
"""

import logging
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from models.state import MeetingQAState
from utils.content_filter import detect_content_filter, create_safe_response

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """답변 생성 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """script_id 기준 최고 관련도만 유지"""
        seen = {}
        for source in sources:
            script_id = source["script_id"]
            if script_id not in seen or source["relevance_score"] > seen[script_id]["relevance_score"]:
                seen[script_id] = source
        return list(seen.values())
    
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
            
            # 요약본 추가 (항상 포함)
            for summary in relevant_summaries[:3]:
                summary_text = summary.get('summary_text', '')
                if summary_text:
                    context_parts.append(f"[요약본] {summary_text}")
            
            # 관련 청크 추가 (있는 경우에만)
            for chunk in relevant_chunks[:5]:
                chunk_text = chunk.get('chunk_text', '')
                if chunk_text:
                    context_parts.append(f"[원본] {chunk_text}")
            
            # 원본 청크가 없으면 요약본만으로 답변 생성
            if not relevant_chunks and relevant_summaries:
                logger.info("원본 청크가 없어서 요약본만으로 답변 생성")
            
            context = "\n\n".join(context_parts)
            
            # 대화 메모리 포함한 답변 생성 프롬프트
            memory_context = f"\n\n이전 대화 맥락: {conversation_memory}" if conversation_memory else ""
            
            answer_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 기반 “추출형” QA 시스템입니다.
                **형식 규칙:**
                - 답변은 자연스러운 문장으로 구성하되, 근거가 되는 인용문은 반드시 별도 줄에 작성하세요.
                - 인용문은 ("인용내용", 화자명) 형식으로 표기하고, 본문과 한 줄 띄워서 구분하세요.
                - 각 주요 포인트마다 관련 인용문을 바로 아래에 배치하세요.
                - 최종 답변은 5문장 이내로 간결하게 구성하세요.
                - 답변은 "~입니다.", "~습니다."와 같은 존댓말을 사용해 주세요.

                **예시 형식:**
                명함 교환 시에는 서서 가벼운 인사와 함께 글자쪽을 상대방에게 향하게 하여 전달해야 합니다.

                ("명함은 내 얼굴이라고 생각해 주시면 돼요…글자쪽을 상대방을 향하게 손가락으로 가리지 않게 옆부분을 잡고", 화자01)

                교환 순서는 아랫사람이 윗사람에게 먼저 주는 것이 예의입니다.

                ("건네는 순서는 이제 아랫사람이 윗사람한테 먼저 하고…오시는 분들이 먼저 이렇게 주세요", 화자01)
                '''
            )
            
            formatted_prompt = answer_prompt.format(
                question=user_question,
                memory_context=memory_context,
                context=context
            )
            
            response = self.llm.invoke(formatted_prompt)
            final_answer = response.content.strip()
            
            # fallback 메시지 처리 제거 (단순한 에러 처리로 변경)
            
            # 출처 정보 생성 (매핑 딕셔너리 사용)
            script_metadata = state.get("script_metadata", {})
            sources = []
            for chunk in relevant_chunks[:10]:  # 더 많은 청크에서 선별
                script_id = chunk["script_id"]
                metadata = script_metadata.get(script_id, {})
                
                source = {
                    "script_id": script_id,
                    "chunk_index": chunk["chunk_index"],
                    "relevance_score": chunk["relevance_score"],
                    "meeting_title": metadata.get("title", ""),  # 매핑에서 제목 조회
                    "meeting_date": metadata.get("timestamp", "")  # 매핑에서 날짜 조회
                }
                sources.append(source)
            
            # script_id 기준 중복 제거 (최고 관련도만 유지)
            sources = self._deduplicate_sources(sources)
            
            # 관련도 순으로 정렬하여 상위 5개만 유지
            sources.sort(key=lambda x: x["relevance_score"], reverse=True)
            sources = sources[:5]

            # 실제 사용된 문서 ID 계산
            used_script_ids = sorted({s["script_id"] for s in sources})
            
            # 신뢰도 계산 (간단한 버전)
            confidence_score = min(0.9, max(0.1, 
                sum(chunk["relevance_score"] for chunk in relevant_chunks[:3]) / 3
            )) if relevant_chunks else 0.1
            
            logger.info(f"답변 생성 완료 (출처 중복 제거 적용): 신뢰도 {confidence_score:.2f}, 출처 {len(sources)}개")
            
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
            
            # Azure 콘텐츠 필터 감지 (오류 코드 우선)
            filter_info = detect_content_filter(e)
            if filter_info['is_filtered']:
                logger.warning(f"답변 생성 중 콘텐츠 필터 감지: {filter_info}")
                return create_safe_response(state, 'generate_answer', filter_info)
            
            # 일반적인 오류 처리
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
            6. **인용문 형식**: 답변은 자연스러운 문장으로 구성하되, 근거가 되는 인용문은 반드시 별도 줄에 작성하세요.
                            - 인용문은 ("인용내용", 화자명) 형식으로 표기하고, 본문과 한 줄 띄워서 구분하세요.
            7. **규칙 언급 금지**: 답변 내용에 이 규칙들을 다시 언급하지 마세요.

            질문: {question}
            이전 답변: {current_answer}
            참고 자료:
            {context}
            """
            response = self.llm.invoke(improvement_prompt)
            improved_answer = response.content.strip()
            
            # fallback 메시지 처리 제거 (단순한 에러 처리로 변경)
            
            return {
                **state,
                "final_answer": improved_answer,
                "improvement_attempts": improvement_attempts,
                "current_step": "answer_improved"
            }
            
        except Exception as e:
            logger.error(f"답변 개선 실패: {str(e)}")
            
            # Azure 콘텐츠 필터 감지 (오류 코드 우선)
            filter_info = detect_content_filter(e)
            if filter_info['is_filtered']:
                logger.warning(f"답변 개선 중 콘텐츠 필터 감지: {filter_info}")
                safe_response = create_safe_response(state, 'improve_answer', filter_info)
                # improvement_attempts 추가
                safe_response["improvement_attempts"] = int(state.get("improvement_attempts") or 0) + 1
                return safe_response
            
            # 일반적인 오류 처리
            return {
                **state,
                "improvement_attempts": int(state.get("improvement_attempts") or 0) + 1,
                "current_step": "improvement_failed"
            }
