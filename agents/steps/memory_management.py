"""
8단계: 메모리 관리 로직
"""

import logging
from typing import Dict
from models.state import MeetingQAState

logger = logging.getLogger(__name__)

class MemoryManager:
    """메모리 관리 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def summarize_conversation_history(self, state: MeetingQAState) -> MeetingQAState:
        """이전 대화 요약 생성"""
        try:
            current_question = state.get("user_question", "")
            previous_memory = state.get("conversation_memory", "") or ""
            conversation_count = int(state.get("conversation_count") or 0)
            
            if conversation_count == 0 or not previous_memory:
                # 첫 번째 대화
                return {
                    **state,
                    "conversation_memory": "",
                    "conversation_count": conversation_count + 1
                }
            
            # 이전 대화와 현재 질문을 요약
            summary_prompt = f"""
            이전 대화 요약: {previous_memory}
            현재 질문: {current_question}
            
            위 정보를 바탕으로 대화의 맥락을 간단히 요약해주세요.
            중요한 키워드와 주제만 포함하여 2-3문장으로 요약해주세요.
            """
            
            response = self.llm.invoke(summary_prompt)
            new_memory = response.content.strip()
            
            return {
                **state,
                "conversation_memory": new_memory,
                "conversation_count": conversation_count + 1
            }
            
        except Exception as e:
            logger.error(f"대화 요약 실패: {str(e)}")
            return {
                **state,
                "conversation_memory": state.get("conversation_memory", "") or "",
                "conversation_count": int(state.get("conversation_count") or 0) + 1
            }
    
    def determine_conversation_mode(self, state: MeetingQAState) -> MeetingQAState:
        """대화 모드 결정"""
        user_question = state.get("user_question", "")
        selected_script_id = state.get("selected_script_id")
        
        # 특정 스크립트가 선택되어 있으면 specific_script 모드
        if selected_script_id:
            return {**state, "conversation_mode": "specific_script"}
        
        # 전체 맥락에서 대화하는 경우
        return {**state, "conversation_mode": "general"}
    
    def route_rag_search(self, state: MeetingQAState) -> str:
        """RAG 검색 분기"""
        mode = state.get("conversation_mode", "general")
        return "general" if mode == "general" else "specific"
