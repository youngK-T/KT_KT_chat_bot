"""
회의록 QA Agent - 리팩토링된 버전
"""

import logging
from typing import Dict
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from config.settings import AZURE_OPENAI_CONFIG
from models.state import MeetingQAState

# 분리된 모듈들 import
from .steps import (
    QuestionProcessor,
    RAGSearchProcessor,
    ScriptFetcher,
    TextProcessor,
    AnswerGenerator,
    QualityEvaluator,
    MemoryManager
)

logger = logging.getLogger(__name__)

class MeetingQAAgent:
    """회의록 QA Agent - 리팩토링된 버전"""
    
    def __init__(self):
        # LLM 초기화
        self.llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_CONFIG["api_key"],
            azure_endpoint=AZURE_OPENAI_CONFIG["endpoint"],
            api_version=AZURE_OPENAI_CONFIG["api_version"],
            azure_deployment=AZURE_OPENAI_CONFIG["deployment_name"],
            temperature=1
        )
        
        # 분리된 모듈들 초기화
        self.question_processor = QuestionProcessor(self.llm)
        self.rag_processor = RAGSearchProcessor()
        self.script_fetcher = ScriptFetcher()
        self.text_processor = TextProcessor()
        self.answer_generator = AnswerGenerator(self.llm)
        self.quality_evaluator = QualityEvaluator(self.llm)
        self.memory_manager = MemoryManager(self.llm)
        
        # 그래프 구성
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Agent 그래프 구성"""
        builder = StateGraph(MeetingQAState)
        
        # 노드 추가
        builder.add_node("summarize_memory", self.memory_manager.summarize_conversation_history)
        builder.add_node("enhance_question", self.question_processor.enhance_question_with_memory)
        builder.add_node("process_question", self.question_processor.process_question)
        builder.add_node("search_rag", self.rag_processor.get_all_rag_summaries)
        builder.add_node("get_specific_summary", self.rag_processor.get_summary_by_id)
        builder.add_node("fetch_scripts", self.script_fetcher.fetch_original_scripts)
        builder.add_node("process_scripts", self.text_processor.process_original_scripts)
        builder.add_node("select_chunks", self.text_processor.select_relevant_chunks)
        builder.add_node("generate_answer", self.answer_generator.generate_final_answer)
        builder.add_node("evaluate_answer", self.quality_evaluator.evaluate_answer_quality)
        builder.add_node("improve_answer", self.answer_generator.improve_answer)
        
        # 엣지 연결
        builder.set_entry_point("summarize_memory")
        builder.add_edge("summarize_memory", "enhance_question")
        builder.add_edge("enhance_question", "process_question")
        # 조건부 분기: process_question → search_rag 또는 get_specific_summary
        builder.add_conditional_edges(
            "process_question",
            self._route_rag_search,  # 분기 로직 함수
            {
                "general_search": "search_rag",           # 전체 RAG 검색
                "specific_search": "get_specific_summary" # 특정 스크립트 검색
            }
        )
        builder.add_edge("search_rag", "fetch_scripts")
        builder.add_edge("get_specific_summary", "fetch_scripts")
        builder.add_edge("fetch_scripts", "process_scripts")
        builder.add_edge("process_scripts", "select_chunks")
        builder.add_edge("select_chunks", "generate_answer")
        builder.add_edge("generate_answer", "evaluate_answer")
        
        # 조건부 엣지
        builder.add_conditional_edges(
            "evaluate_answer",
            self.quality_evaluator.should_improve_answer,
            {
                "improve": "improve_answer",
                "finish": END
            }
        )
        builder.add_edge("improve_answer", END)
        
        return builder.compile()
    
    def _route_rag_search(self, state: MeetingQAState) -> str:
        """RAG 검색 방식 분기 - 단순한 리스트 체크"""
        user_selected_script_ids = state.get("user_selected_script_ids", [])
        
        if user_selected_script_ids:  # 리스트가 비어있지 않으면
            return "specific_search"  # 상세 챗봇
        else:
            return "general_search"   # 기본 챗봇
    
    async def run(self, initial_state: MeetingQAState) -> MeetingQAState:
        """Agent 실행"""
        try:
            logger.info("Meeting QA Agent 실행 시작")
            final_state = await self.graph.ainvoke(initial_state)
            logger.info("Meeting QA Agent 실행 완료")
            return final_state
        except Exception as e:
            logger.error(f"Agent 실행 실패: {str(e)}")
            return {
                **initial_state,
                "error_message": f"Agent 실행 실패: {str(e)}",
                "current_step": "failed"
            }
