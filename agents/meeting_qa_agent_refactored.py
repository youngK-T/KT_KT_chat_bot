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
        builder.add_node("handle_content_filter", self._handle_content_filter)
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
        
        # 조건부 분기: process_question → 콘텐츠 필터 체크 또는 일반 처리
        builder.add_conditional_edges(
            "process_question",
            self._check_content_filter,  # 콘텐츠 필터 체크 함수
            {
                "content_filter": "handle_content_filter",  # 필터 감지 시
                "normal_flow": "route_rag_search"           # 정상 처리 시
            }
        )
        
        # 콘텐츠 필터 처리 후 종료
        builder.add_edge("handle_content_filter", END)
        
        # 가상 노드를 통한 RAG 검색 분기
        builder.add_node("route_rag_search", self._route_rag_search_node)
        builder.add_conditional_edges(
            "route_rag_search",
            self._route_rag_search,  # 분기 로직 함수
            {
                "general_search": "search_rag",           # 전체 RAG 검색
                "specific_search": "get_specific_summary" # 특정 스크립트 검색
            }
        )
        builder.add_edge("search_rag", "fetch_scripts")
        # get_specific_summary 후 문서 없음 처리
        builder.add_conditional_edges(
            "get_specific_summary",
            self._check_document_found,
            {
                "document_not_found": END,  # 문서 없음 시 즉시 종료
                "document_found": "fetch_scripts"  # 문서 있음 시 계속
            }
        )
        builder.add_edge("fetch_scripts", "process_scripts")
        builder.add_edge("process_scripts", "select_chunks")
        builder.add_edge("select_chunks", "generate_answer")
        
        # generate_answer 후 콘텐츠 필터 체크
        builder.add_conditional_edges(
            "generate_answer",
            self._check_content_filter_after_generation,
            {
                "content_filter": END,  # 필터 감지 시 즉시 종료
                "normal_flow": "evaluate_answer"  # 정상 처리 시 품질 평가
            }
        )
        
        # 조건부 엣지 (품질 평가 후)
        builder.add_conditional_edges(
            "evaluate_answer",
            self.quality_evaluator.should_improve_answer,
            {
                "improve": "improve_answer",
                "finish": END
            }
        )
        
        # improve_answer 후 콘텐츠 필터 체크
        builder.add_conditional_edges(
            "improve_answer",
            self._check_content_filter_after_generation,
            {
                "content_filter": END,  # 필터 감지 시 즉시 종료
                "normal_flow": END      # 정상 처리 시 종료
            }
        )
        
        return builder.compile()
    
    def _check_content_filter(self, state: MeetingQAState) -> str:
        """콘텐츠 필터 감지 체크 (질문 처리 단계)"""
        if state.get("content_filter_triggered", False):
            return "content_filter"
        else:
            return "normal_flow"
    
    def _check_content_filter_after_generation(self, state: MeetingQAState) -> str:
        """콘텐츠 필터 감지 체크 (답변 생성 후)"""
        current_step = state.get("current_step", "")
        
        # 답변 생성 또는 개선 중 콘텐츠 필터 감지 시
        if (state.get("content_filter_triggered", False) or 
            current_step in ["content_filter_in_answer", "content_filter_in_improvement"]):
            logger.info("콘텐츠 필터가 감지되어 안전 응답을 반환합니다.")
            return END  # ← END로 변경
        else:
            return "normal_flow"
    
    def _check_document_found(self, state: MeetingQAState) -> str:
        """문서 존재 여부 확인"""
        current_step = state.get("current_step", "")
        
        if current_step == "document_not_found":
            logger.info("요청된 문서를 찾을 수 없어 프로세스를 종료합니다.")
            return "document_not_found"
        else:
            return "document_found"
    
    def _handle_content_filter(self, state: MeetingQAState) -> MeetingQAState:
        """콘텐츠 필터 감지 시 안전 응답 생성"""
        logger.warning("콘텐츠 필터가 감지되어 안전 응답을 생성합니다.")
        
        return {
            **state,
            "final_answer": "Azure 콘텐츠 필터에 따라 해당 내용의 답을 할 수 없습니다.",
            "sources": [],
            "confidence_score": 0.0,
            "used_script_ids": [],
            "current_step": "content_filter_handled"
        }
    
    def _route_rag_search_node(self, state: MeetingQAState) -> MeetingQAState:
        """RAG 검색 분기를 위한 가상 노드 (상태 그대로 전달)"""
        return state
    
    def _route_rag_search(self, state: MeetingQAState) -> str:
        """RAG 검색 방식 분기 - 단순한 리스트 체크"""
        user_selected_script_ids = state.get("user_selected_script_ids", [])
        
        # 🔍 분기 로직 디버깅
        logger.info(f"🔍 [DEBUG] RAG 검색 분기 결정:")
        logger.info(f"🔍 [DEBUG] - user_selected_script_ids: {user_selected_script_ids}")
        logger.info(f"🔍 [DEBUG] - user_selected_script_ids length: {len(user_selected_script_ids) if user_selected_script_ids else 0}")
        
        if user_selected_script_ids:  # 리스트가 비어있지 않으면
            logger.info(f"🔍 [DEBUG] - 분기 결정: specific_search (상세 챗봇)")
            return "specific_search"  # 상세 챗봇
        else:
            logger.info(f"🔍 [DEBUG] - 분기 결정: general_search (기본 챗봇)")
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
