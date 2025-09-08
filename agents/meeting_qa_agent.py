from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import StateGraph, END
from typing import Literal

from models.state import MeetingQAState
from services.rag_client import RAGClient
from services.postgres_client import PostgreSQLClient
from services.blob_client import AzureBlobClient
from utils.text_processing import chunk_text, clean_text
from utils.embeddings import EmbeddingManager, find_most_relevant_chunks
from config.settings import AZURE_OPENAI_CONFIG, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

class MeetingQAAgent:
    """회의록 QA Agent"""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_CONFIG["deployment_name"],
            api_version=AZURE_OPENAI_CONFIG["api_version"],
            azure_endpoint=AZURE_OPENAI_CONFIG["endpoint"],
            api_key=AZURE_OPENAI_CONFIG["api_key"]
        )
        self.embedding_manager = EmbeddingManager()
        self.blob_client = AzureBlobClient()
        
        # Agent 그래프 구성
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Agent 그래프 구성"""
        builder = StateGraph(MeetingQAState)
        
        # 노드 추가
        builder.add_node("process_question", self.process_question)
        builder.add_node("search_rag", self.search_rag_summaries)
        builder.add_node("fetch_metadata", self.fetch_meeting_metadata)
        builder.add_node("fetch_scripts", self.fetch_original_scripts)
        builder.add_node("process_scripts", self.process_original_scripts)
        builder.add_node("select_chunks", self.select_relevant_chunks)
        builder.add_node("generate_answer", self.generate_final_answer)
        
        # 엣지 연결
        builder.set_entry_point("process_question")
        builder.add_edge("process_question", "search_rag")
        builder.add_edge("search_rag", "fetch_metadata")
        builder.add_edge("fetch_metadata", "fetch_scripts")
        builder.add_edge("fetch_scripts", "process_scripts")
        builder.add_edge("process_scripts", "select_chunks")
        builder.add_edge("select_chunks", "generate_answer")
        builder.add_edge("generate_answer", END)
        
        return builder.compile()
    
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
    
    def process_question(self, state: MeetingQAState) -> MeetingQAState:
        """1단계: 질문 전처리 및 키워드 추출"""
        try:
            user_question = state.get("user_question", "")
            if not user_question:
                raise ValueError("user_question이 비어 있습니다.")

            # 질문 전처리 프롬프트
            question_process_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 검색을 위한 질문을 분석하는 AI입니다.
                다음 사용자 질문을 분석하여 더 명확하고 검색에 최적화된 형태로 전처리해주세요.
                단, 사용자의 의도를 절대로 변경하지 마세요. 질문은 길어져도 괜찮습니다.
                
                사용자 질문: {user_question}
                
                전처리된 질문을 출력해주세요:'''
            )
            
            # 키워드 추출 프롬프트  
            keyword_extract_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 검색을 위한 키워드를 추출하는 AI입니다.
                다음 질문에서 검색에 중요한 핵심 키워드를 5~8개 추출해주세요.
                
                질문: {processed_question}
                
                핵심 키워드만 쉼표로 구분해서 출력해주세요:'''
            )

            # 질문 전처리 실행
            formatted_question_prompt = question_process_prompt.format(user_question=user_question)
            question_response = self.llm.invoke(formatted_question_prompt)
            processed_question = question_response.content.strip()

            # 키워드 추출 실행
            formatted_keyword_prompt = keyword_extract_prompt.format(processed_question=processed_question)
            keyword_response = self.llm.invoke(formatted_keyword_prompt)
            
            parser = CommaSeparatedListOutputParser()
            search_keywords = parser.parse(keyword_response.content)

            logger.info(f"질문 전처리 완료: {len(search_keywords)}개 키워드 추출")
            
            return {
                **state,
                "processed_question": processed_question,
                "search_keywords": search_keywords,
                "current_step": "question_processed"
            }
            
        except Exception as e:
            logger.error(f"질문 전처리 실패: {str(e)}")
            return {
                **state,
                "error_message": f"질문 전처리 실패: {str(e)}",
                "current_step": "process_question_failed"
            }
    
    def search_rag_summaries(self, state: MeetingQAState) -> MeetingQAState:
        """2단계: RAG 서비스에서 관련 요약본 검색"""
        try:
            processed_question = state.get("processed_question", "")
            search_keywords = state.get("search_keywords", [])
            rag_service_url = state.get("rag_service_url", "")
            
            if not all([processed_question, search_keywords, rag_service_url]):
                raise ValueError("필수 매개변수가 누락되었습니다.")
            
            rag_client = RAGClient(rag_service_url)
            relevant_summaries = rag_client.search_summaries(
                query=processed_question,
                keywords=search_keywords,
                top_k=5,
                similarity_threshold=0.7
            )
            
            # meeting_id 추출
            selected_meeting_ids = list(set([
                summary.get("meeting_id") 
                for summary in relevant_summaries 
                if summary.get("meeting_id")
            ]))
            
            logger.info(f"RAG 검색 완료: {len(relevant_summaries)}개 요약본, {len(selected_meeting_ids)}개 회의 ID")
            
            return {
                **state,
                "relevant_summaries": relevant_summaries,
                "selected_meeting_ids": selected_meeting_ids,
                "current_step": "rag_search_completed"
            }
            
        except Exception as e:
            logger.error(f"RAG 검색 실패: {str(e)}")
            return {
                **state,
                "error_message": f"RAG 검색 실패: {str(e)}",
                "current_step": "rag_search_failed"
            }
    
    def fetch_meeting_metadata(self, state: MeetingQAState) -> MeetingQAState:
        """3단계: PostgreSQL에서 회의 메타데이터 조회"""
        try:
            selected_meeting_ids = state.get("selected_meeting_ids", [])
            postgresql_config = state.get("postgresql_config", {})
            
            if not selected_meeting_ids:
                raise ValueError("선택된 회의 ID가 없습니다.")
            
            if not postgresql_config:
                raise ValueError("PostgreSQL 설정이 없습니다.")
            
            with PostgreSQLClient(postgresql_config) as pg_client:
                meeting_metadata = pg_client.fetch_meeting_metadata(selected_meeting_ids)
            
            logger.info(f"메타데이터 조회 완료: {len(meeting_metadata)}개 회의 정보")
            
            return {
                **state,
                "meeting_metadata": meeting_metadata,
                "current_step": "metadata_fetched"
            }
            
        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {str(e)}")
            return {
                **state,
                "error_message": f"메타데이터 조회 실패: {str(e)}",
                "current_step": "fetch_metadata_failed"
            }
    
    def fetch_original_scripts(self, state: MeetingQAState) -> MeetingQAState:
        """4단계: Azure Blob Storage에서 원본 스크립트 다운로드"""
        try:
            meeting_metadata = state.get("meeting_metadata", [])
            
            if not meeting_metadata:
                raise ValueError("회의 메타데이터가 없습니다.")
            
            original_scripts = self.blob_client.download_text_files(meeting_metadata)
            
            logger.info(f"원본 스크립트 다운로드 완료: {len(original_scripts)}개 파일")
            
            return {
                **state,
                "original_scripts": original_scripts,
                "current_step": "scripts_fetched"
            }
            
        except Exception as e:
            logger.error(f"원본 스크립트 다운로드 실패: {str(e)}")
            return {
                **state,
                "error_message": f"원본 스크립트 다운로드 실패: {str(e)}",
                "current_step": "fetch_scripts_failed"
            }
    
    def process_original_scripts(self, state: MeetingQAState) -> MeetingQAState:
        """5단계: 원본 스크립트 청킹 및 임베딩"""
        try:
            original_scripts = state.get("original_scripts", [])
            
            if not original_scripts:
                raise ValueError("원본 스크립트가 없습니다.")
            
            all_chunked_scripts = []
            
            for script in original_scripts:
                meeting_id = script["meeting_id"]
                full_content = script["full_content"]
                
                # 텍스트 정리
                cleaned_content = clean_text(full_content)
                
                # 청킹
                chunks = chunk_text(
                    cleaned_content, 
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP
                )
                
                # 임베딩 추가
                chunks_with_embeddings = self.embedding_manager.add_embeddings_to_chunks(
                    chunks, meeting_id
                )
                
                all_chunked_scripts.extend(chunks_with_embeddings)
            
            logger.info(f"스크립트 처리 완료: {len(all_chunked_scripts)}개 청크 생성")
            
            return {
                **state,
                "chunked_scripts": all_chunked_scripts,
                "current_step": "scripts_processed"
            }
            
        except Exception as e:
            logger.error(f"스크립트 처리 실패: {str(e)}")
            return {
                **state,
                "error_message": f"스크립트 처리 실패: {str(e)}",
                "current_step": "process_scripts_failed"
            }
    
    def select_relevant_chunks(self, state: MeetingQAState) -> MeetingQAState:
        """6단계: 질문과 관련된 청크 선별"""
        try:
            processed_question = state.get("processed_question", "")
            chunked_scripts = state.get("chunked_scripts", [])
            
            if not processed_question or not chunked_scripts:
                raise ValueError("필수 데이터가 누락되었습니다.")
            
            # 질문 임베딩 생성
            query_embedding = self.embedding_manager.embed_query(processed_question)
            
            # 관련 청크 선별
            relevant_chunks = find_most_relevant_chunks(
                query_embedding=query_embedding,
                chunks=chunked_scripts,
                top_k=10,
                similarity_threshold=0.6
            )
            
            logger.info(f"관련 청크 선별 완료: {len(relevant_chunks)}개 청크")
            
            return {
                **state,
                "relevant_chunks": relevant_chunks,
                "current_step": "chunks_selected"
            }
            
        except Exception as e:
            logger.error(f"청크 선별 실패: {str(e)}")
            return {
                **state,
                "error_message": f"청크 선별 실패: {str(e)}",
                "current_step": "select_chunks_failed"
            }
    
    def generate_final_answer(self, state: MeetingQAState) -> MeetingQAState:
        """7단계: 최종 답변 생성"""
        try:
            user_question = state.get("user_question", "")
            relevant_summaries = state.get("relevant_summaries", [])
            relevant_chunks = state.get("relevant_chunks", [])
            
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
            
            # 답변 생성 프롬프트
            answer_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 질의응답 전문 AI입니다. 
                주어진 회의록 내용을 바탕으로 사용자의 질문에 정확하고 구체적으로 답변해주세요.
                
                질문: {question}
                
                관련 회의록 내용:
                {context}
                
                답변 시 주의사항:
                1. 제공된 회의록 내용만을 근거로 답변하세요
                2. 추측이나 가정은 하지 마세요
                3. 관련 정보가 없다면 "제공된 회의록에서 관련 정보를 찾을 수 없습니다"라고 답변하세요
                4. 구체적인 근거와 함께 답변하세요
                
                답변:'''
            )
            
            formatted_prompt = answer_prompt.format(
                question=user_question,
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
                        meeting_metadata = script.get("metadata", {})
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
