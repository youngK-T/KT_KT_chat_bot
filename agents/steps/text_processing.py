"""
5단계: 텍스트 처리 로직
"""

import logging
from typing import Dict, List
from utils.text_processing import chunk_text, clean_text
from utils.embeddings import EmbeddingManager, find_most_relevant_chunks
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from models.state import MeetingQAState

logger = logging.getLogger(__name__)

class TextProcessor:
    """텍스트 처리 클래스"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    
    def process_original_scripts(self, state: MeetingQAState) -> MeetingQAState:
        """5단계: 원본 스크립트 청킹 및 임베딩"""
        try:
            original_scripts = state.get("original_scripts", [])
            
            if not original_scripts:
                raise ValueError("원본 스크립트가 없습니다.")
            
            all_chunked_scripts = []
            processed_script_ids = set()  # 중복 처리 방지
            
            for script in original_scripts:
                script_id = script["script_id"]
                
                # 이미 처리된 스크립트 건너뛰기
                if script_id in processed_script_ids:
                    logger.debug(f"이미 처리된 스크립트 건너뛰기: {script_id}")
                    continue
                processed_script_ids.add(script_id)
                
                full_content = script["content"]
                
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
                    chunks, script_id
                )
                
                all_chunked_scripts.extend(chunks_with_embeddings)
            
            logger.info(f"스크립트 처리 완료: {len(all_chunked_scripts)}개 청크 생성")
            
            
            return {
                **state,  # 이미 script_metadata가 포함되어 있어야 함
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
            # 질문이 비어 있으면 user_question으로 폴백
            processed_question = (state.get("processed_question") or state.get("user_question") or "").strip()
            # 청크 리스트 None 방지
            chunked_scripts = state.get("chunked_scripts") or []
            
            if not processed_question:
                raise ValueError("질문 텍스트가 없습니다.")
            if not chunked_scripts:
                raise ValueError("처리된 원본 청크가 없습니다.")
            
            # 질문 임베딩 생성
            query_embedding = self.embedding_manager.embed_query(processed_question)
            
            # 관련 청크 선별
            relevant_chunks = find_most_relevant_chunks(
                query_embedding=query_embedding,
                chunks=chunked_scripts,
                top_k=10,
                similarity_threshold=0.4  # 0.6에서 0.4로 낮춤
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
    
    def process_with_rag_embeddings(self, state: MeetingQAState) -> MeetingQAState:
        """RAG 임베딩을 사용한 텍스트 처리 (새로운 분기용)"""
        try:
            processed_question = state.get("processed_question", "")
            all_summaries = state.get("all_summaries", {})
            
            if not processed_question or not all_summaries:
                raise ValueError("필수 데이터가 누락되었습니다.")
            
            # 질문 임베딩 생성
            query_embedding = self.embedding_manager.embed_query(processed_question)
            
            # RAG 요약본들과 유사도 계산
            relevant_summaries = []
            for script_id, summary_data in all_summaries.items():
                summary_embedding = summary_data.get("embedding", [])
                if summary_embedding:
                    # 코사인 유사도 계산
                    from utils.embeddings import cosine_similarity
                    similarity = cosine_similarity(query_embedding, summary_embedding)
                    
                    if similarity >= 0.6:  # 임계값
                        relevant_summaries.append({
                            "script_id": script_id,
                            "summary_text": summary_data.get("summary_text", ""),
                            "relevance_score": similarity,
                            "meeting_date": summary_data.get("meeting_date", "")
                        })
            
            # 유사도 순으로 정렬
            relevant_summaries.sort(key=lambda x: x["relevance_score"], reverse=True)
            relevant_summaries = relevant_summaries[:5]  # 상위 5개
            
            logger.info(f"RAG 임베딩 기반 검색 완료: {len(relevant_summaries)}개 요약본")
            
            
            return {
                **state,  # script_metadata 포함되어 있어야 함
                "relevant_summaries": relevant_summaries,
                "current_step": "rag_embeddings_processed"
            }
            
        except Exception as e:
            logger.error(f"RAG 임베딩 처리 실패: {str(e)}")
            return {
                **state,
                "error_message": f"RAG 임베딩 처리 실패: {str(e)}",
                "current_step": "rag_embeddings_failed"
            }
