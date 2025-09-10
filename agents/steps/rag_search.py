"""
2단계: RAG 검색 로직
"""

import logging
from typing import Dict, List
from services.rag_client import RAGClient
from config.settings import RAG_SERVICE_URL
from models.state import MeetingQAState

logger = logging.getLogger(__name__)

class RAGSearchProcessor:
    """RAG 검색 처리 클래스"""
    
    def __init__(self):
        self.rag_client = RAGClient(RAG_SERVICE_URL)
    
    def get_all_rag_summaries(self, state: MeetingQAState) -> MeetingQAState:
        """2단계: RAG 서비스에서 전체 요약본 호출"""
        try:
            processed_question = state.get("processed_question", "")
            
            # 전체 요약본 가져오기
            all_summaries = self.rag_client.get_all_summaries()
            # all_summaries 구조: Dict[str, Dict[str, List[float]]]
            
            # 질문 임베딩 생성 (EmbeddingManager 필요)
            from utils.embeddings import EmbeddingManager
            embedding_manager = EmbeddingManager()
            query_embedding = embedding_manager.embed_query(processed_question)
            
            # 유사도 계산 및 선별
            relevant_summaries = []
            for script_id, summary_data in all_summaries.items():
                embedding = summary_data.get("embedding", [])
                if embedding:
                    # 코사인 유사도 계산
                    from utils.embeddings import cosine_similarity
                    similarity = cosine_similarity(query_embedding, embedding)
                    
                    if similarity > 0.7:  # 유사도 임계값
                        relevant_summaries.append({
                            "script_id": script_id,
                            "relevance_score": similarity
                        })
            
            # 유사도 순으로 정렬
            relevant_summaries.sort(key=lambda x: x["relevance_score"], reverse=True)
            relevant_summaries = relevant_summaries[:5]  # 상위 5개
            
            # script_id 추출
            selected_script_ids = [summary["script_id"] for summary in relevant_summaries]
            
            logger.info(f"RAG 검색 완료: {len(relevant_summaries)}개 요약본, {len(selected_script_ids)}개 회의 ID")
            
            return {
                **state,
                "relevant_summaries": relevant_summaries,
                "selected_script_ids": selected_script_ids,
                "current_step": "rag_search_completed"
            }
            
        except Exception as e:
            logger.error(f"RAG 검색 실패: {str(e)}")
            return {
                **state,
                "error_message": f"RAG 검색 실패: {str(e)}",
                "current_step": "rag_search_failed"
            }
    
    
    def get_summary_by_id(self, state: MeetingQAState) -> MeetingQAState:
        """특정 script_id들의 요약본 조회 및 유사도 검색 (상세 챗봇용)"""
        try:
            user_selected_script_ids = state.get("user_selected_script_ids", [])
            # user_selected_script_ids (사용자가 선택한 스크립트들) : List[str]

            if not user_selected_script_ids:
                raise ValueError("user_selected_script_ids가 없습니다.")

            # 선택된 요약본 가져오기
            selected_summaries = self.rag_client.get_summary_by_ids(user_selected_script_ids)
            # selected_summaries 구조: Dict[str, Dict[str, List[float]]]

            processed_question = state.get("processed_question", "")
            
            # 질문 임베딩 생성
            from utils.embeddings import EmbeddingManager, cosine_similarity
            embedding_manager = EmbeddingManager()
            query_embedding = embedding_manager.embed_query(processed_question)
            
            # 선택된 스크립트들의 요약본 조회 및 유사도 검색
            relevant_summaries = []
            for script_id, summary_data in selected_summaries.items():
                try:                
                    if summary_data and "embedding" in summary_data:
                        embedding = summary_data["embedding"]
                        
                        # 코사인 유사도 계산
                        similarity = cosine_similarity(query_embedding, embedding)
                        
                        if similarity > 0.7:  # 유사도 임계값
                            relevant_summaries.append({
                                "script_id": script_id,
                                "relevance_score": similarity
                            })
                    else:
                        logger.warning(f"스크립트 {script_id}의 임베딩 데이터가 없습니다.")
                        
                except Exception as e:
                    logger.warning(f"스크립트 {script_id} 조회 실패: {str(e)}")
                    continue
            
            # 유사도 순으로 정렬
            relevant_summaries.sort(key=lambda x: x["relevance_score"], reverse=True)
            relevant_summaries = relevant_summaries[:5]  # 상위 5개만 선택
            
            selected_script_ids = [summary["script_id"] for summary in relevant_summaries]
            
            logger.info(f"특정 스크립트 유사도 검색 완료: {len(relevant_summaries)}개 요약본, {len(selected_script_ids)}개 스크립트 ID")
            
            return {
                **state,
                "relevant_summaries": relevant_summaries,
                "selected_script_ids": selected_script_ids,
                "current_step": "specific_rag_search_completed"
            }
            
        except Exception as e:
            logger.error(f"특정 스크립트 유사도 검색 실패: {str(e)}")
            return {
                **state,
                "error_message": f"특정 스크립트 유사도 검색 실패: {str(e)}",
                "current_step": "specific_rag_search_failed"
            }
