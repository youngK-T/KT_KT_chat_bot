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
    
    def _deduplicate_summaries(self, summaries: List[Dict]) -> List[Dict]:
        """script_id 기준 중복 제거 (최고 점수만 유지)"""
        seen = {}
        for summary in summaries:
            script_id = summary["script_id"]
            if script_id not in seen or summary["relevance_score"] > seen[script_id]["relevance_score"]:
                seen[script_id] = summary
        return list(seen.values())
    
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
            
            # script_id 기준 중복 제거 (최고 점수만 유지)
            relevant_summaries = self._deduplicate_summaries(relevant_summaries)
            
            # 유사도 순으로 정렬
            relevant_summaries.sort(key=lambda x: x["relevance_score"], reverse=True)
            relevant_summaries = relevant_summaries[:5]  # 상위 5개
            
            # script_id 추출
            selected_script_ids = [summary["script_id"] for summary in relevant_summaries]
            
            logger.info(f"RAG 검색 완료 (중복 제거 적용): {len(relevant_summaries)}개 요약본, {len(selected_script_ids)}개 회의 ID")
            
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
            
            # 404 오류로 빈 결과가 반환된 경우 예외처리
            if not selected_summaries:
                logger.warning(f"⚠️ 선택된 스크립트를 찾을 수 없음: {user_selected_script_ids}")
                # 요청된 문서 ID 목록 생성 (가독성을 위해 짧게 자르기)
                display_ids = []
                for script_id in user_selected_script_ids:
                    if len(script_id) > 8:  # UUID인 경우 앞 8자리만
                        display_ids.append(f"{script_id[:8]}...")
                    else:
                        display_ids.append(script_id)
                
                ids_text = ", ".join(display_ids)
                
                return {
                    **state,
                    "final_answer": f"요청하신 문서 [{ids_text}]를 찾을 수 없습니다. 다른 문서를 선택하거나 전체 검색을 이용해 주세요.",
                    "sources": [],
                    "used_script_ids": [],
                    "confidence_score": 0.0,
                    "relevant_summaries": [],
                    "selected_script_ids": [],
                    "current_step": "document_not_found"
                }
            # selected_summaries 구조: Dict[str, Dict[str, List[float]]]

            processed_question = state.get("processed_question", "")
            
            # 질문 임베딩 생성
            from utils.embeddings import EmbeddingManager, cosine_similarity
            embedding_manager = EmbeddingManager()
            query_embedding = embedding_manager.embed_query(processed_question)
            
            # 선택된 스크립트들의 요약본 조회 및 유사도 검색
            relevant_summaries = []
            
            # === 디버그 로그: RAG 응답 구조 분석 ===
            logger.info(f"🔍 [DEBUG] user_selected_script_ids: {user_selected_script_ids}")
            logger.info(f"🔍 [DEBUG] selected_summaries type: {type(selected_summaries)}")
            logger.info(f"🔍 [DEBUG] selected_summaries keys: {list(selected_summaries.keys())}")
            
            for key, value in selected_summaries.items():
                logger.info(f"🔍 [DEBUG] key='{key}', value_type={type(value)}")
                if isinstance(value, dict):
                    logger.info(f"🔍 [DEBUG] key='{key}', value_keys={list(value.keys())}")
                elif isinstance(value, list):
                    logger.info(f"🔍 [DEBUG] key='{key}', value_length={len(value)}")
                else:
                    logger.info(f"🔍 [DEBUG] key='{key}', value={str(value)[:100]}...")
            
            for script_id, summary_data in selected_summaries.items():
                try:
                    # 추가 방어 로직: UUID 패턴 검증
                    import re
                    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
                    if not uuid_pattern.match(script_id):
                        logger.info(f"🚫 [DEBUG] 유효하지 않은 script_id 형태 건너뛰기: {script_id}")
                        continue
                        
                    if summary_data and "embedding" in summary_data:
                        embedding = summary_data["embedding"]
                        
                        # 코사인 유사도 계산
                        similarity = cosine_similarity(query_embedding, embedding)
                        
                        if similarity > 0.7:  # 유사도 임계값
                            relevant_summaries.append({
                                "script_id": script_id,
                                "relevance_score": similarity
                            })
                            logger.info(f"✅ [DEBUG] 스크립트 추가: {script_id} (유사도: {similarity:.3f})")
                        else:
                            logger.info(f"❌ [DEBUG] 유사도 부족: {script_id} (유사도: {similarity:.3f})")
                    else:
                        logger.warning(f"⚠️ [DEBUG] 임베딩 없음: {script_id}, summary_data={summary_data}")
                        
                except Exception as e:
                    logger.warning(f"💥 [DEBUG] 처리 실패: {script_id}, 오류={str(e)}")
                    continue
            
            # script_id 기준 중복 제거 (최고 점수만 유지)
            relevant_summaries = self._deduplicate_summaries(relevant_summaries)
            
            # 유사도 순으로 정렬
            relevant_summaries.sort(key=lambda x: x["relevance_score"], reverse=True)
            relevant_summaries = relevant_summaries[:5]  # 상위 5개만 선택
            
            selected_script_ids = [summary["script_id"] for summary in relevant_summaries]
            
            logger.info(f"특정 스크립트 유사도 검색 완료 (중복 제거 적용): {len(relevant_summaries)}개 요약본, {len(selected_script_ids)}개 스크립트 ID")
            
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
