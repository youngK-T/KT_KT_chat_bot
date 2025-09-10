from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict
import numpy as np
import logging
from config.settings import AZURE_OPENAI_CONFIG, AZURE_OPENAI_EMBEDDING_DEPLOYMENT

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """임베딩 관리 클래스"""
    
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_CONFIG["api_version"],
            azure_endpoint=AZURE_OPENAI_CONFIG["endpoint"],
            api_key=AZURE_OPENAI_CONFIG["api_key"]
        )
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환"""
        try:
            if not texts:
                return []
            
            logger.info(f"임베딩 생성 시작: {len(texts)}개 텍스트")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"임베딩 생성 완료: {len(embeddings)}개 벡터")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            raise Exception(f"임베딩 생성 실패: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        try:
            if not query:
                return []
            
            logger.info("쿼리 임베딩 생성 시작")
            embedding = self.embeddings.embed_query(query)
            logger.info("쿼리 임베딩 생성 완료")
            
            return embedding
            
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 실패: {str(e)}")
            raise Exception(f"쿼리 임베딩 생성 실패: {str(e)}")
    
    def add_embeddings_to_chunks(self, chunks: List[Dict], script_id: str) -> List[Dict]:
        """청크 리스트에 임베딩 추가"""
        try:
            if not chunks:
                return []
            
            # 청크 텍스트들 추출
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            
            # 임베딩 생성
            embeddings = self.embed_texts(chunk_texts)
            
            # 청크에 임베딩 추가
            for i, chunk in enumerate(chunks):
                chunk["chunk_embedding"] = embeddings[i]
                chunk["script_id"] = script_id
            
            return chunks
            
        except Exception as e:
            logger.error(f"청크 임베딩 추가 실패: {str(e)}")
            raise Exception(f"청크 임베딩 추가 실패: {str(e)}")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    if not vec1 or not vec2:
        return 0.0
    
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 영벡터 체크
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity)
        
    except Exception as e:
        logger.error(f"코사인 유사도 계산 실패: {str(e)}")
        return 0.0

def find_most_relevant_chunks(
    query_embedding: List[float], 
    chunks: List[Dict], 
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """쿼리와 가장 관련성 높은 청크들 찾기"""
    if not query_embedding or not chunks:
        return []
    
    relevant_chunks = []
    
    for chunk in chunks:
        chunk_embedding = chunk.get("chunk_embedding", [])
        if not chunk_embedding:
            continue
        
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        
        if similarity >= similarity_threshold:
            chunk_with_score = chunk.copy()
            chunk_with_score["relevance_score"] = similarity
            relevant_chunks.append(chunk_with_score)
    
    # 유사도 순으로 정렬
    relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # 상위 k개 반환
    return relevant_chunks[:top_k]
