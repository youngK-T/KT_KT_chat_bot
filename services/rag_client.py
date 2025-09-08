import requests
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class RAGClient:
    """RAG 서비스 클라이언트"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    async def search_summaries(
        self, 
        query: str, 
        keywords: List[str], 
        top_k: int = 5, 
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """RAG 서비스에서 관련 요약본들을 검색"""
        try:
            payload = {
                "query": query,
                "keywords": keywords,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
            
            logger.info(f"RAG 서비스 호출: {self.base_url}/search")
            
            response = self.session.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"RAG 검색 결과: {len(result.get('results', []))}개 요약본 발견")
            
            return result.get("results", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG 서비스 호출 실패: {str(e)}")
            raise Exception(f"RAG 서비스 호출 실패: {str(e)}")
        except Exception as e:
            logger.error(f"RAG 검색 중 오류: {str(e)}")
            raise Exception(f"RAG 검색 중 오류: {str(e)}")
    
    def health_check(self) -> bool:
        """RAG 서비스 헬스체크"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
