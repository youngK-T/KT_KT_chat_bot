import requests
import json
from typing import List, Dict, Any
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
    
    def _normalize_summaries(self, data: Any) -> Dict[str, Dict[str, List[float]]]:
        """서버 응답을 {script_id: {"embedding": [...]}} 형태로 정규화"""
        normalized: Dict[str, Dict[str, List[float]]] = {}

        try:
            # 1) 래핑 키 처리
            if isinstance(data, dict):
                if "all_summaries" in data:
                    data = data.get("all_summaries", {})
                elif "selected_summary" in data:
                    data = data.get("selected_summary", {})

            # 2) 이미 dict 매핑 형태인 경우
            if isinstance(data, dict):
                # {"ID": {"embedding": [...]}} 또는 {"ID": [...]} 양식 모두 지원
                for key, value in data.items():
                    if isinstance(value, dict) and "embedding" in value:
                        embedding = value.get("embedding")
                    else:
                        embedding = value
                    if isinstance(embedding, list):
                        normalized[str(key)] = {"embedding": embedding}
                return normalized

            # 3) 리스트 형태인 경우: [{"scriptId": "...", "embedding": [...]}]
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("scriptId") or item.get("script_id") or item.get("id")
                    embedding = item.get("embedding") or item.get("vector")
                    if sid and isinstance(embedding, list):
                        normalized[str(sid)] = {"embedding": embedding}
                return normalized

        except Exception:
            pass

        return normalized

    def get_all_summaries(self) -> Dict[str, Dict[str, List[float]]]:
        """전체 요약본 임베딩 조회 (GET /api/rag/script-summaries)"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/rag/script-summaries",
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info("전체 요약본 조회 완료(GET)")
            return self._normalize_summaries(result)
        except requests.exceptions.RequestException as e:
            logger.error(f"전체 요약본 조회 실패: {str(e)}")
            raise Exception(f"전체 요약본 조회 실패: {str(e)}")
        except Exception as e:
            logger.error(f"전체 요약본 조회 중 오류: {str(e)}")
            raise Exception(f"전체 요약본 조회 중 오류: {str(e)}")

    def get_summary_by_ids(self, script_ids: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """특정 script_id들의 요약본 임베딩 조회 (GET, 쉼표 구분 다중 필터)"""
        try:
            if not script_ids:
                return {}

            params = {
                "scriptIds": ",".join(script_ids)
            }
            response = self.session.get(
                f"{self.base_url}/api/rag/script-summaries",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"특정 요약본 조회 완료(GET, 다중 필터): {script_ids}")
            return self._normalize_summaries(result)
        except requests.exceptions.RequestException as e:
            logger.error(f"특정 요약본 조회 실패: {str(e)}")
            raise Exception(f"특정 요약본 조회 실패: {str(e)}")
    
    def health_check(self) -> bool:
        """RAG 서비스 헬스체크"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
