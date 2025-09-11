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
        """서버 응답을 {script_id: {"embedding": [...]}} 형태로 정규화
        
        지원 형태:
        1. {"all_summaries": {...}} 또는 {"selected_summary": {...}} - 래핑된 응답
        2. {"script_id": {"embedding": [...]}} - 기존 dict 매핑
        3. {"script_id": [...]} - 직접 임베딩 배열
        4. [{"scriptId": "...", "embedding": [...]}] - 배열 형태
        5. {"scriptId": "...", "embedding": [...]} - 단일 객체 (신규)
        """
        normalized: Dict[str, Dict[str, List[float]]] = {}

        try:
            logger.info(f"🔧 [NORMALIZE] 입력 데이터 타입: {type(data)}")
            
            # 1) 래핑 키 처리
            if isinstance(data, dict):
                if "all_summaries" in data:
                    logger.info("🔧 [NORMALIZE] all_summaries 래핑 해제")
                    data = data.get("all_summaries", {})
                elif "selected_summary" in data:
                    logger.info("🔧 [NORMALIZE] selected_summary 래핑 해제")
                    data = data.get("selected_summary", {})

            # 2) 신규 케이스: 단일 객체 {"scriptId": "...", "embedding": [...]}
            if isinstance(data, dict) and "scriptId" in data and "embedding" in data:
                logger.info("🔧 [NORMALIZE] 단일 객체 형태 감지")
                script_id = data["scriptId"]
                embedding = data["embedding"]
                if isinstance(embedding, list):
                    normalized[str(script_id)] = {"embedding": embedding}
                    logger.info(f"🔧 [NORMALIZE] 단일 객체 처리 완료: {script_id}")
                return normalized

            # 3) 배열 형태: [{"scriptId": "...", "embedding": [...]}]
            if isinstance(data, list):
                logger.info(f"🔧 [NORMALIZE] 배열 형태 감지 (길이: {len(data)})")
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("scriptId") or item.get("script_id") or item.get("id")
                    embedding = item.get("embedding") or item.get("vector")
                    if sid and isinstance(embedding, list):
                        normalized[str(sid)] = {"embedding": embedding}
                        logger.info(f"🔧 [NORMALIZE] 배열[{i}] 처리: {sid}")
                return normalized

            # 4) 기존 dict 매핑 형태: {"script_id": {"embedding": [...]}} 또는 {"script_id": [...]}
            if isinstance(data, dict):
                logger.info(f"🔧 [NORMALIZE] Dict 매핑 형태 감지 (키 개수: {len(data)})")
                for key, value in data.items():
                    # "embedding" 키는 스크립트 ID가 아님 (문제 케이스)
                    if key == "embedding":
                        logger.warning(f"🔧 [NORMALIZE] 'embedding' 키 건너뛰기")
                        continue
                        
                    if isinstance(value, dict) and "embedding" in value:
                        # {"script_id": {"embedding": [...]}} 형태
                        embedding = value.get("embedding")
                        if isinstance(embedding, list):
                            normalized[str(key)] = {"embedding": embedding}
                            logger.info(f"🔧 [NORMALIZE] Dict 중첩 처리: {key}")
                    elif isinstance(value, list):
                        # {"script_id": [...]} 직접 임베딩 형태
                        normalized[str(key)] = {"embedding": value}
                        logger.info(f"🔧 [NORMALIZE] Dict 직접 처리: {key}")
                return normalized

        except Exception as e:
            logger.error(f"🔧 [NORMALIZE] 처리 중 오류: {str(e)}")
            pass

        logger.warning(f"🔧 [NORMALIZE] 처리할 수 없는 데이터 형태: {type(data)}")
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

            
            # === 디버그 로그: RAG 서비스 원본 응답 분석 ===
            logger.info(f"🔍 [DEBUG] RAG 원본 응답 type: {type(result)}")
            logger.info(f"🔍 [DEBUG] RAG 원본 응답 keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
            logger.info(f"🔍 [DEBUG] RAG 원본 응답 (첫 100자): {str(result)[:100]}...")
            
            normalized = self._normalize_summaries(result)
            
            # === 디버그 로그: 정규화 후 결과 분석 ===
            logger.info(f"🔍 [DEBUG] 정규화 후 type: {type(normalized)}")
            logger.info(f"🔍 [DEBUG] 정규화 후 keys: {list(normalized.keys())}")
            for key, value in normalized.items():
                logger.info(f"🔍 [DEBUG] 정규화 결과 - key='{key}', value_type={type(value)}")
                if isinstance(value, dict):
                    logger.info(f"🔍 [DEBUG] 정규화 결과 - key='{key}', value_keys={list(value.keys())}")
            
            logger.info(f"특정 요약본 조회 완료(GET, 다중 필터): {script_ids}")
            return normalized
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"⚠️ 특정 요약본 404 오류: {script_ids} - 빈 결과 반환")
                return {}  # 404 시 빈 딕셔너리 반환 (fallback 가능하게)
            else:
                logger.error(f"특정 요약본 조회 HTTP 오류: {str(e)}")
                raise Exception(f"특정 요약본 조회 실패: {str(e)}")
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
