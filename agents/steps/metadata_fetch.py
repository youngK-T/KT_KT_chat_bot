"""
3단계: 메타데이터 조회 로직
"""

import logging
import httpx
from typing import Dict, List
from config.settings import MEETING_API_URL
from models.state import MeetingQAState

logger = logging.getLogger(__name__)

class MetadataFetcher:
    """메타데이터 조회 처리 클래스"""
    
    def __init__(self):
        self.meeting_api_url = MEETING_API_URL
    
    def fetch_meeting_metadata(self, state: MeetingQAState) -> MeetingQAState:
        """3단계: 외부 회의록 API에서 원본 스크립트(및 메타) 조회

        사양:
        - 전체:  GET /api/scripts                      → 배열
        - 단일:  GET /api/scripts?scriptIds=abc123     → 객체
        - 다중:  GET /api/scripts?scriptIds=a,b,c      → 배열
        응답 필드: { "scriptId", "storageUrl", "scriptText" }
        """
        try:
            selected_script_ids = state.get("selected_script_ids", [])
            
            if not selected_script_ids:
                raise ValueError("선택된 회의 ID가 없습니다.")
            
            # 외부 API 호출: GET /api/scripts?scriptIds=a,b,c (쉼표 구분 다중 필터)
            with httpx.Client(timeout=30) as client:
                params = {"scriptIds": ",".join(selected_script_ids)}
                response = client.get(f"{self.meeting_api_url}/api/scripts", params=params)
                
                if response.status_code != 200:
                    raise Exception(f"API 호출 실패: {response.status_code}")
                result = response.json()

            # 단일 조회 시 객체로 올 수 있으므로 리스트로 정규화
            items = result if isinstance(result, list) else [result]

            # meeting_metadata + original_scripts 구성
            meeting_metadata = []
            original_scripts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                script_id = item.get("scriptId") or item.get("id") or item.get("meeting_id")
                storage_url = item.get("storageUrl")
                script_text = item.get("scriptText") or ""
                if not script_id:
                    continue

                meta = {
                    "meeting_id": script_id,
                    "storage_url": storage_url
                }
                meeting_metadata.append(meta)

                original_scripts.append({
                    "meeting_id": script_id,
                    "content": script_text,
                    "filename": f"meeting_{script_id}.txt",
                    "meeting_metadata": meta
                })

            logger.info(f"원본 스크립트 조회 완료: {len(original_scripts)}개")
            
            return {
                **state,
                "meeting_metadata": meeting_metadata,
                "original_scripts": original_scripts,
                "current_step": "metadata_fetched"
            }
            
        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {str(e)}")
            return {
                **state,
                "error_message": f"메타데이터 조회 실패: {str(e)}",
                "current_step": "fetch_metadata_failed"
            }
    
    def fetch_all_metadata(self, state: MeetingQAState) -> MeetingQAState:
        """전체 메타데이터 조회 (새로운 분기용)"""
        try:
            # 외부 API 호출: GET /api/scripts
            with httpx.Client(timeout=30) as client:
                response = client.get(f"{self.meeting_api_url}/api/scripts")
                
                if response.status_code != 200:
                    raise Exception(f"API 호출 실패: {response.status_code}")
                
                all_meetings = response.json()
            
            logger.info(f"전체 메타데이터 조회 완료: {len(all_meetings)}개 회의 정보")
            
            return {
                **state,
                "all_meeting_metadata": all_meetings,
                "current_step": "all_metadata_fetched"
            }
            
        except Exception as e:
            logger.error(f"전체 메타데이터 조회 실패: {str(e)}")
            return {
                **state,
                "error_message": f"전체 메타데이터 조회 실패: {str(e)}",
                "current_step": "fetch_all_metadata_failed"
            }
