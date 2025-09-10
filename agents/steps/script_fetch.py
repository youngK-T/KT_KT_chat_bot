"""
4단계: 원본 스크립트 조회 로직
"""

import logging
import httpx
from typing import Dict, List
from config.settings import MEETING_API_URL
from models.state import MeetingQAState

logger = logging.getLogger(__name__)

class ScriptFetcher:
    """원본 스크립트 조회 처리 클래스"""
    
    def __init__(self):
        self.meeting_api_url = MEETING_API_URL
    
    def fetch_original_scripts(self, state: MeetingQAState) -> MeetingQAState:
        """4단계: 외부 회의록 API에서 원본 스크립트 직접 조회

        사양:
        - 전체:  GET /api/scripts                      → 배열
        - 단일:  GET /api/scripts?scriptIds=abc123     → 객체
        - 다중:  GET /api/scripts?scriptIds=a,b,c      → 배열
        응답 필드: { "scriptId", "storageUrl", "scriptText" }
        """
        try:
            selected_script_ids = state.get("selected_script_ids", [])
            if not selected_script_ids:
                raise ValueError("selected_script_ids가 없습니다.")

            with httpx.Client(timeout=30) as client:
                params = {"scriptIds": ",".join(selected_script_ids)}
                response = client.get(f"{self.meeting_api_url}/api/scripts", params=params)
                if response.status_code != 200:
                    raise Exception(f"API 호출 실패: {response.status_code}")
                result = response.json()

            items = result if isinstance(result, list) else [result]
            original_scripts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                script_id = item.get("scriptId") or item.get("id") or item.get("meeting_id")
                if not script_id:
                    continue

                # 1) 기본: scriptText 사용
                script_text = item.get("scriptText")

                # 2) 대안: segments 배열 → speaker: text 로 합쳐서 원문 구성
                if not script_text:
                    segments = item.get("segments")
                    if isinstance(segments, list):
                        lines = []
                        for seg in segments:
                            try:
                                speaker = (seg.get("speaker") or "").strip()
                                text = (seg.get("text") or "").strip()
                                if not text:
                                    continue
                                line = f"{speaker}: {text}" if speaker else text
                                lines.append(line)
                            except Exception:
                                continue
                        script_text = "\n".join(lines)

                script_text = script_text or ""

                original_scripts.append({
                    "meeting_id": script_id,
                    "content": script_text,
                    "filename": f"meeting_{script_id}.txt"
                })
            
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
    
    # 개별 by_id 메서드는 다중 GET로 대체되므로 제거 (필요 시 복구)
