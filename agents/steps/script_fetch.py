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
        - 단일:  GET /api/scripts?ids=abc123           → 객체
        - 다중:  GET /api/scripts?ids=a,b,c            → 배열 (요청 순서 보장)
        응답 필드: { "scriptId", "title", "timestamp", "segments" | "scriptText" }
        """
        try:
            selected_script_ids = state.get("selected_script_ids", [])
            
            if not selected_script_ids:
                raise ValueError("selected_script_ids가 없습니다.")

            # API 호출
            params = {"ids": ",".join(selected_script_ids)}
            api_url = f"{self.meeting_api_url}/api/scripts"

            with httpx.Client(timeout=30) as client:
                response = client.get(api_url, params=params)
                if response.status_code != 200:
                    raise Exception(f"API 호출 실패: {response.status_code}")
                result = response.json()

            # 배열이 아닐 수 있어 보정
            items = result if isinstance(result, list) else [result]

            # 요청 순서 보장: 응답이 순서를 보장한다고 했지만 안전하게 재정렬
            by_id = {}
            for it in items:
                if isinstance(it, dict):
                    sid = it.get("scriptId") or it.get("id") or it.get("meeting_id")
                    if sid:
                        by_id[str(sid)] = it
            items = [by_id[sid] for sid in selected_script_ids if sid in by_id]
            original_scripts = []
            seen_script_ids = set()  # 중복 방지용 집합
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                script_id = item.get("scriptId") or item.get("id") or item.get("meeting_id")
                if not script_id:
                    continue
                
                # 중복 스크립트 ID 건너뛰기
                if script_id in seen_script_ids:
                    logger.debug(f"중복된 스크립트 ID 건너뛰기: {script_id}")
                    continue
                seen_script_ids.add(script_id)

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
                
                # 제목과 타임스탬프 추출 (디버깅 로그 추가)
                title = item.get("title", "")
                timestamp = item.get("timestamp", "")
                
                
                original_scripts.append({
                    "script_id": script_id,
                    "content": script_text,
                    "title": title,
                    "timestamp": timestamp,  # 추가 (날짜 정보)
                    "filename": f"meeting_{script_id}.txt"
                })
            
            # 제목 매핑 딕셔너리 생성 (프론트엔드 출처 표시용)
            script_metadata = {}
            for script in original_scripts:
                script_metadata[script["script_id"]] = {
                    "title": script["title"],
                    "timestamp": script["timestamp"]
                }
            
            logger.info(f"원본 스크립트 다운로드 완료: {len(original_scripts)}개 파일")
            
            return {
                **state,
                "original_scripts": original_scripts,
                "script_metadata": script_metadata,  # 제목 매핑 딕셔너리 추가
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
