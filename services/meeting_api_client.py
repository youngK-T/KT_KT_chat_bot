import requests
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class MeetingAPIClient:
    """회의록 API 서비스 클라이언트"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def fetch_meeting_metadata(self, meeting_ids: List[str]) -> List[Dict]:
        """회의 ID 목록으로 메타데이터 조회"""
        try:
            logger.info(f"회의 메타데이터 조회: {len(meeting_ids)}개 회의 ID")
            
            # 전체 스크립트 조회
            response = self.session.get(
                f"{self.base_url}/api/scripts",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            all_scripts = response.json()
            
            # 요청된 meeting_ids와 일치하는 스크립트들만 필터링
            filtered_metadata = []
            
            for script in all_scripts:
                script_id = script.get("id") or script.get("meeting_id")
                if script_id in meeting_ids:
                    # API 응답을 기존 형식에 맞게 변환
                    metadata = {
                        "meeting_id": script_id,
                        "meeting_title": script.get("title", ""),
                        "meeting_date": script.get("date", ""),
                        "file_size": len(script.get("content", "").encode('utf-8')),
                        "created_at": script.get("created_at", ""),
                        "updated_at": script.get("updated_at", ""),
                        # API에서 직접 내용을 제공하므로 URL 정보는 참조용
                        "api_url": f"{self.base_url}/api/scripts/{script_id}",
                        "script_data": script  # 전체 스크립트 데이터 저장
                    }
                    filtered_metadata.append(metadata)
            
            logger.info(f"메타데이터 조회 완료: {len(filtered_metadata)}개 회의 정보")
            return filtered_metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"회의 API 호출 실패: {str(e)}")
            raise Exception(f"회의 API 호출 실패: {str(e)}")
        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {str(e)}")
            raise Exception(f"메타데이터 조회 실패: {str(e)}")
    
    def download_text_files(self, meeting_metadata: List[Dict]) -> List[Dict]:
        """메타데이터를 사용해 회의록 텍스트 파일들 다운로드"""
        original_scripts = []
        
        for metadata in meeting_metadata:
            try:
                meeting_id = metadata.get("meeting_id")
                
                if not meeting_id:
                    logger.warning(f"회의 메타데이터에 meeting_id가 없음")
                    continue
                
                # 이미 메타데이터에 스크립트 데이터가 있는 경우 사용
                if "script_data" in metadata:
                    script_data = metadata["script_data"]
                    content = script_data.get("content", "")
                else:
                    # 개별 스크립트 조회 API 호출
                    logger.info(f"스크립트 다운로드 시작: {meeting_id}")
                    
                    response = self.session.get(
                        f"{self.base_url}/api/scripts/{meeting_id}",
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    script_data = response.json()
                    content = script_data.get("content", "")
                
                original_script = {
                    "meeting_id": meeting_id,
                    "full_content": content,
                    "api_url": f"{self.base_url}/api/scripts/{meeting_id}",
                    "file_size": len(content.encode('utf-8')),
                    "metadata": metadata,
                    "script_data": script_data
                }
                
                original_scripts.append(original_script)
                
                logger.info(f"스크립트 다운로드 완료: {meeting_id}, 크기: {original_script['file_size']} bytes")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"회의 {meeting_id} 스크립트 API 호출 실패: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"회의 {meeting_id} 스크립트 다운로드 실패: {str(e)}")
                continue
        
        logger.info(f"총 {len(original_scripts)}개 원본 스크립트 다운로드 완료")
        return original_scripts
    
    def download_single_file(self, meeting_id: str) -> str:
        """단일 파일 다운로드"""
        try:
            logger.info(f"단일 스크립트 다운로드: {meeting_id}")
            
            response = self.session.get(
                f"{self.base_url}/api/scripts/{meeting_id}",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            script_data = response.json()
            content = script_data.get("content", "")
            
            logger.info(f"단일 스크립트 다운로드 완료: {meeting_id}, 크기: {len(content.encode('utf-8'))} bytes")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"스크립트 API 호출 실패: {str(e)}")
            raise Exception(f"스크립트 API 호출 실패: {str(e)}")
        except Exception as e:
            logger.error(f"스크립트 다운로드 실패: {str(e)}")
            raise Exception(f"스크립트 다운로드 실패: {str(e)}")
    
    def health_check(self) -> bool:
        """회의록 API 헬스체크"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/scripts",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def get_all_scripts(self) -> List[Dict]:
        """모든 스크립트 조회"""
        try:
            logger.info("전체 스크립트 목록 조회")
            
            response = self.session.get(
                f"{self.base_url}/api/scripts",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            scripts = response.json()
            
            logger.info(f"전체 스크립트 조회 완료: {len(scripts)}개")
            return scripts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"전체 스크립트 API 호출 실패: {str(e)}")
            raise Exception(f"전체 스크립트 API 호출 실패: {str(e)}")
        except Exception as e:
            logger.error(f"전체 스크립트 조회 실패: {str(e)}")
            raise Exception(f"전체 스크립트 조회 실패: {str(e)}")
