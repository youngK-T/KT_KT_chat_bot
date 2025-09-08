from azure.storage.blob import BlobServiceClient
from typing import List, Dict
import logging
import io

logger = logging.getLogger(__name__)

class AzureBlobClient:
    """Azure Blob Storage 클라이언트"""
    
    def __init__(self):
        self.blob_service_client = None
    
    def download_text_files(self, meeting_metadata: List[Dict]) -> List[Dict]:
        """메타데이터를 사용해 Blob에서 텍스트 파일들 다운로드"""
        original_scripts = []
        
        for metadata in meeting_metadata:
            try:
                blob_url = metadata.get("blob_url")
                blob_key = metadata.get("blob_key")
                meeting_id = metadata.get("meeting_id")
                
                if not blob_url or not blob_key:
                    logger.warning(f"회의 {meeting_id}: blob_url 또는 blob_key가 없음")
                    continue
                
                # Blob Service Client 생성 (URL에서 추출)
                if not self.blob_service_client:
                    # blob_url에서 account_url 추출
                    # 예: https://storageaccount.blob.core.windows.net/container/filename.txt
                    parts = blob_url.split('/')
                    account_url = f"https://{parts[2]}"
                    
                    self.blob_service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=blob_key
                    )
                
                # 컨테이너와 blob 이름 추출
                container_name = parts[3]
                blob_name = '/'.join(parts[4:])
                
                # Blob 다운로드
                blob_client = self.blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )
                
                logger.info(f"Blob 다운로드 시작: {meeting_id}")
                
                # 텍스트 내용 다운로드
                download_stream = blob_client.download_blob()
                content = download_stream.readall().decode('utf-8')
                
                original_script = {
                    "meeting_id": meeting_id,
                    "full_content": content,
                    "blob_url": blob_url,
                    "file_size": len(content.encode('utf-8')),
                    "metadata": metadata
                }
                
                original_scripts.append(original_script)
                
                logger.info(f"Blob 다운로드 완료: {meeting_id}, 크기: {original_script['file_size']} bytes")
                
            except Exception as e:
                logger.error(f"회의 {meeting_id} Blob 다운로드 실패: {str(e)}")
                continue
        
        logger.info(f"총 {len(original_scripts)}개 원본 스크립트 다운로드 완료")
        return original_scripts
    
    def download_single_file(self, blob_url: str, blob_key: str) -> str:
        """단일 파일 다운로드"""
        try:
            parts = blob_url.split('/')
            account_url = f"https://{parts[2]}"
            container_name = parts[3]
            blob_name = '/'.join(parts[4:])
            
            blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=blob_key
            )
            
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            download_stream = blob_client.download_blob()
            content = download_stream.readall().decode('utf-8')
            
            return content
            
        except Exception as e:
            logger.error(f"Blob 파일 다운로드 실패: {str(e)}")
            raise Exception(f"Blob 파일 다운로드 실패: {str(e)}")
    
    def health_check(self, blob_url: str, blob_key: str) -> bool:
        """Blob Storage 헬스체크"""
        try:
            parts = blob_url.split('/')
            account_url = f"https://{parts[2]}"
            
            blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=blob_key
            )
            
            # 계정 정보 조회로 연결 테스트
            account_info = blob_service_client.get_account_information()
            return True
        except:
            return False
