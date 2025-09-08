import psycopg2
import psycopg2.extras
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PostgreSQLClient:
    """PostgreSQL 클라이언트"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connection = None
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = psycopg2.connect(
                host=self.config["host"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                port=self.config.get("port", 5432)
            )
            logger.info("PostgreSQL 연결 성공")
        except Exception as e:
            logger.error(f"PostgreSQL 연결 실패: {str(e)}")
            raise Exception(f"PostgreSQL 연결 실패: {str(e)}")
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("PostgreSQL 연결 해제")
    
    def fetch_meeting_metadata(self, meeting_ids: List[str]) -> List[Dict]:
        """회의 ID 목록으로 메타데이터 조회"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # IN 절을 위한 플레이스홀더 생성
            placeholders = ','.join(['%s'] * len(meeting_ids))
            
            query = f"""
                SELECT 
                    meeting_id,
                    meeting_title,
                    meeting_date,
                    blob_url,
                    blob_key,
                    file_size,
                    created_at,
                    updated_at
                FROM meeting_files 
                WHERE meeting_id IN ({placeholders})
                ORDER BY meeting_date DESC
            """
            
            cursor.execute(query, meeting_ids)
            results = cursor.fetchall()
            
            # RealDictRow를 일반 dict로 변환
            metadata = [dict(row) for row in results]
            
            logger.info(f"메타데이터 조회 완료: {len(metadata)}개 회의 정보")
            
            return metadata
            
        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {str(e)}")
            raise Exception(f"메타데이터 조회 실패: {str(e)}")
        finally:
            if cursor:
                cursor.close()
    
    def health_check(self) -> bool:
        """PostgreSQL 헬스체크"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            return result[0] == 1
        except:
            return False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
