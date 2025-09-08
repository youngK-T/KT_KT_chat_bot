#!/usr/bin/env python3
"""
Meeting QA API 서버 실행 엔트리포인트
"""

import uvicorn
import sys
import os

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import API_HOST, API_PORT

def main():
    """메인 실행 함수"""
    print("🚀 Meeting QA API 서버를 시작합니다...")
    print(f"📍 서버 주소: http://{API_HOST}:{API_PORT}")
    print(f"📖 API 문서: http://{API_HOST}:{API_PORT}/docs")
    print("🔄 개발 모드: 파일 변경 시 자동 재시작")
    print("-" * 50)
    
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
