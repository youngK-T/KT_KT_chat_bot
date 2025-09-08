"""
원본 코드 수정 없이 Mock 데이터로 전체 플로우 테스트
Import Mock을 사용해서 패키지가 없어도 import 가능하게 함
"""

import asyncio
import sys
import os
from unittest.mock import patch, MagicMock
import random

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===== Import 전에 Mock 모듈들을 sys.modules에 등록 =====
def setup_mock_modules():
    """누락된 패키지들을 Mock으로 등록"""
    
    # psycopg2 Mock
    mock_psycopg2 = MagicMock()
    mock_psycopg2.extras = MagicMock()
    mock_psycopg2.extras.RealDictCursor = MagicMock()
    sys.modules['psycopg2'] = mock_psycopg2
    sys.modules['psycopg2.extras'] = mock_psycopg2.extras
    
    # Azure Storage Mock (Azure OpenAI는 실제 사용하므로 Storage만 Mock)
    mock_azure_storage = MagicMock()
    mock_azure_storage.blob = MagicMock()
    mock_azure_storage.blob.BlobServiceClient = MagicMock()
    sys.modules['azure.storage'] = mock_azure_storage
    sys.modules['azure.storage.blob'] = mock_azure_storage.blob
    # 주의: azure 전체는 Mock하지 않음 (Azure OpenAI 사용을 위해)
    
    # PyMuPDF Mock (fitz)
    mock_fitz = MagicMock()
    sys.modules['fitz'] = mock_fitz
    
    # python-docx Mock
    mock_docx = MagicMock()
    sys.modules['docx'] = mock_docx
    
    # chromadb Mock
    mock_chromadb = MagicMock()
    sys.modules['chromadb'] = mock_chromadb
    
    # Wikipedia Mock
    mock_wikipedia = MagicMock()
    sys.modules['wikipedia'] = mock_wikipedia
    
    # OpenAI Mock 제거 - Azure OpenAI 실제 사용을 위해
    # sys.modules['openai'] = mock_openai  # 주석처리
    
    # 기타 가능한 Mock들
    mock_pandas = MagicMock()
    sys.modules['pandas'] = mock_pandas
    
    print("✅ 모든 Mock 모듈 등록 완료")

# Mock 모듈들 등록 실행
setup_mock_modules()

# 이제 안전하게 import 가능
from models.state import MeetingQAState
from agents.meeting_qa_agent import MeetingQAAgent
from tests.mock_data import (
    MOCK_RAG_SUMMARIES, 
    MOCK_MEETING_METADATA, 
    MOCK_ORIGINAL_SCRIPTS,
    TEST_QUESTIONS
)

class MockTester:
    """원본 코드 수정 없이 Mock 테스트"""
    
    def __init__(self):
        self.test_results = []
    
    def create_mock_rag_client(self):
        """Mock RAG 클라이언트 생성 - async를 동기로 변환"""
        mock_rag = MagicMock()
        
        # 동기 함수로 만들어서 async 문제 해결
        def mock_search_summaries(query, keywords, top_k=5, similarity_threshold=0.7):
            print(f"🔍 Mock RAG 검색: 질문='{query}', 키워드={keywords}")
            # 키워드 기반 간단한 필터링
            results = []
            for summary in MOCK_RAG_SUMMARIES:
                if any(keyword.lower() in summary["summary_text"].lower() for keyword in keywords):
                    results.append(summary)
                    print(f"   ✅ 매칭: {summary['meeting_title']}")
            
            final_results = results[:top_k] or MOCK_RAG_SUMMARIES[:1]  # 최소 1개는 반환
            print(f"   📊 반환: {len(final_results)}개 요약본")
            return final_results
        
        # async 함수를 완전히 동기 함수로 대체
        mock_rag.search_summaries = mock_search_summaries
        return mock_rag
    
    def create_mock_postgres_client(self):
        """Mock PostgreSQL 클라이언트 생성"""
        mock_pg = MagicMock()
        
        def mock_fetch_metadata(meeting_ids):
            print(f"💾 Mock PostgreSQL 조회: meeting_ids={meeting_ids}")
            results = [metadata for metadata in MOCK_MEETING_METADATA 
                      if metadata["meeting_id"] in meeting_ids]
            print(f"   📊 반환: {len(results)}개 메타데이터")
            return results
        
        mock_pg.fetch_meeting_metadata = mock_fetch_metadata
        mock_pg.__enter__ = lambda self: mock_pg
        mock_pg.__exit__ = lambda self, *args: None
        return mock_pg
    
    def create_mock_blob_client(self):
        """Mock Azure Blob 클라이언트 생성"""
        mock_blob = MagicMock()
        
        def mock_download_files(meeting_metadata):
            print(f"📄 Mock Blob 다운로드: {len(meeting_metadata)}개 메타데이터")
            results = []
            for metadata in meeting_metadata:
                meeting_id = metadata["meeting_id"]
                if meeting_id in MOCK_ORIGINAL_SCRIPTS:
                    results.append({
                        "meeting_id": meeting_id,
                        "full_content": MOCK_ORIGINAL_SCRIPTS[meeting_id],
                        "blob_url": metadata["blob_url"],
                        "file_size": len(MOCK_ORIGINAL_SCRIPTS[meeting_id]),
                        "metadata": metadata
                    })
                    print(f"   ✅ 스크립트 로드: {meeting_id}")
            print(f"   📊 반환: {len(results)}개 원본 스크립트")
            return results
        
        mock_blob.download_text_files = mock_download_files
        return mock_blob
    
    def create_mock_llm(self):
        """Mock LLM 생성"""
        mock_llm = MagicMock()
        
        def mock_invoke(prompt):
            mock_response = MagicMock()
            
            # 프롬프트 내용에 따라 다른 응답
            if "키워드" in prompt and "추출" in prompt:
                mock_response.content = "마케팅, 전략, 예산, 증액, Q1"
            elif "전처리" in prompt:
                mock_response.content = prompt.split("사용자 질문: ")[-1].split("\n")[0] if "사용자 질문: " in prompt else "전처리된 질문"
            else:
                # 답변 생성
                mock_response.content = f"Mock 테스트 답변: 제공된 회의록 내용을 바탕으로 답변드리겠습니다. {random.choice(['예산 30% 증액', '4월 출시 연기', '고객 유지율 95% 목표'])}이 주요 내용입니다."
            
            return mock_response
        
        mock_llm.invoke = mock_invoke
        return mock_llm
    
    def create_mock_embedding_manager(self):
        """Mock 임베딩 매니저 생성"""
        mock_embedding = MagicMock()
        
        def mock_embed_query(query):
            # 간단한 Mock 임베딩 (실제로는 숫자 벡터)
            return [0.1, 0.2, 0.3, 0.4, 0.5] * 20  # 100차원 가정
        
        def mock_add_embeddings_to_chunks(chunks, meeting_id):
            for i, chunk in enumerate(chunks):
                chunk["chunk_embedding"] = [0.1 + i*0.01] * 100
                chunk["meeting_id"] = meeting_id
            return chunks
        
        mock_embedding.embed_query = mock_embed_query
        mock_embedding.add_embeddings_to_chunks = mock_add_embeddings_to_chunks
        return mock_embedding
    
    async def test_full_flow(self, question: str):
        """전체 플로우 테스트"""
        print(f"\n🤖 테스트 질문: {question}")
        print("=" * 60)
        
        try:
            # Mock 객체들 생성 (외부 서비스만)
            print("🔧 외부 서비스 Mock 객체들 생성 중...")
            mock_rag = self.create_mock_rag_client()
            mock_pg = self.create_mock_postgres_client()
            mock_blob = self.create_mock_blob_client()
            # mock_llm과 mock_embedding은 실제 Azure OpenAI 사용
            print("✅ 외부 서비스 Mock 객체 생성 완료")
            
            print("🔄 Monkey Patching 시작...")
            print("🤖 LLM만 실제 Azure OpenAI 호출")
            print("📦 외부 서비스(RAG, DB, Storage, Embedding)는 Mock 사용")
            # 외부 서비스만 Mock으로 교체, LLM과 임베딩은 실제 호출
            with patch('agents.meeting_qa_agent.RAGClient', return_value=mock_rag) as patch_rag, \
                 patch('agents.meeting_qa_agent.PostgreSQLClient', return_value=mock_pg) as patch_pg, \
                 patch('agents.meeting_qa_agent.AzureBlobClient', return_value=mock_blob) as patch_blob, \
                 patch('utils.text_processing.chunk_text') as mock_chunk, \
                 patch('utils.embeddings.find_most_relevant_chunks') as mock_find_chunks, \
                 patch('agents.meeting_qa_agent.EmbeddingManager') as mock_embedding_manager:
                
                print("✅ 모든 패치 적용 완료")
                
                # 텍스트 처리 Mock 설정 (실제 회의 내용 사용)
                mock_chunk.return_value = [
                    {"chunk_text": "김철수: Q1 전략은 Z세대와 밀레니얼 세대를 중심으로 한 소셜미디어 캠페인에 집중하는 것이 좋겠습니다. 예산은 어떻게 배분할까요?", "chunk_index": 0},
                    {"chunk_text": "정우성: 기존 예산 대비 30% 증액을 제안합니다. 인스타그램 광고에 40%, 틱톡 캠페인에 35%, 유튜브 쇼츠에 25% 배분하는 것이 효과적일 것 같습니다.", "chunk_index": 1}
                ]
                
                # 임베딩 Mock 설정 (Rate Limit 방지)
                mock_embedding_instance = MagicMock()
                mock_embedding_instance.create_embeddings.return_value = [0.1] * 1536  # 가짜 임베딩 벡터
                mock_embedding_instance.embed_query.return_value = [0.1] * 1536  # 가짜 쿼리 임베딩
                
                # add_embeddings_to_chunks Mock 구현
                def mock_add_embeddings_to_chunks(chunks, meeting_id):
                    result = []
                    for i, chunk in enumerate(chunks):
                        result.append({
                            "chunk_text": chunk["chunk_text"],
                            "chunk_index": chunk["chunk_index"],
                            "meeting_id": meeting_id,
                            "embedding": [0.1] * 1536,  # 가짜 임베딩
                            "relevance_score": 0.8  # 기본 관련성 점수
                        })
                    return result
                
                mock_embedding_instance.add_embeddings_to_chunks = mock_add_embeddings_to_chunks
                mock_embedding_manager.return_value = mock_embedding_instance
                print("📊 임베딩은 임시로 Mock 처리 (Rate Limit 방지)")
                
                # 청크 선별 Mock 설정 (간소화)
                mock_find_chunks.return_value = [
                    {"chunk_text": "Q1 마케팅 전략을 정리하면: 1) 소셜미디어 중심의 디지털 마케팅 강화, 2) Z세대 타겟팅 집중, 3) 예산 30% 증액, 4) 플랫폼별 차별화된 콘텐츠 전략입니다.", 
                     "chunk_index": 0, "meeting_id": "MEETING_2024_0115_001", "relevance_score": 0.9},
                    {"chunk_text": "기존 예산 대비 30% 증액을 제안합니다. 인스타그램 광고에 40%, 틱톡 캠페인에 35%, 유튜브 쇼츠에 25% 배분하는 것이 효과적일 것 같습니다.", 
                     "chunk_index": 1, "meeting_id": "MEETING_2024_0115_001", "relevance_score": 0.85}
                ]
                
                # Agent 생성 및 실행
                print("🤖 MeetingQAAgent 생성 중...")
                agent = MeetingQAAgent()
                print("✅ Agent 생성 완료")
                
                # 초기 상태 설정
                print("⚙️ 초기 상태 설정 중...")
                initial_state: MeetingQAState = {
                    "user_question": question,
                    "processed_question": "",
                    "search_keywords": [],
                    "relevant_summaries": [],
                    "selected_meeting_ids": [],
                    "meeting_metadata": [],
                    "original_scripts": [],
                    "chunked_scripts": [],
                    "relevant_chunks": [],
                    "context_chunks": [],
                    "final_answer": "",
                    "sources": [],
                    "confidence_score": 0.0,
                    "rag_service_url": "http://mock-rag-service:8080",
                    "postgresql_config": {
                        "host": "mock-host",
                        "database": "mock_db",
                        "user": "mock_user",
                        "password": "mock_pass"
                    },
                    "current_step": "initialized",
                    "error_message": ""
                }
                
                print("📝 Agent 실행 시작...")
                print(f"   질문: {question}")
                print(f"   RAG URL: {initial_state['rag_service_url']}")
                
                # Agent 실행
                print("🚀 agent.run() 호출...")
                final_state = await agent.run(initial_state)
                print("✅ agent.run() 완료")
                
                # 결과 출력
                print("✅ 전체 플로우 완료!")
                print("-" * 40)
                print(f"📊 현재 단계: {final_state.get('current_step', 'unknown')}")
                print(f"🔍 관련 요약본: {len(final_state.get('relevant_summaries', []))}개")
                print(f"💾 회의 메타데이터: {len(final_state.get('meeting_metadata', []))}개") 
                print(f"📄 원본 스크립트: {len(final_state.get('original_scripts', []))}개")
                print(f"📝 관련 청크: {len(final_state.get('relevant_chunks', []))}개")
                print(f"🎯 신뢰도: {final_state.get('confidence_score', 0):.2f}")
                
                if final_state.get('error_message'):
                    print(f"❌ 오류: {final_state['error_message']}")
                else:
                    print(f"💬 답변: {final_state.get('final_answer', '답변 없음')}")
                
                # 출처 정보
                sources = final_state.get('sources', [])
                if sources:
                    print(f"📚 출처: {len(sources)}개 회의록")
                    for source in sources[:2]:  # 처음 2개만 표시
                        print(f"   - {source.get('meeting_title', 'Unknown')} ({source.get('meeting_date', 'Unknown')})")
                
                return final_state
                
        except Exception as e:
            print(f"❌ 테스트 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

async def main():
    """Mock 테스트 메인 실행"""
    print("🚀 Mock 데이터 전체 플로우 테스트 시작")
    print("=" * 60)
    print("📌 원본 코드는 전혀 수정하지 않고 Monkey Patching 사용")
    print("=" * 60)
    
    tester = MockTester()
    
    # 여러 질문으로 테스트
    for i, question in enumerate(TEST_QUESTIONS[:2], 1):  # 처음 2개만 테스트
        print(f"\n🔄 테스트 {i}/{len(TEST_QUESTIONS[:2])}")
        result = await tester.test_full_flow(question)
        
        if result:
            print("✅ 테스트 성공")
        else:
            print("❌ 테스트 실패")
        
        print("\n" + "="*60)
    
    print("\n🎉 전체 Mock 테스트 완료!")
    print("💡 원본 코드는 전혀 수정되지 않았습니다.")

if __name__ == "__main__":
    asyncio.run(main())