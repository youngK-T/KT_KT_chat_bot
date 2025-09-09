#!/usr/bin/env python3
"""
터미널 기반 Agent 테스트
"""

import logging
from typing import TypedDict, List, Optional, Literal
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END 
from utils.text_processing import chunk_text, clean_text
from utils.embeddings import EmbeddingManager, find_most_relevant_chunks
from config.settings import AZURE_OPENAI_CONFIG, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMeetingQAState(TypedDict, total=False):
    user_question: str
    processed_question: str
    original_scripts: List[dict]
    chunked_scripts: List[dict]
    relevant_chunks: List[dict]
    final_answer: str
    answer_quality_score: int
    improvement_attempts: int
    current_step: str
    error_message: str

class TestAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_CONFIG["deployment_name"],
            api_version=AZURE_OPENAI_CONFIG["api_version"],
            azure_endpoint=AZURE_OPENAI_CONFIG["endpoint"],
            api_key=AZURE_OPENAI_CONFIG["api_key"]
        )
        self.embedding_manager = EmbeddingManager()
        self.user_scripts = []
        
        # Agent 그래프 구성
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Agent 그래프 구성"""
        builder = StateGraph(TestMeetingQAState)
        
        # 노드 추가
        builder.add_node("process_question", self.process_question)
        builder.add_node("load_user_scripts", self.load_user_scripts)
        builder.add_node("process_scripts", self.process_original_scripts)
        builder.add_node("select_chunks", self.select_relevant_chunks)
        builder.add_node("generate_answer", self.generate_final_answer)
        builder.add_node("evaluate_answer", self.evaluate_answer_quality)
        builder.add_node("improve_answer", self.improve_answer)
        
        # 엣지 연결
        builder.set_entry_point("process_question")
        builder.add_edge("process_question", "load_user_scripts")
        builder.add_edge("load_user_scripts", "process_scripts")
        builder.add_edge("process_scripts", "select_chunks")
        builder.add_edge("select_chunks", "generate_answer")
        builder.add_edge("generate_answer", "evaluate_answer")
        
        
        # 조건부 엣지 (딕셔너리 매핑과 함께)
        builder.add_conditional_edges(
            "evaluate_answer",
            self.should_improve_answer,
            {
                "improve": "improve_answer",
                "finish": END
            }
        )
        builder.add_edge("improve_answer", END)        
        
        return builder.compile()
    
    def add_script(self, content: str, title: str = ""):
        """스크립트 추가"""
        script_data = {
            "meeting_id": f"script_{len(self.user_scripts) + 1}",
            "content": content,
            "filename": f"{title or f'스크립트_{len(self.user_scripts) + 1}'}.txt"
        }
        self.user_scripts.append(script_data)
        print(f"✅ 스크립트 추가: {title or f'스크립트 {len(self.user_scripts)}'}")
    
    def process_question(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """1단계: 질문 전처리"""
        print("🔍 1단계: 질문 전처리")
        user_question = state.get("user_question", "")
        
        if not user_question:
            return {**state, "error_message": "질문이 없습니다.", "current_step": "process_question_failed"}
        
        processed_question = clean_text(user_question)
        return {**state, "processed_question": processed_question, "current_step": "question_processed"}
    
    def load_user_scripts(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """2단계: 사용자 스크립트 로드"""
        print("📝 2단계: 사용자 스크립트 로드")
        print(f"   등록된 스크립트: {len(self.user_scripts)}개")
        
        if not self.user_scripts:
            return {**state, "error_message": "등록된 스크립트가 없습니다.", "current_step": "load_scripts_failed"}
        
        return {**state, "original_scripts": self.user_scripts, "current_step": "user_scripts_loaded"}
    
    def process_original_scripts(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """3단계: 원본 스크립트 청킹 및 임베딩"""
        print("⚙️ 3단계: 스크립트 청킹 및 임베딩")
        original_scripts = state.get("original_scripts", [])
        
        all_chunked_scripts = []
        for script in original_scripts:
            content = script.get("content", "")
            meeting_id = script.get("meeting_id", "")
            
            chunks = chunk_text(content, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            print(f"   {meeting_id}: {len(chunks)}개 청크 생성")
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_text_content = chunk["chunk_text"]
                    embedding = self.embedding_manager.embed_query(chunk_text_content)
                    all_chunked_scripts.append({
                        "meeting_id": meeting_id,
                        "chunk_text": chunk_text_content,
                        "chunk_index": i,
                        "chunk_embedding": embedding
                    })
                except Exception as e:
                    print(f"   ❌ 임베딩 실패: {e}")
                    return {**state, "chunked_scripts": [], "current_step": "scripts_processing_failed"}
        
        print(f"   총 {len(all_chunked_scripts)}개 청크 생성")
        return {**state, "chunked_scripts": all_chunked_scripts, "current_step": "scripts_processed"}
    
    def select_relevant_chunks(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """4단계: 질문과 관련된 청크 선별"""
        print("🎯 4단계: 관련 청크 선별")
        processed_question = state.get("processed_question", "")
        chunked_scripts = state.get("chunked_scripts", [])
        
        try:
            query_embedding = self.embedding_manager.embed_query(processed_question)
            relevant_chunks = find_most_relevant_chunks(
                query_embedding=query_embedding,
                chunks=chunked_scripts,
                top_k=5,
                similarity_threshold=0.6
            )
            print(f"   {len(relevant_chunks)}개 관련 청크 선별")
            return {**state, "relevant_chunks": relevant_chunks, "current_step": "chunks_selected"}
        except Exception as e:
            print(f"   ❌ 청크 선별 실패: {e}")
            return {**state, "error_message": f"청크 선별 실패: {e}", "current_step": "select_chunks_failed"}
    
    def generate_final_answer(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """5단계: 최종 답변 생성"""
        print("🤖 5단계: 답변 생성")
        question = state.get("processed_question", "")
        relevant_chunks = state.get("relevant_chunks", [])
        
        context = "\n\n".join([chunk["chunk_text"] for chunk in relevant_chunks])
        
        prompt = f"""
        다음 회의록 내용을 바탕으로 질문에 답변해주세요.
        
        질문: {question}
        
        참고 자료:
        {context}
        
        정확하고 도움이 되는 답변을 한국어로 작성해주세요.
        """
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            print(f"   답변 생성 완료")
            return {**state, "final_answer": answer, "current_step": "answer_generated"}
        except Exception as e:
            print(f"   ❌ 답변 생성 실패: {e}")
            return {**state, "error_message": f"답변 생성 실패: {e}", "current_step": "generate_answer_failed"}
    
    def evaluate_answer_quality(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """6단계: 답변 품질 평가"""
        print("📊 6단계: 답변 품질 평가")
        answer = state.get("final_answer", "")
        improvement_attempts = state.get("improvement_attempts", 0)
        
        if not answer:
            return {**state, "answer_quality_score": 5, "current_step": "quality_evaluated"}
        
        try:
            evaluation_prompt = f"""
            다음 답변의 품질을 1-5점으로 평가해주세요.
            
            답변: {answer}
            
            점수만 숫자로 답변해주세요 (예: 4)
            """
            
            response = self.llm.invoke(evaluation_prompt)
            score_text = response.content.strip()
            digits = "".join(ch for ch in score_text if ch.isdigit())
            quality_score = int(digits) if digits else 5
            quality_score = max(1, min(5, quality_score))
            
            print(f"   품질 점수: {quality_score}/5")
            return {**state, "answer_quality_score": quality_score, "improvement_attempts": improvement_attempts, "current_step": "quality_evaluated"}
        except Exception as e:
            print(f"   ❌ 품질 평가 실패: {e}")
            return {**state, "answer_quality_score": 5, "current_step": "quality_evaluation_failed"}
    
    def should_improve_answer(self, state: TestMeetingQAState) -> Literal["improve", "finish"]:
        """답변 개선 여부 결정"""
        score = state.get("answer_quality_score", 5)
        tries = state.get("improvement_attempts", 0)
        return "finish" if tries >= 1 or score > 3 else "improve"
    
    def improve_answer(self, state: TestMeetingQAState) -> TestMeetingQAState:
        """7단계: 답변 개선"""
        print("🔄 7단계: 답변 개선")
        question = state.get("processed_question", "")
        current_answer = state.get("final_answer", "")
        relevant_chunks = state.get("relevant_chunks", [])
        
        improvement_attempts = state.get("improvement_attempts", 0) + 1
        context = "\n\n".join([chunk["chunk_text"] for chunk in relevant_chunks])
        
        improvement_prompt = f"""
        다음 답변을 더 정확하고 유용하게 개선해주세요.
        
        질문: {question}
        현재 답변: {current_answer}
        
        참고 자료:
        {context}
        
        개선된 답변을 생성해주세요.
        """
        
        try:
            response = self.llm.invoke(improvement_prompt)
            improved_answer = response.content.strip()
            print("   답변 개선 완료")
            return {**state, "final_answer": improved_answer, "improvement_attempts": improvement_attempts, "current_step": "answer_improved"}
        except Exception as e:
            print(f"   ❌ 답변 개선 실패: {e}")
            return {**state, "improvement_attempts": improvement_attempts, "answer_quality_score": 5, "current_step": "answer_improvement_failed"}
    
    def run_agent(self, question: str):
        """Agent 실행"""
        initial_state = {"user_question": question, "current_step": "started", "improvement_attempts": 0}
        try:
            result = self.graph.invoke(initial_state, config={"recursion_limit": 100})
            if result is None:
                print("❌ Agent 실행 결과가 None입니다.")
                return
            
            print(f"✅ 최종 단계: {result.get('current_step')}, 시도: {result.get('improvement_attempts')}, 점수: {result.get('answer_quality_score')}")
            print(f"🤖 최종 답변: {result.get('final_answer','')}")
        except Exception as e:
            print(f"❌ run_agent 예외: {e}")

def main():
    print("🤖 터미널 기반 Agent 테스트")
    print("=" * 40)
    
    agent = TestAgent()
    
    # 테스트 스크립트 추가
    test_script = """
    안녕하세요. 오늘은 2024년 1월 정기 회의입니다.
    
    주요 안건:
    1. 2023년 사업 실적 검토
    2. 2024년 사업 계획 수립
    3. 신규 프로젝트 추진 방안
    4. 예산 편성 계획
    
    먼저 2023년 사업 실적을 검토해보겠습니다.
    매출은 전년 대비 15% 증가했으며, 신규 고객 확보에 성공했습니다.
    
    2024년에는 AI 기술 도입을 통한 업무 효율성 향상에 집중하겠습니다.
    챗봇 개발 프로젝트를 진행 중이며, 현재 80% 완료되었습니다.
    """
    
    agent.add_script(test_script, "2024년 1월 회의록")
    
    # 테스트 질문
    test_question = "주요 안건은 무엇인가요?"
    agent.run_agent(test_question)

if __name__ == "__main__":
    main()