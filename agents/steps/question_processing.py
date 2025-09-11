"""
1단계: 질문 전처리 및 키워드 추출
"""

import logging
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from models.state import MeetingQAState
from utils.content_filter import detect_content_filter, create_safe_response

logger = logging.getLogger(__name__)

class QuestionProcessor:
    """질문 전처리 및 키워드 추출 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def process_question(self, state: MeetingQAState) -> MeetingQAState:
        """1단계: 질문 전처리 및 키워드 추출"""
        try:
            user_question = state.get("user_question", "")
            if not user_question:
                raise ValueError("user_question이 비어 있습니다.")

            # 질문 전처리 프롬프트
            question_process_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 검색을 위한 질문을 분석하는 AI입니다.
                다음 사용자 질문을 분석하여 더 명확하고 검색에 최적화된 형태로 전처리해주세요.
                단, 사용자의 의도를 절대로 변경하지 마세요. 질문은 길어져도 괜찮습니다.
                
                사용자 질문: {user_question}
                
                전처리된 질문을 출력해주세요:'''
            )
            
            # 키워드 추출 프롬프트  
            keyword_extract_prompt = ChatPromptTemplate.from_template(
                '''당신은 회의록 검색을 위한 키워드를 추출하는 AI입니다.
                다음 질문에서 검색에 중요한 핵심 키워드를 5~8개 추출해주세요.
                
                질문: {processed_question}
                
                핵심 키워드만 쉼표로 구분해서 출력해주세요:'''
            )

            # 질문 전처리 실행
            formatted_question_prompt = question_process_prompt.format(user_question=user_question)
            question_response = self.llm.invoke(formatted_question_prompt)
            processed_question = question_response.content.strip()
            
            # 키워드 추출 실행
            formatted_keyword_prompt = keyword_extract_prompt.format(processed_question=processed_question)
            keyword_response = self.llm.invoke(formatted_keyword_prompt)
            keywords_text = keyword_response.content.strip()
            
            # 키워드 파싱
            search_keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            logger.info(f"질문 전처리 완료: {len(search_keywords)}개 키워드 추출")
            
            return {
                **state,
                "processed_question": processed_question,
                "search_keywords": search_keywords,
                "current_step": "question_processed"
            }
            
        except Exception as e:
            logger.error(f"질문 전처리 실패: {str(e)}")
            
            # Azure 콘텐츠 필터 감지 (오류 코드 우선)
            filter_info = detect_content_filter(e)
            if filter_info['is_filtered']:
                logger.warning(f"질문 전처리 중 콘텐츠 필터 감지: {filter_info}")
                return create_safe_response(state, 'question_processing', filter_info)
            
            # 일반적인 오류 처리
            return {
                **state,
                "error_message": f"질문 전처리 실패: {str(e)}",
                "current_step": "question_processing_failed"
            }
    
    def enhance_question_with_memory(self, state: MeetingQAState) -> MeetingQAState:
        """메모리를 활용하여 질문 보강"""
        try:
            original_question = state.get("user_question", "")
            memory = state.get("conversation_memory", "")
            
            if not memory:
                # 메모리가 없으면 원본 질문 그대로 사용
                return {
                    **state,
                    "processed_question": original_question
                }
            
            # 메모리를 활용하여 질문 보강
            enhanced_prompt = f"""
            이전 대화 맥락: {memory}
            현재 질문: {original_question}
            
            위 맥락을 고려하여 현재 질문을 더 명확하고 구체적으로 만들어주세요.
            이전 대화와의 연관성을 유지하면서 질문을 개선해주세요.
            """
            
            response = self.llm.invoke(enhanced_prompt)
            enhanced_question = response.content.strip()
            
            return {
                **state,
                "processed_question": enhanced_question
            }
            
        except Exception as e:
            logger.error(f"질문 보강 실패: {str(e)}")
            return {
                **state,
                "processed_question": state.get("user_question", "")
            }
