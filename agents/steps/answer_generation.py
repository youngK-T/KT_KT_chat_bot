"""
6단계: 답변 생성 로직
"""

import json
import re
import logging
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from models.state import MeetingQAState
from utils.content_filter import detect_content_filter, create_safe_response

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """답변 생성 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def _stabilize_chunks(self, chunks: List[Dict], max_count: int = None) -> List[Dict]:
        """청크를 안정적으로 정렬하여 일관성 확보"""
        if not chunks:
            return []
        
        # 1차: relevance_score 내림차순 (높은 점수 우선)
        # 2차: script_id 오름차순 (동일 점수일 때 일관된 순서)
        # 3차: chunk_index 오름차순 (동일 script_id일 때 일관된 순서)
        sorted_chunks = sorted(chunks, 
                              key=lambda x: (
                                  -x.get("relevance_score", 0.0),  # 음수로 내림차순
                                  x.get("script_id", ""),          # 오름차순
                                  x.get("chunk_index", 0)          # 오름차순
                              ))
        
        # 최대 개수 제한
        if max_count:
            return sorted_chunks[:max_count]
        
        return sorted_chunks

    def _stabilize_summaries(self, summaries: List[Dict], max_count: int = None) -> List[Dict]:
        """요약본을 안정적으로 정렬하여 일관성 확보"""
        if not summaries:
            return []
        
        # script_id로 정렬하여 일관된 순서 보장
        sorted_summaries = sorted(summaries, key=lambda x: x.get("script_id", ""))
        
        # 최대 개수 제한
        if max_count:
            return sorted_summaries[:max_count]
        
        return sorted_summaries

    def _build_script_metadata(self, original_scripts: List[Dict]) -> Dict[str, Dict]:
        """original_scripts에서 script_metadata 매핑 생성 (중복 제거용 유틸리티)"""
        script_metadata = {}
        for script in original_scripts:
            script_id = script.get("script_id")
            if script_id:
                script_metadata[script_id] = {
                    "title": script.get("title", ""),
                    "meeting_date": script.get("timestamp", "")
                }
        return script_metadata
    
    def _build_context(self, relevant_summaries: List[Dict], relevant_chunks: List[Dict]) -> str:
        """컨텍스트 생성 로직 통합 (안정적인 정렬 적용)"""
        context_parts = []
        
        # 요약본 추가 (안정적인 정렬)
        stable_summaries = self._stabilize_summaries(relevant_summaries, 3)
        for summary in stable_summaries:
            summary_text = summary.get('summary_text', '').strip()
            if summary_text:
                context_parts.append(f"[요약본] {summary_text}")
        
        # 청크 추가 (안정적인 정렬)
        stable_chunks = self._stabilize_chunks(relevant_chunks, 5)
        for chunk in stable_chunks:
            chunk_text = chunk.get('chunk_text', '').strip()
            if chunk_text:
                context_parts.append(f"[원본] {chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # 빈 컨텍스트 처리
        if not context.strip():
            return "[정보 없음] 관련된 회의록 내용을 찾을 수 없습니다."
        
        return context
    
    def _handle_empty_context(self, question: str) -> Tuple[str, List[Dict]]:
        """빈 컨텍스트일 때 명시적인 답변 생성"""
        fallback_answer = f"죄송합니다. '{question}'에 대한 관련 회의록 내용을 찾을 수 없습니다. 다른 질문을 시도해보시기 바랍니다."
        return fallback_answer, []
    
    def _build_sources(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """Sources 생성 (안정적인 정렬 적용)"""
        if not relevant_chunks:
            return []
        
        # 안정적인 정렬 적용
        stable_chunks = self._stabilize_chunks(relevant_chunks, 5)
        
        # 중복 제거 (타입 안전성 강화)
        seen_scripts = {}
        for chunk in stable_chunks:
            script_id = chunk.get("script_id")
            if not script_id:
                continue
                
            chunk_index = chunk.get("chunk_index", 0)
            relevance_score = chunk.get("relevance_score", 0.0)
            
            if script_id not in seen_scripts or relevance_score > seen_scripts[script_id]["relevance_score"]:
                seen_scripts[script_id] = {
                    "script_id": script_id,
                    "chunk_index": chunk_index,
                    "relevance_score": relevance_score
                }
        
        return list(seen_scripts.values())
    
    def _calculate_confidence(self, relevant_chunks: List[Dict]) -> float:
        """신뢰도 계산 개선 - 청크 개수와 품질 고려"""
        if not relevant_chunks:
            return 0.1
        
        # 청크 개수와 평균 관련도를 고려
        chunk_count = len(relevant_chunks)
        top_chunks = relevant_chunks[:3]  # 상위 3개만 사용
        
        avg_score = sum(chunk.get("relevance_score", 0.0) for chunk in top_chunks) / len(top_chunks)
        
        # 청크 개수 보너스 (최대 0.2)
        count_bonus = min(0.2, chunk_count * 0.05)
        
        # 최종 신뢰도 계산
        confidence = min(0.9, max(0.3, avg_score + count_bonus))
        return confidence
    
    def _convert_quotes_to_evidence(self, quotes: List[Dict], relevant_chunks: List[Dict], original_scripts: List[Dict]) -> List[Dict]:
        """구조화된 quotes를 evidence_quotes 형식으로 변환 (안정적인 정렬 적용)"""
        evidence_quotes = []
        
        # script_metadata 매핑 생성
        script_metadata = self._build_script_metadata(original_scripts)
        
        # 청크를 안정적으로 정렬
        stable_chunks = self._stabilize_chunks(relevant_chunks)
        
        # 청크 검색 최적화 - 인덱스 사전 생성
        chunk_text_map = {}
        for chunk in stable_chunks:
            chunk_text = chunk.get("chunk_text", "")
            if chunk_text:
                chunk_text_map[chunk_text] = chunk
        
        # quotes를 evidence_quotes로 변환 (최적화된 검색)
        for quote_data in quotes:
            quote_text = quote_data.get("text", "")
            speaker = quote_data.get("speaker", "")
            
            # 다단계 유연한 매칭 로직
            source_chunk = None
            matching_method = ""
            
            # 1단계: 정확한 매칭
            for chunk_text, chunk in chunk_text_map.items():
                if quote_text in chunk_text:
                    source_chunk = chunk
                    matching_method = "정확한 매칭"
                    break
            
            # 2단계: 정규화된 매칭 (공백, 특수문자 제거)
            if not source_chunk:
                normalized_quote = re.sub(r'[\s\.,!?]+', ' ', quote_text).strip()
                for chunk_text, chunk in chunk_text_map.items():
                    normalized_chunk = re.sub(r'[\s\.,!?]+', ' ', chunk_text).strip()
                    if normalized_quote in normalized_chunk:
                        source_chunk = chunk
                        matching_method = "정규화된 매칭"
                        break
            
            # 3단계: 핵심 단어 매칭 (최소 3단어)
            if not source_chunk:
                quote_words = normalized_quote.split()
                if len(quote_words) >= 3:
                    for chunk_text, chunk in chunk_text_map.items():
                        normalized_chunk = re.sub(r'[\s\.,!?]+', ' ', chunk_text).strip()
                        # 핵심 단어들이 모두 포함되는지 확인
                        if all(word in normalized_chunk for word in quote_words[:3]):
                            source_chunk = chunk
                            matching_method = "핵심 단어 매칭"
                            break
            
            # 4단계: 부분 단어 매칭 (2단어)
            if not source_chunk:
                quote_words = normalized_quote.split()
                if len(quote_words) >= 2:
                    for chunk_text, chunk in chunk_text_map.items():
                        normalized_chunk = re.sub(r'[\s\.,!?]+', ' ', chunk_text).strip()
                        # 2단어 이상 매칭되는지 확인
                        matched_words = sum(1 for word in quote_words if word in normalized_chunk)
                        if matched_words >= 2:
                            source_chunk = chunk
                            matching_method = "부분 단어 매칭"
                            break
            
            # 5단계: Fallback - 첫 번째 청크 사용 (최소한의 메타데이터라도 제공)
            if not source_chunk and stable_chunks:
                source_chunk = stable_chunks[0]
                matching_method = "Fallback (첫 번째 청크)"
                logger.warning(f"⚠️ 인용문 매칭 실패, Fallback 사용: '{quote_text[:30]}...'")
            
            # 매칭 결과 로깅
            if source_chunk:
                logger.debug(f"✅ 인용문 매칭 성공 ({matching_method}): '{quote_text[:30]}...' -> chunk {source_chunk.get('chunk_index', 0)}")
            
            # 매칭된 청크가 있으면 메타데이터 추출
            if source_chunk:
                script_id = source_chunk.get("script_id", "")
                metadata = script_metadata.get(script_id, {})
                
                evidence_quotes.append({
                    "quote": quote_text,
                    "speaker": speaker,
                    "script_id": script_id,
                    "meeting_title": metadata.get("title", ""),
                    "meeting_date": metadata.get("meeting_date", ""),
                    "chunk_index": source_chunk.get("chunk_index", 0),
                    "relevance_score": source_chunk.get("relevance_score", 0.0)
                })
            else:
                # 최후의 수단: 빈 메타데이터로 저장
                evidence_quotes.append({
                    "quote": quote_text,
                    "speaker": speaker,
                    "script_id": "",
                    "meeting_title": "",
                    "meeting_date": "",
                    "chunk_index": 0,
                    "relevance_score": 0.0
                })
                logger.warning(f"⚠️ 인용문 완전 매칭 실패: '{quote_text[:30]}...'")
        return evidence_quotes
    
    def _generate_structured_answer(self, question: str, context: str, memory: str = "") -> Tuple[str, List[Dict]]:
        """구조화된 JSON 출력으로 답변 생성"""
        
        memory_context = f"\n\n이전 대화 맥락: {memory}" if memory else ""
        
        structured_prompt = f'''당신은 회의록 기반 QA 시스템입니다.
        회의록을 기반으로 해서 사용자 질문에 대한 답변을 생성합니다.
        반드시 다음 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

        {{
            "answer": "회의록 기반 최종 답변 (5문장 이내)",
            "quotes": [
                {{"text": "회의록에서 추출한 인용문", "speaker": "화자01"}},
                {{"text": "추가 인용문", "speaker": "화자02"}}
            ]
        }}

        질문: {question}{memory_context}

        회의록 내용:
        {context}

        JSON:'''

        raw_content = ""
        try:
            # JSON Mode로 응답 생성 시도
            response = self.llm.invoke(structured_prompt)
            raw_content = response.content.strip()
            
            # JSON 파싱
            data = json.loads(raw_content)
            answer = str(data.get("answer", "")) if data.get("answer") else ""
            quotes = data.get("quotes", []) if isinstance(data.get("quotes"), list) else []
            
            # JSON 파싱이 성공했으면 결과를 그대로 사용 (빈 결과도 유효한 결과)
            logger.debug(f"✅ JSON 파싱 성공: 답변 {len(answer)}자, 인용문 {len(quotes)}개")
            return answer, quotes
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON 파싱 실패: {e}")
            return self._simple_fallback_parsing(raw_content)
        except Exception as e:
            logger.error(f"❌ 구조화된 답변 생성 실패: {e}")
            return self._simple_fallback_parsing(raw_content) if raw_content else ("", [])

    def _simple_fallback_parsing(self, raw_content: str) -> Tuple[str, List[Dict]]:
        """JSON 파싱 실패 시 간단한 백업 파싱"""
        
        # 간단한 정규식으로 인용문만 추출 (유연한 공백 처리)
        quote_pattern = r'\(\s*"([^"]+)"\s*,\s*(화자\d+)\s*\)'
        matches = re.findall(quote_pattern, raw_content)
        
        # 인용문 제거하여 순수 답변 추출
        clean_answer = re.sub(quote_pattern, '', raw_content).strip()
        clean_answer = re.sub(r'\n\s*\n', '\n', clean_answer).strip()
        
        # quotes 형식으로 변환
        quotes = [{"text": quote, "speaker": speaker} for quote, speaker in matches]
        
        logger.debug(f"🔄 백업 파싱 완료: 답변 {len(clean_answer)}자, 인용문 {len(quotes)}개")
        return clean_answer, quotes

    def generate_final_answer(self, state: MeetingQAState) -> MeetingQAState:
        """7단계: 최종 답변 생성"""
        try:
            user_question = state.get("user_question", "")
            relevant_summaries = state.get("relevant_summaries", [])
            relevant_chunks = state.get("relevant_chunks", [])
            conversation_memory = state.get("conversation_memory", "")
            
            if not user_question:
                raise ValueError("사용자 질문이 없습니다.")
            
            # 공통 함수로 컨텍스트 생성
            context = self._build_context(relevant_summaries, relevant_chunks)
            
            # 빈 컨텍스트 처리 (주요 문제 해결)
            if "[정보 없음]" in context:
                logger.warning("⚠️ 빈 컨텍스트 감지 - 명시적 답변 생성")
                structured_answer, structured_quotes = self._handle_empty_context(user_question)
            else:
                # 🚀 구조화된 JSON 답변 생성 (새로운 방식!)
                logger.debug("🔄 구조화된 JSON 답변 생성 시작")
                structured_answer, structured_quotes = self._generate_structured_answer(
                    question=user_question,
                    context=context,
                    memory=conversation_memory
                )
            
            # 공통 함수로 evidence_quotes 변환
            original_scripts = state.get("original_scripts", [])
            evidence_quotes = self._convert_quotes_to_evidence(structured_quotes, relevant_chunks, original_scripts)
            
            final_answer = structured_answer
            
            # fallback 메시지 처리 제거 (단순한 에러 처리로 변경)
            
            # 공통 함수로 sources 생성 (단순화됨)
            sources = self._build_sources(relevant_chunks)

            # 실제 사용된 문서 ID 계산
            used_script_ids = sorted({s["script_id"] for s in sources})
            
            # 신뢰도 계산 (개선됨 - 청크 개수와 품질 고려)
            confidence_score = self._calculate_confidence(relevant_chunks)
            
            
            # 최종 응답 state 구성
            final_state = {
                **state,
                "context_chunks": context.split("\n\n") if context else [],
                "final_answer": final_answer,  # 순수 답변만
                "evidence_quotes": evidence_quotes,  # 근거 인용문들 (제목 정보 포함)
                "sources": sources,  # 청킹 관련 정보만
                "used_script_ids": used_script_ids,
                "confidence_score": confidence_score,
                "current_step": "completed"
            }
            
            # 간소화된 로깅 (운영 환경 최적화)
            logger.info(f"✅ 답변 생성 완료: 신뢰도 {confidence_score:.2f}")
            logger.info(f"📊 Evidence Quotes: {len(evidence_quotes)}개, Sources: {len(sources)}개")
            
            # 상세 구조는 DEBUG 레벨로
            logger.debug(f"🔍 상세 구조: {json.dumps(final_state, ensure_ascii=False, indent=2)}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {str(e)}")
            
            # Azure 콘텐츠 필터 감지 (오류 코드 우선)
            filter_info = detect_content_filter(e)
            if filter_info['is_filtered']:
                logger.warning(f"답변 생성 중 콘텐츠 필터 감지: {filter_info}")
                return create_safe_response(state, 'generate_answer', filter_info)
            
            # 일반적인 오류 처리
            return {
                **state,
                "error_message": f"답변 생성 실패: {str(e)}",
                "current_step": "generate_answer_failed"
            }
    
    def improve_answer(self, state: MeetingQAState) -> MeetingQAState:
        """답변 개선"""
        try:
            question = state.get("processed_question", "")
            current_answer = state.get("final_answer", "")
            quality_score = state.get("answer_quality_score", 0)
            relevant_summaries = state.get("relevant_summaries", [])
            relevant_chunks = state.get("relevant_chunks", [])
            improvement_attempts = int(state.get("improvement_attempts") or 0) + 1
            
            # 공통 함수로 컨텍스트 생성
            context = self._build_context(relevant_summaries, relevant_chunks)
            
            # 개선된 답변 생성
            improvement_prompt = f'''당신은 회의록 기반 QA 시스템입니다.
            회의록을 기반으로 해서 사용자 질문에 대한 답변을 개선합니다.
            반드시 다음 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

            이전 답변의 품질이 낮습니다 (점수: {quality_score}/5). 더 정확하고 유용한 답변으로 개선해주세요.

            **개선 규칙:**
            - 정확성: 참고 자료에 기반하여 부정확한 부분 수정
            - 명확성: 모호한 표현 제거하고 핵심 정보 명확히 전달
            - 간결성: 5문장 이내로 요약하되 필요한 정보는 모두 포함
            - 추측 금지: 참고 자료에 없는 내용 절대 추가 금지

            질문: {question}
            이전 답변: {current_answer}
            참고 자료:
            {context}

            **응답 형식 (JSON만):**
            {{
                "answer": "개선된 순수 답변 (인용문 없이)",
                "quotes": [
                    {{"text": "인용 내용", "speaker": "화자01"}},
                    {{"text": "다른 인용 내용", "speaker": "화자02"}}
                ]
            }}
            '''
            response = self.llm.invoke(improvement_prompt)
            raw_content = response.content.strip()
            
            # JSON 파싱 시도
            try:
                data = json.loads(raw_content)
                improved_answer = data.get("answer", "")
                improved_quotes = data.get("quotes", [])
                
                logger.debug(f"✅ 개선 답변 JSON 파싱 성공: 답변 {len(improved_answer)}자, 인용문 {len(improved_quotes)}개")
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ 개선 답변 JSON 파싱 실패: {e}")
                improved_answer, improved_quotes = self._simple_fallback_parsing(raw_content)
            except Exception as e:
                logger.error(f"❌ 개선 답변 생성 실패: {e}")
                improved_answer, improved_quotes = self._simple_fallback_parsing(raw_content) if raw_content else (current_answer, [])
            
            # 공통 함수로 evidence_quotes 변환
            original_scripts = state.get("original_scripts", [])
            relevant_chunks = state.get("relevant_chunks", [])
            evidence_quotes = self._convert_quotes_to_evidence(improved_quotes, relevant_chunks, original_scripts)
            
            # 일관된 로깅 형식 적용
            logger.info(f"✅ 답변 개선 완료: 신뢰도 개선 예상")
            logger.info(f"📊 Evidence Quotes: {len(evidence_quotes)}개 생성")
            
            return {
                **state,
                "final_answer": improved_answer,
                "evidence_quotes": evidence_quotes,
                "improvement_attempts": improvement_attempts,
                "current_step": "answer_improved"
            }
            
        except Exception as e:
            logger.error(f"답변 개선 실패: {str(e)}")
            
            # Azure 콘텐츠 필터 감지 (오류 코드 우선)
            filter_info = detect_content_filter(e)
            if filter_info['is_filtered']:
                logger.warning(f"답변 개선 중 콘텐츠 필터 감지: {filter_info}")
                safe_response = create_safe_response(state, 'improve_answer', filter_info)
                # improvement_attempts 추가
                safe_response["improvement_attempts"] = int(state.get("improvement_attempts") or 0) + 1
                return safe_response
            
            # 일반적인 오류 처리
            return {
                **state,
                "improvement_attempts": int(state.get("improvement_attempts") or 0) + 1,
                "current_step": "improvement_failed"
            }
