"""
Azure OpenAI 콘텐츠 필터 감지 및 처리 유틸리티
"""

import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

def detect_content_filter(exception: Exception) -> Dict[str, Any]:
    """
    Azure OpenAI 콘텐츠 필터 감지 및 상세 정보 반환
    
    Args:
        exception: LLM 호출 중 발생한 예외
        
    Returns:
        dict: {
            'is_filtered': bool,
            'categories': dict,  # 필터 카테고리별 정보
            'severity': str,     # 'high', 'medium', 'low'
            'method': str        # 'error_code' 또는 'fallback'
        }
    """
    try:
        # Option A: 오류 코드 기반 감지 (우선)
        if hasattr(exception, 'response') and exception.response:
            try:
                error_data = exception.response.json()
                error_info = error_data.get('error', {})
                
                # Azure OpenAI 콘텐츠 필터 오류 코드 확인
                if error_info.get('code') == 'content_filter':
                    inner_error = error_info.get('innererror', {})
                    filter_result = inner_error.get('content_filter_result', {})
                    
                    # 최고 심각도 계산
                    max_severity = 'safe'
                    filtered_categories = []
                    
                    for category, details in filter_result.items():
                        if isinstance(details, dict) and details.get('filtered', False):
                            filtered_categories.append(category)
                            severity = details.get('severity', 'safe')
                            if severity in ['high', 'medium', 'low'] and severity != 'safe':
                                if max_severity == 'safe' or (severity == 'high' and max_severity != 'high'):
                                    max_severity = severity
                    
                    logger.warning(f"콘텐츠 필터 감지 (오류코드): 카테고리={filtered_categories}, 심각도={max_severity}")
                    
                    return {
                        'is_filtered': True,
                        'categories': filter_result,
                        'filtered_categories': filtered_categories,
                        'severity': max_severity,
                        'method': 'error_code'
                    }
                    
            except (json.JSONDecodeError, AttributeError, KeyError) as parse_error:
                logger.debug(f"오류 응답 파싱 실패, fallback 사용: {parse_error}")
                pass
                
    except Exception as detect_error:
        logger.debug(f"오류 코드 감지 실패, fallback 사용: {detect_error}")
        pass
    
    # Option B: Fallback - 문자열 매칭
    error_message = str(exception).lower()
    content_filter_keywords = [
        "content filter being triggered",
        "content management policy",
        "content_filter",
        "responsibleaipolicyviolation"
    ]
    
    if any(keyword in error_message for keyword in content_filter_keywords):
        logger.warning(f"콘텐츠 필터 감지 (문자열매칭): {error_message[:100]}...")
        return {
            'is_filtered': True,
            'categories': {},
            'filtered_categories': ['unknown'],
            'severity': 'unknown',
            'method': 'fallback'
        }
    
    return {
        'is_filtered': False,
        'method': 'none'
    }

def create_safe_response(state: Dict[str, Any], step_name: str, filter_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    콘텐츠 필터 감지 시 안전한 응답 생성
    
    Args:
        state: 현재 상태
        step_name: 필터가 감지된 단계명
        filter_info: detect_content_filter 결과
        
    Returns:
        dict: 안전 응답이 포함된 새로운 상태
    """
    safe_message = "Azure 콘텐츠 필터에 따라 해당 내용의 답을 할 수 없습니다."
    
    # 필터 정보 로깅
    if filter_info.get('method') == 'error_code':
        categories = filter_info.get('filtered_categories', [])
        severity = filter_info.get('severity', 'unknown')
        logger.info(f"콘텐츠 필터 안전응답 생성: 단계={step_name}, 카테고리={categories}, 심각도={severity}")
    else:
        logger.info(f"콘텐츠 필터 안전응답 생성: 단계={step_name}, 방법={filter_info.get('method')}")
    
    # 단계별 맞춤 응답 생성
    if step_name in ['generate_answer', 'improve_answer']:
        # 답변 생성/개선 단계
        return {
            **state,
            "final_answer": safe_message,
            "sources": state.get("sources", []) if step_name == 'improve_answer' else [],
            "used_script_ids": state.get("used_script_ids", []) if step_name == 'improve_answer' else [],
            "confidence_score": 0.0,
            "content_filter_triggered": True,
            "current_step": f"content_filter_in_{step_name}"
        }
    elif step_name == 'quality_evaluation':
        # 품질 평가 단계
        return {
            **state,
            "final_answer": safe_message,
            "answer_quality_score": 0,
            "confidence_score": 0.0,
            "content_filter_triggered": True,
            "current_step": "content_filter_in_evaluation"
        }
    elif step_name == 'question_processing':
        # 질문 처리 단계
        return {
            **state,
            "processed_question": state.get("user_question", ""),
            "search_keywords": [],
            "content_filter_triggered": True,
            "current_step": "content_filter_detected"
        }
    elif step_name == 'memory_management':
        # 메모리 관리 단계
        return {
            **state,
            "conversation_memory": "",  # 빈 메모리로 처리
            "content_filter_triggered": True,
            "current_step": "content_filter_in_memory"
        }
    else:
        # 기본 처리
        return {
            **state,
            "content_filter_triggered": True,
            "current_step": f"content_filter_in_{step_name}"
        }
