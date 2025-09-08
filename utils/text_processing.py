from typing import List, Dict
import re

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """텍스트를 청크로 분할"""
    if not text:
        return []
    
    chunks = []
    text_length = len(text)
    start = 0
    chunk_index = 0
    
    while start < text_length:
        # 청크 끝 위치 계산
        end = min(start + chunk_size, text_length)
        
        # 문장 경계에서 분할하도록 조정
        if end < text_length:
            # 마지막 문장 끝을 찾기
            last_period = text.rfind('.', start, end)
            last_exclamation = text.rfind('!', start, end)
            last_question = text.rfind('?', start, end)
            
            sentence_end = max(last_period, last_exclamation, last_question)
            
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "chunk_text": chunk_text,
                "chunk_index": chunk_index,
                "start_pos": start,
                "end_pos": end
            })
            chunk_index += 1
        
        # 다음 청크 시작 위치 (오버랩 고려)
        start = max(start + 1, end - chunk_overlap)
    
    return chunks

def clean_text(text: str) -> str:
    """텍스트 정리"""
    if not text:
        return ""
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 연속된 줄바꿈 제거
    text = re.sub(r'\n+', '\n', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    """간단한 키워드 추출 (빈도 기반)"""
    if not text:
        return []
    
    # 기본적인 전처리
    text = clean_text(text.lower())
    
    # 단어 추출 (한글, 영문, 숫자만)
    words = re.findall(r'[가-힣a-z0-9]+', text)
    
    # 불용어 제거 (간단한 버전)
    stopwords = {
        '그리고', '그런데', '하지만', '그러나', '또한', '따라서', '그래서', '이런', '저런',
        '이것', '저것', '그것', '여기서', '거기서', '저기서', '때문에', '그런지',
        'and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
    }
    
    words = [word for word in words if word not in stopwords and len(word) > 1]
    
    # 빈도 계산
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # 빈도순 정렬
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 키워드 반환
    keywords = [word for word, freq in sorted_words[:max_keywords]]
    
    return keywords

def calculate_text_similarity(text1: str, text2: str) -> float:
    """간단한 텍스트 유사도 계산 (단어 겹침 기반)"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(re.findall(r'[가-힣a-z0-9]+', text1.lower()))
    words2 = set(re.findall(r'[가-힣a-z0-9]+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0
