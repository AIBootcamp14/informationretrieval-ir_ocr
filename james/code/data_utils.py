import json
import os
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI

# Solar API 설정 (과학/비과학 분류용)
SOLAR_API_KEY = os.environ.get("SOLAR_API_KEY", "REDACTED_SOLAR_KEY")
client = OpenAI(
    api_key=SOLAR_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

DOCUMENT_CATEGORIES = [
    'ARC_Challenge',
    'conceptual_physics',
    'human_aging',
    'nutrition',
    'high_school_biology',
    'astronomy',
    'electrical_engineering',
    'high_school_chemistry',
    'virology',
    'college_medicine',
    'high_school_physics',
    'global_facts',
    'human_sexuality',
    'medical_genetics',
    'computer_security',
    'anatomy',
    'college_biology',
    'college_physics',
    'college_chemistry',
    'college_computer_science'
]

def extract_category_from_src(src: str) -> str:
    if not src:
        return "general"
    parts = src.split("__")
    if len(parts) >= 2:
        cat = parts[1]
    else:
        cat = src
    if cat in DOCUMENT_CATEGORIES:
        return cat
    if "ARC" in cat:
        return "ARC_Challenge"
    return "general"

def load_documents(documents_path: str) -> Dict[str, Dict[str, str]]:
    """문서 로딩: docid -> {content, src, category} 매핑 반환"""
    print(f"문서 로딩 중: {documents_path}")
    documents = {}
    with open(documents_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="문서 읽는 중"):
            doc = json.loads(line.strip())
            src = doc.get('src', '')
            documents[doc['docid']] = {
                "content": doc['content'],
                "src": src,
                "category": extract_category_from_src(src)
            }
    print(f"총 {len(documents)}개의 문서가 로딩되었습니다.")
    return documents

def load_eval_queries(eval_path: str) -> List[Dict]:
    """평가 쿼리 로딩"""
    print(f"평가 쿼리 로딩 중: {eval_path}")
    queries = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data = json.loads(line.strip())
            queries.append(eval_data)
    print(f"총 {len(queries)}개의 쿼리가 로딩되었습니다.")
    return queries

def process_multi_turn_query(msg_list: List[Dict]) -> str:
    """
    멀티턴 대화 메시지를 하나의 쿼리로 변환
    단순하게 User 메시지들을 합치거나, 마지막 질문에 집중하되 문맥을 포함하도록 처리
    """
    user_messages = [msg['content'] for msg in msg_list if msg['role'] == 'user']
    
    if not user_messages:
        return ""
        
    # 싱글턴인 경우
    if len(user_messages) == 1:
        return user_messages[0]
    
    # 멀티턴인 경우: 모든 문맥을 포함하여 하나의 쿼리로 만듦
    # 예: "Q1" -> "A1" -> "Q2" => "Q1 Q2" (단순 연결이 검색에 유리할 때가 많음)
    # 필요시 LLM을 이용해 요약(Reformulation) 할 수도 있음
    return " ".join(user_messages)

def classify_science_query(query: str) -> bool:
    """
    Solar LLM을 사용하여 질문이 과학 상식 관련인지 판단
    True: 과학 질문, False: 일상 대화/비과학 질문
    """
    # 간단한 룰베이스 필터링 (비용 절약)
    common_greetings = ["안녕", "반가워", "누구니", "뭐해", "이름이", "고마워", "사랑해", "잘자", "좋은 아침"]
    if any(greeting in query for greeting in common_greetings) and len(query) < 20:
        return False

    system_prompt = """당신은 질문 분류기입니다. 주어진 질문이 '과학적 지식', '사실 확인', '현상 설명', '학술적 내용'을 묻는 경우 'science'라고 답하고, 단순한 '인사', '일상 대화', '농담', '주관적 의견', '창작 요청'인 경우 'chat'이라고 답하세요.

예시:
Q: "사과는 왜 떨어져?" -> science
Q: "지구는 몇 살이야?" -> science
Q: "안녕 반가워" -> chat
Q: "너는 누구니?" -> chat
Q: "오늘 기분 어때?" -> chat
Q: "물은 화학식으로 뭐야?" -> science
"""

    try:
        completion = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        answer = completion.choices[0].message.content.lower()
        return "science" in answer
    except Exception as e:
        print(f"분류 중 에러 발생: {e}")
        return True # 에러 시 기본적으로 검색 수행

def generate_hypothetical_document(query: str) -> str:
    """
    HyDE: 질문에 대한 가상의 답변(Hypothetical Document)을 생성
    이 답변을 임베딩하여 검색에 활용하면 의미적 매칭 확률이 높아짐
    """
    system_prompt = "당신은 과학 전문 지식인입니다. 사용자의 질문에 대해 정확하고 논리적인 답변을 간략하게 작성해주세요. 답변은 검색 엔진이 문서를 잘 찾을 수 있도록 핵심 키워드와 설명을 포함해야 합니다."
    
    try:
        completion = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7 # 창의적인 답변 생성을 위해 약간 높임
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"HyDE 생성 중 에러: {e}")
        return query # 에러 시 원래 쿼리 반환

def classify_query_category(query: str) -> str:
    """
    Solar LLM을 사용해 질문을 문서 카테고리 중 하나로 분류
    """
    system_prompt = f"""당신은 과학 주제 분류기입니다. 아래 목록에서 질문과 가장 관련 있는 카테고리를 하나 선택해 소문자로 출력하세요.
카테고리 목록: {', '.join(DOCUMENT_CATEGORIES)}. 
확신이 없다면 'general'이라고만 답하세요."""
    try:
        completion = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        answer = completion.choices[0].message.content.strip().lower()
        normalized = answer.replace(" ", "_")
        if normalized in [c.lower() for c in DOCUMENT_CATEGORIES]:
            # 원본 리스트와 케이스 일치
            for cat in DOCUMENT_CATEGORIES:
                if cat.lower() == normalized:
                    return cat
        return "general"
    except Exception as e:
        print(f"카테고리 분류 에러: {e}")
        return "general"

def preprocess_queries(queries: List[Dict], cache_path: str = "query_classifications.json") -> List[Dict]:
    """
    쿼리 전처리: 멀티턴 처리 + 과학/비과학 태깅
    결과는 캐싱하여 재사용
    """
    print("쿼리 전처리 및 과학 질문 분류 수행 중...")
    
    # 캐시 확인
    classifications = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            classifications = json.load(f)
            
    processed_queries = []
    
    for q in tqdm(queries, desc="쿼리 분석"):
        eval_id = str(q['eval_id'])
        query_text = process_multi_turn_query(q['msg'])
        
        # 분류 수행 (캐시에 없으면 API 호출)
        if eval_id in classifications:
            is_science = classifications[eval_id]
        else:
            is_science = classify_science_query(query_text)
            classifications[eval_id] = is_science
            
        processed_queries.append({
            "eval_id": q['eval_id'],
            "query_text": query_text,
            "is_science": is_science,
            "original_msg": q['msg']
        })
        
    # 캐시 저장
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(classifications, f, ensure_ascii=False, indent=2)
        
    return processed_queries


