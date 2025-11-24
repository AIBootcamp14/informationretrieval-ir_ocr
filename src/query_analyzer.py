# src/query_analyzer.py

from openai import OpenAI
from pydantic import BaseModel, Field
import json
import config

client = OpenAI()

# Pydantic 모델을 사용하여 LLM 응답을 구조화 (안정성 확보)
class QueryAnalysisResult(BaseModel):
    """질의 분석 결과를 담는 Pydantic 모델"""
    is_scientific_query: bool = Field(description="질문이 과학 상식(MMLU, ARC)에 관한 것이면 True, 일반 대화나 가치 판단 질문이면 False.")
    standalone_query: str = Field(description="이전 대화 맥락을 포함하여 검색 엔진에 바로 넣을 수 있는 독립적인 질의문.")

def analyze_query(messages: list) -> QueryAnalysisResult:
    """
    대화 메시지를 분석하여 과학 상식 질문 여부와 독립적인 질의를 생성합니다.
    """
    # 1. 멀티턴 대화 처리: 전체 대화 기록을 LLM에 전달
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    system_prompt = f"""
    당신은 RAG 시스템의 질의 분석 전문가입니다.
    사용자의 전체 대화 기록을 검토하고 다음 두 가지를 판단하세요.
    
    **[최우선 규칙: MAP 점수 확보]**
    - 질문이 '니가 대답을 잘해줘서 너무 신나!', '통학 버스의 가치'처럼 **감정, 의견, 가치 판단, 일상 잡담**에 관한 것이라면 **무조건 is_scientific_query를 False**로 반환해야 합니다.
    - 질문이 **정의, 원리, 현상, 사실** (예: '헬륨이 반응을 안 하는 이유', '기억 상실증 원인')을 묻는 과학 지식에 관한 것이어야만 True입니다.

    **[Standalone Query 생성]**
    - 이전 대화의 맥락을 완전히 포함하는 **독립적인 질의(Standalone Query)**를 생성하세요.
    - 독립적인 질의는 검색 엔진에 최적화된, 간결하고 구체적인 키워드 형태여야 합니다. (예: "기억 상실증의 원인")
    """

    try:
        response = client.chat.completions.create(
            model=config.CLASSIFIER_MODEL,
            response_model=QueryAnalysisResult, # Pydantic 모델을 사용한 구조화된 출력 요청
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"대화 기록: \n---\n{history_str}"}
            ]
        )
        return response
    
    except Exception as e:
        print(f"LLM 질의 분석 오류: {e}")
        # 오류 발생 시: 안전을 위해 검색을 시도하도록 True, 마지막 메시지를 쿼리로 반환
        last_message = messages[-1]['content']
        return QueryAnalysisResult(is_scientific_query=True, standalone_query=last_message)