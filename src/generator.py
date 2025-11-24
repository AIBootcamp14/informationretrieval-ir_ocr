# src/generator.py

from openai import OpenAI
import config
from typing import List
from elasticsearch import Elasticsearch

client = OpenAI()

def get_document_contents(doc_ids: List[str]) -> str:
    """docid 리스트를 사용하여 Elasticsearch에서 문서 내용을 가져옵니다."""
    if not doc_ids:
        return ""
        
    es_client = Elasticsearch(
        hosts=[{"host": config.ES_HOST, "port": config.ES_PORT, "scheme": "http"}],
        basic_auth=(config.ES_USERNAME, config.ES_PASSWORD),
        request_timeout=30
    )
    
    # MGET API 사용 (네트워크 효율성 최적화)
    try:
        response = es_client.mget(index=config.ES_INDEX_NAME, ids=doc_ids)
        # 찾은 문서만 추출
        docs = [item['_source']['content'] for item in response['docs'] if item.get('found', False)]
    except Exception as e:
        print(f"MGET 로드 오류: {e}")
        return ""
    
    
    return "\n\n--- 참고 문서 ---\n\n" + "\n\n---\n\n".join(docs)


def generate_answer(query: str, doc_ids: List[str]) -> str:
    """
    검색된 문서를 참고하여 최종 답변을 생성합니다.
    """
    context = get_document_contents(doc_ids)
    
    if not context:
        # 검색 결과가 없는 경우
        return f"죄송합니다. 요청하신 '{query}'에 대한 과학 상식 정보를 찾지 못했습니다."
        
    system_prompt = f"""
    당신은 신뢰할 수 있는 RAG 시스템의 답변 생성 전문가입니다.
    제공된 [참고 문서]만을 기반으로 사용자의 질문에 답하세요.

    **[핵심 규칙]**
    1. 반드시 [참고 문서]에 있는 사실만을 사용하여 답변을 구성해야 합니다.
    2. 답변은 초등학생도 알기 쉽게 풀어서, 짧고 보기 좋게 간결하게 정리해서 보여줘야 합니다. (사용자 요청 반영)
    3. [참고 문서]에 답변할 수 있는 내용이 없다면, **"죄송하지만 제공된 문서만으로는 답변할 수 없습니다."**라고 명확히 밝히세요.
    4. 답변에는 출처(docid)나 [참고 문서]라는 문구를 언급하지 마세요.
    """

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[참고 문서]:\n{context}\n\n[질문]: {query}"}
            ]
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"LLM 답변 생성 오류: {e}")
        return "죄송합니다. 답변 생성 중 시스템 오류가 발생했습니다."