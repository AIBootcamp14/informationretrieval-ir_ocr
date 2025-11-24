# src/retriever.py

from elasticsearch import Elasticsearch
from openai import OpenAI
import config
from src.query_analyzer import QueryAnalysisResult
import json
import re
from typing import List

client = OpenAI()

def connect_es():
    """Elasticsearch에 연결합니다."""
    es_client = Elasticsearch(
        hosts=[{"host": config.ES_HOST, "port": config.ES_PORT, "scheme": "http"}],
        basic_auth=(config.ES_USERNAME, config.ES_PASSWORD),
        request_timeout=30
    )
    if not es_client.ping():
        raise ConnectionError("Elasticsearch 연결 실패.")
    return es_client

def generate_ocr_noise_query(query: str) -> str:
    """
    [OCR 전문가 전략] LLM을 사용해 한국어 OCR 오류 패턴을 시뮬레이션한 변형된 질의를 생성합니다.
    (예: 띄어쓰기 오류, 시각적 유사 글자 치환)
    """
    system_prompt = f"""
    당신은 OCR(광학 문자 인식) 시스템이 한국어 텍스트를 인식할 때 발생하는 오류를 시뮬레이션하는 전문가입니다.
    주어진 질의를 **하나**의 새로운, 그럴듯한 OCR 오류가 포함된 질의로 변형하세요.

    **규칙:**
    1.  원본 질의의 의미는 유지하되, **시각적 오류** 또는 **띄어쓰기 오류**를 최소 1회 이상 적용합니다. (예: '나무 분류'를 '나 무분류'로, '바'를 '봐'로, 'ㅁ'을 'ㅂ'으로)
    2.  변형된 질의 **하나**만 출력합니다.
    """
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"변형할 질의: {query}"}
            ],
            temperature=0.8 # 높은 온도로 다양한 오류 패턴 유도 (Recall 증대)
        )
        ocr_query = response.choices[0].message.content.strip()
        # LLM 응답 정제 (불필요한 따옴표, 마크다운 제거)
        ocr_query = re.sub(r'^["\'`]*|["\'`]*$', '', ocr_query) 
        return ocr_query
    except Exception as e:
        # 오류 시 폴백(Fallback): 띄어쓰기만 제거하여 기본적인 노이즈 쿼리 반환
        return query.replace(" ", "")

def search_documents(query_result: QueryAnalysisResult, initial_k=15) -> List[str]:
    """
    OCR 앙상블 검색을 통해 문서를 찾고, LLM Reranking을 통해 최종 topk를 결정합니다.
    """
    if not query_result.is_scientific_query:
        return [] # 과학 상식 질문이 아니면 빈 리스트 반환 (MAP 1점 전략)
    
    es_client = connect_es()
    original_query = query_result.standalone_query
    
    # 1. OCR 앙상블 검색 (Recall 최대화)
    ocr_query = generate_ocr_noise_query(original_query)
    search_queries = list(set([original_query, ocr_query])) # 중복 제거
    
    all_hits = {} 
    
    for q in search_queries:
        if not q: continue
        try:
            # Match 쿼리 사용 (BM25 기본 검색)
            res = es_client.search(
                index=config.ES_INDEX_NAME,
                query={"match": {"content": q}},
                size=initial_k 
            )
            for hit in res['hits']['hits']:
                doc_id = hit['_source']['docid']
                # 두 쿼리 중 점수가 높은 결과만 유지
                if doc_id not in all_hits or hit['_score'] > all_hits[doc_id]['score']:
                    all_hits[doc_id] = {
                        'docid': doc_id,
                        'content': hit['_source']['content'],
                        'score': hit['_score']
                    }
        except Exception as e:
            print(f"Elasticsearch 검색 오류 ({q}): {e}")

    documents_to_rerank = list(all_hits.values())
    if not documents_to_rerank:
        return []

    # 2. LLM Reranking (Precision 최대화)
    # LLM에게 텍스트의 실제 연관성을 평가하게 함
    rerank_prompt = f"""
    주어진 질문과 문서 목록을 검토하여, 질문에 대한 **정확한 답변을 포함**하거나 **가장 밀접한 과학적 사실**을 담고 있는 문서 3개를 선정하세요. 선정된 문서의 'docid'를 **가장 관련성 높은 순서대로** 배열하여 JSON 리스트로 반환하세요.
    
    **질문:** {original_query}
    
    **문서 목록:**
    {json.dumps([{'docid': doc['docid'], 'score': doc['score'], 'snippet': doc['content'][:100] + '...'} for doc in documents_to_rerank], ensure_ascii=False, indent=2)}

    **출력 형식:** ['docid1', 'docid2', 'docid3']
    """

    try:
        rerank_response = client.chat.completions.create(
            model=config.RERANKER_MODEL,
            messages=[
                {"role": "system", "content": "당신은 RAG 시스템의 Reranking 전문가입니다. 질문과 문서 내용을 철저히 비교하여 가장 정확한 문서 3개의 ID만 JSON 배열 형태로 반환하세요."},
                {"role": "user", "content": rerank_prompt}
            ]
        )
        reranked_docids = json.loads(rerank_response.choices[0].message.content)
        # LLM이 이상한 ID를 반환할 경우를 대비하여 유효성 검사 및 3개로 자름
        valid_docids = [did for did in reranked_docids if did in all_hits][:3] 
        return valid_docids
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Reranking 오류 발생. ES 점수 순으로 대체: {e}")
        # 오류 시, ES 점수가 높은 순으로 정렬하여 반환 (안전 장치)
        sorted_hits = sorted(documents_to_rerank, key=lambda x: x['score'], reverse=True)
        return [hit['docid'] for hit in sorted_hits[:3]]