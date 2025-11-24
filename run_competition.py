# run_competition.py

from src.query_analyzer import analyze_query
from src.retriever import search_documents
from src.generator import generate_answer
import json
import pandas as pd
from tqdm import tqdm
import os
from elasticsearch import Elasticsearch

# config 파일 import를 통해 ES, OpenAI 키 자동 로드
import config 
import openai # OpenAI 클라이언트 연결

client = openai.OpenAI()


def run_rag_pipeline(eval_file_path="data/eval.jsonl", output_file="results/submission.csv"):
    """
    최종 RAG 파이프라인을 실행하고 결과를 CSV 파일로 저장합니다.
    """
    print("--- RAG 파이프라인 실행 시작 (단 한번의 완벽한 시도) ---")
    
    # 1. ES 연결 및 헬스 체크
    try:
        es_client = Elasticsearch(
            hosts=[{"host": config.ES_HOST, "port": config.ES_PORT, "scheme": "http"}],
            basic_auth=(config.ES_USERNAME, config.ES_PASSWORD),
            request_timeout=30
        )
        if not es_client.ping():
             raise ConnectionError("Elasticsearch 연결 실패.")
        print("Elasticsearch 연결 성공.")
    except Exception as e:
        print(f"심각한 오류: Elasticsearch 연결 실패. {e}")
        return

    # 2. 결과 저장 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 3. 평가 데이터 로드
    eval_data = []
    try:
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"오류: {eval_file_path} 파일이 없습니다. data/ 폴더에 eval.jsonl을 확인하세요.")
        return

    results = []
    
    # 4. RAG 파이프라인 실행 (평가 데이터 반복)
    for item in tqdm(eval_data, desc="총 220개 평가 데이터 처리 중"):
        eval_id = item['eval_id']
        messages = item['msg']
        
        # 4.1. 질의 분석 (의도 판단 & Standalone Query 생성)
        query_analysis_result = analyze_query(messages)
        standalone_query = query_analysis_result.standalone_query
        is_scientific = query_analysis_result.is_scientific_query

        topk = []
        answer = ""
        
        if is_scientific:
            # 4.2. 과학 상식 질문: 검색 및 Reranking 수행
            topk = search_documents(query_analysis_result, initial_k=15) 
            
            # 4.3. 답변 생성
            answer = generate_answer(standalone_query, topk)
        else:
            # 4.4. 비과학 상식 질문: 검색 안 함 (MAP 1점 확보 전략)
            topk = [] 
            
            # 일반 대화 답변 생성 (LLM에게 요청)
            last_message = messages[-1]['content']
            try:
                # 일반 대화에는 사용자 정의 페르소나 적용 (친절하고 간결하게)
                system_persona = "당신은 파이썬 전문가이자 파이썬 초보자에게 친절하고 초등학생도 알기 쉽게 설명해주는 AI 어시스턴트입니다. 모든 말은 반말로 하고 짧고 간결하게 정리해서 보여줍니다."
                
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_persona},
                        {"role": "user", "content": last_message}
                    ]
                )
                answer = response.choices[0].message.content.strip()
            except Exception:
                answer = "죄송합니다. 일반 대화 답변 생성 중 오류가 발생했습니다."


        results.append({
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": topk, 
            "answer": answer
        })
        
    # 5. 최종 제출 파일 생성 (CSV 확장자 필수)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n--- 파이프라인 완료 ---")
    print(f"최종 결과 파일: '{output_file}'이 생성되었습니다. 이 파일을 제출하세요.")


if __name__ == "__main__":
    run_rag_pipeline()