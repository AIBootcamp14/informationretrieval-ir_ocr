# scripts/index_data.py

from elasticsearch import Elasticsearch
from tqdm import tqdm
import json
import config

# ES 연결 헬퍼 함수
def connect_es():
    """Elasticsearch에 연결합니다."""
    es_client = Elasticsearch(
        hosts=[{"host": config.ES_HOST, "port": config.ES_PORT, "scheme": "http"}],
        basic_auth=(config.ES_USERNAME, config.ES_PASSWORD),
        request_timeout=30 
    )
    if not es_client.ping():
        # 연결 실패 시 즉시 중단
        raise ConnectionError("Elasticsearch 연결 실패. config.py 또는 .env 설정을 확인하세요.")
    return es_client

def create_index(es_client):
    """인덱스를 생성하고 매핑을 설정합니다."""
    # MAP 점수 최대화를 위해, 키워드 검색(docid, src)과 전문 검색(content) 매핑 분리
    mapping = {
        "properties": {
            "docid": {"type": "keyword"},
            "src": {"type": "keyword"},
            "content": {
                "type": "text", 
                "analyzer": "standard" # 표준 분석기 사용 (Nori 불필요 시)
            } 
        }
    }
    
    if es_client.indices.exists(index=config.ES_INDEX_NAME):
        es_client.indices.delete(index=config.ES_INDEX_NAME)

    es_client.indices.create(index=config.ES_INDEX_NAME, mappings=mapping)
    print(f"인덱스 '{config.ES_INDEX_NAME}' 생성 완료.")

def index_documents(es_client, file_path="data/documents.jsonl"):
    """JSONL 파일의 문서를 Elasticsearch에 색인합니다."""
    print(f"'{file_path}' 파일에서 문서 로드 및 색인 시작...")
    
    actions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                actions.append({
                    "_index": config.ES_INDEX_NAME,
                    "_id": doc["docid"],
                    "_source": doc
                })
            except json.JSONDecodeError:
                continue

    # 대규모 데이터 처리를 위한 Bulk API 사용 (성능 최적화)
    from elasticsearch.helpers import bulk
    success, failed = bulk(es_client, actions, raise_on_error=False, request_timeout=60)
    
    if failed:
         print(f"경고: {len(failed)}개의 문서 색인 실패.")
         # 현업이라면 실패 원인 로깅 (여기선 생략)
    
    print(f"총 {len(actions)}개 중 {success}개 문서 색인 성공.")


if __name__ == "__main__":
    try:
        es_client = connect_es()
        create_index(es_client)
        index_documents(es_client)
    except ConnectionError as e:
        print(f"오류: {e}. 데이터 색인 실패.")