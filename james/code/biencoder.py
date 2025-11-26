import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import pickle
import os
import argparse
from typing import List, Dict, Tuple

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class QwenBiEncoder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B", use_flash_attention=False, device=None):
        print(f"Qwen Bi-Encoder 모델 로딩 중: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        model_kwargs = {}
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.device_map_auto = False
        if device is None:
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    model_kwargs["device_map"] = "auto"
                    self.device_map_auto = True
                    self.input_device = torch.device("cuda:0")
                else:
                    self.input_device = torch.device("cuda:0")
            else:
                # CPU 환경
                self.input_device = torch.device("cpu")
        else:
            self.input_device = torch.device(device)

        # 모델 로딩
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float16, **model_kwargs)
        self.model.eval() # 모델을 평가 모드로 설정
        
        self.max_length = 8192
        self.task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        
        print(f"모델 로딩 완료. 입력 텐서는 {self.input_device}로 전송됩니다.")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        all_embeddings = []
        
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="임베딩 계산 중")
        
        for batch_texts in batches:
            # 토크나이징
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            batch_dict = {k: v.to(self.input_device) for k, v in batch_dict.items()}
            
            # 임베딩 계산
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # 정규화
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
            
            if self.input_device.type == "cuda":
                torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        formatted_queries = [get_detailed_instruct(self.task_description, query) for query in queries]
        return self.encode_texts(formatted_queries, batch_size, show_progress=True)
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        return self.encode_texts(documents, batch_size, show_progress=True)
    
    def compute_similarities(self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        return np.dot(query_embeddings, doc_embeddings.T)

def load_documents(documents_path: str) -> Tuple[List[str], List[str]]:
    print(f"문서 로딩 중: {documents_path}")
    doc_ids = []
    doc_contents = []
    
    with open(documents_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="문서 로딩"):
            doc = json.loads(line.strip())
            doc_ids.append(doc['docid'])
            doc_contents.append(doc['content'])
    
    print(f"총 {len(doc_ids)}개의 문서가 로딩되었습니다.")
    return doc_ids, doc_contents

def load_eval_queries(eval_path: str) -> List[Dict]:
    print(f"평가 쿼리 로딩 중: {eval_path}")
    queries = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data = json.loads(line.strip())
            queries.append(eval_data)
    print(f"총 {len(queries)}개의 쿼리가 로딩되었습니다.")
    return queries

def extract_query_text(msg_list: List[Dict]) -> str:
    user_messages = []
    for msg in msg_list:
        if msg['role'] == 'user':
            user_messages.append(msg['content'])
    
    # 모든 사용자 메시지를 연결하여 컨텍스트 포함 쿼리 생성
    if len(user_messages) > 1:
        query = " ".join(user_messages)
    else:
        query = user_messages[0] if user_messages else ""
    
    return query

def save_embeddings(embeddings: np.ndarray, filepath: str):
    print(f"임베딩을 {filepath}에 저장 중...")
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"임베딩 저장 완료: {embeddings.shape}")

def load_embeddings(filepath: str) -> np.ndarray:
    print(f"임베딩을 {filepath}에서 로딩 중...")
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"임베딩 로딩 완료: {embeddings.shape}")
    return embeddings

def main():
    parser = argparse.ArgumentParser(description='Qwen Bi-Encoder를 사용한 문서 검색')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-8B', help='모델 이름 (기본값: Qwen/Qwen3-Embedding-8B)')
    parser.add_argument('--documents_path', type=str, default='../documents.jsonl', help='문서 JSONL 파일 경로')
    parser.add_argument('--eval_path', type=str, default='../eval.jsonl', help='평가 쿼리 JSONL 파일 경로')
    parser.add_argument('--output_path', type=str, default='biencoder_submission.jsonl', help='결과 JSONL 파일 경로')
    parser.add_argument('--scores_path', type=str, default='biencoder_scores.csv', help='유사도 점수 CSV 파일 경로 (점수는 CSV 유지)')
    parser.add_argument('--doc_embeddings_path', type=str, default='doc_embeddings.pkl', help='문서 임베딩 저장 경로')
    parser.add_argument('--top_k', type=int, default=3, help='상위 K개 문서 선택 (기본값: 3)')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기 (기본값: 32)')
    parser.add_argument('--use_flash_attention', action='store_true', help='Flash Attention 2 사용')
    parser.add_argument('--recompute_embeddings', action='store_true', help='문서 임베딩 재계산 강제')
    
    args = parser.parse_args()
    
    print("=== Qwen3-Embedding-8B Bi-Encoder 시스템 ===")
    
    # 문서와 쿼리 로딩
    doc_ids, doc_contents = load_documents(args.documents_path)
    queries = load_eval_queries(args.eval_path)
    
    # Bi-Encoder 초기화
    encoder = QwenBiEncoder(model_name=args.model_name, use_flash_attention=args.use_flash_attention)
    
    # 문서 임베딩 계산 또는 로딩
    if os.path.exists(args.doc_embeddings_path) and not args.recompute_embeddings:
        print("기존 문서 임베딩을 사용합니다.")
        doc_embeddings = load_embeddings(args.doc_embeddings_path)
    else:
        print("문서 임베딩을 새로 계산합니다.")
        doc_embeddings = encoder.encode_documents(doc_contents, batch_size=args.batch_size)
        save_embeddings(doc_embeddings, args.doc_embeddings_path)
    
    # 쿼리 처리
    print("\n쿼리 임베딩 계산 중...")
    query_texts = []
    eval_ids = []
    
    for query_data in queries:
        eval_id = query_data['eval_id']
        query_text = extract_query_text(query_data['msg'])
        
        eval_ids.append(eval_id)
        query_texts.append(query_text)
    
    # 쿼리 임베딩 계산
    query_embeddings = encoder.encode_queries(query_texts, batch_size=args.batch_size)
    
    # 유사도 계산
    print("유사도 계산 중...")
    similarities = encoder.compute_similarities(query_embeddings, doc_embeddings)
    
    # 결과 생성 - JSONL 형식으로 변경
    print("결과 생성 중...")
    submission_results = []
    score_results = []
    
    for i, (eval_id, query_text) in enumerate(tqdm(zip(eval_ids, query_texts), desc="결과 처리")):
        # 해당 쿼리의 유사도 점수
        query_scores = similarities[i]
        
        # 점수와 문서 ID를 쌍으로 만들어 정렬
        doc_score_pairs = list(zip(doc_ids, query_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Top-K 문서 선택
        top_k_docs = doc_score_pairs[:args.top_k]
        
        # 제출용 결과 생성 - JSONL 형식
        topk_doc_ids = [doc_id for doc_id, score in top_k_docs]
        references = [{"score": float(score), "content": doc_contents[doc_ids.index(doc_id)]} for doc_id, score in top_k_docs]
        
        submission_results.append({
            "eval_id": eval_id,
            "standalone_query": query_text,
            "topk": topk_doc_ids,
            "answer": "",  # Bi-encoder는 답변을 생성하지 않으므로 빈 문자열
            "references": references
        })
        
        # 점수 결과 저장 (Top-10만)
        for rank, (doc_id, score) in enumerate(doc_score_pairs[:10], 1):
            score_results.append({
                'eval_id': eval_id,
                'query': query_text,
                'doc_id': doc_id,
                'score': float(score),
                'rank': rank
            })
    
    # 결과를 JSONL로 저장
    print(f"\n결과를 {args.output_path}에 저장 중...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for result in submission_results:
            f.write(f'{json.dumps(result, ensure_ascii=False)}\n')
    
    print(f"유사도 점수를 {args.scores_path}에 저장 중...")
    scores_df = pd.DataFrame(score_results)
    scores_df.to_csv(args.scores_path, index=False)
    
    print("\n=== 완료 ===")
    print(f"총 {len(submission_results)}개의 쿼리에 대한 결과가 생성되었습니다.")
    print(f"총 {len(score_results)}개의 점수가 저장되었습니다.")
    print(f"쿼리당 {args.top_k}개 문서 추천")
    
    # 성능 통계
    scores_df = pd.DataFrame(score_results)
    if not scores_df.empty:
        avg_top1_score = scores_df[scores_df['rank'] == 1]['score'].mean()
        print(f"평균 Top-1 유사도 점수: {avg_top1_score:.4f}")

if __name__ == "__main__":
    main()
    