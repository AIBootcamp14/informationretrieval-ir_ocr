import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from typing import List, Tuple, Dict

from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, doc_contents: List[str], model_name="bert-base-multilingual-cased"):
        print("BM25 인덱싱 시작...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenized_docs = [self.tokenize(doc) for doc in tqdm(doc_contents, desc="토큰화")]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25 인덱싱 완료!")
        
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
        
    def retrieve(self, query: str, top_k: int = 50) -> Tuple[List[int], List[float]]:
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Top-K 추출
        indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indices]
        
        return indices.tolist(), top_scores.tolist()

class BGEBiEncoder:
    def __init__(self, model_name="BAAI/bge-m3", device=None):
        print(f"Bi-Encoder 모델 로딩 중: {model_name}")
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"모델 로딩 완료. 디바이스: {self.device}")
        
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 512, show_progress: bool = True) -> np.ndarray:
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="임베딩 계산")
            
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # BGE-M3는 CLS 토큰을 사용
                embeddings = outputs.last_hidden_state[:, 0]
                # 정규화
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

def save_embeddings(embeddings: np.ndarray, filepath: str):
    print(f"임베딩 저장 중: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filepath: str) -> np.ndarray:
    print(f"임베딩 로딩 중: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def retrieve_documents(encoder: BGEBiEncoder, 
                      queries: List[str], 
                      doc_embeddings: np.ndarray, 
                      top_k: int = 100, 
                      batch_size: int = 32) -> Tuple[List[List[int]], List[List[float]]]:
    """
    쿼리에 대해 관련 문서 검색 (Brute-force Cosine Similarity)
    반환: (indices, scores)
    """
    print(f"쿼리 {len(queries)}개에 대한 검색 수행 (Top-{top_k})...")
    
    # 쿼리 임베딩
    query_embeddings = encoder.encode(queries, batch_size=batch_size, show_progress=True)
    
    # 유사도 계산 (Matrix Multiplication)
    # query: (Q, D), doc: (N, D) -> (Q, N)
    similarities = np.dot(query_embeddings, doc_embeddings.T)
    
    # Top-K 추출
    top_indices = []
    top_scores = []
    
    print("Top-K 정렬 중...")
    for score_row in tqdm(similarities, desc="Ranking"):
        # 상위 K개 인덱스 추출 (주의: argpartition은 정렬되지 않음, 뒤에서 정렬 필요)
        # 데이터가 작으므로(4200개) 그냥 argsort해도 빠름
        indices = np.argsort(score_row)[::-1][:top_k]
        scores = score_row[indices]
        
        top_indices.append(indices)
        top_scores.append(scores)
        
    return top_indices, top_scores


