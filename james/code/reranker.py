import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device=None):
        print(f"Reranker 모델 로딩 중: {model_name}")
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Reranker 로딩 완료. 디바이스: {self.device}")
        
    def rerank(self, query: str, docs: List[str], batch_size: int = 16) -> List[float]:
        """
        하나의 쿼리와 여러 문서에 대한 연관성 점수 계산
        """
        pairs = [[query, doc] for doc in docs]
        scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Cross-Encoder는 logits를 점수로 사용 (Sigmoid 적용 전)
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.view(-1).float()
                scores.extend(batch_scores.cpu().numpy().tolist())
                
        return scores

    def rerank_batch(self, queries: List[str], doc_lists: List[List[str]], batch_size: int = 16) -> List[List[float]]:
        """
        여러 쿼리에 대해 각각의 문서 리스트 점수 계산
        """
        all_scores = []
        for q, docs in tqdm(zip(queries, doc_lists), total=len(queries), desc="Reranking"):
            scores = self.rerank(q, docs, batch_size=batch_size)
            all_scores.append(scores)
        return all_scores
