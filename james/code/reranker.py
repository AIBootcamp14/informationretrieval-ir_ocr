import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output

class QwenReranker:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-8B", use_flash_attention=False, device=None):
        print(f"Qwen Reranker 모델 로딩 중: {model_name}")
        
        # 토크나이저 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # 모델 로딩
        if use_flash_attention:
            # device_map="auto"를 사용하면 모델이 여러 GPU에 자동으로 분산됩니다.
            # 이때는 모델 전체를 특정 device로 다시 옮기지 않습니다.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                attn_implementation="flash_attention_2",
                device_map="auto"
            ).eval()
            # 입력 텐서를 배치할 기본 장치 설정. 보통 cuda:0이 됩니다.
            # model.device를 사용하면 분산 모델의 경우 'cpu'를 반환할 수 있으므로, 명시적으로 설정
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"모델이 'auto' device_map으로 로딩되어 여러 GPU에 분산되었습니다. 입력은 {self.device}로 이동됩니다.")
        else:
            # Flash Attention을 사용하지 않는 경우, 모델을 지정된 단일 장치로 옮깁니다.
            self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
            # 디바이스 설정
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model = self.model.to(self.device) # 모델을 단일 장치로 이동
            print(f"모델이 {self.device}로 로딩되었습니다.")
        
        # 토큰 ID 설정
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        # 프리픽스와 서픽스 설정
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
            
    def process_inputs(self, pairs):
        """입력 텍스트들을 토크나이즈하고 처리"""
        # attention_mask도 함께 반환하도록 설정
        inputs = self.tokenizer(
            pairs, 
            padding=False, # initial padding=False
            truncation='longest_first',
            return_attention_mask=True, # attention_mask를 반환하도록 설정
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # 프리픽스와 서픽스 추가 및 attention_mask 업데이트
        for i in range(len(inputs['input_ids'])):
            original_input_ids = inputs['input_ids'][i]
            original_attention_mask = inputs['attention_mask'][i]

            inputs['input_ids'][i] = self.prefix_tokens + original_input_ids + self.suffix_tokens
            # 새로 추가된 토큰에 맞춰 attention_mask 갱신
            inputs['attention_mask'][i] = [1] * len(self.prefix_tokens) + original_attention_mask + [1] * len(self.suffix_tokens)
        
        # 최종 패딩 (input_ids와 attention_mask 모두 처리)
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def compute_logits(self, inputs):
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rerank_documents(self, query, documents, batch_size=32, task=None):
        if task is None:
            task = 'Given a web search query, retrieve relevant passages that answer the query'
        
        # 모든 쿼리-문서 쌍을 형식화
        pairs = [format_instruction(task, query, doc) for doc in documents]
        
        all_scores = []
        
        # 배치 단위로 처리
        for i in tqdm(range(0, len(pairs), batch_size), desc="문서 재순위 매기는 중"):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self.process_inputs(batch_pairs)
            scores = self.compute_logits(inputs)
            all_scores.extend(scores)
        
        return all_scores

def load_documents(documents_path):
    print(f"문서 로딩 중: {documents_path}")
    documents = {}
    with open(documents_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents[doc['docid']] = doc['content']
    print(f"총 {len(documents)}개의 문서가 로딩되었습니다.")
    return documents

def load_eval_queries(eval_path):
    print(f"평가 쿼리 로딩 중: {eval_path}")
    queries = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data = json.loads(line.strip())
            queries.append(eval_data)
    print(f"총 {len(queries)}개의 쿼리가 로딩되었습니다.")
    return queries

def extract_query_text(msg_list):
    user_messages = []
    for msg in msg_list:
        if msg['role'] == 'user':
            user_messages.append(msg['content'])
    
    # 모든 사용자 메시지를 연결하여 컨텍스트 포함 쿼리 생성
    if len(user_messages) > 1:
        # 멀티턴인 경우 전체 대화 컨텍스트를 포함
        query = " ".join(user_messages)
    else:
        # 싱글턴인 경우 해당 쿼리만 사용
        query = user_messages[0] if user_messages else ""
    
    return query

def main():
    parser = argparse.ArgumentParser(description='Qwen Reranker를 사용한 문서 재순위 매기기')
    parser.add_argument('--documents_path', type=str, required=True, help='문서 JSONL 파일 경로')
    parser.add_argument('--eval_path', type=str, required=True, help='평가 쿼리 JSONL 파일 경로')
    parser.add_argument('--output_path', type=str, required=True, help='결과 JSONL 파일 경로')
    parser.add_argument('--scores_path', type=str, required=True, help='유사도 점수 CSV 파일 경로')
    parser.add_argument('--top_k', type=int, default=3, help='상위 K개 문서 선택 (기본값: 3)')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기 (기본값: 32)')
    parser.add_argument('--use_flash_attention', action='store_true', help='Flash Attention 2 사용')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-Reranker-8B", help='모델 이름 (기본값: Qwen/Qwen3-Reranker-8B)')
    
    args = parser.parse_args()
    
    # 문서와 쿼리 로딩
    documents = load_documents(args.documents_path)
    queries = load_eval_queries(args.eval_path)
    
    # Reranker 초기화
    reranker = QwenReranker(
        model_name=args.model_name,
        use_flash_attention=args.use_flash_attention
    )
    
    # 결과 저장용 리스트 - JSONL 형식으로 변경
    submission_results = []
    score_results = []
    
    # 각 쿼리에 대해 재순위 매기기 수행
    for query_data in tqdm(queries, desc="쿼리 처리 중"):
        eval_id = query_data['eval_id']
        query_text = extract_query_text(query_data['msg'])
        
        print(f"\n쿼리 {eval_id} 처리 중: {query_text}")
        
        # 모든 문서에 대해 유사도 계산
        doc_ids = list(documents.keys())
        doc_contents = [documents[doc_id] for doc_id in doc_ids]
        
        scores = reranker.rerank_documents(query_text, doc_contents, batch_size=args.batch_size)
        
        # 점수와 문서 ID를 쌍으로 만들어 정렬
        doc_score_pairs = list(zip(doc_ids, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Top-K 문서 선택
        top_k_docs = doc_score_pairs[:args.top_k]
        
        # 제출용 결과 생성 - JSONL 형식
        topk_doc_ids = [doc_id for doc_id, score in top_k_docs]
        references = [{"score": score, "content": documents[doc_id]} for doc_id, score in top_k_docs]
        
        submission_results.append({
            "eval_id": eval_id,
            "standalone_query": query_text,
            "topk": topk_doc_ids,
            "answer": "",  # Reranker는 답변을 생성하지 않으므로 빈 문자열
            "references": references
        })
        
        # 점수 결과 저장 (모든 문서의 점수)
        for doc_id, score in doc_score_pairs:
            score_results.append({
                'eval_id': eval_id,
                'query': query_text,
                'doc_id': doc_id,
                'score': score
            })
    
    # 결과를 JSONL로 저장
    print(f"\n결과를 {args.output_path}에 저장 중...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for result in submission_results:
            f.write(f'{json.dumps(result, ensure_ascii=False)}\n')
    
    print(f"유사도 점수를 {args.scores_path}에 저장 중...")
    scores_df = pd.DataFrame(score_results)
    scores_df.to_csv(args.scores_path, index=False)
    
    print("완료!")
    print(f"총 {len(submission_results)}개의 쿼리에 대한 결과가 생성되었습니다.")
    print(f"총 {len(score_results)}개의 점수가 저장되었습니다.")


if __name__ == "__main__":
    main()