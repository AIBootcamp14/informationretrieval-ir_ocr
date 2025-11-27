import argparse
import json
import os
from collections import defaultdict
from typing import List

from tqdm import tqdm

from data_utils import (
    generate_hypothetical_document,
    load_documents,
    load_eval_queries,
    preprocess_queries,
)
from retriever import (
    BGEBiEncoder,
    BM25Retriever,
    load_embeddings,
    retrieve_documents,
    save_embeddings,
)
from reranker import CrossEncoderReranker


def rrf_fusion(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion."""
    scores = defaultdict(float)
    for ranks in rank_lists:
        if not ranks:
            continue
        for position, doc_idx in enumerate(ranks):
            scores[int(doc_idx)] += 1.0 / (k + position + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_idx for doc_idx, _ in sorted_docs]


def main():
    parser = argparse.ArgumentParser(description="IR Contest RAG Pipeline")
    parser.add_argument("--documents_path", type=str, default="data/documents.jsonl")
    parser.add_argument("--eval_path", type=str, default="data/eval.jsonl")
    parser.add_argument("--output_path", type=str, default="final_submission_rrf.jsonl")
    parser.add_argument(
        "--top_k_retrieval",
        type=int,
        default=50,
        help="경로별 1차 검색 후보 수",
    )
    parser.add_argument(
        "--top_k_fusion",
        type=int,
        default=100,
        help="RRF 융합 후 유지할 후보 수",
    )
    parser.add_argument(
        "--top_k_final",
        type=int,
        default=3,
        help="최종 제출할 문서 수",
    )
    parser.add_argument("--recompute_embeddings", action="store_true")
    parser.add_argument(
        "--skip_classification",
        action="store_true",
        help="과학 질문 분류 생략",
    )
    parser.add_argument(
        "--use_hyde",
        dest="use_hyde",
        action="store_true",
        default=True,
        help="HyDE 경로 사용",
    )
    parser.add_argument(
        "--no_hyde", dest="use_hyde", action="store_false", help="HyDE 비활성화"
    )
    parser.add_argument(
        "--use_bm25",
        dest="use_bm25",
        action="store_true",
        default=True,
        help="BM25 경로 사용",
    )
    parser.add_argument(
        "--no_bm25", dest="use_bm25", action="store_false", help="BM25 비활성화"
    )

    args = parser.parse_args()

    # 1. 데이터 로딩
    doc_map = load_documents(args.documents_path)
    doc_ids = list(doc_map.keys())
    doc_contents = [doc_map[doc_id]["content"] for doc_id in doc_ids]

    raw_queries = load_eval_queries(args.eval_path)

    # 2. 쿼리 전처리
    if args.skip_classification:
        processed_queries = []
        for q in raw_queries:
            msgs = [m["content"] for m in q["msg"] if m["role"] == "user"]
            query_text = msgs[-1] if msgs else ""
            processed_queries.append(
                {"eval_id": q["eval_id"], "query_text": query_text, "is_science": True}
            )
    else:
        processed_queries = preprocess_queries(raw_queries)

    target_queries = [q["query_text"] for q in processed_queries]

    # 2.5 HyDE 생성
    hyde_texts = [""] * len(processed_queries)
    if args.use_hyde:
        print("HyDE (교과서 스타일) 생성 중...")
        for idx, q in enumerate(tqdm(processed_queries, desc="HyDE Generation")):
            if q["is_science"]:
                hyde_texts[idx] = generate_hypothetical_document(q["query_text"])

    # 3. Retrieval
    print(">>> 1차 검색 (Retrieval) 시작 <<<")
    encoder = BGEBiEncoder(model_name="BAAI/bge-m3")
    emb_path = "doc_embeddings_bge_m3.pkl"
    if os.path.exists(emb_path) and not args.recompute_embeddings:
        doc_embeddings = load_embeddings(emb_path)
    else:
        doc_embeddings = encoder.encode(doc_contents)
        save_embeddings(doc_embeddings, emb_path)

    dense_indices, _ = retrieve_documents(
        encoder, target_queries, doc_embeddings, top_k=args.top_k_retrieval
    )
    dense_indices = [list(idx) for idx in dense_indices]

    # HyDE 경로
    hyde_indices = [[] for _ in processed_queries]
    if args.use_hyde:
        valid_pairs = [
            (idx, text) for idx, text in enumerate(hyde_texts) if text.strip()
        ]
        if valid_pairs:
            _, hyde_queries_only = zip(*valid_pairs)
            hyde_results, _ = retrieve_documents(
                encoder, list(hyde_queries_only), doc_embeddings, top_k=args.top_k_retrieval
            )
            for (idx, _), retrieved in zip(valid_pairs, hyde_results):
                hyde_indices[idx] = list(retrieved)

    # BM25 경로
    bm25_indices = [[] for _ in processed_queries]
    if args.use_bm25:
        print("BM25 검색 수행 중...")
        bm25 = BM25Retriever(doc_contents)
        bm25_indices = []
        for q in tqdm(target_queries, desc="BM25"):
            indices, _ = bm25.retrieve(q, top_k=args.top_k_retrieval)
            bm25_indices.append(indices)

    # 4. RRF + Re-ranking
    print(">>> 2차 재순위 (Reranking) 시작 <<<")
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")

    final_results = []
    for idx, q_info in enumerate(processed_queries):
        eval_id = q_info["eval_id"]
        query_text = q_info["query_text"]

        if not q_info["is_science"]:
            final_results.append(
                {
                    "eval_id": eval_id,
                    "standalone_query": query_text,
                    "topk": [],
                    "answer": "",
                    "references": [],
                }
            )
            continue

        rank_lists = [dense_indices[idx]]
        if args.use_hyde:
            rank_lists.append(hyde_indices[idx])
        if args.use_bm25:
            rank_lists.append(bm25_indices[idx])

        fusion_indices = rrf_fusion(rank_lists)
        fusion_indices = fusion_indices[: args.top_k_fusion]

        if not fusion_indices:
            final_results.append(
                {
                    "eval_id": eval_id,
                    "standalone_query": query_text,
                    "topk": [],
                    "answer": "",
                    "references": [],
                }
            )
            continue

        candidate_docs = [doc_contents[i] for i in fusion_indices]
        candidate_ids = [doc_ids[i] for i in fusion_indices]

        rerank_scores = reranker.rerank(query_text, candidate_docs)
        scored_candidates = sorted(
            zip(candidate_ids, candidate_docs, rerank_scores),
            key=lambda x: x[2],
            reverse=True,
        )
        final_candidates = scored_candidates[: args.top_k_final]

        final_results.append(
            {
                "eval_id": eval_id,
                "standalone_query": query_text,
                "topk": [c[0] for c in final_candidates],
                "answer": "",
                "references": [
                    {"score": float(c[2]), "content": c[1]} for c in final_candidates
                ],
            }
        )

    print(f"결과 저장 중: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in final_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("완료!")


if __name__ == "__main__":
    main()

