"""
Retrieval Evaluation Module.
Measures Hit@K (Hit@1, Hit@3, Hit@5), Mean Reciprocal Rank (MRR),
and Context Precision across golden benchmark test queries.
"""

from typing import Dict, List, Set
import numpy as np


def evaluate_retrieval(
    queries: List[Dict],
    retriever,
    top_k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Evaluates retriever over a list of benchmark query dicts:
    each query dict has:
    - query: str
    - product_id: str
    - ground_truth_relevant_review_ids: List[str]
    """
    max_k = max(top_k_values)
    hit_records = {k: [] for k in top_k_values}
    reciprocal_ranks = []
    precisions = []

    for item in queries:
        q = item["query"]
        pid = item.get("product_id")
        gold_rids: Set[str] = set(item.get("ground_truth_relevant_review_ids", []))

        # Retrieve top max_k
        retrieved_chunks = retriever.retrieve(query=q, top_k=max_k, product_id_filter=pid)
        retrieved_rids = [c["review_id"] for c in retrieved_chunks]

        # 1. Hit@K
        for k in top_k_values:
            top_k_rids = set(retrieved_rids[:k])
            hit = 1.0 if top_k_rids.intersection(gold_rids) else 0.0
            hit_records[k].append(hit)

        # 2. MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank_pos, rid in enumerate(retrieved_rids, 1):
            if rid in gold_rids:
                rr = 1.0 / rank_pos
                break
        reciprocal_ranks.append(rr)

        # 3. Context Precision at max_k
        top_k_rids_all = set(retrieved_rids[:max_k])
        matches = len(top_k_rids_all.intersection(gold_rids))
        precisions.append(matches / float(max_k) if max_k > 0 else 0.0)

    summary = {
        f"hit@{k}": float(np.mean(hit_records[k])) for k in top_k_values
    }
    summary["mrr"] = float(np.mean(reciprocal_ranks))
    summary["context_precision@5"] = float(np.mean(precisions))
    summary["n_queries_evaluated"] = len(queries)
    return summary
