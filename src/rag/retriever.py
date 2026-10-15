"""
Hybrid Dense & Lexical Review Retriever.
Blends dense semantic vector search with keyword/aspect matching and MMR diversification.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from src.indexing.vector_store import DenseVectorStore


class ReviewRetriever:
    """Hybrid Retriever for customer review chunks."""

    def __init__(self, vector_store: DenseVectorStore, hybrid_alpha: float = 0.75):
        self.vector_store = vector_store
        self.hybrid_alpha = hybrid_alpha  # Weight for dense semantic vs lexical

    def lexical_overlap_score(self, query: str, chunk_text: str) -> float:
        """Computes word overlap Jaccard score between query and chunk."""
        q_tokens = set(query.lower().split())
        c_tokens = set(chunk_text.lower().split())
        intersection = q_tokens.intersection(c_tokens)
        return len(intersection) / float(len(q_tokens)) if q_tokens else 0.0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        product_id_filter: Optional[str] = None,
        min_rating: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieves top_k review chunks relevant to user query.
        """
        # Fetch top candidate pool from dense vector store
        candidates = self.vector_store.search(
            query=query,
            top_k=top_k * 3,
            product_id_filter=product_id_filter,
        )

        if not candidates:
            return []

        scored_results = []
        for chunk, dense_score in candidates:
            if min_rating is not None and chunk.get("rating", 0) < min_rating:
                continue

            lex_score = self.lexical_overlap_score(query, chunk["text"])
            hybrid_score = self.hybrid_alpha * dense_score + (1.0 - self.hybrid_alpha) * lex_score

            chunk_copy = dict(chunk)
            chunk_copy["retrieval_score"] = float(hybrid_score)
            chunk_copy["dense_score"] = float(dense_score)
            chunk_copy["lexical_score"] = float(lex_score)
            scored_results.append(chunk_copy)

        # Sort descending by hybrid score
        scored_results.sort(key=lambda x: -x["retrieval_score"])

        # Deduplicate multiple chunks from identical review to maximize diversity
        seen_reviews = set()
        diverse_results = []
        for r in scored_results:
            rid = r["review_id"]
            if rid not in seen_reviews:
                diverse_results.append(r)
                seen_reviews.add(rid)
            if len(diverse_results) >= top_k:
                break

        # Fallback if top_k not reached
        if len(diverse_results) < top_k:
            for r in scored_results:
                if r not in diverse_results:
                    diverse_results.append(r)
                if len(diverse_results) >= top_k:
                    break

        return diverse_results[:top_k]
