"""
Answer Faithfulness & Grounding Verification Module.
Validates that generated answers are strictly faithful to retrieved customer review context,
computing Faithfulness Rate and Hallucination Risk.
"""

import re
from typing import Dict, List, Set, Tuple
import numpy as np


class FaithfulnessEvaluator:
    """Verifies grounding and claim substantiation for RAG responses."""

    def __init__(self, citation_regex: str = r"\[Review #(REV_\d+)\]"):
        self.citation_pattern = re.compile(citation_regex)

    def extract_claims(self, answer_text: str) -> List[str]:
        """Splits answer into individual factual sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        return [s.strip() for s in sentences if len(s.strip().split()) >= 4]

    def verify_claim_grounding(
        self,
        claim: str,
        retrieved_chunks: List[Dict],
    ) -> Tuple[bool, float, List[str]]:
        """
        Verifies if a claim is supported by the retrieved context.
        Returns (is_supported, support_score, matched_review_ids).
        """
        # Extract explicit citations from the claim if present
        cited_in_claim = self.citation_pattern.findall(claim)

        # Normalize claim words (remove stopwords and citations)
        clean_claim = self.citation_pattern.sub("", claim).lower()
        claim_words = set(re.findall(r"\b[a-z]{3,}\b", clean_claim))

        if not claim_words:
            return True, 1.0, cited_in_claim

        best_overlap_ratio = 0.0
        supporting_rids = []

        for chunk in retrieved_chunks:
            rid = chunk["review_id"]
            chunk_words = set(re.findall(r"\b[a-z]{3,}\b", chunk["text"].lower()))
            overlap = len(claim_words.intersection(chunk_words))
            ratio = overlap / float(len(claim_words))

            # If explicitly cited, give strong verification boost
            if rid in cited_in_claim:
                ratio = min(1.0, ratio * 1.35)

            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio

            if ratio >= 0.40:
                supporting_rids.append(rid)

        is_supported = best_overlap_ratio >= 0.40
        return is_supported, float(best_overlap_ratio), supporting_rids

    def evaluate_answer_faithfulness(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict],
    ) -> Dict:
        """
        Evaluates the faithfulness of a generated answer.
        """
        claims = self.extract_claims(answer)
        if not claims:
            return {
                "faithfulness_score": 1.0,
                "n_claims": 0,
                "supported_claims": 0,
                "hallucinated_claims": 0,
            }

        supported_count = 0
        claim_details = []

        for claim in claims:
            supported, score, rids = self.verify_claim_grounding(claim, retrieved_chunks)
            if supported:
                supported_count += 1
            claim_details.append({
                "claim": claim,
                "supported": supported,
                "support_score": round(score, 3),
                "supporting_reviews": rids,
            })

        faithfulness = supported_count / float(len(claims))
        return {
            "faithfulness_score": float(faithfulness),
            "n_claims": len(claims),
            "supported_claims": supported_count,
            "hallucinated_claims": len(claims) - supported_count,
            "details": claim_details,
        }

    def evaluate_benchmark(
        self,
        qa_pairs: List[Dict],
    ) -> Dict[str, float]:
        """
        Computes average faithfulness across a batch of evaluated QA responses.
        """
        scores = []
        for pair in qa_pairs:
            res = self.evaluate_answer_faithfulness(
                query=pair["query"],
                answer=pair["answer"],
                retrieved_chunks=pair["retrieved_chunks"],
            )
            scores.append(res["faithfulness_score"])

        return {
            "mean_faithfulness": float(np.mean(scores)),
            "n_evaluated_responses": len(qa_pairs),
        }
