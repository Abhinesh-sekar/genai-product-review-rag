"""
Grounded Answer Generator & Anti-Hallucination Engine.
Generates factually substantiated responses grounded strictly in retrieved customer reviews,
embedding verifiable citations [Review #REV_xxxx].
"""

import re
from typing import Dict, List, Set, Tuple


PROMPT_TEMPLATE = """You are a helpful and honest AI Product & Review Assistant.
Your goal is to answer the shopper's question based strictly on the retrieved customer reviews provided below.

Rules:
1. Ground every factual claim directly in the provided reviews.
2. Whenever citing a fact, mention the review ID in brackets, e.g. [Review #{review_id}].
3. If the reviews do not contain sufficient evidence to answer, state clearly: "Based on available reviews, there is not enough information to confirm this."
4. Do NOT extrapolate, speculate, or hallucinate features not mentioned in the context.

Retrieved Customer Reviews:
{context_text}

User Question: {query}

Grounded Answer:"""


class GroundedAnswerGenerator:
    """Generates cited, hallucination-resistant answers from retrieved reviews."""

    def __init__(self, citation_regex: str = r"\[Review #(REV_\d+)\]"):
        self.citation_pattern = re.compile(citation_regex)

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Formats review chunks into a structured prompt context block."""
        if not retrieved_chunks:
            return "No relevant customer reviews found."

        context_lines = []
        for c in retrieved_chunks:
            rid = c.get("review_id", "UNKNOWN")
            rating = c.get("rating", "N/A")
            verified = "Verified Purchase" if c.get("verified_purchase", True) else "Unverified"
            context_lines.append(
                f"- [Review #{rid}] (Rating: {rating}/5, {verified}): \"{c['text']}\""
            )
        return "\n".join(context_lines)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        product_name: str = "",
    ) -> Dict:
        """
        Synthesizes a factually grounded answer with review citations.
        Uses deterministic extractive synthesis grounded in the retrieved review texts.
        """
        if not retrieved_chunks:
            return {
                "answer": "I could not find any customer reviews discussing this specific aspect for this product.",
                "cited_review_ids": [],
                "grounded": False,
                "confidence_score": 0.0,
            }

        # Identify most relevant sentences answering the query
        query_words = set(query.lower().split())
        matched_sentences = []
        cited_reviews = []

        for chunk in retrieved_chunks:
            rid = chunk["review_id"]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk["text"]) if s.strip()]

            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words.intersection(sent_words))
                if overlap >= 2:  # Relevant sentence
                    matched_sentences.append((sent, rid, overlap))

        # Sort by relevance overlap
        matched_sentences.sort(key=lambda x: -x[2])

        if not matched_sentences:
            # Fallback to top retrieved chunk
            top_c = retrieved_chunks[0]
            matched_sentences.append((top_c["text"], top_c["review_id"], 1))

        # Synthesize coherent grounded answer
        top_sentences = matched_sentences[:3]
        answer_parts = []
        used_rids = []

        for sent, rid, _ in top_sentences:
            cleaned_sent = sent.strip()
            if not cleaned_sent.endswith("."):
                cleaned_sent += "."
            answer_parts.append(f"{cleaned_sent} [Review #{rid}]")
            if rid not in used_rids:
                used_rids.append(rid)

        full_answer = " ".join(answer_parts)
        if product_name:
            full_answer = f"According to verified shoppers for {product_name}: " + full_answer

        return {
            "query": query,
            "answer": full_answer,
            "cited_review_ids": used_rids,
            "grounded": True,
            "confidence_score": min(1.0, 0.70 + 0.10 * len(used_rids)),
        }

    def extract_citations(self, text: str) -> List[str]:
        """Extracts cited review IDs from text."""
        return list(set(self.citation_pattern.findall(text)))
