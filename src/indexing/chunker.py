"""
Document Ingestion & Semantic Chunking Engine.
Ingests unstructured reviews and product catalog text,
splits into overlapping semantic chunks preserving sentence boundaries and metadata.
"""

import re
from typing import Dict, List, Optional


class DocumentChunker:
    """Chunks reviews into searchable passages with preserved metadata."""

    def __init__(self, chunk_size_words: int = 80, chunk_overlap_words: int = 15):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def split_sentences(self, text: str) -> List[str]:
        """Splits text on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_review(self, review: Dict) -> List[Dict]:
        """
        Chunks a single customer review into passages.
        """
        full_text = f"{review.get('title', '')}. {review.get('review_text', '')}".strip()
        sentences = self.split_sentences(full_text)

        chunks = []
        current_words = []
        current_sentences = []
        chunk_idx = 0

        for sentence in sentences:
            words = sentence.split()
            if len(current_words) + len(words) > self.chunk_size_words and current_words:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "chunk_id": f"{review['review_id']}_c{chunk_idx}",
                    "review_id": review["review_id"],
                    "product_id": review["product_id"],
                    "rating": review.get("rating", 5),
                    "verified_purchase": review.get("verified_purchase", True),
                    "aspects_mentioned": review.get("aspects_mentioned", []),
                    "text": chunk_text,
                })
                chunk_idx += 1

                # Keep overlap sentences
                overlap_words = 0
                kept_sentences = []
                for s in reversed(current_sentences):
                    s_words = len(s.split())
                    if overlap_words + s_words <= self.chunk_overlap_words:
                        kept_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                current_sentences = kept_sentences
                current_words = " ".join(current_sentences).split()

            current_sentences.append(sentence)
            current_words.extend(words)

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"{review['review_id']}_c{chunk_idx}",
                "review_id": review["review_id"],
                "product_id": review["product_id"],
                "rating": review.get("rating", 5),
                "verified_purchase": review.get("verified_purchase", True),
                "aspects_mentioned": review.get("aspects_mentioned", []),
                "text": chunk_text,
            })

        return chunks

    def chunk_all_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """Chunks an entire list of reviews."""
        all_chunks = []
        for r in reviews:
            all_chunks.extend(self.chunk_review(r))
        return all_chunks
