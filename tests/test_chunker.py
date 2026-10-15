"""Unit tests for document chunker."""

import pytest
from src.indexing.chunker import DocumentChunker


def test_chunker_basic():
    chunker = DocumentChunker(chunk_size_words=20, chunk_overlap_words=5)
    review = {
        "review_id": "REV_001",
        "product_id": "PROD_001",
        "rating": 5,
        "verified_purchase": True,
        "title": "Great sound",
        "review_text": "The battery lasts for 30 hours. Active noise cancellation is superb. Very comfortable memory foam earcups.",
    }
    chunks = chunker.chunk_review(review)
    assert len(chunks) >= 1
    assert chunks[0]["review_id"] == "REV_001"
    assert chunks[0]["product_id"] == "PROD_001"
    assert "battery" in chunks[0]["text"].lower()


def test_chunk_all_reviews():
    chunker = DocumentChunker(chunk_size_words=30)
    reviews = [
        {"review_id": "R1", "product_id": "P1", "rating": 5, "title": "T1", "review_text": "Text one."},
        {"review_id": "R2", "product_id": "P2", "rating": 4, "title": "T2", "review_text": "Text two."},
    ]
    all_chunks = chunker.chunk_all_reviews(reviews)
    assert len(all_chunks) == 2
