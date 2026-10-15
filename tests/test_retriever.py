"""Unit tests for dense vector store and hybrid retriever."""

import pytest
from src.indexing.vector_store import DenseVectorStore
from src.rag.retriever import ReviewRetriever


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "c1",
            "review_id": "REV_01",
            "product_id": "P1",
            "rating": 5,
            "verified_purchase": True,
            "text": "The battery life is fantastic, lasting over 30 hours of continuous music.",
        },
        {
            "chunk_id": "c2",
            "review_id": "REV_02",
            "product_id": "P1",
            "rating": 4,
            "verified_purchase": True,
            "text": "Active noise cancellation blocks background noise effectively on trains.",
        },
        {
            "chunk_id": "c3",
            "review_id": "REV_03",
            "product_id": "P2",
            "rating": 2,
            "verified_purchase": False,
            "text": "The shoes ran small and caused blisters after a three mile run.",
        },
    ]


def test_vector_store_search(sample_chunks):
    store = DenseVectorStore(embedding_dim=16)
    store.fit_index(sample_chunks)

    results = store.search(query="battery playback hours", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk["review_id"] == "REV_01"
    assert score > 0.0


def test_retriever_filter_and_hybrid(sample_chunks):
    store = DenseVectorStore(embedding_dim=16)
    store.fit_index(sample_chunks)
    retriever = ReviewRetriever(store, hybrid_alpha=0.7)

    # Filter by product P2
    p2_results = retriever.retrieve(query="shoes size", top_k=1, product_id_filter="P2")
    assert len(p2_results) == 1
    assert p2_results[0]["product_id"] == "P2"
    assert p2_results[0]["review_id"] == "REV_03"
