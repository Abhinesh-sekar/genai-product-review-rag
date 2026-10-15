"""Unit tests for retrieval Hit@K and answer faithfulness evaluation."""

import pytest
from src.evaluation.retrieval_eval import evaluate_retrieval
from src.evaluation.faithfulness_eval import FaithfulnessEvaluator


class DummyRetriever:
    def retrieve(self, query, top_k=5, product_id_filter=None, min_rating=None):
        return [
            {"review_id": "REV_01", "text": "Battery lasts 30 hours continuously."},
            {"review_id": "REV_02", "text": "Great noise cancellation mode."},
        ]


def test_retrieval_evaluation():
    retriever = DummyRetriever()
    queries = [
        {
            "query": "How long is battery?",
            "product_id": "P1",
            "ground_truth_relevant_review_ids": ["REV_01"],
        },
        {
            "query": "Noise cancellation quality?",
            "product_id": "P1",
            "ground_truth_relevant_review_ids": ["REV_99"],  # Miss
        },
    ]

    metrics = evaluate_retrieval(queries, retriever, top_k_values=[1, 3])
    # Query 1 has REV_01 at rank 1 (hit@1 = 1, hit@3 = 1, rr = 1.0)
    # Query 2 misses REV_99 (hit@1 = 0, hit@3 = 0, rr = 0.0)
    assert metrics["hit@1"] == pytest.approx(0.5)
    assert metrics["hit@3"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)


def test_faithfulness_evaluator():
    evaluator = FaithfulnessEvaluator()
    chunks = [
        {"review_id": "REV_01", "text": "The battery delivers 30 hours of continuous music playback with ANC active."}
    ]

    # Faithful answer
    ans_faithful = "The battery delivers 30 hours of continuous music playback [Review #REV_01]."
    res_f = evaluator.evaluate_answer_faithfulness("battery life", ans_faithful, chunks)
    assert res_f["faithfulness_score"] == 1.0
    assert res_f["supported_claims"] == 1

    # Hallucinated answer
    ans_hallucinated = "The battery lasts for three whole weeks and supports wireless solar charging."
    res_h = evaluator.evaluate_answer_faithfulness("battery life", ans_hallucinated, chunks)
    assert res_h["faithfulness_score"] == 0.0
    assert res_h["hallucinated_claims"] == 1
