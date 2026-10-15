"""
FastAPI Microservice for GenAI Product & Review Assistant (RAG).
Provides endpoints for conversational grounded Q&A, semantic review search, and telemetry evaluation.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.indexing.chunker import DocumentChunker
from src.indexing.vector_store import DenseVectorStore
from src.rag.retriever import ReviewRetriever
from src.rag.generator import GroundedAnswerGenerator
from src.evaluation.retrieval_eval import evaluate_retrieval
from src.evaluation.faithfulness_eval import FaithfulnessEvaluator
from src.api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse, EvalResponse

# Global singletons
state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load datasets and build vector index
    data_dir = os.environ.get("DATA_DIR", "data/raw")
    reviews_file = os.path.join(data_dir, "reviews.json")
    products_file = os.path.join(data_dir, "products.json")
    eval_file = os.path.join(data_dir, "golden_evaluation_set.json")

    print("[FastAPI] Initializing RAG Knowledge Base...")
    with open(reviews_file, "r") as f:
        reviews = json.load(f)
    with open(products_file, "r") as f:
        products = json.load(f)
    with open(eval_file, "r") as f:
        eval_queries = json.load(f)

    prod_map = {p["product_id"]: p["name"] for p in products}

    chunker = DocumentChunker(chunk_size_words=80, chunk_overlap_words=15)
    chunks = chunker.chunk_all_reviews(reviews)

    vector_store = DenseVectorStore(embedding_dim=128)
    vector_store.fit_index(chunks)

    retriever = ReviewRetriever(vector_store, hybrid_alpha=0.75)
    generator = GroundedAnswerGenerator()
    faithfulness_eval = FaithfulnessEvaluator()

    state["reviews"] = reviews
    state["products"] = prod_map
    state["eval_queries"] = eval_queries
    state["retriever"] = retriever
    state["generator"] = generator
    state["faithfulness_eval"] = faithfulness_eval
    print(f"[FastAPI] Knowledge Base Ready: {len(products)} products, {len(reviews)} reviews, {len(chunks)} chunks.")
    yield
    state.clear()


app = FastAPI(
    title="GenAI Product & Review Assistant API",
    description="RAG microservice for answering e-commerce shopper questions grounded in verified reviews with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "genai-product-review-rag",
        "version": "1.0.0",
        "indexed_products": len(state.get("products", {})),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    retriever = state.get("retriever")
    generator = state.get("generator")
    products = state.get("products", {})

    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    prod_name = products.get(req.product_id, "") if req.product_id else ""
    chunks = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        product_id_filter=req.product_id,
        min_rating=req.min_rating,
    )

    result = generator.generate_answer(
        query=req.query,
        retrieved_chunks=chunks,
        product_name=prod_name,
    )

    return AskResponse(
        query=req.query,
        product_id=req.product_id,
        answer=result["answer"],
        cited_review_ids=result["cited_review_ids"],
        retrieved_reviews=chunks,
        grounded=result["grounded"],
        confidence_score=result["confidence_score"],
    )


@app.post("/search", response_model=SearchResponse)
def search_reviews(req: SearchRequest):
    retriever = state.get("retriever")
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    chunks = retriever.retrieve(query=req.query, top_k=req.top_k, product_id_filter=req.product_id)
    return SearchResponse(
        query=req.query,
        total_results=len(chunks),
        results=chunks,
    )


@app.get("/evaluate", response_model=EvalResponse)
def evaluate_rag():
    retriever = state.get("retriever")
    generator = state.get("generator")
    eval_evaluator = state.get("faithfulness_eval")
    eval_queries = state.get("eval_queries", [])

    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    # 1. Retrieval Metrics
    retrieval_res = evaluate_retrieval(eval_queries, retriever, top_k_values=[1, 3, 5])

    # 2. Faithfulness across sample of queries
    evaluated_pairs = []
    for item in eval_queries[:25]:
        chunks = retriever.retrieve(query=item["query"], top_k=5, product_id_filter=item["product_id"])
        ans_res = generator.generate_answer(item["query"], chunks, item.get("product_name", ""))
        evaluated_pairs.append({
            "query": item["query"],
            "answer": ans_res["answer"],
            "retrieved_chunks": chunks,
        })

    faith_res = eval_evaluator.evaluate_benchmark(evaluated_pairs)

    return EvalResponse(
        hit_at_1=retrieval_res["hit@1"],
        hit_at_3=retrieval_res["hit@3"],
        hit_at_5=retrieval_res["hit@5"],
        mrr=retrieval_res["mrr"],
        context_precision_at_5=retrieval_res["context_precision@5"],
        answer_faithfulness=faith_res["mean_faithfulness"],
        n_queries_evaluated=len(eval_queries),
    )
