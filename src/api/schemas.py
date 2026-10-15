"""
Pydantic v2 Data Models for FastAPI GenAI Review Assistant.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., description="User shopping question about a product or feature")
    product_id: Optional[str] = Field(None, description="Optional product ID to filter reviews (e.g. PROD_001)")
    top_k: int = Field(5, ge=1, le=10, description="Number of review chunks to retrieve")
    min_rating: Optional[int] = Field(None, ge=1, le=5, description="Filter by minimum star rating")


class RetrievedReviewChunk(BaseModel):
    chunk_id: str
    review_id: str
    product_id: str
    rating: int
    verified_purchase: bool
    text: str
    retrieval_score: float


class AskResponse(BaseModel):
    query: str
    product_id: Optional[str]
    answer: str
    cited_review_ids: List[str]
    retrieved_reviews: List[RetrievedReviewChunk]
    grounded: bool
    confidence_score: float


class SearchRequest(BaseModel):
    query: str
    product_id: Optional[str] = None
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[RetrievedReviewChunk]


class EvalResponse(BaseModel):
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    context_precision_at_5: float
    answer_faithfulness: float
    n_queries_evaluated: int
