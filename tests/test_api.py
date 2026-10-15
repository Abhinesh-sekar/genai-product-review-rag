"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["indexed_products"] == 50


def test_search_endpoint(client):
    payload = {"query": "battery life", "product_id": "PROD_001", "top_k": 3}
    res = client.post("/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 3


def test_ask_endpoint(client):
    payload = {
        "query": "Does the battery last long on flights?",
        "product_id": "PROD_001",
        "top_k": 3,
    }
    res = client.post("/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "cited_review_ids" in data
    assert data["grounded"] is True
