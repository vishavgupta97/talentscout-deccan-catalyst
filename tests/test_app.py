import httpx
import pytest

from app.app import app


@pytest.mark.anyio
async def test_home_page_loads():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "TalentScout" in response.text


@pytest.mark.anyio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_pipeline_api_returns_ranked_candidates():
    payload = {
        "title": "Senior AI Engineer",
        "company": "Deccan AI Labs",
        "location": "Remote - India",
        "employment_type": "Full-time",
        "description": (
            "We are hiring a Senior AI Engineer with 4+ years experience in Python, FastAPI, "
            "LLM systems, LangGraph, RAG, Postgres, and remote collaboration."
        ),
        "top_k": 4,
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job"]["title"] == "Senior AI Engineer"
    assert len(data["ranked_candidates"]) == 4
    assert data["ranked_candidates"][0]["final_score"] >= data["ranked_candidates"][-1]["final_score"]
