"""One end-to-end check of /claims/review against the real OpenAI API and the real index.

Skipped by default. Run it with: pytest -m live tests/test_review_live.py
It needs OPENAI_API_KEY and a built index (python -m scripts.ingest_policy), and costs a few cents.
"""
import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_index_client
from app.main import app
from app.rag.chunker import policy_hash
from app.rag.index import COLLECTION_NAME

CLAIM = (
    "Hi, this is Marcus Bell, policy HG-2231-88. On 12 March 2026 a pipe burst under the kitchen sink "
    "and flooded the floor. The plumber quoted EUR 800 for repairs."
)


@pytest.mark.live
def test_a_real_claim_goes_through_the_whole_pipeline():
    if COLLECTION_NAME not in {c.name for c in get_index_client().list_collections()}:
        pytest.skip("No policy index. Build it with: python -m scripts.ingest_policy")

    response = TestClient(app).post("/claims/review", json={"text": CLAIM})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["extraction"]["claimant_name"] == "Marcus Bell"
    assert body["extraction"]["estimated_amount"] == 800
    assert body["coverage"]["status"] in ("covered", "excluded", "cannot_determine")
    assert body["final_route"] in ("auto_approve_candidate", "escalate")
    assert body["versions"]["policy_hash"] == policy_hash()
    assert response.headers["X-Request-ID"]
    # any citation must be a clause the model was actually shown
    shown = {c["chunk_id"] for c in body["coverage"]["retrieved"]}
    assert {c["chunk_id"] for c in body["coverage"]["cited_clauses"]} <= shown
