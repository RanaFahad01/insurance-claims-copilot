"""/claims/review and /policy/clauses with a fake embedder, an in-memory index and a fake coverage model."""
import logging

import chromadb
import pytest
from fakes import FakeEmbedder
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import dependencies, main
from app.dependencies import get_collection, get_coverage_model, get_embedder, get_index_client
from app.extractor import ExtractionError
from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.coverage import CoverageAnswer, CoverageError
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingError
from app.rag.index import COLLECTION_NAME, ingest
from app.schemas import ClaimExtraction

client = TestClient(main.app)

VALID_TEXT = "Hi, this is Marcus Bell, policy HG-2231-88. A pipe burst under the kitchen sink."
CHUNKS = {chunk.chunk_id: chunk for chunk in load_policy_chunks()}
CLOSE_TO_C2_1 = CHUNKS["en-C2.1"].document   # as a summary, this matches clause C2.1 almost exactly
FAR_FROM_THE_POLICY = "zzzz qqqq xxxx"       # shares no words with the policy


def make_extraction(**overrides) -> ClaimExtraction:
    base = dict(incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell",
                policy_number="HG-2231-88", damage_type="other", estimated_amount=800.0, currency="EUR",
                language="en", summary=CLOSE_TO_C2_1)
    base.update(overrides)
    return ClaimExtraction(**base)


class FakeCoverageModel:
    model = "fake-coverage"
    prompt_version = "fake-prompt"

    def __init__(self):
        self.calls = 0

    def assess(self, extraction, clauses):
        self.calls += 1
        return CoverageAnswer(coverage_status="covered", cited_clause_ids=[clauses[0].chunk_id], rationale="Fits.")


class FailingCoverageModel(FakeCoverageModel):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def assess(self, extraction, clauses):
        raise CoverageError(self.kind, "simulated")


class FailingEmbedder(FakeEmbedder):
    def __init__(self, kind):
        super().__init__(model=EMBEDDING_MODEL)
        self.kind = kind

    def embed(self, texts):
        raise EmbeddingError(self.kind, "simulated")


def clear_index(chroma):
    if COLLECTION_NAME in {c.name for c in chroma.list_collections()}:
        chroma.delete_collection(COLLECTION_NAME)


@pytest.fixture
def chroma():
    """An in-memory Chroma client. It shares state inside the process, so the policy collection is removed around each test."""
    chroma = chromadb.EphemeralClient()
    clear_index(chroma)
    yield chroma
    clear_index(chroma)


@pytest.fixture
def api(chroma):
    """The app wired to a fake embedder, a fake coverage model and a built index. Returns the fakes for inspection."""
    embedder = FakeEmbedder(model=EMBEDDING_MODEL)
    ingest(chroma, load_policy_chunks(), embedder, policy_hash())
    coverage_model = FakeCoverageModel()
    main.app.dependency_overrides[get_index_client] = lambda: chroma
    main.app.dependency_overrides[get_embedder] = lambda: embedder
    main.app.dependency_overrides[get_coverage_model] = lambda: coverage_model
    yield coverage_model
    main.app.dependency_overrides.clear()


class FakeExtraction:
    """Stands in for extract_claim: records the texts it receives and returns whatever extraction it was given."""

    def __init__(self):
        self.received: list[str] = []
        self.result = make_extraction()

    def __call__(self, text: str) -> ClaimExtraction:
        self.received.append(text)
        return self.result

    def __eq__(self, other):   # lets a test write `extraction_calls == [text]`
        return self.received == other


@pytest.fixture
def extraction_calls(monkeypatch):
    fake = FakeExtraction()
    monkeypatch.setattr(main, "extract_claim", fake)
    return fake


# ---- /claims/review

def test_a_clean_covered_claim_returns_a_dossier_with_an_auto_approve_candidate(api, extraction_calls):
    response = client.post("/claims/review", json={"text": VALID_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert extraction_calls == [VALID_TEXT]
    assert body["final_route"] == "auto_approve_candidate"
    assert body["reasons"] == []
    assert body["extraction"]["claimant_name"] == "Marcus Bell"
    assert body["rules_decision"]["route"] == "auto_approve_candidate"
    assert body["coverage"]["status"] == "covered" and body["coverage"]["gate_triggered"] == "none"
    cited = body["coverage"]["cited_clauses"]
    assert [c["chunk_id"] for c in cited] == ["en-C2.1"]
    assert cited[0]["text"] == CHUNKS["en-C2.1"].text
    retrieved = {c["chunk_id"]: c for c in body["coverage"]["retrieved"]}
    assert retrieved["en-C2.1"]["distance"] == pytest.approx(0, abs=1e-3)
    assert retrieved["en-C1.3"]["added_by_rule"] is True
    assert body["versions"]["embedding_model"] == EMBEDDING_MODEL
    assert body["versions"]["policy_hash"] == policy_hash()
    assert body["versions"]["coverage_model"] == "fake-coverage"
    assert api.calls == 1


def test_the_rules_can_escalate_a_claim_the_coverage_step_likes(api, extraction_calls):
    extraction_calls.result = make_extraction(estimated_amount=9000.0)
    body = client.post("/claims/review", json={"text": VALID_TEXT}).json()
    assert body["coverage"]["status"] == "covered"
    assert body["final_route"] == "escalate"
    assert body["reasons"] == ["amount_above_high_threshold"]


def test_a_claim_far_from_the_policy_escalates_without_calling_the_coverage_model(api, extraction_calls):
    extraction_calls.result = make_extraction(summary=FAR_FROM_THE_POLICY)
    body = client.post("/claims/review", json={"text": VALID_TEXT}).json()
    assert body["coverage"]["gate_triggered"] == "low_retrieval_confidence"
    assert body["final_route"] == "escalate"
    assert body["reasons"] == ["retrieval_low_confidence"]
    assert api.calls == 0


def test_a_too_short_claim_is_rejected_before_anything_runs(api, extraction_calls):
    assert client.post("/claims/review", json={"text": "short"}).status_code == 422
    assert extraction_calls == []


@pytest.mark.parametrize("kind, status", [
    ("rate_limited", 503), ("timeout", 503), ("connection", 503),
    ("api_error", 502), ("incomplete", 502), ("refusal", 502), ("no_parsed_output", 502),
])
def test_extraction_errors_map_to_http_status(api, monkeypatch, kind, status):
    def failing(text):
        raise ExtractionError(kind, "simulated")

    monkeypatch.setattr(main, "extract_claim", failing)
    response = client.post("/claims/review", json={"text": VALID_TEXT})
    assert response.status_code == status
    assert response.json()["detail"].startswith(f"{kind}:")


@pytest.mark.parametrize("kind, status", [
    ("rate_limited", 503), ("timeout", 503), ("connection", 503),
    ("api_error", 502), ("incomplete", 502), ("refusal", 502), ("no_parsed_output", 502),
])
def test_coverage_errors_map_to_http_status(api, extraction_calls, kind, status):
    main.app.dependency_overrides[get_coverage_model] = lambda: FailingCoverageModel(kind)
    assert client.post("/claims/review", json={"text": VALID_TEXT}).status_code == status


@pytest.mark.parametrize("kind, status", [
    ("rate_limited", 503), ("timeout", 503), ("connection", 503), ("api_error", 502), ("bad_response", 502),
])
def test_embedding_errors_map_to_http_status(api, extraction_calls, kind, status):
    main.app.dependency_overrides[get_embedder] = lambda: FailingEmbedder(kind)
    assert client.post("/claims/review", json={"text": VALID_TEXT}).status_code == status


# ---- a missing or unusable index

def use_index(chroma):
    main.app.dependency_overrides[get_index_client] = lambda: chroma
    main.app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(model=EMBEDDING_MODEL)
    main.app.dependency_overrides[get_coverage_model] = lambda: FakeCoverageModel()


def break_index(chroma, how):
    if how == "empty":
        chroma.create_collection(COLLECTION_NAME, embedding_function=None,
                                 metadata={"embedding_model": EMBEDDING_MODEL, "policy_hash": "x"})
    elif how == "model_mismatch":
        ingest(chroma, load_policy_chunks(), FakeEmbedder(model="some-other-model"), policy_hash())


@pytest.mark.parametrize("how", ["missing", "empty", "model_mismatch"])
def test_an_unusable_index_gives_503_and_no_language_model_call(chroma, extraction_calls, how):
    break_index(chroma, how)
    use_index(chroma)
    try:
        response = client.post("/claims/review", json={"text": VALID_TEXT})
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"].startswith(f"index_{how}:")
    assert "ingest_policy" in response.json()["detail"]
    assert extraction_calls == []


# ---- request ID and logging

def test_the_request_id_header_matches_the_log_line_and_the_log_has_no_claim_text(api, extraction_calls, caplog):
    secret_text = "Hi, this is Marcus Bell, policy HG-2231-88. A pipe burst under the kitchen sink."
    with caplog.at_level(logging.INFO, logger="claims.review"):
        response = client.post("/claims/review", json={"text": secret_text})
    request_id = response.headers["X-Request-ID"]
    lines = [r.getMessage() for r in caplog.records if r.name == "claims.review"]
    assert len(lines) == 1
    line = lines[0]
    assert f"request_id={request_id}" in line
    assert "route=auto_approve_candidate" in line and "coverage=covered" in line and "gate=none" in line
    assert "en-C2.1:0.00" in line
    for private in ("Marcus", "HG-2231-88", "kitchen", CLOSE_TO_C2_1[:30]):
        assert private not in line


# ---- /policy/clauses

def test_clause_lookup_returns_the_exact_clause(api):
    response = client.get("/policy/clauses/en-C2.1")
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_id"] == "en-C2.1" and body["clause_id"] == "C2.1" and body["language"] == "en"
    assert body["text"] == CHUNKS["en-C2.1"].text


def test_clause_lookup_works_in_spanish(api):
    body = client.get("/policy/clauses/es-C2.1").json()
    assert body["language"] == "es" and body["text"] == CHUNKS["es-C2.1"].text


@pytest.mark.parametrize("clause_id", ["en-C99.9", "C2.1", "nonsense"])
def test_unknown_or_ambiguous_clause_ids_return_404(api, clause_id):
    response = client.get(f"/policy/clauses/{clause_id}")
    assert response.status_code == 404
    assert "en-C2.1" in response.json()["detail"]


def test_an_overlong_clause_id_is_rejected(api):
    assert client.get("/policy/clauses/" + "a" * 41).status_code == 422


def test_clause_lookup_needs_neither_an_embedder_nor_a_coverage_model(api):
    def boom():
        raise AssertionError("the clause lookup must not build an embedder or a coverage model")

    main.app.dependency_overrides[get_embedder] = boom
    main.app.dependency_overrides[get_coverage_model] = boom
    assert client.get("/policy/clauses/en-C2.1").status_code == 200


def test_clause_lookup_gives_503_when_the_index_is_missing(chroma):
    use_index(chroma)
    try:
        response = client.get("/policy/clauses/en-C2.1")
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"].startswith("index_missing:")


# ---- the real providers

def test_providers_give_503_when_there_is_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dependencies._embedder.cache_clear()
    dependencies._coverage_model.cache_clear()
    for provider in (get_embedder, get_coverage_model):
        with pytest.raises(HTTPException) as raised:
            provider()
        assert raised.value.status_code == 503 and raised.value.detail.startswith("not_configured:")


def test_get_collection_is_the_dependency_the_endpoints_use():
    # guards against an endpoint being wired to a different function that tests would then fail to override
    for route in main.app.routes:
        if getattr(route, "path", "") in ("/claims/review", "/policy/clauses/{clause_id}"):
            assert get_collection in [d.call for d in route.dependant.dependencies]
