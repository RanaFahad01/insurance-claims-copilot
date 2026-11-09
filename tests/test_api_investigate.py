"""/claims/investigate, /runs, /reviews: the whole flow through the API with fakes in place of every OpenAI call."""
import json
import logging
import threading
from types import SimpleNamespace

import chromadb
import pytest
from fakes import FakeAgentModel, FakeEmbedder, FakeInvoiceModel, agent_response, fn_call
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import agent_routes, dependencies, main
from app.agent.documents import InvoiceExtraction
from app.agent.loop import AgentModelError
from app.agent.reviews import ReviewStore
from app.agent.trace import TraceWriter
from app.dependencies import (
    get_agent_model,
    get_collection,
    get_coverage_model,
    get_embedder,
    get_index_client,
    get_invoice_model,
    get_review_store,
    get_tool_deps,
    get_trace_writer,
)
from app.extractor import ExtractionError
from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.coverage import CoverageAnswer, CoverageError
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingError
from app.rag.index import COLLECTION_NAME, ingest
from app.schemas import ClaimExtraction

client = TestClient(main.app)

CLAIM_TEXT = "Hi, this is Marcus Bell, policy HG-2231-88. A pipe burst under the kitchen sink."
CHUNKS = {chunk.chunk_id: chunk for chunk in load_policy_chunks()}
CLOSE_TO_C2_1 = CHUNKS["en-C2.1"].document   # as a summary, this matches clause C2.1 almost exactly
AGENT_TEXT = "The invoice agrees with the claim and the plumber is licensed."


def make_extraction(**overrides) -> ClaimExtraction:
    base = dict(incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell", policy_number="HG-2231-88",
                damage_type="other", estimated_amount=800.0, currency="EUR", language="en", summary=CLOSE_TO_C2_1)
    base.update(overrides)
    return ClaimExtraction(**base)


def reading(**overrides) -> InvoiceExtraction:
    base = dict(vendor="Hartwell & Pike Plumbing Ltd", document_date="2026-03-14", total=800.0, currency="EUR",
                customer_name="Marcus Bell", policy_number="HG-2231-88", remarks="Paid in full.")
    base.update(overrides)
    return InvoiceExtraction(**base)


class FakeCoverageModel:
    model = "fake-coverage"
    prompt_version = "fake-prompt"

    def __init__(self):
        self.calls = 0

    def assess(self, extraction, claim_text, clauses):
        self.calls += 1
        return CoverageAnswer(coverage_status="covered", cited_clause_ids=[clauses[0].chunk_id], rationale="Fits.")


def compare(doc_id, call_id="c1"):
    return fn_call("compare_documents", json.dumps({"document_id": doc_id}), call_id)


def clean_script(doc_id="doc_14"):
    return [agent_response(compare(doc_id)), agent_response(text=AGENT_TEXT)]


def clear_index(chroma):
    if COLLECTION_NAME in {c.name for c in chroma.list_collections()}:
        chroma.delete_collection(COLLECTION_NAME)


@pytest.fixture
def chroma():
    chroma = chromadb.EphemeralClient()
    clear_index(chroma)
    yield chroma
    clear_index(chroma)


@pytest.fixture
def api(chroma, tmp_path, monkeypatch):
    """The app wired to fakes and a temporary folder. Returns the fakes, so a test can script them and look at what they received."""
    embedder = FakeEmbedder(model=EMBEDDING_MODEL)
    ingest(chroma, load_policy_chunks(), embedder, policy_hash())
    fakes = SimpleNamespace(
        embedder=embedder, coverage=FakeCoverageModel(), invoice=FakeInvoiceModel(reading()), agent=FakeAgentModel(clean_script()),
        reviews=ReviewStore(tmp_path / "reviews.sqlite"), writer=TraceWriter(tmp_path / "traces"), tmp=tmp_path,
        extracted=[], extraction=make_extraction(),
    )

    def fake_extract(text):
        fakes.extracted.append(text)
        return fakes.extraction

    monkeypatch.setattr(agent_routes, "extract_claim", fake_extract)
    overrides = main.app.dependency_overrides
    overrides[get_index_client] = lambda: chroma
    overrides[get_embedder] = lambda: embedder
    overrides[get_coverage_model] = lambda: fakes.coverage
    overrides[get_invoice_model] = lambda: fakes.invoice
    overrides[get_agent_model] = lambda: fakes.agent
    overrides[get_review_store] = lambda: fakes.reviews
    overrides[get_trace_writer] = lambda: fakes.writer
    yield fakes
    overrides.clear()


def investigate(documents=("doc_14",), text=CLAIM_TEXT):
    return client.post("/claims/investigate", json={"text": text, "document_ids": list(documents)})


def no_model_was_called(api):
    return api.extracted == [] and api.coverage.calls == 0 and api.agent.requests == [] and api.invoice.texts == []


def nothing_was_stored(api):
    return not (api.tmp / "traces").exists() and api.reviews.list() == []


# ---- a clean document

def test_a_clean_document_keeps_the_fast_track_and_returns_a_run_id(api):
    response = investigate()
    assert response.status_code == 200
    body = response.json()
    assert body["final_route"] == "auto_approve_candidate" and body["reasons"] == []
    assert body["documents_attached"] == ["doc_14"] and body["documents_compared"] == ["doc_14"] and body["mismatches"] == []
    assert body["findings"] == AGENT_TEXT
    assert body["agent"]["status"] == "finished" and body["agent"]["steps"] == 2
    assert body["dossier"]["coverage"]["status"] == "covered"
    assert body["review_created"] is None and api.reviews.list() == []
    assert body["run_id"] == response.headers["X-Run-ID"] and len(body["run_id"]) == 16
    assert "tool_calls" not in body        # the calls are in the trace, not in the answer


def test_the_trace_endpoint_returns_the_run_in_order_with_totals(api):
    run_id = investigate().json()["run_id"]
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    trace = response.json()
    assert trace["run_id"] == run_id
    assert [(e["type"], e["step"]) for e in trace["events"]] == [("turn", 1), ("tool_call", 1), ("turn", 2)]
    call = trace["events"][1]
    assert call["tool"] == "compare_documents" and call["outcome"] == "ok" and '"document_id": "doc_14"' in call["arguments"]
    assert '"agrees_with_claim": true' in call["result"]
    run = trace["run"]
    assert run["status"] == "completed" and run["final_route"] == "auto_approve_candidate" and run["reasons"] == []
    assert (run["steps"], run["tool_calls"], run["agent_status"]) == (2, 1, "finished")
    assert (run["input_tokens"], run["output_tokens"]) == (20, 10)
    assert run["documents"] == ["doc_14"] and run["mismatches"] == [] and run["review_created"] is None
    assert run["versions"]["agent_model"] == "fake-agent"
    assert [e["seq"] for e in trace["events"]] + [run["seq"]] == [0, 1, 2, 3]


def test_the_trace_file_holds_no_claim_text_and_not_what_the_agent_wrote(api):
    run_id = investigate().json()["run_id"]
    raw = (api.tmp / "traces" / f"{run_id}.jsonl").read_text(encoding="utf-8")
    for private in ("kitchen sink", "pipe burst", AGENT_TEXT, CLOSE_TO_C2_1[:40]):
        assert private not in raw


def test_the_log_line_has_the_run_id_and_no_claim_text(api, caplog):
    with caplog.at_level(logging.INFO, logger="claims.investigate"):
        run_id = investigate().json()["run_id"]
    (line,) = [r.getMessage() for r in caplog.records if r.name == "claims.investigate"]
    assert f"run_id={run_id}" in line and "route=auto_approve_candidate" in line and "steps=2" in line and "tokens=30" in line
    for private in ("Marcus", "HG-2231-88", "kitchen", AGENT_TEXT):
        assert private not in line


# ---- a document that disagrees

def test_a_mismatch_escalates_and_creates_a_review_that_points_at_the_run(api):
    api.invoice.result = reading(total=1500.0)
    api.agent.script = clean_script("doc_22")
    body = investigate(["doc_22"]).json()
    assert body["final_route"] == "escalate" and "document_mismatch" in body["reasons"]
    assert [(m["document_id"], m["field"], m["severity"]) for m in body["mismatches"]] == [("doc_22", "amount", "high")]
    assert body["review_created"] is True

    (listed,) = client.get("/reviews").json()
    assert listed["claim_id"] == body["claim_id"] and listed["status"] == "pending"
    assert listed["run_id"] == body["run_id"] and "document_mismatch" in listed["reasons"]

    trace = client.get(f"/runs/{body['run_id']}").json()
    assert trace["run"]["final_route"] == "escalate" and trace["run"]["review_created"] is True
    assert trace["run"]["mismatches"] == [{"document_id": "doc_22", "field": "amount", "severity": "high"}]
    last = trace["events"][-1]
    assert (last["tool"], last["step"], last["side_effect"]) == ("escalate_to_human", 0, True)


def test_an_agent_that_never_compares_the_document_still_escalates(api):
    api.agent.script = [agent_response(text="Looks fine to me.")]
    body = investigate().json()
    assert body["final_route"] == "escalate" and "required_check_missing" in body["reasons"]
    assert body["review_created"] is True


# ---- no documents

def test_without_documents_the_agent_does_not_run(api):
    response = investigate([])
    body = response.json()
    assert response.status_code == 200 and body["agent"] is None and body["documents_attached"] == []
    assert body["final_route"] == "auto_approve_candidate"
    assert api.agent.requests == [] and api.invoice.texts == []
    run = client.get(f"/runs/{body['run_id']}").json()["run"]
    assert (run["agent_status"], run["steps"], run["tool_calls"], run["input_tokens"]) == ("not_run", 0, 0, 0)


def test_the_documents_field_is_optional(api):
    response = client.post("/claims/investigate", json={"text": CLAIM_TEXT})
    assert response.status_code == 200 and response.json()["documents_attached"] == []


def test_a_document_listed_twice_is_attached_once(api):
    body = investigate(["doc_14", "doc_14"]).json()
    assert body["documents_attached"] == ["doc_14"]


# ---- requests that are refused before anything runs

@pytest.mark.parametrize("bad", ["doc_1", "DOC_01", "doc_01\n", "../x", "doc_01; drop", "doc_00001", "", "doc_ab", " doc_01"])
def test_a_malformed_document_id_is_rejected_before_any_model_call(api, bad):
    response = investigate([bad])
    assert response.status_code == 422
    assert no_model_was_called(api) and nothing_was_stored(api)


def test_too_many_documents_are_rejected(api):
    assert investigate([f"doc_{n:02d}" for n in range(1, 7)]).status_code == 422
    assert no_model_was_called(api) and nothing_was_stored(api)


def test_a_document_that_does_not_exist_is_rejected_with_its_name_before_any_model_call(api):
    response = investigate(["doc_14", "doc_98", "doc_99"])
    assert response.status_code == 422
    assert response.json()["detail"] == "unknown_document: doc_98, doc_99"
    assert no_model_was_called(api) and nothing_was_stored(api)


@pytest.mark.parametrize("body", [{"text": "short"}, {"document_ids": ["doc_14"]}, {"text": "x" * 20001}, {"text": CLAIM_TEXT, "document_ids": "doc_14"}])
def test_a_bad_body_is_rejected(api, body):
    assert client.post("/claims/investigate", json=body).status_code == 422
    assert no_model_was_called(api)


@pytest.mark.parametrize("how", ["missing", "empty"])
def test_an_unusable_index_gives_503_and_no_model_call(chroma, tmp_path, monkeypatch, how):
    clear_index(chroma)
    if how == "empty":
        chroma.create_collection(COLLECTION_NAME, embedding_function=None, metadata={"embedding_model": EMBEDDING_MODEL, "policy_hash": "x"})
    fakes = SimpleNamespace(extracted=[], coverage=FakeCoverageModel(), agent=FakeAgentModel([]), invoice=FakeInvoiceModel(reading()))
    monkeypatch.setattr(agent_routes, "extract_claim", lambda text: fakes.extracted.append(text))
    overrides = main.app.dependency_overrides
    overrides[get_index_client] = lambda: chroma
    overrides[get_embedder] = lambda: FakeEmbedder(model=EMBEDDING_MODEL)
    overrides[get_coverage_model] = lambda: fakes.coverage
    overrides[get_invoice_model] = lambda: fakes.invoice
    overrides[get_agent_model] = lambda: fakes.agent
    overrides[get_review_store] = lambda: ReviewStore(tmp_path / "r.sqlite")
    overrides[get_trace_writer] = lambda: TraceWriter(tmp_path / "traces")
    try:
        response = investigate()
    finally:
        overrides.clear()
    assert response.status_code == 503 and response.json()["detail"].startswith(f"index_{how}:")
    assert fakes.extracted == [] and fakes.coverage.calls == 0 and fakes.agent.requests == [] and not (tmp_path / "traces").exists()


# ---- failures upstream

@pytest.mark.parametrize("kind, status", [("rate_limited", 503), ("timeout", 503), ("connection", 503), ("api_error", 502), ("refusal", 502)])
def test_an_extraction_error_maps_to_http_status_and_leaves_a_failed_trace(api, monkeypatch, kind, status):
    def failing(text):
        raise ExtractionError(kind, "simulated")

    monkeypatch.setattr(agent_routes, "extract_claim", failing)
    response = investigate()
    assert response.status_code == status and response.json()["detail"].startswith(f"{kind}:")
    run = client.get(f"/runs/{response.headers['X-Run-ID']}").json()
    assert run["events"] == []
    assert (run["run"]["status"], run["run"]["error_stage"], run["run"]["error_kind"]) == ("failed", "extraction", kind)
    assert api.reviews.list() == [] and api.agent.requests == []


@pytest.mark.parametrize("kind, status", [("rate_limited", 503), ("timeout", 503), ("api_error", 502), ("no_parsed_output", 502)])
def test_a_coverage_error_maps_to_http_status_and_leaves_a_failed_trace(api, kind, status):
    class Failing(FakeCoverageModel):
        def assess(self, extraction, claim_text, clauses):
            raise CoverageError(kind, "simulated")

    api.coverage = Failing()
    response = investigate()
    assert response.status_code == status
    run = client.get(f"/runs/{response.headers['X-Run-ID']}").json()["run"]
    assert (run["status"], run["error_stage"], run["error_kind"]) == ("failed", "coverage", kind)
    assert api.agent.requests == [] and api.reviews.list() == []


@pytest.mark.parametrize("kind, status", [("rate_limited", 503), ("connection", 503), ("api_error", 502), ("bad_response", 502)])
def test_an_embedding_error_maps_to_http_status(api, kind, status):
    class Failing(FakeEmbedder):
        def embed(self, texts):
            raise EmbeddingError(kind, "simulated")

    main.app.dependency_overrides[get_embedder] = lambda: Failing(model=EMBEDDING_MODEL)
    response = investigate()
    assert response.status_code == status
    run = client.get(f"/runs/{response.headers['X-Run-ID']}").json()["run"]
    assert (run["error_stage"], run["error_kind"]) == ("embedding", kind)


def test_an_unexpected_error_is_a_500_and_still_leaves_a_failed_trace(api, monkeypatch):
    def broken(text):
        raise RuntimeError("bug")

    monkeypatch.setattr(agent_routes, "extract_claim", broken)
    response = TestClient(main.app, raise_server_exceptions=False).post("/claims/investigate", json={"text": CLAIM_TEXT, "document_ids": []})
    assert response.status_code == 500
    (path,) = (api.tmp / "traces").iterdir()
    run = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert (run["status"], run["error_stage"], run["error_kind"]) == ("failed", "unexpected", "RuntimeError")


def test_an_agent_model_failure_never_becomes_an_approval(api):
    api.agent.script = [AgentModelError("timeout", "simulated")]
    response = investigate()
    body = response.json()
    assert response.status_code == 200
    assert body["agent"]["status"] == "error" and body["agent"]["error_kind"] == "timeout"
    assert body["final_route"] == "escalate" and {"agent_incomplete", "required_check_missing"} <= set(body["reasons"])
    assert body["review_created"] is True
    run = client.get(f"/runs/{body['run_id']}").json()["run"]
    assert (run["agent_status"], run["agent_error_kind"], run["final_route"]) == ("error", "timeout", "escalate")


def test_a_trace_that_cannot_be_written_does_not_break_the_answer(api, caplog):
    class Broken(TraceWriter):
        def write(self, run_id, trace, summary):
            raise OSError("disk full")

    main.app.dependency_overrides[get_trace_writer] = lambda: Broken(api.tmp / "traces")
    with caplog.at_level(logging.ERROR, logger="claims.investigate"):
        response = investigate()
    assert response.status_code == 200 and response.json()["final_route"] == "auto_approve_candidate"
    assert any("trace_not_written=OSError" in r.getMessage() for r in caplog.records)


def test_the_same_claim_investigated_twice_keeps_one_review_and_two_traces(api):
    api.invoice.result = reading(total=1500.0)
    api.agent.script = clean_script("doc_22")
    first = investigate(["doc_22"]).json()
    api.agent.script = clean_script("doc_22")
    second = investigate(["doc_22"]).json()
    assert first["run_id"] != second["run_id"] and first["claim_id"] == second["claim_id"]
    assert (first["review_created"], second["review_created"]) == (True, False)
    assert len(api.reviews.list()) == 1 and api.reviews.list()[0].run_id == first["run_id"]
    assert len(list((api.tmp / "traces").iterdir())) == 2


# ---- GET /runs

def test_an_unknown_run_is_404(api):
    assert client.get("/runs/0123456789abcdef").status_code == 404


@pytest.mark.parametrize("bad", ["x", "0123456789ABCDEF", "0123456789abcdef0", "%2e%2e%2fsecret", "..%2fsecret", "0123456789abcdef%0A",
                                 "0123456789abcdef.jsonl", "..", "%2e%2e", "secret"])
def test_anything_that_is_not_a_run_id_is_404_and_never_reads_outside_the_folder(api, bad):
    (api.tmp / "secret.jsonl").write_text('{"type": "run", "secret": true}\n', encoding="utf-8")
    (api.tmp / "traces").mkdir(exist_ok=True)
    response = client.get(f"/runs/{bad}")
    assert response.status_code == 404 and "secret" not in response.text


def test_a_corrupt_trace_file_is_a_500_with_a_plain_message(api):
    (api.tmp / "traces").mkdir()
    (api.tmp / "traces" / "0123456789abcdef.jsonl").write_text("{broken\n", encoding="utf-8")
    response = client.get("/runs/0123456789abcdef")
    assert response.status_code == 500 and response.json()["detail"] == "trace_unreadable"


# ---- the review queue

def escalate_two(api):
    api.invoice.result = reading(total=1500.0)
    api.agent.script = clean_script("doc_22")
    first = investigate(["doc_22"]).json()
    api.agent.script = clean_script("doc_14")
    api.invoice.result = reading(customer_name="Someone Else")
    second = investigate(["doc_14"], text=CLAIM_TEXT + " It happened on Tuesday.").json()
    return first, second


def test_the_queue_lists_reviews_and_filters_by_status(api):
    first, second = escalate_two(api)
    assert {r["claim_id"] for r in client.get("/reviews").json()} == {first["claim_id"], second["claim_id"]}
    assert len(client.get("/reviews?status=pending").json()) == 2 and client.get("/reviews?status=resolved").json() == []
    client.post(f"/reviews/{first['claim_id']}/resolve", json={"resolution": "approved"})
    assert [r["claim_id"] for r in client.get("/reviews?status=pending").json()] == [second["claim_id"]]
    assert [r["claim_id"] for r in client.get("/reviews?status=resolved").json()] == [first["claim_id"]]


def test_an_unknown_status_filter_is_rejected(api):
    assert client.get("/reviews?status=banana").status_code == 422


def test_an_empty_queue_is_an_empty_list(api):
    assert client.get("/reviews").json() == []


def test_resolving_records_the_decision_and_changes_nothing_else(api):
    first, _ = escalate_two(api)
    before_review = next(r for r in client.get("/reviews").json() if r["claim_id"] == first["claim_id"])
    before_trace = client.get(f"/runs/{first['run_id']}").json()

    response = client.post(f"/reviews/{first['claim_id']}/resolve", json={"resolution": "rejected", "note": "Invoice is inflated."})
    assert response.status_code == 200
    review = response.json()
    assert (review["status"], review["resolution"], review["resolution_note"]) == ("resolved", "rejected", "Invoice is inflated.")
    assert review["resolved_at"] is not None
    for field in ("claim_id", "reasons", "summary", "run_id", "created_at"):
        assert review[field] == before_review[field]
    assert client.get(f"/runs/{first['run_id']}").json() == before_trace      # the machine's record of the run is untouched


def test_resolving_twice_is_a_409_that_shows_the_first_decision(api):
    first, _ = escalate_two(api)
    assert client.post(f"/reviews/{first['claim_id']}/resolve", json={"resolution": "approved", "note": "ok"}).status_code == 200
    again = client.post(f"/reviews/{first['claim_id']}/resolve", json={"resolution": "rejected"})
    assert again.status_code == 409
    detail = again.json()["detail"]
    assert detail["error"] == "already_resolved" and detail["review"]["resolution"] == "approved" and detail["review"]["resolution_note"] == "ok"


@pytest.mark.parametrize("claim_id", ["doesnotexist", "ABC", "x'; DROP TABLE reviews; --", "a" * 65])
def test_resolving_an_unknown_claim_is_404(api, claim_id):
    assert client.post(f"/reviews/{claim_id}/resolve", json={"resolution": "approved"}).status_code == 404


@pytest.mark.parametrize("body", [{}, {"resolution": "maybe"}, {"resolution": "approved", "note": "n" * 501}, {"resolution": None}, {"note": "x"}])
def test_a_bad_resolution_is_rejected_and_the_review_stays_pending(api, body):
    first, _ = escalate_two(api)
    assert client.post(f"/reviews/{first['claim_id']}/resolve", json=body).status_code == 422
    assert len(client.get("/reviews?status=pending").json()) == 2


def test_the_three_decisions_are_accepted(api):
    first, second = escalate_two(api)
    assert client.post(f"/reviews/{first['claim_id']}/resolve", json={"resolution": "needs_more_information"}).json()["resolution"] == "needs_more_information"
    assert client.post(f"/reviews/{second['claim_id']}/resolve", json={"resolution": "approved"}).status_code == 200


def test_two_people_resolving_at_once_give_one_200_and_one_409(api):
    first, _ = escalate_two(api)
    codes = []

    def resolve(decision):
        codes.append(TestClient(main.app).post(f"/reviews/{first['claim_id']}/resolve", json={"resolution": decision}).status_code)

    threads = [threading.Thread(target=resolve, args=(d,)) for d in ["approved", "rejected"] * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [200] + [409] * 5


# ---- wiring

def test_the_providers_give_503_when_there_is_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dependencies._agent_model.cache_clear()
    dependencies._invoice_model.cache_clear()
    for provider in (get_agent_model, get_invoice_model):
        with pytest.raises(HTTPException) as raised:
            provider()
        assert raised.value.status_code == 503 and raised.value.detail.startswith("not_configured:")


def test_the_investigate_endpoint_checks_the_index_before_anything_else():
    # The first dependency decides what fails first. The index must come before the models, so a missing index costs no model call.
    route = next(r for r in agent_routes.router.routes if getattr(r, "path", "") == "/claims/investigate")
    first = route.dependant.dependencies[0].call
    assert first is get_tool_deps
    names = [d.call for d in dependencies_of(get_tool_deps)]
    assert names[0] is get_collection


def dependencies_of(function):
    from fastapi.dependencies.utils import get_dependant
    return get_dependant(path="/", call=function).dependencies


def test_the_review_and_run_endpoints_read_from_the_injected_stores():
    paths = {getattr(r, "path", None): r for r in agent_routes.router.routes}
    assert get_review_store in [d.call for d in paths["/reviews"].dependant.dependencies]
    assert get_review_store in [d.call for d in paths["/reviews/{claim_id}/resolve"].dependant.dependencies]
    assert get_trace_writer in [d.call for d in paths["/runs/{run_id}"].dependant.dependencies]
