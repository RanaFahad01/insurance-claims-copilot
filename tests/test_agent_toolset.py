"""Each tool, called the way the loop calls it (through execute_tool), with fakes in place of the API."""
import json
import uuid
from types import SimpleNamespace

import chromadb
import pytest
from fakes import FakeEmbedder, FakeInvoiceModel

from app.agent.documents import DocumentStore, InvoiceError, InvoiceExtraction
from app.agent.policy_terms import load_policy_terms
from app.agent.reviews import ReviewStore
from app.agent.state import InvestigationState, RunContext
from app.agent.tools import execute_tool, model_schemas
from app.agent.toolset import ToolDeps, default_registry
from app.agent.trace import Trace
from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.embeddings import EmbeddingError
from app.rag.index import ingest
from app.schemas import ClaimExtraction

EXTRACTION = ClaimExtraction(
    incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell", policy_number="HG-2231-88",
    damage_type="water", estimated_amount=800.0, currency="EUR", language="en", summary="A pipe burst under the kitchen sink.",
)


def reading(**overrides) -> InvoiceExtraction:
    base = dict(vendor="Hartwell & Pike Plumbing Ltd", document_date="2026-03-14", total=800.0, currency="EUR",
                customer_name="Marcus Bell", policy_number="HG-2231-88", remarks="Paid in full.")
    base.update(overrides)
    return InvoiceExtraction(**base)


@pytest.fixture(scope="module")
def embedder():
    return FakeEmbedder()


@pytest.fixture
def collection(embedder):
    client = chromadb.EphemeralClient()
    name = f"test_{uuid.uuid4().hex}"
    ingest(client, load_policy_chunks(), embedder, policy_hash(), collection_name=name)
    yield client.get_collection(name, embedding_function=None)
    client.delete_collection(name)


@pytest.fixture
def invoice_model():
    return FakeInvoiceModel(reading())


@pytest.fixture
def make_ctx(collection, embedder, invoice_model, tmp_path):
    def build(document_ids=("doc_14",), deps_overrides=None, extraction=EXTRACTION):
        deps = ToolDeps(
            collection=collection, embedder=embedder, documents=DocumentStore(), invoice_model=invoice_model,
            terms=load_policy_terms(), reviews=ReviewStore(tmp_path / "reviews.sqlite"),
        )
        for key, value in (deps_overrides or {}).items():
            setattr(deps, key, value)
        state = InvestigationState(claim_id="abc123", document_ids=list(document_ids))
        return RunContext(extraction=extraction, state=state, deps=deps)
    return build


@pytest.fixture
def registry():
    return default_registry()


def call(registry, ctx, name, arguments, from_model=True):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    trace = Trace()
    result = execute_tool(SimpleNamespace(name=name, arguments=raw, call_id="call_1"), registry, ctx, trace, from_model=from_model)
    return result, trace


# --- the registry ----------------------------------------------------------------------------------------------

def assert_closed(node):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            assert node["additionalProperties"] is False and node["required"] == list(node.get("properties", {}))
        for value in node.values():
            assert_closed(value)
    elif isinstance(node, list):
        for value in node:
            assert_closed(value)


def test_the_model_sees_five_read_only_tools_and_the_escalation_tool_is_hidden(registry):
    assert [s["name"] for s in model_schemas(registry)] == [
        "search_policy_clauses", "get_clause", "compare_documents", "get_policy_terms", "calculate_payout"]
    assert registry["escalate_to_human"].side_effect and not registry["escalate_to_human"].model_visible
    assert not any(t.side_effect for t in registry.values() if t.model_visible)


def test_every_schema_is_closed_and_every_description_says_what_the_tool_is_for(registry):
    for schema in model_schemas(registry):
        assert_closed(schema["parameters"])
        assert len(schema["description"]) > 60 and schema["strict"] is True


def test_no_tool_takes_the_claim_as_an_argument(registry):
    for tool in registry.values():
        assert "claim_id" not in tool.args_model.model_fields


def test_a_tool_run_without_its_dependencies_fails_as_data(registry):
    ctx = RunContext(extraction=EXTRACTION, state=InvestigationState(claim_id="abc123"))
    result, trace = call(registry, ctx, "get_clause", {"clause_id": "en-C2.1"})
    assert result == {"error": "tool_failed"} and trace.records[0].detail == "AttributeError"


# --- policy search and lookup ----------------------------------------------------------------------------------

def test_search_returns_at_most_four_clauses_with_full_text_and_notes_them(registry, make_ctx):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "search_policy_clauses", {"query": "water escapes from a pipe", "language": "en"})
    clauses = result["clauses"]
    assert 1 <= len(clauses) <= 4
    assert all(set(c) == {"clause_id", "title", "text", "distance"} for c in clauses)
    assert all(c["clause_id"].startswith("en-") and c["text"] for c in clauses)
    assert [c["distance"] for c in clauses] == sorted(c["distance"] for c in clauses)
    assert ctx.state.clause_ids_seen == [c["clause_id"] for c in clauses]
    assert len(json.dumps(result, ensure_ascii=False)) < 4000


def test_search_can_be_limited_to_spanish_or_cover_both_languages(registry, make_ctx):
    es, _ = call(registry, make_ctx(), "search_policy_clauses", {"query": "agua de una tubería", "language": "es"})
    assert es["clauses"] and all(c["clause_id"].startswith("es-") for c in es["clauses"])
    both, _ = call(registry, make_ctx(), "search_policy_clauses", {"query": "water escapes from a pipe", "language": None})
    assert both["clauses"]


def test_search_that_cannot_embed_reports_unavailable_and_notes_nothing(registry, make_ctx):
    class Broken:
        model = "broken"

        def embed(self, texts):
            raise EmbeddingError("rate_limited", "simulated")

    ctx = make_ctx(deps_overrides={"embedder": Broken()})
    result, _ = call(registry, ctx, "search_policy_clauses", {"query": "a pipe burst", "language": None})
    assert result == {"error": "search_unavailable"} and ctx.state.clause_ids_seen == []


@pytest.mark.parametrize("arguments", [
    {"query": "ab", "language": None}, {"query": "x" * 301, "language": None},
    {"query": "a pipe burst", "language": "fr"}, {"query": "a pipe burst"},
])
def test_search_refuses_bad_arguments(registry, make_ctx, arguments):
    result, _ = call(registry, make_ctx(), "search_policy_clauses", arguments)
    assert result["error"] == "invalid_arguments"


def test_get_clause_returns_the_clause_and_notes_it(registry, make_ctx):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "get_clause", {"clause_id": "en-C7.2"})
    assert result["clause_id"] == "en-C7.2" and "Deductible" in result["title"] and "250" in result["text"]
    assert ctx.state.clause_ids_seen == ["en-C7.2"]


def test_get_clause_ignores_surrounding_spaces(registry, make_ctx):
    result, _ = call(registry, make_ctx(), "get_clause", {"clause_id": "  es-C7.2 "})
    assert result["clause_id"] == "es-C7.2"


@pytest.mark.parametrize("clause_id", ["C7.2", "en-C99.9", "nonsense", "   "])
def test_get_clause_reports_clause_not_found_with_a_hint(registry, make_ctx, clause_id):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "get_clause", {"clause_id": clause_id})
    assert result["error"] == "clause_not_found" and "en-C2.1" in result["hint"]
    assert ctx.state.clause_ids_seen == []


# --- compare_documents -----------------------------------------------------------------------------------------

def test_a_matching_document_is_compared_and_marked_done(registry, make_ctx, invoice_model):
    ctx = make_ctx()
    result, trace = call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    assert result["agrees_with_claim"] is True and result["mismatches"] == []
    assert result["document"] == {"vendor": "Hartwell & Pike Plumbing Ltd", "date": "2026-03-14", "total": 800.0, "currency": "EUR"}
    assert result["remarks"] == "Paid in full."
    assert ctx.state.compared_document_ids == ["doc_14"] and ctx.state.required_checks_done and ctx.state.mismatches == []
    assert "Marcus Bell" in invoice_model.texts[0]       # the model was given the real document text
    assert trace.records[0].outcome == "ok"


def test_mismatches_are_returned_and_recorded(registry, make_ctx, invoice_model):
    invoice_model.result = reading(total=1450.0, customer_name="Daniel Cooper")
    ctx = make_ctx()
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    assert result["agrees_with_claim"] is False
    assert sorted((m["field"], m["severity"]) for m in result["mismatches"]) == [("amount", "high"), ("name", "high")]
    assert all("document_id" not in m for m in result["mismatches"])
    assert sorted(m.field for m in ctx.state.mismatches) == ["amount", "name"]
    assert ctx.state.required_checks_done


def test_comparing_the_same_document_twice_does_not_count_its_mismatches_twice(registry, make_ctx, invoice_model):
    invoice_model.result = reading(total=1450.0)
    ctx = make_ctx()
    call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    assert [m.field for m in ctx.state.mismatches] == ["amount"]


def test_a_document_that_is_not_attached_is_refused_and_nothing_is_read(registry, make_ctx, invoice_model):
    ctx = make_ctx(document_ids=("doc_14",))
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "doc_22"})
    assert result == {"error": "document_not_attached", "attached": ["doc_14"]}
    assert invoice_model.texts == [] and ctx.state.compared_document_ids == []


def test_an_attached_id_that_is_not_in_the_store_is_not_found(registry, make_ctx, invoice_model):
    ctx = make_ctx(document_ids=("doc_99",))
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "doc_99"})
    assert result == {"error": "document_not_found"}
    assert invoice_model.texts == [] and not ctx.state.required_checks_done


def test_an_attached_id_that_is_not_a_document_id_is_refused_by_the_store(registry, make_ctx):
    ctx = make_ctx(document_ids=("../manifest",))
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "../manifest"})
    assert result == {"error": "document_invalid_id"}


def test_an_unreadable_document_is_not_marked_compared(registry, make_ctx, invoice_model):
    invoice_model.result = InvoiceExtraction(vendor=None, document_date=None, total=None, currency=None,
                                             customer_name=None, policy_number=None, remarks=None)
    ctx = make_ctx()
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    assert result == {"error": "document_unreadable"}
    assert ctx.state.compared_document_ids == [] and not ctx.state.required_checks_done and ctx.state.mismatches == []


def test_a_reader_that_fails_leaves_the_state_unchanged(registry, make_ctx, invoice_model):
    invoice_model.result = InvoiceError("rate_limited", "simulated")
    ctx = make_ctx()
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    assert result == {"error": "document_reader_unavailable"}
    assert ctx.state.compared_document_ids == [] and not ctx.state.required_checks_done


def test_a_reading_that_cannot_be_trusted_by_the_comparison_does_not_crash_it(registry, make_ctx, invoice_model):
    invoice_model.result = reading(document_date="not a date", total=None)
    ctx = make_ctx()
    result, _ = call(registry, ctx, "compare_documents", {"document_id": "doc_14"})
    assert result["agrees_with_claim"] is True and ctx.state.required_checks_done


def test_text_from_the_document_is_clipped_before_the_agent_sees_it(registry, make_ctx, invoice_model):
    invoice_model.result = reading(vendor="V" * 500, remarks="R" * 2000, customer_name="N" * 500)
    result, _ = call(registry, make_ctx(), "compare_documents", {"document_id": "doc_14"})
    assert len(result["document"]["vendor"]) == 83 and result["document"]["vendor"].endswith("...")
    assert len(result["remarks"]) == 303 and result["remarks"].endswith("...")
    assert len(result["mismatches"][0]["document_value"]) == 83
    assert len(json.dumps(result, ensure_ascii=False)) < 1500


def test_a_claim_id_argument_is_rejected(registry, make_ctx, invoice_model):
    result, _ = call(registry, make_ctx(), "compare_documents", {"document_id": "doc_14", "claim_id": "other-claim"})
    assert result["error"] == "invalid_arguments" and invoice_model.texts == []


# --- policy terms and payout -----------------------------------------------------------------------------------

@pytest.mark.parametrize("insured_object, deductible, limit", [("home", 250, 300000), ("contents", 250, 50000), ("vehicle", 500, None)])
def test_policy_terms_come_from_the_terms_file(registry, make_ctx, insured_object, deductible, limit):
    result, _ = call(registry, make_ctx(), "get_policy_terms", {"insured_object": insured_object})
    assert (result["deductible"], result["limit"]) == (deductible, limit)
    assert result["notification"]["report_damage_within_days"] == 30


def test_an_unknown_insured_object_is_an_argument_error(registry, make_ctx):
    result, _ = call(registry, make_ctx(), "get_policy_terms", {"insured_object": "boat"})
    assert result["error"] == "invalid_arguments"


def test_payout_is_calculated_in_code_and_kept_in_the_state(registry, make_ctx):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "calculate_payout", {"insured_object": "home", "amount": 800, "currency": "EUR"})
    assert result["payable"] == 550.0 and result["deductible"] == 250.0 and result["over_limit"] is False
    assert ctx.state.payout.payable == 550.0


def test_payout_in_another_currency_is_refused_and_leaves_the_state_alone(registry, make_ctx):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "calculate_payout", {"insured_object": "home", "amount": 800, "currency": "USD"})
    assert result == {"error": "currency_not_supported", "supported": "EUR"} and ctx.state.payout is None


@pytest.mark.parametrize("amount", [-5, 1e12, "NaN", "Infinity", "lots"])
def test_payout_refuses_amounts_that_are_not_sensible(registry, make_ctx, amount):
    raw = '{"insured_object": "home", "amount": %s, "currency": "EUR"}' % (amount if isinstance(amount, (int, float)) else (amount if amount in ("NaN", "Infinity") else json.dumps(amount)))
    ctx = make_ctx()
    result, _ = call(registry, ctx, "calculate_payout", raw)
    assert result["error"] == "invalid_arguments" and ctx.state.payout is None


# --- escalation: the surrounding code only ---------------------------------------------------------------------

ESCALATION = {"summary": "The invoice total is higher than the claim.", "reasons": ["document_mismatch"]}


def test_the_model_cannot_escalate(registry, make_ctx):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "escalate_to_human", ESCALATION)
    assert result == {"error": "unknown_tool"}
    assert ctx.deps.reviews.list() == []


def test_the_surrounding_code_escalates_once_per_claim(registry, make_ctx):
    ctx = make_ctx()
    first, trace = call(registry, ctx, "escalate_to_human", ESCALATION, from_model=False)
    second, _ = call(registry, ctx, "escalate_to_human", {**ESCALATION, "summary": "again"}, from_model=False)
    assert first == {"review_id": "abc123", "created": True} and second == {"review_id": "abc123", "created": False}
    assert trace.records[0].side_effect is True
    (review,) = ctx.deps.reviews.list()
    assert review.reasons == ["document_mismatch"] and review.summary == ESCALATION["summary"] and review.status == "pending"


def test_escalation_uses_the_claim_of_the_run_not_one_from_the_arguments(registry, make_ctx):
    ctx = make_ctx()
    result, _ = call(registry, ctx, "escalate_to_human", {**ESCALATION, "claim_id": "other-claim"}, from_model=False)
    assert result["error"] == "invalid_arguments" and ctx.deps.reviews.list() == []
