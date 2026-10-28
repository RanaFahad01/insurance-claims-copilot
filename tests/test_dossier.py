import uuid

import chromadb
import pytest
from fakes import FakeEmbedder

from app import extractor
from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.coverage import CoverageAnswer, CoverageResult
from app.rag.dossier import (
    GATE_REASONS,
    Dossier,
    Versions,
    build_dossier,
    collect_versions,
    final_route,
    render_markdown,
    review_claim,
)
from app.rag.index import index_info, ingest
from app.rag.retriever import DISTANCE_CUTOFF, RetrievedClause
from app.rules import triage
from app.schemas import ClaimExtraction, RouteDecision


def make_extraction(**overrides) -> ClaimExtraction:
    base = dict(incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell",
                policy_number="HG-2231-88", damage_type="water", estimated_amount=800.0, currency="EUR",
                language="en", summary="A pipe burst under the kitchen sink and flooded the floor.")
    base.update(overrides)
    return ClaimExtraction(**base)


def clause(chunk_id="en-C2.1", distance=0.40, added_by_rule=False) -> RetrievedClause:
    language, clause_id = chunk_id.split("-", 1)
    return RetrievedClause(chunk_id=chunk_id, clause_id=clause_id, language=language, title=f"Title of {clause_id}",
                           text=f"Text of {clause_id}.", distance=distance, added_by_rule=added_by_rule)


def coverage_result(status="covered", gate="none", cited=("en-C2.1",)) -> CoverageResult:
    retrieved = [clause("en-C2.1", 0.40), clause("en-C2.3", 0.45), clause("en-C1.3", 0.70, added_by_rule=True)]
    return CoverageResult(status=status, cited_clauses=[c for c in retrieved if c.chunk_id in cited],
                          rationale="Because of the clauses.", retrieved=retrieved, gate_triggered=gate)


CLAIM_TEXT = "Hi, this is Marcus Bell, policy HG-2231-88. A pipe burst under the kitchen sink."

AUTO = RouteDecision(route="auto_approve_candidate", severity_score=10, reasons=[])
ESCALATE = RouteDecision(route="escalate", severity_score=60, reasons=["amount_above_high_threshold", "missing_incident_date"])

VERSIONS = Versions(extraction_model="x-model", extraction_prompt="extraction-vX", coverage_model="c-model",
                    coverage_prompt="coverage-vY", embedding_model="e-model", policy_hash="a" * 64, distance_cutoff=0.65)


# ---- final routing: every combination

# (status, gate) pairs the coverage step can produce. A gate always comes with cannot_determine.
COVERAGE_OUTCOMES = [
    ("covered", "none", []),
    ("excluded", "none", ["coverage_possible_exclusion"]),
    ("cannot_determine", "none", ["coverage_undetermined"]),
    ("cannot_determine", "low_retrieval_confidence", ["retrieval_low_confidence"]),
    ("cannot_determine", "citation_check_failed", ["citation_check_failed"]),
]


@pytest.mark.parametrize("rules", [AUTO, ESCALATE], ids=["rules_auto", "rules_escalate"])
@pytest.mark.parametrize("status, gate, coverage_reasons", COVERAGE_OUTCOMES)
def test_every_combination_of_rules_and_coverage(rules, status, gate, coverage_reasons):
    route, reasons = final_route(rules, coverage_result(status, gate))
    expected_route = "auto_approve_candidate" if rules is AUTO and status == "covered" and gate == "none" else "escalate"
    assert route == expected_route
    assert reasons == rules.reasons + coverage_reasons


def test_only_one_combination_ends_as_auto_approve_candidate():
    routes = [final_route(rules, coverage_result(status, gate))[0]
              for rules in (AUTO, ESCALATE) for status, gate, _ in COVERAGE_OUTCOMES]
    assert routes.count("auto_approve_candidate") == 1


def test_phase_2_reasons_come_first_and_in_their_original_order():
    _, reasons = final_route(ESCALATE, coverage_result("excluded"))
    assert reasons == ["amount_above_high_threshold", "missing_incident_date", "coverage_possible_exclusion"]


def test_final_route_does_not_change_its_inputs():
    rules, coverage = ESCALATE.model_copy(deep=True), coverage_result("excluded")
    final_route(rules, coverage)
    assert rules == ESCALATE and rules.reasons == ["amount_above_high_threshold", "missing_incident_date"]


def test_a_gate_escalates_even_if_the_status_somehow_says_covered():
    route, reasons = final_route(AUTO, coverage_result("covered", "citation_check_failed"))
    assert route == "escalate"
    assert reasons == ["citation_check_failed"]


def test_gate_reasons_cover_every_gate_name():
    assert set(GATE_REASONS) == {"low_retrieval_confidence", "citation_check_failed"}


def test_the_route_is_never_approved_or_denied():
    for rules in (AUTO, ESCALATE):
        for status, gate, _ in COVERAGE_OUTCOMES:
            route, reasons = final_route(rules, coverage_result(status, gate))
            assert route in ("auto_approve_candidate", "escalate")
            assert not any(word in " ".join(reasons) for word in ("approved", "denied"))


# ---- versions and the dossier

class FakeCoverageModel:
    model = "fake-coverage"
    prompt_version = "fake-prompt"

    def __init__(self):
        self.calls = 0
        self.claim_texts: list[str] = []

    def assess(self, extraction, claim_text, clauses):
        self.calls += 1
        self.claim_texts.append(claim_text)
        return CoverageAnswer(coverage_status="covered", cited_clause_ids=[clauses[0].chunk_id], rationale="Fits.")


def test_versions_come_from_the_models_and_the_index():
    details = {"embedding_model": "emb-1", "policy_hash": "f" * 64, "clauses": 58}
    versions = collect_versions(FakeCoverageModel(), details, cutoff=0.5)
    assert versions == Versions(
        extraction_model=extractor.MODEL, extraction_prompt=extractor.PROMPT_VERSION,
        coverage_model="fake-coverage", coverage_prompt="fake-prompt",
        embedding_model="emb-1", policy_hash="f" * 64, distance_cutoff=0.5,
    )


def test_versions_use_the_default_cutoff():
    details = {"embedding_model": "emb-1", "policy_hash": "f" * 64, "clauses": 58}
    assert collect_versions(FakeCoverageModel(), details).distance_cutoff == DISTANCE_CUTOFF


def test_dossier_holds_everything_and_the_decision():
    dossier = build_dossier(make_extraction(), ESCALATE, coverage_result("excluded"), VERSIONS)
    assert dossier.final_route == "escalate"
    assert dossier.reasons == ["amount_above_high_threshold", "missing_incident_date", "coverage_possible_exclusion"]
    assert dossier.rules_decision == ESCALATE
    assert dossier.versions == VERSIONS
    assert [c.chunk_id for c in dossier.coverage.cited_clauses] == ["en-C2.1"]
    assert len(dossier.coverage.retrieved) == 3


def test_dossier_survives_a_json_round_trip():
    dossier = build_dossier(make_extraction(), AUTO, coverage_result("covered"), VERSIONS)
    assert Dossier.model_validate_json(dossier.model_dump_json()) == dossier


# ---- the whole path with fakes

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


def test_review_of_a_clean_covered_claim_is_an_auto_approve_candidate(collection, embedder):
    model = FakeCoverageModel()
    dossier = review_claim(make_extraction(), CLAIM_TEXT, collection, embedder, model, cutoff=2.0)
    assert dossier.final_route == "auto_approve_candidate"
    assert dossier.reasons == []
    assert dossier.coverage.status == "covered" and dossier.coverage.cited_clauses
    assert dossier.rules_decision == triage(make_extraction())
    assert model.calls == 1
    assert model.claim_texts == [CLAIM_TEXT]


def test_review_records_the_index_versions(collection, embedder):
    dossier = review_claim(make_extraction(), CLAIM_TEXT, collection, embedder, FakeCoverageModel(), cutoff=2.0)
    assert dossier.versions.embedding_model == embedder.model
    assert dossier.versions.policy_hash == index_info(collection)["policy_hash"] == policy_hash()


def test_review_escalates_when_the_rules_escalate_even_if_coverage_is_fine(collection, embedder):
    dossier = review_claim(make_extraction(estimated_amount=9000.0), CLAIM_TEXT, collection, embedder, FakeCoverageModel(), cutoff=2.0)
    assert dossier.coverage.status == "covered"
    assert dossier.final_route == "escalate"
    assert dossier.reasons == ["amount_above_high_threshold"]


def test_review_escalates_without_calling_the_model_when_retrieval_is_not_confident(collection, embedder):
    model = FakeCoverageModel()
    dossier = review_claim(make_extraction(), CLAIM_TEXT, collection, embedder, model, cutoff=-1.0)
    assert dossier.coverage.gate_triggered == "low_retrieval_confidence"
    assert dossier.final_route == "escalate"
    assert dossier.reasons == ["retrieval_low_confidence"]
    assert model.calls == 0


# ---- the readable version

def test_markdown_shows_the_route_reasons_and_cited_text():
    dossier = build_dossier(make_extraction(incident_location="Florida"), ESCALATE, coverage_result("excluded"), VERSIONS)
    text = render_markdown(dossier)
    assert "**Recommendation: escalate**" in text
    assert "coverage_possible_exclusion" in text
    assert "Incident location: Florida" in text
    assert "**[en-C2.1] Title of C2.1** (distance 0.40)\nText of C2.1." in text
    assert "- [en-C2.3] Title of C2.3 (distance 0.45)" in text
    assert "- [en-C1.3] Title of C1.3 (distance 0.70, added because it is always checked)" in text
    assert "extraction-vX" in text and "aaaaaaaaaaaa" in text


def test_markdown_says_not_stated_for_missing_values_and_none_for_no_citations():
    extraction = make_extraction(incident_date=None, claimant_name=None, policy_number=None,
                                 estimated_amount=None, currency=None)
    dossier = build_dossier(extraction, AUTO, coverage_result("cannot_determine", cited=()), VERSIONS)
    text = render_markdown(dossier)
    assert "Claimant: not stated" in text and "Amount: not stated" in text
    assert "### Cited clauses\nNone." in text


def test_markdown_never_says_approved_or_denied():
    for status, gate, _ in COVERAGE_OUTCOMES:
        for rules in (AUTO, ESCALATE):
            text = render_markdown(build_dossier(make_extraction(), rules, coverage_result(status, gate), VERSIONS)).lower()
            assert "approved" not in text and "denied" not in text
