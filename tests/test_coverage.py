from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.rag.coverage import (
    COVERAGE_MODEL,
    LOW_CONFIDENCE_RATIONALE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    CoverageAnswer,
    CoverageError,
    OpenAICoverageModel,
    assess_coverage,
    build_user_message,
    verify_citations,
)
from app.rag.retriever import DISTANCE_CUTOFF, RetrievedClause
from app.schemas import ClaimExtraction

REQUEST = httpx2.Request("POST", "http://example.invalid/v1/responses")


def make_extraction(**overrides) -> ClaimExtraction:
    base = dict(incident_date=None, incident_location=None, claimant_name=None, policy_number=None, damage_type="water",
                estimated_amount=None, currency=None, language="en",
                summary="A pipe burst under the kitchen sink and flooded the floor.")
    base.update(overrides)
    return ClaimExtraction(**base)


def clause(chunk_id="en-C2.1", distance=0.40, added_by_rule=False) -> RetrievedClause:
    language, clause_id = chunk_id.split("-", 1)
    return RetrievedClause(chunk_id=chunk_id, clause_id=clause_id, language=language, title=f"Title of {clause_id}",
                           text=f"Text of {clause_id}.", distance=distance, added_by_rule=added_by_rule)


class FakeCoverageModel:
    model = "fake-model"
    prompt_version = "fake-prompt"

    def __init__(self, answer: CoverageAnswer):
        self.answer = answer
        self.calls = 0

    def assess(self, extraction, clauses):
        self.calls += 1
        return self.answer


def answer(status="covered", cited=("en-C2.1",), rationale="Because the pipe burst suddenly.") -> CoverageAnswer:
    return CoverageAnswer(coverage_status=status, cited_clause_ids=list(cited), rationale=rationale)


RETRIEVED = [clause("en-C2.1", 0.40), clause("en-C2.3", 0.45), clause("en-C1.3", 0.70, added_by_rule=True)]


# ---- gate 2: citation verification

def test_valid_citation_keeps_the_answer_and_returns_the_clause():
    result = verify_citations(answer("covered", ["en-C2.1"]), RETRIEVED)
    assert result.status == "covered"
    assert result.gate_triggered == "none"
    assert [c.chunk_id for c in result.cited_clauses] == ["en-C2.1"]
    assert result.retrieved == RETRIEVED


def test_invented_clause_id_downgrades_to_cannot_determine_and_keeps_the_original_rationale():
    result = verify_citations(answer("covered", ["en-C9.9"], "The model's reasoning."), RETRIEVED)
    assert result.status == "cannot_determine"
    assert result.gate_triggered == "citation_check_failed"
    assert result.cited_clauses == []
    assert "en-C9.9" in result.rationale
    assert "The model's reasoning." in result.rationale
    assert "covered" in result.rationale


def test_one_invented_id_among_valid_ones_still_fails():
    result = verify_citations(answer("excluded", ["en-C2.1", "en-C7.7"]), RETRIEVED)
    assert (result.status, result.gate_triggered) == ("cannot_determine", "citation_check_failed")


@pytest.mark.parametrize("status", ["covered", "excluded"])
def test_covered_or_excluded_without_any_citation_fails(status):
    result = verify_citations(answer(status, []), RETRIEVED)
    assert (result.status, result.gate_triggered) == ("cannot_determine", "citation_check_failed")
    assert "no clause was cited" in result.rationale


def test_cannot_determine_without_citations_is_left_untouched():
    result = verify_citations(answer("cannot_determine", [], "Nothing addresses fences."), RETRIEVED)
    assert (result.status, result.gate_triggered, result.cited_clauses) == ("cannot_determine", "none", [])
    assert result.rationale == "Nothing addresses fences."


def test_cannot_determine_with_an_invented_id_is_untouched_but_the_invented_clause_is_not_shown():
    result = verify_citations(answer("cannot_determine", ["en-C2.1", "en-C9.9"]), RETRIEVED)
    assert result.gate_triggered == "none"
    assert [c.chunk_id for c in result.cited_clauses] == ["en-C2.1"]


def test_a_clause_added_by_rule_can_be_cited():
    result = verify_citations(answer("excluded", ["en-C1.3"]), RETRIEVED)
    assert (result.status, result.gate_triggered) == ("excluded", "none")


def test_plain_clause_id_is_accepted_when_only_one_language_is_retrieved():
    result = verify_citations(answer("covered", ["C2.1"]), RETRIEVED)
    assert result.status == "covered"
    assert [c.chunk_id for c in result.cited_clauses] == ["en-C2.1"]


def test_plain_clause_id_is_rejected_when_it_could_mean_two_languages():
    both = [clause("en-C2.1"), clause("es-C2.1")]
    result = verify_citations(answer("covered", ["C2.1"]), both)
    assert result.gate_triggered == "citation_check_failed"


def test_duplicate_and_padded_citations_are_collapsed():
    result = verify_citations(answer("covered", ["en-C2.1", " en-C2.1 ", "C2.1"]), RETRIEVED)
    assert [c.chunk_id for c in result.cited_clauses] == ["en-C2.1"]


# ---- gate 1: low retrieval confidence

def test_best_match_beyond_the_cutoff_skips_the_model():
    model = FakeCoverageModel(answer())
    far = [clause("en-C2.1", DISTANCE_CUTOFF + 0.01), clause("en-C2.3", 0.9)]
    result = assess_coverage(make_extraction(), far, model)
    assert (result.status, result.gate_triggered) == ("cannot_determine", "low_retrieval_confidence")
    assert result.rationale == LOW_CONFIDENCE_RATIONALE
    assert model.calls == 0
    assert result.retrieved == far


def test_best_match_exactly_at_the_cutoff_still_reaches_the_model():
    model = FakeCoverageModel(answer())
    result = assess_coverage(make_extraction(), [clause("en-C2.1", DISTANCE_CUTOFF)], model)
    assert result.gate_triggered == "none"
    assert model.calls == 1


def test_nothing_retrieved_skips_the_model():
    model = FakeCoverageModel(answer())
    result = assess_coverage(make_extraction(), [], model)
    assert result.gate_triggered == "low_retrieval_confidence"
    assert model.calls == 0


def test_clauses_added_by_rule_do_not_count_toward_the_confidence_gate():
    model = FakeCoverageModel(answer("excluded", ["en-C1.3"]))
    only_added = [clause("en-C1.3", 0.05, added_by_rule=True)]
    assert assess_coverage(make_extraction(), only_added, model).gate_triggered == "low_retrieval_confidence"
    assert model.calls == 0

    far_search_close_added = [clause("en-C2.1", 0.9), clause("en-C1.3", 0.05, added_by_rule=True)]
    assert assess_coverage(make_extraction(), far_search_close_added, model).gate_triggered == "low_retrieval_confidence"


def test_a_distant_clause_added_by_rule_does_not_trigger_the_gate():
    model = FakeCoverageModel(answer("covered", ["en-C2.1"]))
    result = assess_coverage(make_extraction(), RETRIEVED, model)
    assert result.gate_triggered == "none" and result.status == "covered"
    assert model.calls == 1


def test_the_full_path_runs_the_citation_check_on_the_models_answer():
    model = FakeCoverageModel(answer("covered", ["en-C9.9"]))
    result = assess_coverage(make_extraction(), RETRIEVED, model)
    assert (result.status, result.gate_triggered) == ("cannot_determine", "citation_check_failed")
    assert model.calls == 1


# ---- prompt and message

def test_user_message_contains_the_summary_and_every_clause_with_its_id():
    message = build_user_message(make_extraction(), RETRIEVED)
    assert "A pipe burst under the kitchen sink" in message
    for c in RETRIEVED:
        assert f"[{c.chunk_id}] {c.title}" in message and c.text in message


def test_user_message_does_not_contain_claimant_details_or_the_raw_claim():
    extraction = make_extraction(claimant_name="Marcus Bell", policy_number="HG-2231-88", estimated_amount=800.0)
    message = build_user_message(extraction, RETRIEVED)
    assert "Marcus" not in message and "HG-2231-88" not in message and "800" not in message


def test_user_message_gives_the_incident_location_when_there_is_one():
    message = build_user_message(make_extraction(incident_location="Florida"), RETRIEVED)
    assert "Incident location (from the extraction): Florida" in message


def test_user_message_says_not_stated_when_there_is_no_location():
    message = build_user_message(make_extraction(incident_location=None), RETRIEVED)
    assert "Incident location (from the extraction): not stated" in message


def test_system_prompt_defines_the_three_statuses_and_has_a_version():
    for status in ("covered", "excluded", "cannot_determine"):
        assert status in SYSTEM_PROMPT
    assert PROMPT_VERSION == "coverage-v2" and COVERAGE_MODEL


def test_system_prompt_ties_the_incident_location_to_the_territory_clause():
    assert "incident location" in SYSTEM_PROMPT
    assert "territory clause" in SYSTEM_PROMPT
    assert "If no location is stated, assume the event happened inside the territory" in SYSTEM_PROMPT


# ---- the OpenAI wrapper

def parsed_response(parsed, status="completed", output=()):
    return SimpleNamespace(status=status, incomplete_details=None, output=list(output), output_parsed=parsed)


class FakeResponsesAPI:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def make_model(**kwargs):
    api = FakeResponsesAPI(**kwargs)
    return OpenAICoverageModel(client=SimpleNamespace(responses=api)), api


def test_wrapper_returns_the_parsed_answer_and_sends_the_expected_request():
    expected = answer()
    model, api = make_model(response=parsed_response(expected))
    assert model.assess(make_extraction(), RETRIEVED) == expected
    assert api.kwargs["model"] == COVERAGE_MODEL
    assert api.kwargs["text_format"] is CoverageAnswer
    assert api.kwargs["temperature"] == 0
    assert api.kwargs["input"][0]["content"] == SYSTEM_PROMPT
    assert "[en-C2.1]" in api.kwargs["input"][1]["content"]


def test_wrapper_reports_prompt_version_and_model():
    model, _ = make_model(response=parsed_response(answer()))
    assert (model.model, model.prompt_version) == (COVERAGE_MODEL, PROMPT_VERSION)


def test_incomplete_response_is_an_error():
    response = parsed_response(None, status="incomplete")
    response.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    model, _ = make_model(response=response)
    with pytest.raises(CoverageError) as raised:
        model.assess(make_extraction(), RETRIEVED)
    assert raised.value.kind == "incomplete" and "max_output_tokens" in str(raised.value)


def test_refusal_is_an_error():
    refusal = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="I cannot help")])
    model, _ = make_model(response=parsed_response(None, output=[refusal]))
    with pytest.raises(CoverageError) as raised:
        model.assess(make_extraction(), RETRIEVED)
    assert raised.value.kind == "refusal"


def test_missing_parsed_output_is_an_error():
    model, _ = make_model(response=parsed_response(None))
    with pytest.raises(CoverageError) as raised:
        model.assess(make_extraction(), RETRIEVED)
    assert raised.value.kind == "no_parsed_output"


@pytest.mark.parametrize(
    "error, kind",
    [
        (RateLimitError("simulated", response=httpx2.Response(429, request=REQUEST), body=None), "rate_limited"),
        (APITimeoutError(request=REQUEST), "timeout"),
        (APIConnectionError(request=REQUEST), "connection"),
        (APIStatusError("simulated", response=httpx2.Response(401, request=REQUEST), body=None), "api_error"),
    ],
)
def test_api_errors_are_mapped_to_coverage_errors(error, kind):
    model, _ = make_model(error=error)
    with pytest.raises(CoverageError) as raised:
        model.assess(make_extraction(), RETRIEVED)
    assert raised.value.kind == kind
