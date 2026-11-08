import pytest

from app.agent.state import InvestigationState, Mismatch, RunContext
from app.schemas import ClaimExtraction


def make_state(**kwargs) -> InvestigationState:
    return InvestigationState(claim_id="abc123", **kwargs)


def make_mismatch(document_id: str = "doc_1") -> Mismatch:
    return Mismatch(document_id=document_id, field="amount", claim_value="800", document_value="1450", severity="high")


def test_a_new_state_is_empty_and_has_not_run():
    state = make_state()
    assert state.agent_status == "not_run"
    assert state.compared_document_ids == [] and state.mismatches == [] and state.clause_ids_seen == []
    assert state.steps == 0 and state.tool_calls == 0 and state.agent_text is None and state.payout is None


def test_with_no_documents_there_is_nothing_to_check():
    state = make_state()
    assert not state.documents_attached
    assert state.required_checks_done


def test_checks_are_not_done_until_every_document_is_compared():
    state = make_state(document_ids=["doc_1", "doc_2"])
    assert state.documents_attached
    assert not state.required_checks_done
    state.mark_compared("doc_1")
    assert not state.required_checks_done
    state.mark_compared("doc_2")
    assert state.required_checks_done


def test_marking_a_document_twice_counts_once():
    state = make_state(document_ids=["doc_1"])
    state.mark_compared("doc_1")
    state.mark_compared("doc_1")
    assert state.compared_document_ids == ["doc_1"]


def test_a_document_that_is_not_attached_cannot_be_marked_compared():
    state = make_state(document_ids=["doc_1"])
    with pytest.raises(ValueError):
        state.mark_compared("doc_9")
    assert state.compared_document_ids == []
    assert not state.required_checks_done


def test_duplicate_attached_ids_are_collapsed_in_order():
    state = make_state(document_ids=["doc_2", "doc_1", "doc_2"])
    assert state.document_ids == ["doc_2", "doc_1"]


def test_mismatches_and_seen_clauses_accumulate():
    state = make_state(document_ids=["doc_1"])
    state.add_mismatches([make_mismatch()])
    state.note_clause("en-C2.1")
    state.note_clause("en-C2.1")
    state.note_clause("en-C7.2")
    assert len(state.mismatches) == 1
    assert state.clause_ids_seen == ["en-C2.1", "en-C7.2"]


def test_a_mismatch_needs_a_known_severity():
    with pytest.raises(ValueError):
        Mismatch(document_id="d", field="amount", claim_value="1", document_value="2", severity="critical")


def test_the_status_must_be_a_known_one():
    with pytest.raises(ValueError):
        make_state(agent_status="done")


def test_the_run_context_reads_ids_from_the_state():
    extraction = ClaimExtraction(
        incident_date=None, incident_location=None, claimant_name=None, policy_number=None, damage_type="water",
        estimated_amount=None, currency=None, language="en", summary="A pipe burst.",
    )
    ctx = RunContext(extraction=extraction, state=make_state(document_ids=["doc_1"]))
    assert ctx.claim_id == "abc123"
    assert ctx.document_ids == ["doc_1"]


def test_recording_a_comparison_replaces_that_documents_earlier_mismatches_and_marks_it_done():
    state = make_state(document_ids=["doc_1", "doc_2"])
    state.record_comparison("doc_1", [make_mismatch("doc_1")])
    state.record_comparison("doc_2", [make_mismatch("doc_2")])
    state.record_comparison("doc_1", [])
    assert [m.document_id for m in state.mismatches] == ["doc_2"]
    assert state.compared_document_ids == ["doc_1", "doc_2"] and state.required_checks_done


def test_recording_a_comparison_for_a_document_that_is_not_attached_changes_nothing():
    state = make_state(document_ids=["doc_1"])
    with pytest.raises(ValueError):
        state.record_comparison("doc_9", [make_mismatch("doc_9")])
    assert state.mismatches == [] and state.compared_document_ids == []


def test_a_mismatch_must_name_a_known_field():
    with pytest.raises(ValueError):
        Mismatch(document_id="d", field="vendor", claim_value="1", document_value="2", severity="low")
