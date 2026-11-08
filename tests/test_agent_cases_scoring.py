"""The per-case check, tested on small invented runs. No API calls."""
import copy
import json
from pathlib import Path

import pytest

from scripts.run_agent_cases import transcript
from scripts.score_agent_cases import check_run, load_cases

CASES = load_cases()
CLEAN = CASES["case_01"]        # clean document on a fast-track claim
OVERCHARGED = CASES["case_11"]  # amount mismatch
NO_DOCS = CASES["case_25"]
UNREADABLE = CASES["case_23"]


def good_run(case: dict) -> dict:
    """A run that matches its case exactly."""
    docs = case["documents"]
    return {
        "case": case["id"], "run": 1, "final_route": case["expected_route"], "reasons": list(case["reasons_include"]),
        "mismatches": [{"document": m["document"], "field": m["field"]} for m in case["expected_mismatches"]],
        "tools_called": list(case["must_call"]),
        "agent": {"status": "finished", "steps": 2, "tool_calls": len(docs)} if docs else None,
    }


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_a_run_that_matches_its_case_passes(case_id):
    assert check_run(CASES[case_id], good_run(CASES[case_id])) == []


def test_a_wrong_route_is_reported():
    run = good_run(CLEAN)
    run["final_route"] = "escalate"
    assert check_run(CLEAN, run) == ["route escalate, expected auto_approve_candidate"]


def test_missing_and_unexpected_reasons_are_reported():
    run = good_run(OVERCHARGED)
    run["reasons"] = ["amount_above_high_threshold"]
    assert check_run(OVERCHARGED, run) == ["missing reason document_mismatch"]
    run = good_run(CLEAN)
    run["reasons"] = ["document_mismatch"]
    assert "unexpected reason document_mismatch" in check_run(CLEAN, run)


def test_a_missed_or_invented_mismatch_is_reported():
    run = good_run(OVERCHARGED)
    run["mismatches"] = []
    assert any(p.startswith("mismatches []") for p in check_run(OVERCHARGED, run))
    run = good_run(CLEAN)
    run["mismatches"] = [{"document": "doc_14", "field": "name"}]
    assert any("expected []" in p for p in check_run(CLEAN, run))


def test_a_mismatch_on_the_wrong_document_is_reported():
    mixed = CASES["case_22"]
    run = good_run(mixed)
    (m,) = run["mismatches"]
    run["mismatches"] = [{"document": [d for d in mixed["documents"] if d != m["document"]][0], "field": m["field"]}]
    assert check_run(mixed, run)


def test_the_agent_must_finish_within_the_step_limit():
    run = good_run(CLEAN)
    run["agent"] = {"status": "max_steps_exceeded", "steps": 8, "tool_calls": 4}
    problems = check_run(CLEAN, run)
    assert "agent status max_steps_exceeded" in problems and f"8 steps, limit {CLEAN['max_steps']}" in problems


def test_a_required_tool_that_was_not_called_is_reported():
    run = good_run(CLEAN)
    run["tools_called"] = ["get_clause"]
    assert check_run(CLEAN, run) == ["did not call compare_documents"]


def test_a_forbidden_tool_is_reported():
    case = copy.deepcopy(CLEAN)
    case["must_not_call"] = ["calculate_payout"]
    run = good_run(case)
    run["tools_called"] = ["compare_documents", "calculate_payout"]
    assert check_run(case, run) == ["called calculate_payout"]


def test_a_case_without_documents_expects_no_agent():
    assert check_run(NO_DOCS, good_run(NO_DOCS)) == []
    run = good_run(NO_DOCS)
    run["agent"] = {"status": "finished", "steps": 1, "tool_calls": 0}
    assert check_run(NO_DOCS, run) == ["the agent ran although no documents were attached"]


def test_a_case_with_documents_expects_the_agent_to_have_run():
    run = good_run(CLEAN)
    run["agent"] = None
    assert "the agent did not run" in check_run(CLEAN, run)


def test_a_run_that_failed_is_a_failure_with_its_error():
    assert check_run(CLEAN, {"case": "case_01", "run": 1, "error": "timeout: Request timed out"}) == ["run failed: timeout: Request timed out"]


def test_the_unreadable_case_expects_the_safety_net_and_no_comparison():
    run = good_run(UNREADABLE)
    assert run["reasons"] == ["required_check_missing"] and run["mismatches"] == []
    assert check_run(UNREADABLE, run) == []


def test_the_transcript_shows_the_calls_the_result_and_the_findings():
    run = {
        "case": "case_11", "run": 1, "claim": "claims/claim_01_water_en.txt", "documents": ["doc_22"],
        "phase3_route": "auto_approve_candidate", "phase3_reasons": [], "coverage_status": "covered",
        "tool_trace": [{"step": 1, "tool": "compare_documents", "arguments": '{"document_id": "doc_22"}', "outcome": "ok",
                        "result": '{"agrees_with_claim": false}', "latency_ms": 1200}],
        "agent": {"status": "finished", "steps": 2, "tool_calls": 1, "input_tokens": 900, "output_tokens": 80, "elapsed_ms": 4000},
        "findings": "The total is higher than the claim.", "final_route": "escalate", "reasons": ["document_mismatch"], "review_created": True,
    }
    text = transcript(run)
    for expected in ["case_11 run 1", "compare_documents", "doc_22", "agrees_with_claim", "900+80 tokens", "The total is higher",
                     "FINAL: escalate ['document_mismatch']"]:
        assert expected in text
    assert "ERROR timeout" in transcript({"case": "c", "run": 1, "claim": "x", "documents": [], "error": "timeout: slow"})
    assert "agent: not run" in transcript({**run, "agent": None})
