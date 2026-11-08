"""The agent test cases must be consistent with the documents and the claim labels. No API calls."""
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
CASES = json.loads((DATA / "agent_cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((DATA / "expected_documents.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((DATA / "documents" / "manifest.json").read_text(encoding="utf-8"))
COVERAGE = json.loads((DATA / "expected_coverage.json").read_text(encoding="utf-8"))

KEYS = {"id", "name", "split", "claim", "documents", "agent_runs", "expected_route", "reasons_include", "reasons_exclude",
        "expected_mismatches", "must_call", "must_not_call", "max_steps", "note"}
TOOLS = {"search_policy_clauses", "get_clause", "compare_documents", "get_policy_terms", "calculate_payout"}
ROUTES = {"auto_approve_candidate", "escalate"}


def test_there_are_enough_cases_and_a_small_development_split():
    assert 25 <= len(CASES) <= 35
    assert [c["id"] for c in CASES] == [f"case_{i:02d}" for i in range(1, len(CASES) + 1)]
    dev = [c for c in CASES if c["split"] == "dev"]
    assert len(dev) == 5 and all(c["split"] in ("dev", "test") for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_is_well_formed(case):
    assert set(case) == KEYS
    assert (DATA / case["claim"]).exists() and case["claim"] in COVERAGE
    assert case["expected_route"] in ROUTES
    assert set(case["must_call"]) <= TOOLS and set(case["must_not_call"]) <= TOOLS
    assert not set(case["must_call"]) & set(case["must_not_call"])
    assert case["agent_runs"] == bool(case["documents"])
    assert set(case["reasons_include"]).isdisjoint(case["reasons_exclude"])
    assert 2 <= case["max_steps"] <= 8
    assert len(set(case["documents"])) == len(case["documents"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_agrees_with_the_documents_it_attaches(case):
    expected = sorted((m["document"], m["field"]) for m in case["expected_mismatches"])
    from_docs = sorted((d, m["field"]) for d in case["documents"] for m in EXPECTED[d]["mismatches"])
    assert expected == from_docs
    unreadable = any(not EXPECTED[d]["readable"] for d in case["documents"])
    assert ("required_check_missing" in case["reasons_include"]) == unreadable
    assert ("document_mismatch" in case["reasons_include"]) == bool(expected)
    if case["documents"]:
        assert "compare_documents" in case["must_call"]
    else:
        assert case["must_call"] == []


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_expected_route_follows_from_the_reasons(case):
    assert (case["expected_route"] == "escalate") == bool(case["reasons_include"])
    if case["expected_route"] == "auto_approve_candidate":
        assert COVERAGE[case["claim"]]["status"] == "covered"


def test_documents_are_attached_to_the_claims_they_were_written_for_unless_a_case_says_otherwise():
    for case in CASES:
        for doc_id in case["documents"]:
            assert MANIFEST[doc_id]["claim"] == case["claim"], (case["id"], doc_id)


def test_the_development_cases_cover_clean_planted_mixed_and_excluded():
    dev = [c for c in CASES if c["split"] == "dev"]
    assert any(c["expected_route"] == "auto_approve_candidate" and c["documents"] for c in dev)
    assert any(len(c["expected_mismatches"]) == 1 and len(c["documents"]) == 1 for c in dev)
    assert any(len(c["documents"]) == 2 and c["expected_mismatches"] for c in dev)
    assert any("coverage_possible_exclusion" in c["reasons_include"] and not c["expected_mismatches"] for c in dev)


def test_every_kind_of_situation_appears_in_the_test_split():
    test = [c for c in CASES if c["split"] == "test"]
    assert any(not c["agent_runs"] for c in test)
    assert any("required_check_missing" in c["reasons_include"] for c in test)
    assert any(c["expected_route"] == "auto_approve_candidate" and c["documents"] for c in test)
    assert any(c["expected_route"] == "escalate" and c["documents"] and not c["expected_mismatches"] and "required_check_missing" not in c["reasons_include"] for c in test)
