import pytest
from fastapi.testclient import TestClient

from app import main
from app.extractor import ExtractionError
from app.schemas import ClaimExtraction

client = TestClient(main.app)

VALID_TEXT = "Hi, this is Marcus Bell, policy HG-2231-88. A pipe burst under the kitchen sink."


def make_extraction() -> ClaimExtraction:
    return ClaimExtraction(
        incident_date="2026-03-12",
        claimant_name="Marcus Bell",
        policy_number="HG-2231-88",
        damage_type="water",
        estimated_amount=800.0,
        currency="EUR",
        language="en",
        summary="Burst pipe flooded the kitchen.",
    )


@pytest.fixture
def calls(monkeypatch):
    """Replace the LLM call with a stub and record every text it receives."""
    received: list[str] = []

    def fake_extract(text: str) -> ClaimExtraction:
        received.append(text)
        return make_extraction()

    monkeypatch.setattr(main, "extract_claim", fake_extract)
    return received


def test_valid_claim_returns_extraction_and_decision(calls):
    response = client.post("/claims/triage", json={"text": VALID_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["extraction"]["claimant_name"] == "Marcus Bell"
    assert body["decision"]["route"] == "auto_approve_candidate"
    assert body["decision"]["severity_score"] == 10
    assert body["decision"]["reasons"] == []
    assert calls == [VALID_TEXT]


@pytest.mark.parametrize(
    "kind, expected_status",
    [
        ("rate_limited", 503),
        ("timeout", 503),
        ("connection", 503),
        ("api_error", 502),
        ("incomplete", 502),
        ("refusal", 502),
        ("no_parsed_output", 502),
    ],
)
def test_extraction_errors_map_to_http_status(monkeypatch, kind, expected_status):
    def failing_extract(text: str):
        raise ExtractionError(kind, "something went wrong")

    monkeypatch.setattr(main, "extract_claim", failing_extract)
    response = client.post("/claims/triage", json={"text": VALID_TEXT})
    assert response.status_code == expected_status
    assert response.json()["detail"] == f"{kind}: something went wrong"


def test_text_under_minimum_length_is_rejected_before_llm_call(calls):
    response = client.post("/claims/triage", json={"text": "x" * 19})
    assert response.status_code == 422
    assert calls == []


def test_text_over_maximum_length_is_rejected_before_llm_call(calls):
    response = client.post("/claims/triage", json={"text": "x" * 20001})
    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize("length", [20, 20000])
def test_text_at_length_limits_is_accepted(calls, length):
    response = client.post("/claims/triage", json={"text": "x" * length})
    assert response.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [{}, {"text": None}, {"text": 12345678901234567890}, {"other": VALID_TEXT}])
def test_missing_or_wrong_type_text_is_rejected(calls, payload):
    response = client.post("/claims/triage", json=payload)
    assert response.status_code == 422
    assert calls == []
