import pytest
from pydantic import ValidationError

from app.extractor import PROMPT_VERSION, SYSTEM_PROMPT
from app.schemas import ClaimExtraction

VALID = dict(
    incident_date=None, incident_location=None, claimant_name=None, policy_number=None, damage_type="water",
    estimated_amount=None, currency=None, language="en", summary="A pipe burst.",
)


def test_every_field_is_required_with_no_default():
    # Strict structured outputs reject fields that have defaults, so every field must be required.
    schema = ClaimExtraction.model_json_schema()
    assert set(schema["required"]) == set(ClaimExtraction.model_fields)
    assert all("default" not in prop for prop in schema["properties"].values())


def test_incident_location_must_be_given_even_when_it_is_null():
    without = {k: v for k, v in VALID.items() if k != "incident_location"}
    with pytest.raises(ValidationError):
        ClaimExtraction(**without)
    assert ClaimExtraction(**VALID).incident_location is None


def test_incident_location_sits_right_after_incident_date():
    assert list(ClaimExtraction.model_fields)[:2] == ["incident_date", "incident_location"]


def test_incident_location_is_free_text_or_null():
    assert ClaimExtraction(**{**VALID, "incident_location": "Lisbon, Portugal"}).incident_location == "Lisbon, Portugal"


def test_extraction_prompt_has_a_version_and_covers_the_location_rules():
    assert PROMPT_VERSION == "extraction-v2"
    assert "incident_location" in SYSTEM_PROMPT
    for rule in ("letterhead", "claimant's own address", "Do not translate", "Never add a country"):
        assert rule in SYSTEM_PROMPT
