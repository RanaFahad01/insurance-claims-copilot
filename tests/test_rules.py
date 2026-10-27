from app import rules
from app.rules import triage
from app.schemas import ClaimExtraction


def make_claim(**overrides) -> ClaimExtraction:
    base = dict(
        incident_date="2026-03-12",
        incident_location="Valencia",
        claimant_name="Marcus Bell",
        policy_number="HG-2231-88",
        damage_type="water",
        estimated_amount=800.0,
        currency="EUR",
        language="en",
        summary="Burst pipe flooded the kitchen.",
    )
    base.update(overrides)
    return ClaimExtraction(**base)


def test_clean_small_claim_is_auto_approve_candidate():
    assert triage(make_claim()).route == "auto_approve_candidate"


def test_missing_date_escalates():
    result = triage(make_claim(incident_date=None))
    assert result.route == "escalate"
    assert "missing_incident_date" in result.reasons


def test_amount_above_threshold_escalates():
    assert triage(make_claim(estimated_amount=1500.0)).route == "escalate"


def test_non_eur_currency_escalates():
    assert triage(make_claim(currency="USD")).route == "escalate"


def test_unclear_damage_escalates():
    assert triage(make_claim(damage_type="unclear")).route == "escalate"


def test_zero_amount_escalates():
    assert triage(make_claim(estimated_amount=0.0)).route == "escalate"


def test_clean_claim_has_no_reasons_and_base_score():
    result = triage(make_claim())
    assert result.reasons == []
    assert result.severity_score == 10


def test_amount_exactly_at_auto_threshold_is_auto_approve_candidate():
    result = triage(make_claim(estimated_amount=1000.0))
    assert result.route == "auto_approve_candidate"
    assert result.reasons == []


def test_amount_just_above_auto_threshold_escalates():
    result = triage(make_claim(estimated_amount=1000.01))
    assert result.route == "escalate"
    assert result.reasons == ["amount_above_auto_threshold"]
    assert result.severity_score == 30


def test_amount_exactly_at_high_threshold_is_only_above_auto_threshold():
    result = triage(make_claim(estimated_amount=5000.0))
    assert result.reasons == ["amount_above_auto_threshold"]
    assert result.severity_score == 30


def test_amount_just_above_high_threshold_is_high():
    result = triage(make_claim(estimated_amount=5000.01))
    assert result.reasons == ["amount_above_high_threshold"]
    assert result.severity_score == 50


def test_several_missing_fields_are_all_listed():
    result = triage(make_claim(incident_date=None, claimant_name=None, policy_number=None))
    assert result.route == "escalate"
    assert result.reasons == ["missing_incident_date", "missing_claimant_name", "missing_policy_number"]
    assert result.severity_score == 40


def test_missing_amount_escalates_without_currency_reason():
    result = triage(make_claim(estimated_amount=None, currency=None))
    assert result.route == "escalate"
    assert result.reasons == ["missing_estimated_amount"]
    assert result.severity_score == 20


def test_missing_currency_with_an_amount_escalates():
    result = triage(make_claim(currency=None))
    assert result.reasons == ["currency_not_eur"]
    assert result.severity_score == 30


def test_large_non_eur_amount_only_gets_currency_reason():
    result = triage(make_claim(estimated_amount=50000.0, currency="USD"))
    assert result.reasons == ["currency_not_eur"]
    assert result.severity_score == 30


def test_negative_amount_escalates():
    result = triage(make_claim(estimated_amount=-50.0))
    assert result.route == "escalate"
    assert result.reasons == ["non_positive_amount"]


def test_zero_amount_reason():
    assert triage(make_claim(estimated_amount=0.0)).reasons == ["non_positive_amount"]


def test_clean_other_damage_type_is_auto_approve_candidate():
    result = triage(make_claim(damage_type="other"))
    assert result.route == "auto_approve_candidate"
    assert result.severity_score == 5


def test_unclear_damage_reason_and_score():
    result = triage(make_claim(damage_type="unclear"))
    assert result.reasons == ["damage_type_unclear"]
    assert result.severity_score == 15


def test_clean_fire_claim_score():
    result = triage(make_claim(damage_type="fire"))
    assert result.route == "auto_approve_candidate"
    assert result.severity_score == 20


def test_multiple_reasons_accumulate_in_order():
    result = triage(make_claim(incident_date=None, damage_type="unclear", estimated_amount=6000.0))
    assert result.reasons == ["missing_incident_date", "damage_type_unclear", "amount_above_high_threshold"]
    assert result.severity_score == 65


def test_missing_incident_location_does_not_escalate():
    # Most claims never state a place, so the location is not a required field.
    result = triage(make_claim(incident_location=None))
    assert result.route == "auto_approve_candidate"
    assert result.reasons == []


def test_location_outside_the_policy_territory_does_not_change_this_decision():
    # The territory is checked in the coverage step, not by these rules.
    result = triage(make_claim(incident_location="Miami"))
    assert result.route == "auto_approve_candidate"
    assert result.severity_score == 10


def test_severity_score_is_capped_at_100(monkeypatch):
    # With the real weights the highest possible score is 90, so raise a weight to reach the cap.
    monkeypatch.setitem(rules.TYPE_WEIGHT, "fire", 200)
    assert triage(make_claim(damage_type="fire")).severity_score == 100
