import json
from pathlib import Path

import pytest

from app.agent.compare import (
    AMOUNT_HIGH_SEVERITY,
    AMOUNT_TOLERANCE,
    LATE_DAYS,
    VALUE_MAX_CHARS,
    compare_invoice,
    name_words,
    policy_key,
)
from app.agent.documents import InvoiceExtraction
from app.schemas import ClaimExtraction

DATA = Path(__file__).resolve().parent.parent / "data"


def claim(**overrides) -> ClaimExtraction:
    base = dict(incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell", policy_number="HG-2231-88",
                damage_type="water", estimated_amount=800.0, currency="EUR", language="en", summary="A pipe burst.")
    base.update(overrides)
    return ClaimExtraction(**base)


def invoice(**overrides) -> InvoiceExtraction:
    base = dict(vendor="Hartwell & Pike Plumbing Ltd", document_date="2026-03-14", total=800.0, currency="EUR",
                customer_name="Marcus Bell", policy_number="HG-2231-88", remarks="Paid in full.")
    base.update(overrides)
    return InvoiceExtraction(**base)


def found(c: ClaimExtraction, i: InvoiceExtraction) -> list[tuple[str, str]]:
    return [(m.field, m.severity) for m in compare_invoice("doc_01", c, i)]


def test_constants_are_the_ones_in_the_labeling_rules():
    assert str(AMOUNT_TOLERANCE) == "0.05" and str(AMOUNT_HIGH_SEVERITY) == "0.25" and LATE_DAYS == 90


def test_a_matching_document_has_no_mismatches():
    assert found(claim(), invoice()) == []


# --- amount ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("total", [800.0, 840.0, 760.0, 812.0, 839.99])
def test_amounts_inside_the_tolerance_agree(total):
    assert found(claim(), invoice(total=total)) == []


@pytest.mark.parametrize("total, severity", [
    (840.01, "low"), (759.99, "low"), (700.0, "low"), (1000.0, "low"),      # 1000 is exactly 25% above
    (1000.01, "high"), (600.0, "low"), (599.99, "high"), (520.0, "high"), (1450.0, "high"), (0.0, "high"),
])
def test_amounts_outside_the_tolerance_are_mismatches_with_the_right_severity(total, severity):
    assert found(claim(), invoice(total=total)) == [("amount", severity)]


def test_decimal_arithmetic_keeps_the_exact_boundary_exact():
    # 0.05 * 460 = 23 exactly. In binary floating point the product can land just above or below 23.
    assert found(claim(estimated_amount=460.0), invoice(total=483.0)) == []
    assert found(claim(estimated_amount=460.0), invoice(total=483.01)) == [("amount", "low")]


def test_with_a_zero_claim_any_other_total_is_a_mismatch():
    assert found(claim(estimated_amount=0.0), invoice(total=0.0)) == []
    assert found(claim(estimated_amount=0.0), invoice(total=10.0)) == [("amount", "high")]


def test_different_currencies_report_only_the_currency():
    assert found(claim(), invoice(currency="USD", total=5000.0)) == [("currency", "high")]


def test_an_unstated_currency_on_either_side_does_not_block_the_amount_check():
    assert found(claim(currency=None), invoice(total=1450.0)) == [("amount", "high")]
    assert found(claim(), invoice(currency=None, total=1450.0)) == [("amount", "high")]


# --- date ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("written", ["2026-03-12", "2026-03-13", "2026-06-10"])    # the day itself, and day 90
def test_dates_from_the_incident_day_to_ninety_days_later_agree(written):
    assert found(claim(), invoice(document_date=written)) == []


def test_a_document_dated_before_the_incident_is_a_high_mismatch():
    assert found(claim(), invoice(document_date="2026-03-11")) == [("date", "high")]


def test_a_document_dated_after_ninety_days_is_a_low_mismatch():
    assert found(claim(), invoice(document_date="2026-06-11")) == [("date", "low")]


@pytest.mark.parametrize("written", [None, "", "yesterday", "2026-13-40", "12/03/2026"])
def test_a_date_that_is_missing_or_unusable_is_not_compared(written):
    assert found(claim(), invoice(document_date=written)) == []


def test_a_claim_without_a_date_is_not_compared():
    assert found(claim(incident_date=None), invoice(document_date="2020-01-01")) == []


# --- names and policy numbers ----------------------------------------------------------------------------------

@pytest.mark.parametrize("written", ["marcus bell", "MARCUS BELL", "BELL, Marcus", "Bell Marcus", "  Marcus   Bell "])
def test_names_ignore_case_order_and_punctuation(written):
    assert found(claim(), invoice(customer_name=written)) == []


def test_names_ignore_accents():
    assert found(claim(claimant_name="Raúl Domínguez"), invoice(customer_name="RAUL DOMINGUEZ")) == []


@pytest.mark.parametrize("written", ["Marcus Belle", "Daniel Cooper", "M. Bell", "Marcus"])
def test_a_different_name_is_a_high_mismatch(written):
    assert found(claim(), invoice(customer_name=written)) == [("name", "high")]


@pytest.mark.parametrize("written", ["hg-2231-88", "HG 2231 88", "HG2231 88", " HG-2231-88 "])
def test_policy_numbers_ignore_spaces_hyphens_and_case(written):
    assert found(claim(), invoice(policy_number=written)) == []


@pytest.mark.parametrize("written", ["HG-2231-89", "HG-2231-98", "HG-2213-88", "HG-2231"])
def test_a_different_policy_number_is_a_high_mismatch(written):
    assert found(claim(), invoice(policy_number=written)) == [("policy_number", "high")]


def test_helpers_normalise_as_described():
    assert name_words("BELL, Marcus") == name_words("marcus bell") == ["bell", "marcus"]
    assert policy_key("HG 4812 05") == policy_key("hg-4812-05") == "hg481205"


# --- missing fields --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["document_date", "total", "currency", "customer_name", "policy_number"])
def test_a_field_missing_in_the_document_is_never_a_mismatch(field):
    assert found(claim(), invoice(**{field: None})) == []


@pytest.mark.parametrize("field, other", [
    ("incident_date", "date"), ("claimant_name", "name"), ("policy_number", "policy_number"), ("estimated_amount", "amount"),
])
def test_a_field_missing_in_the_claim_is_never_a_mismatch(field, other):
    wrong_everywhere = invoice(total=99999.0, document_date="2020-01-01", customer_name="Nobody", policy_number="X")
    result = [name for name, _ in found(claim(**{field: None}), wrong_everywhere)]
    assert other not in result
    assert len(result) == 3      # the other three comparisons still run


def test_several_problems_are_all_reported_in_a_fixed_order():
    result = found(claim(), invoice(total=1450.0, document_date="2026-03-01", customer_name="Daniel Cooper", policy_number="HG-0000-00"))
    assert result == [("amount", "high"), ("date", "high"), ("name", "high"), ("policy_number", "high")]


# --- what the agent is shown -----------------------------------------------------------------------------------

def test_document_text_is_clipped_before_it_is_returned():
    long_name = "A" * 500
    (mismatch,) = compare_invoice("doc_01", claim(), invoice(customer_name=long_name))
    assert len(mismatch.document_value) == VALUE_MAX_CHARS + 3 and mismatch.document_value.endswith("...")
    assert mismatch.claim_value == "Marcus Bell" and mismatch.document_id == "doc_01"


def test_mismatch_values_are_text():
    (mismatch,) = compare_invoice("doc_01", claim(), invoice(total=1450.0))
    assert (mismatch.claim_value, mismatch.document_value) == ("800.0", "1450.0")


# --- the answer key --------------------------------------------------------------------------------------------

EXPECTED_DOCUMENTS = json.loads((DATA / "expected_documents.json").read_text(encoding="utf-8"))
EXPECTED_EXTRACTIONS = json.loads((DATA / "expected_extractions.json").read_text(encoding="utf-8"))
READABLE = sorted(d for d, e in EXPECTED_DOCUMENTS.items() if e["readable"])


@pytest.mark.parametrize("doc_id", READABLE)
def test_the_comparison_reproduces_the_labeled_mismatches_for_every_document(doc_id):
    entry = EXPECTED_DOCUMENTS[doc_id]
    labels = {k: v for k, v in EXPECTED_EXTRACTIONS[entry["claim"].removeprefix("claims/")].items() if k != "note"}
    facts = entry["facts"]
    result = compare_invoice(
        doc_id,
        ClaimExtraction(**labels, summary="A claim."),
        InvoiceExtraction(vendor=facts["vendor"], document_date=facts["document_date"], total=facts["total"],
                          currency=facts["currency"], customer_name=facts["customer_name"],
                          policy_number=facts["policy_number"], remarks=facts["remarks"]),
    )
    assert sorted((m.field, m.severity) for m in result) == sorted((m["field"], m["severity"]) for m in entry["mismatches"])
