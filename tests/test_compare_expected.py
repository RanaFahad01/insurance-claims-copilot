import pytest

from scripts.compare_expected import FIELDS, field_matches


def test_incident_location_is_one_of_the_scored_fields():
    assert "incident_location" in FIELDS


@pytest.mark.parametrize(
    "field, got, expected",
    [
        ("incident_date", "2026-03-12", "2026-03-12"),
        ("estimated_amount", 800.0, 800.0),
        ("policy_number", "HG 4812 05", "HG 4812 05"),
        ("currency", None, None),
    ],
)
def test_other_fields_match_only_on_exact_equality(field, got, expected):
    assert field_matches(field, got, expected)


@pytest.mark.parametrize(
    "field, got, expected",
    [
        ("incident_date", "2026-03-13", "2026-03-12"),
        ("claimant_name", "marcus bell", "Marcus Bell"),   # case still matters outside the location field
        ("policy_number", "HG-4812-05", "HG 4812 05"),
        ("currency", "EUR", None),
    ],
)
def test_other_fields_are_not_matched_loosely(field, got, expected):
    assert not field_matches(field, got, expected)


@pytest.mark.parametrize(
    "got, expected",
    [
        ("Miami", "Miami"),
        ("MIAMI", "miami"),
        ("Londres", "Londres"),
        ("Valéncia", "Valencia"),          # accents are ignored
        ("  Miami,   Florida ", "Miami"),  # more detail than expected still counts
        ("Lisbon, Portugal", "Lisbon, Portugal"),
        (None, None),
    ],
)
def test_location_matches_ignore_case_accents_and_extra_detail(got, expected):
    assert field_matches("incident_location", got, expected)


@pytest.mark.parametrize(
    "got, expected",
    [
        ("Lisbon", "Lisbon, Portugal"),     # missing the country the claim stated
        ("Barcelona, Spain", "Orlando"),    # the spoofed location
        ("Madrid", "Lyon"),
        ("Dublin", None),                   # a place where none is expected
        (None, "Florida"),                  # a place that was missed
    ],
)
def test_location_mismatches_are_detected(got, expected):
    assert not field_matches("incident_location", got, expected)
