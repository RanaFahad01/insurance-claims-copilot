"""data/policy_terms.json must say what the policy clauses say, in both languages. No API calls."""
import json
import re
from pathlib import Path

import pytest

from app.rag.chunker import load_policy_chunks

DATA = Path(__file__).resolve().parent.parent / "data"
TERMS = json.loads((DATA / "policy_terms.json").read_text(encoding="utf-8"))
CHUNKS = {chunk.chunk_id: chunk for chunk in load_policy_chunks()}
OBJECTS = ("home", "contents", "vehicle")


def clause_text(language: str, clause_ids: list[str]) -> str:
    return "\n".join(CHUNKS[f"{language}-{cid}"].text for cid in clause_ids)


def has_number(text: str, value: int, language: str) -> bool:
    """The number written as the clause writes it: 300,000 in English and 300.000 in Spanish."""
    written = f"{value:,}" if language == "en" else f"{value:,}".replace(",", ".")
    return re.search(rf"(?<![\d.,]){re.escape(written)}(?![\d.,])", text) is not None


def test_there_is_one_entry_per_insured_object():
    assert set(TERMS["insured_objects"]) == set(OBJECTS)
    assert TERMS["currency"] == "EUR"


@pytest.mark.parametrize("name", OBJECTS)
def test_every_listed_clause_exists_in_both_languages(name):
    for clause_id in TERMS["insured_objects"][name]["clauses"]:
        assert f"en-{clause_id}" in CHUNKS and f"es-{clause_id}" in CHUNKS


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("name", OBJECTS)
def test_deductible_and_limits_appear_in_the_clause_text(name, language):
    entry = TERMS["insured_objects"][name]
    text = clause_text(language, entry["clauses"])
    assert has_number(text, entry["deductible"], language), f"deductible {entry['deductible']}"
    if entry["limit"] is not None:
        assert has_number(text, entry["limit"], language), f"limit {entry['limit']}"
    for label, value in entry["sub_limits"].items():
        assert has_number(text, value, language), f"{label} {value}"


def test_the_vehicle_has_no_number_because_the_policy_pays_market_value():
    vehicle = TERMS["insured_objects"]["vehicle"]
    assert vehicle["limit"] is None and vehicle["limit_basis"] == "market_value"
    assert "market value" in clause_text("en", ["C7.1"]) and "valor de mercado" in clause_text("es", ["C7.1"])
    for name in ("home", "contents"):
        assert TERMS["insured_objects"][name]["limit_basis"] == "sum_insured"


def test_notification_periods_appear_in_the_clause_text():
    notification = TERMS["notification"]
    en, es = clause_text("en", notification["clauses"]), clause_text("es", notification["clauses"])
    assert f"{notification['report_damage_within_days']} days" in en
    assert f"{notification['report_damage_within_days']} días" in es
    assert f"{notification['police_report_within_hours']} hours" in en
    assert f"{notification['police_report_within_hours']} horas" in es


def test_the_two_languages_agree_on_every_number_in_c7():
    for clause_id in ("C7.1", "C7.2", "C7.3"):
        def numbers(language):
            text = CHUNKS[f"{language}-{clause_id}"].text
            return sorted(int(n.replace(",", "").replace(".", "")) for n in re.findall(r"\d[\d.,]*\d|\d", text))
        assert numbers("en") == numbers("es"), clause_id
