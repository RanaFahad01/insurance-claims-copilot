"""The supporting documents and their answer key must stay consistent with each other and with the claim labels.

The comparison below is written out again here from data/documents/DOCUMENT_LABELING.md, independently of the
comparison code in app/agent, so a mistake in a label or in the code shows up as a disagreement. No API calls.
"""
import json
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS = DATA / "documents"
MANIFEST = json.loads((DOCS / "manifest.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((DATA / "expected_documents.json").read_text(encoding="utf-8"))
EXTRACTIONS = json.loads((DATA / "expected_extractions.json").read_text(encoding="utf-8"))

AMOUNT_TOLERANCE = 0.05
AMOUNT_HIGH = 0.25
LATE_DAYS = 90
FIELDS = {"amount", "currency", "date", "name", "policy_number"}


def claim_extraction(claim_key: str) -> dict:
    return EXTRACTIONS[claim_key.removeprefix("claims/")]


def name_words(name: str) -> list[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return sorted("".join(c if c.isalnum() else " " for c in ascii_name).split())


def policy_key(number: str) -> str:
    return "".join(c for c in number if c.isalnum()).lower()


def reference_mismatches(claim: dict, facts: dict) -> list[tuple[str, str]]:
    found = []
    if claim["currency"] and facts["currency"] and claim["currency"] != facts["currency"]:
        found.append(("currency", "high"))
    elif claim["estimated_amount"]:
        difference = abs(facts["total"] - claim["estimated_amount"]) / claim["estimated_amount"]
        if difference > AMOUNT_TOLERANCE:
            found.append(("amount", "high" if difference > AMOUNT_HIGH else "low"))
    if claim["incident_date"]:
        incident, written = date.fromisoformat(claim["incident_date"]), date.fromisoformat(facts["document_date"])
        if written < incident:
            found.append(("date", "high"))
        elif written > incident + timedelta(days=LATE_DAYS):
            found.append(("date", "low"))
    if claim["claimant_name"] and name_words(claim["claimant_name"]) != name_words(facts["customer_name"]):
        found.append(("name", "high"))
    if claim["policy_number"] and policy_key(claim["policy_number"]) != policy_key(facts["policy_number"]):
        found.append(("policy_number", "high"))
    return sorted(found)


def written_total(text: str, language: str) -> float:
    line = next(l for l in text.splitlines() if l.startswith("TOTAL"))
    number = re.search(r"\d[\d.,]*\d", line).group()
    if language == "es":
        return float(number.replace(".", "").replace(",", "."))
    return float(number.replace(",", ""))


MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
}


def labelled(text: str, *labels: str) -> str:
    """The whole value after one of the labels, so "Marcus Belle" cannot pass for "Marcus Bell"."""
    for line in text.splitlines():
        for label in labels:
            if line.startswith(label):
                return line[len(label):].strip()
    raise AssertionError(f"no line starting with one of {labels}")


def written_date(text: str, language: str) -> date:
    value = labelled(text, "Date: ", "Fecha: ")
    if re.fullmatch(r"\d{4}-\d\d-\d\d", value):
        return date.fromisoformat(value)
    if re.fullmatch(r"\d\d/\d\d/\d{4}", value):
        day, month, year = map(int, value.split("/"))
        return date(year, month, day)
    pattern = r"(\d+) de (\w+) de (\d{4})" if language == "es" else r"(\d+) (\w+) (\d{4})"
    day, month, year = re.fullmatch(pattern, value).groups()
    return date(int(year), MONTHS[language].index(month) + 1, int(day))


def test_every_file_is_in_the_manifest_and_the_answer_key():
    files = {p.stem for p in DOCS.glob("doc_*.txt")}
    assert files == set(MANIFEST) == set(EXPECTED)
    assert all(re.fullmatch(r"doc_\d\d", doc_id) for doc_id in MANIFEST)


@pytest.mark.parametrize("doc_id", sorted(MANIFEST))
def test_manifest_entry_is_well_formed(doc_id):
    entry = MANIFEST[doc_id]
    assert set(entry) == {"file", "claim", "kind", "language"}
    assert entry["file"] == f"{doc_id}.txt" and (DOCS / entry["file"]).exists()
    assert (DATA / entry["claim"]).exists() and entry["claim"].removeprefix("claims/") in EXTRACTIONS
    assert entry["kind"] in {"repair_invoice", "receipt", "payment_receipt", "scan"}
    assert entry["language"] in {"en", "es"}
    assert EXPECTED[doc_id]["claim"] == entry["claim"]


@pytest.mark.parametrize("doc_id", sorted(EXPECTED))
def test_the_document_says_what_the_answer_key_says(doc_id):
    entry, text = EXPECTED[doc_id], (DOCS / f"{doc_id}.txt").read_text(encoding="utf-8")
    if not entry["readable"]:
        assert entry["facts"] is None and entry["mismatches"] == []
        assert not re.search(r"HG|EUR|USD|\d{4}", text), "an unreadable scan must not contain recoverable facts"
        return
    facts, language = entry["facts"], MANIFEST[doc_id]["language"]
    assert text.splitlines()[0] == facts["vendor"]
    assert labelled(text, "Billed to: ", "Received from: ", "Cliente: ") == facts["customer_name"]
    assert labelled(text, "Insurance policy: ", "Póliza de seguro: ") == facts["policy_number"]
    assert labelled(text, "Remarks: ", "Observaciones: ") == facts["remarks"] != ""
    assert written_date(text, language).isoformat() == facts["document_date"]
    assert written_total(text, language) == facts["total"]
    assert facts["currency"] in next(l for l in text.splitlines() if l.startswith("TOTAL"))


@pytest.mark.parametrize("doc_id", sorted(EXPECTED))
def test_the_planted_mismatches_are_exactly_what_the_labeling_rules_produce(doc_id):
    entry = EXPECTED[doc_id]
    if not entry["readable"]:
        return
    expected = sorted((m["field"], m["severity"]) for m in entry["mismatches"])
    assert all(field in FIELDS and severity in ("low", "high") for field, severity in expected)
    assert reference_mismatches(claim_extraction(entry["claim"]), entry["facts"]) == expected


def test_the_set_covers_every_kind_of_mismatch_and_enough_clean_documents():
    readable = [e for e in EXPECTED.values() if e["readable"]]
    clean = [e for e in readable if not e["mismatches"]]
    planted = [e for e in readable if e["mismatches"]]
    assert len(clean) >= 8 and len(planted) >= 10
    assert all(len(e["mismatches"]) == 1 for e in planted), "one planted problem per document keeps the answer unambiguous"
    seen = {(m["field"], m["severity"]) for e in planted for m in e["mismatches"]}
    assert seen >= {("amount", "high"), ("amount", "low"), ("currency", "high"), ("date", "high"), ("date", "low"),
                    ("name", "high"), ("policy_number", "high")}
    assert sum(not e["readable"] for e in EXPECTED.values()) >= 1
    assert {m["language"] for m in MANIFEST.values()} == {"en", "es"}


def test_a_claim_with_a_clean_document_and_a_planted_one_exists():
    by_claim: dict[str, list[dict]] = {}
    for entry in EXPECTED.values():
        by_claim.setdefault(entry["claim"], []).append(entry)
    assert any(any(not e["mismatches"] for e in docs) and any(e["mismatches"] for e in docs) for docs in by_claim.values())
