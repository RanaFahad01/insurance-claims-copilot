"""The coverage labels must stay consistent with the claim files and the policy. No API calls."""
import json
from pathlib import Path

import pytest

from app.rag.chunker import load_policy_chunks

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = json.loads((DATA / "expected_coverage.json").read_text(encoding="utf-8"))
POLICY_CLAUSE_IDS = {chunk.clause_id for chunk in load_policy_chunks()}
STATUSES = {"covered", "excluded", "cannot_determine"}


def test_every_claim_file_has_a_label_and_every_label_has_a_file():
    claim_files = {f"claims/{p.name}" for p in (DATA / "claims").glob("*.txt")}
    labelled_claims = {key for key in LABELS if key.startswith("claims/")}
    assert claim_files == labelled_claims
    for key in LABELS:
        assert (DATA / key).exists(), key


def test_every_coverage_attack_has_a_label():
    attacks = {f"adversarial/{p.name}" for p in (DATA / "adversarial").glob("cov_*.txt")}
    assert attacks and attacks <= set(LABELS)


@pytest.mark.parametrize("key", sorted(LABELS))
def test_label_is_well_formed(key):
    label = LABELS[key]
    assert set(label) == {"status", "clause_ids", "borderline", "note"}
    assert label["status"] in STATUSES
    assert isinstance(label["borderline"], bool) and label["note"]
    assert set(label["clause_ids"]) <= POLICY_CLAUSE_IDS, "clause IDs are written without a language prefix and must exist"
    assert len(set(label["clause_ids"])) == len(label["clause_ids"])


def test_covered_and_excluded_labels_name_at_least_one_clause():
    for key, label in LABELS.items():
        if label["status"] in ("covered", "excluded"):
            assert label["clause_ids"], key


def test_every_claim_and_coverage_attack_also_has_expected_extraction_values():
    extractions = json.loads((DATA / "expected_extractions.json").read_text(encoding="utf-8"))
    adversarial = json.loads((DATA / "expected_adversarial.json").read_text(encoding="utf-8"))
    for key in LABELS:
        folder, name = key.split("/")
        assert name in (extractions if folder == "claims" else adversarial), key
