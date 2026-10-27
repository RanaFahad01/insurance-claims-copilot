import uuid

import chromadb
import pytest
from fakes import FakeEmbedder

from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.index import ingest
from app.rag.retriever import (
    DISTANCE_CUTOFF,
    Clause,
    available_languages,
    build_query,
    get_clause,
    retrieve_clauses,
    retrieve_context,
    retrieve_for_claim,
    search_language,
)
from app.schemas import ClaimExtraction


def make_extraction(**overrides) -> ClaimExtraction:
    base = dict(
        incident_date="2026-03-12",
        incident_location=None,
        claimant_name="Marcus Bell",
        policy_number="HG-2231-88",
        damage_type="water",
        estimated_amount=800.0,
        currency="EUR",
        language="en",
        summary="A pipe burst under the kitchen sink and flooded the floor.",
    )
    base.update(overrides)
    return ClaimExtraction(**base)


@pytest.fixture(scope="module")
def embedder():
    return FakeEmbedder()


@pytest.fixture
def collection(embedder):
    client = chromadb.EphemeralClient()
    name = f"test_{uuid.uuid4().hex}"
    ingest(client, load_policy_chunks(), embedder, policy_hash(), collection_name=name)
    yield client.get_collection(name, embedding_function=None)
    client.delete_collection(name)


# ---- query construction

def test_query_combines_damage_type_and_summary():
    assert build_query(make_extraction()) == "water damage: A pipe burst under the kitchen sink and flooded the floor."


def test_query_for_other_and_unclear_uses_only_the_summary():
    for damage_type in ("other", "unclear"):
        assert build_query(make_extraction(damage_type=damage_type)) == make_extraction().summary


def test_query_does_not_use_the_claimant_name_or_policy_number():
    query = build_query(make_extraction())
    assert "Marcus" not in query and "HG-2231-88" not in query


# ---- language handling

def test_available_languages_come_from_the_index(collection):
    assert available_languages(collection) == {"en", "es"}


def test_search_language_filters_only_when_the_policy_has_that_language():
    available = {"en", "es"}
    assert search_language("en", available) == "en"
    assert search_language("es", available) == "es"
    assert search_language("other", available) is None


def test_english_claim_gets_english_clauses_only(collection, embedder):
    clauses = retrieve_for_claim(collection, embedder, make_extraction(language="en"))
    assert len(clauses) == 4
    assert {c.language for c in clauses} == {"en"}
    assert all(c.chunk_id == f"en-{c.clause_id}" for c in clauses)


def test_spanish_claim_gets_spanish_clauses_only(collection, embedder):
    clauses = retrieve_for_claim(collection, embedder, make_extraction(language="es"))
    assert {c.language for c in clauses} == {"es"}


def test_unsupported_language_searches_all_languages_without_duplicate_clauses(collection, embedder):
    clauses = retrieve_for_claim(collection, embedder, make_extraction(language="other"))
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids)) == 4


# ---- ranking and shape

def test_results_are_nearest_first_with_valid_distances(collection, embedder):
    clauses = retrieve_for_claim(collection, embedder, make_extraction())
    distances = [c.distance for c in clauses]
    assert distances == sorted(distances)
    assert all(0.0 <= d <= 2.0 for d in distances)


def test_a_clause_searched_by_its_own_text_comes_back_first_at_distance_zero(collection, embedder):
    target = next(c for c in load_policy_chunks() if c.chunk_id == "en-C5.3")
    clauses = retrieve_clauses(collection, embedder, target.document, language="en")
    assert clauses[0].chunk_id == "en-C5.3"
    assert 0.0 <= clauses[0].distance < 1e-6   # never slightly negative, which would print as -0.00
    assert clauses[0].title == target.title
    assert clauses[0].text == target.text


def test_unfiltered_search_returns_the_best_language_version_of_a_clause(collection, embedder):
    target = next(c for c in load_policy_chunks() if c.chunk_id == "es-C5.3")
    clauses = retrieve_clauses(collection, embedder, target.document, language=None)
    assert clauses[0].chunk_id == "es-C5.3"


def test_k_can_exceed_the_number_of_clauses_without_error(collection, embedder):
    clauses = retrieve_clauses(collection, embedder, "water", language="en", k=100)
    assert len(clauses) == 29


def test_empty_index_returns_nothing(embedder):
    client = chromadb.EphemeralClient()
    name = f"test_{uuid.uuid4().hex}"
    empty = client.create_collection(name, embedding_function=None)
    assert retrieve_clauses(empty, embedder, "water") == []
    client.delete_collection(name)


# ---- context for the coverage step: search results plus required clauses

def clause_text_extraction(chunk_id: str, language: str):
    """An extraction whose search query is exactly one clause's text, so that clause is the top result."""
    target = next(c for c in load_policy_chunks() if c.chunk_id == chunk_id)
    return make_extraction(damage_type="other", language=language, summary=target.document)


def test_context_adds_the_required_clause_when_the_search_did_not_return_it(collection, embedder):
    extraction = clause_text_extraction("en-C5.3", "en")
    context = retrieve_context(collection, embedder, extraction, k=1)
    assert [c.chunk_id for c in context] == ["en-C5.3", "en-C1.3"]
    assert [c.added_by_rule for c in context] == [False, True]
    assert context[0].distance < 1e-6
    assert 0.0 <= context[1].distance <= 2.0  # a real distance to the query, not a placeholder


def test_context_does_not_duplicate_a_required_clause_the_search_already_found(collection, embedder):
    target = next(c for c in load_policy_chunks() if c.chunk_id == "en-C1.3")
    extraction = make_extraction(damage_type="other", summary=target.document)
    context = retrieve_context(collection, embedder, extraction)
    assert [c.clause_id for c in context].count("C1.3") == 1
    assert all(c.added_by_rule is False for c in context)


def test_context_for_a_spanish_claim_adds_the_spanish_required_clause(collection, embedder):
    context = retrieve_context(collection, embedder, clause_text_extraction("es-C5.3", "es"), k=1)
    assert [c.chunk_id for c in context] == ["es-C5.3", "es-C1.3"]
    assert context[1].added_by_rule is True


def test_context_for_an_unsupported_language_adds_the_english_required_clause(collection, embedder):
    context = retrieve_context(collection, embedder, clause_text_extraction("en-C5.3", "other"), k=1)
    assert [c.chunk_id for c in context] == ["en-C5.3", "en-C1.3"]
    assert context[1].added_by_rule is True


def test_context_from_an_empty_index_is_empty(embedder):
    client = chromadb.EphemeralClient()
    name = f"test_{uuid.uuid4().hex}"
    empty = client.create_collection(name, embedding_function=None)
    assert retrieve_context(empty, embedder, make_extraction()) == []
    client.delete_collection(name)


def test_distance_cutoff_is_a_plausible_cosine_distance():
    assert 0.0 < DISTANCE_CUTOFF < 1.0


# ---- exact lookup

def test_get_clause_returns_the_exact_clause(collection):
    clause = get_clause(collection, "en-C2.1")
    assert isinstance(clause, Clause)
    assert (clause.clause_id, clause.language, clause.title) == ("C2.1", "en", "Water damage: covered")
    assert "sudden and unforeseen" in clause.text


def test_get_clause_works_for_spanish_and_ignores_surrounding_whitespace(collection):
    clause = get_clause(collection, "  es-C2.1 ")
    assert clause is not None and clause.language == "es"


@pytest.mark.parametrize("missing", ["en-C99.9", "C2.1", "", "   ", "nonsense"])
def test_get_clause_returns_none_for_unknown_ids(collection, missing):
    assert get_clause(collection, missing) is None
