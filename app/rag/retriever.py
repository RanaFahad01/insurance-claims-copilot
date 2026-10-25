"""Find the policy clauses that matter for a claim. Plain functions: the caller passes in the collection and embedder."""
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel

from ..schemas import ClaimExtraction
from .embeddings import Embedder

TOP_K = 4

# If even the best match is further than this (cosine distance), nothing in the policy is treated as relevant.
# Calibrated on 20 in-scope, 5 gap and 2 off-topic cases (see docs/phase-notes/phase3.md). It is a backstop for text
# far from the policy, not a gap detector: gaps sit as close to the policy as in-scope claims do.
DISTANCE_CUTOFF = 0.65

# How the extracted damage type is phrased in the search query. 'other' and 'unclear' add no words.
DAMAGE_PHRASES = {
    "water": "water damage",
    "fire": "fire damage",
    "theft": "theft",
    "collision": "collision damage to a vehicle or building",
    "other": "",
    "unclear": "",
}


class Clause(BaseModel):
    chunk_id: str     # e.g. "en-C4.1"
    clause_id: str    # e.g. "C4.1", the same in every language
    language: str
    title: str
    text: str


class RetrievedClause(Clause):
    distance: float   # cosine distance: 0 is identical, around 1 is unrelated


def build_query(extraction: ClaimExtraction) -> str:
    """The search query comes from the extraction, not from the raw claim text."""
    phrase = DAMAGE_PHRASES[extraction.damage_type]
    return f"{phrase}: {extraction.summary}" if phrase else extraction.summary


def available_languages(collection: Collection) -> set[str]:
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    return {m["language"] for m in metadatas}


def search_language(extracted_language: str, available: set[str]) -> str | None:
    """Filter to the claim's language when the policy exists in it; otherwise search every language."""
    return extracted_language if extracted_language in available else None


def _to_clause_fields(chunk_id: str, document: str, metadata: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "clause_id": metadata["clause_id"],
        "language": metadata["language"],
        "title": metadata["title"],
        "text": document,
    }


def retrieve_clauses(
    collection: Collection,
    embedder: Embedder,
    query: str,
    language: str | None = None,
    k: int = TOP_K,
) -> list[RetrievedClause]:
    """The k clauses closest to the query, nearest first.

    With a language, only that language's clauses are searched. Without one, every language is
    searched and a clause that exists in several languages is returned once, with its best distance.
    """
    languages = available_languages(collection)
    over_fetch = 1 if language else max(len(languages), 1)
    n_results = min(k * over_fetch, collection.count())
    if n_results == 0:
        return []

    result = collection.query(
        query_embeddings=embedder.embed([query]),
        n_results=n_results,
        where={"language": language} if language else None,
        include=["documents", "metadatas", "distances"],
    )

    found: list[RetrievedClause] = []
    seen_clauses: set[str] = set()
    for chunk_id, document, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if metadata["clause_id"] in seen_clauses:
            continue
        seen_clauses.add(metadata["clause_id"])
        found.append(RetrievedClause(**_to_clause_fields(chunk_id, document, metadata), distance=distance))
    return found[:k]


def retrieve_for_claim(
    collection: Collection,
    embedder: Embedder,
    extraction: ClaimExtraction,
    k: int = TOP_K,
) -> list[RetrievedClause]:
    """Search the policy for a claim, in the claim's language when the policy has that language."""
    language = search_language(extraction.language, available_languages(collection))
    return retrieve_clauses(collection, embedder, build_query(extraction), language=language, k=k)

def get_clause(collection: Collection, chunk_id: str) -> Clause | None:
    """Exact lookup by chunk ID such as 'en-C2.1'. No embeddings: they are weak at exact identifiers."""
    chunk_id = chunk_id.strip()
    if not chunk_id:
        return None
    result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return Clause(**_to_clause_fields(result["ids"][0], result["documents"][0], result["metadatas"][0]))
