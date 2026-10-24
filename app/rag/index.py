"""The local vector index of policy clauses (Chroma). Ingest is repeatable and checks what built the index."""
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel

from .chunker import PolicyChunk
from .embeddings import Embedder

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
COLLECTION_NAME = "policy_clauses"
HNSW_SPACE = "cosine"


class PolicyIndexError(Exception):
    """The index cannot be used. `kind` is one of: missing, empty, model_mismatch."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class IngestReport(BaseModel):
    embedding_model: str
    policy_hash: str
    total: int       # clauses in the index afterwards
    upserted: int    # clauses embedded and written in this run
    removed: int     # clauses deleted because they are no longer in the policy
    skipped: bool    # True when the index was already up to date and nothing was embedded


def open_client(path: Path = DEFAULT_INDEX_DIR) -> ClientAPI:
    return chromadb.PersistentClient(path=str(path))


def _collection_names(client: ClientAPI) -> set[str]:
    return {c.name for c in client.list_collections()}


def _create(client: ClientAPI, name: str, embedding_model: str, policy_hash: str) -> Collection:
    return client.create_collection(
        name=name,
        embedding_function=None,  # vectors are always passed in explicitly
        configuration={"hnsw": {"space": HNSW_SPACE}},
        metadata={"embedding_model": embedding_model, "policy_hash": policy_hash},
    )


def ingest(
    client: ClientAPI,
    chunks: list[PolicyChunk],
    embedder: Embedder,
    policy_hash: str,
    collection_name: str = COLLECTION_NAME,
    rebuild: bool = False,
) -> IngestReport:
    """Write the clauses into the index. Safe to run again: the same clause always has the same ID.

    - Nothing is embedded when the index already matches this policy and embedding model.
    - An index built with a different embedding model is refused unless rebuild=True.
    - Clauses that are no longer in the policy are removed.
    """
    names = _collection_names(client)
    if rebuild and collection_name in names:
        client.delete_collection(collection_name)
        names.discard(collection_name)

    if collection_name not in names:
        collection = _create(client, collection_name, embedder.model, policy_hash)
    else:
        collection = client.get_collection(collection_name, embedding_function=None)
        built_with = (collection.metadata or {}).get("embedding_model")
        if built_with != embedder.model:
            raise PolicyIndexError(
                "model_mismatch",
                f"The index was built with embedding model '{built_with}' but '{embedder.model}' is configured. "
                f"Rebuild it with: python -m scripts.ingest_policy --rebuild",
            )
        unchanged = (collection.metadata or {}).get("policy_hash") == policy_hash and collection.count() == len(chunks)
        if unchanged:
            return IngestReport(embedding_model=embedder.model, policy_hash=policy_hash, total=collection.count(),
                                upserted=0, removed=0, skipped=True)

    vectors = embedder.embed([chunk.document for chunk in chunks])
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=vectors,
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata() for chunk in chunks],
    )

    current_ids = {chunk.chunk_id for chunk in chunks}
    stale = [i for i in collection.get(include=[])["ids"] if i not in current_ids]
    if stale:
        collection.delete(ids=stale)

    collection.modify(metadata={"embedding_model": embedder.model, "policy_hash": policy_hash})
    return IngestReport(embedding_model=embedder.model, policy_hash=policy_hash, total=collection.count(),
                        upserted=len(chunks), removed=len(stale), skipped=False)


def check_index(client: ClientAPI, embedding_model: str, collection_name: str = COLLECTION_NAME) -> Collection:
    """Return the collection if it can be searched, otherwise raise PolicyIndexError."""
    if collection_name not in _collection_names(client):
        raise PolicyIndexError("missing", "The policy index does not exist. Build it with: python -m scripts.ingest_policy")
    collection = client.get_collection(collection_name, embedding_function=None)
    if collection.count() == 0:
        raise PolicyIndexError("empty", "The policy index is empty. Build it with: python -m scripts.ingest_policy")
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != embedding_model:
        raise PolicyIndexError(
            "model_mismatch",
            f"The index was built with embedding model '{built_with}' but '{embedding_model}' is configured. "
            f"Rebuild it with: python -m scripts.ingest_policy --rebuild",
        )
    return collection


def index_info(collection: Collection) -> dict[str, str | int]:
    """What built this index, for the versions block of every result."""
    metadata = collection.metadata or {}
    return {
        "embedding_model": str(metadata.get("embedding_model", "")),
        "policy_hash": str(metadata.get("policy_hash", "")),
        "clauses": collection.count(),
    }
