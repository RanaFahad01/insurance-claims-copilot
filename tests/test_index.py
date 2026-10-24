import uuid

import chromadb
import pytest
from fakes import FakeEmbedder

from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.index import PolicyIndexError, check_index, index_info, ingest


@pytest.fixture
def client():
    return chromadb.EphemeralClient()  # in-memory, and shared by all clients in this process


@pytest.fixture
def name(client):
    """A unique collection name per test, so tests cannot leak into each other."""
    collection_name = f"test_{uuid.uuid4().hex}"
    yield collection_name
    if collection_name in {c.name for c in client.list_collections()}:
        client.delete_collection(collection_name)


@pytest.fixture(scope="module")
def chunks():
    return load_policy_chunks()


@pytest.fixture(scope="module")
def version():
    return policy_hash()


def test_ingest_writes_every_clause_with_metadata(client, name, chunks, version):
    report = ingest(client, chunks, FakeEmbedder(), version, collection_name=name)
    assert (report.total, report.upserted, report.removed, report.skipped) == (58, 58, 0, False)

    collection = client.get_collection(name, embedding_function=None)
    stored = collection.get(ids=["en-C2.1"], include=["metadatas", "documents"])
    assert stored["metadatas"][0] == {
        "clause_id": "C2.1", "title": "Water damage: covered", "language": "en",
        "source_file": "policy_en.md", "section": 2,
    }
    assert "sudden and unforeseen" in stored["documents"][0]


def test_index_uses_cosine_distance_and_records_what_built_it(client, name, chunks, version):
    ingest(client, chunks, FakeEmbedder(model="m1"), version, collection_name=name)
    collection = client.get_collection(name, embedding_function=None)
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert collection.metadata == {"embedding_model": "m1", "policy_hash": version}


def test_running_ingest_twice_does_not_duplicate_or_re_embed(client, name, chunks, version):
    embedder = FakeEmbedder()
    first = ingest(client, chunks, embedder, version, collection_name=name)
    second = ingest(client, chunks, embedder, version, collection_name=name)
    assert first.total == second.total == 58
    assert second.skipped is True and second.upserted == 0
    assert embedder.calls == 1  # the second run embedded nothing


def test_changed_policy_is_re_embedded_and_removed_clauses_are_deleted(client, name, chunks, version):
    embedder = FakeEmbedder()
    ingest(client, chunks, embedder, version, collection_name=name)

    edited = [c.model_copy(update={"text": c.text + " Changed wording."}) if c.chunk_id == "en-C1.1" else c
              for c in chunks if c.chunk_id != "es-C9.2"]
    report = ingest(client, edited, embedder, "another-hash", collection_name=name)

    assert (report.upserted, report.removed, report.total, report.skipped) == (57, 1, 57, False)
    collection = client.get_collection(name, embedding_function=None)
    assert collection.get(ids=["es-C9.2"])["ids"] == []
    assert "Changed wording." in collection.get(ids=["en-C1.1"])["documents"][0]
    assert collection.metadata["policy_hash"] == "another-hash"


def test_ingest_refuses_an_index_built_with_a_different_embedding_model(client, name, chunks, version):
    ingest(client, chunks, FakeEmbedder(model="m1"), version, collection_name=name)
    with pytest.raises(PolicyIndexError) as raised:
        ingest(client, chunks, FakeEmbedder(model="m2"), version, collection_name=name)
    assert raised.value.kind == "model_mismatch"
    assert "--rebuild" in str(raised.value)


def test_rebuild_replaces_an_index_built_with_a_different_model(client, name, chunks, version):
    ingest(client, chunks, FakeEmbedder(model="m1"), version, collection_name=name)
    report = ingest(client, chunks, FakeEmbedder(model="m2"), version, collection_name=name, rebuild=True)
    assert report.total == 58 and report.skipped is False
    assert check_index(client, "m2", name).count() == 58


def test_check_index_reports_a_missing_index(client, name):
    with pytest.raises(PolicyIndexError) as raised:
        check_index(client, "m1", name)
    assert raised.value.kind == "missing"


def test_check_index_reports_an_empty_index(client, name):
    client.create_collection(name, embedding_function=None, metadata={"embedding_model": "m1"})
    with pytest.raises(PolicyIndexError) as raised:
        check_index(client, "m1", name)
    assert raised.value.kind == "empty"


def test_check_index_reports_a_model_mismatch(client, name, chunks, version):
    ingest(client, chunks, FakeEmbedder(model="m1"), version, collection_name=name)
    with pytest.raises(PolicyIndexError) as raised:
        check_index(client, "m2", name)
    assert raised.value.kind == "model_mismatch"


def test_check_index_returns_a_usable_collection_and_info(client, name, chunks, version):
    ingest(client, chunks, FakeEmbedder(model="m1"), version, collection_name=name)
    collection = check_index(client, "m1", name)
    assert index_info(collection) == {"embedding_model": "m1", "policy_hash": version, "clauses": 58}


def test_each_vector_is_stored_under_the_right_clause(client, name, chunks, version):
    embedder = FakeEmbedder()
    ingest(client, chunks, embedder, version, collection_name=name)
    collection = client.get_collection(name, embedding_function=None)
    target = next(c for c in chunks if c.chunk_id == "en-C3.3")
    result = collection.query(query_embeddings=embedder.embed([target.document]), n_results=1,
                              include=["distances"])
    assert result["ids"][0] == ["en-C3.3"]
    assert result["distances"][0][0] < 1e-6
