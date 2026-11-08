import sqlite3
import threading

import pytest

from app.agent.reviews import MAX_REASON_CHARS, MAX_REASONS, MAX_SUMMARY_CHARS, ReviewStore


@pytest.fixture
def store(tmp_path):
    clock = iter(f"2026-11-0{n}T10:00:00+00:00" for n in range(1, 9))
    return ReviewStore(tmp_path / "runs" / "reviews.sqlite", now=lambda: next(clock))


def test_the_database_and_its_folder_are_created_on_first_use(tmp_path):
    path = tmp_path / "does" / "not" / "exist" / "reviews.sqlite"
    ReviewStore(path)
    assert path.exists()


def test_a_new_review_is_pending(store):
    review, created = store.create("abc123", ["document_mismatch"], "The invoice total differs.", run_id="run-1")
    assert created is True
    assert (review.claim_id, review.status, review.reasons, review.summary, review.run_id) == (
        "abc123", "pending", ["document_mismatch"], "The invoice total differs.", "run-1")
    assert review.created_at == "2026-11-01T10:00:00+00:00"
    assert review.resolution is None and review.resolved_at is None and review.resolution_note is None


def test_escalating_the_same_claim_twice_keeps_one_record_with_the_first_summary(store):
    first, created_first = store.create("abc123", ["document_mismatch"], "first summary")
    second, created_second = store.create("abc123", ["document_mismatch"], "second summary")
    assert (created_first, created_second) == (True, False)
    assert second == first and second.summary == "first summary"
    assert len(store.list()) == 1


def test_a_reason_found_later_is_added_to_a_pending_record_and_never_lost(store):
    store.create("abc123", ["document_mismatch"], "first summary")
    second, created = store.create("abc123", ["required_check_missing", "document_mismatch"], "second summary")
    assert created is False
    assert second.reasons == ["document_mismatch", "required_check_missing"]
    assert second.summary == "first summary" and second.created_at == "2026-11-01T10:00:00+00:00"
    assert store.get("abc123").reasons == ["document_mismatch", "required_check_missing"]
    assert len(store.list()) == 1


def test_merged_reasons_stay_within_the_limit(store):
    store.create("abc123", [f"reason_{i}" for i in range(MAX_REASONS)], "s")
    merged, _ = store.create("abc123", ["one_more"], "s")
    assert len(merged.reasons) == MAX_REASONS and "one_more" not in merged.reasons


def test_a_resolved_record_is_left_alone(store, tmp_path):
    store.create("abc123", ["document_mismatch"], "s")
    with sqlite3.connect(tmp_path / "runs" / "reviews.sqlite") as conn:
        conn.execute("UPDATE reviews SET status = 'resolved' WHERE claim_id = 'abc123'")
    again, created = store.create("abc123", ["something_else"], "s")
    assert created is False and again.status == "resolved" and again.reasons == ["document_mismatch"]


def test_different_claims_get_different_records(store):
    store.create("claim-a", ["x"], "a")
    store.create("claim-b", ["y"], "b")
    assert [r.claim_id for r in store.list()] == ["claim-a", "claim-b"]
    assert [r.claim_id for r in store.list(status="pending")] == ["claim-a", "claim-b"]
    assert store.list(status="resolved") == []


def test_a_record_survives_a_new_store_on_the_same_file(store, tmp_path):
    store.create("abc123", ["r"], "s")
    reopened = ReviewStore(tmp_path / "runs" / "reviews.sqlite")
    assert reopened.get("abc123").summary == "s"
    assert reopened.get("missing") is None


def test_text_is_limited(store):
    review, _ = store.create("abc123", [f"reason_{i}_" + "x" * 100 for i in range(50)], "s" * 2000)
    assert len(review.summary) == MAX_SUMMARY_CHARS
    assert len(review.reasons) == MAX_REASONS and all(len(r) <= MAX_REASON_CHARS for r in review.reasons)


@pytest.mark.parametrize("bad", ["", "ABC", "has space", "../x", "a" * 65, "x'; DROP TABLE reviews; --", "-leading", "abc123\n"])
def test_a_claim_id_must_be_a_short_plain_identifier(store, bad):
    with pytest.raises(ValueError):
        store.create(bad, [], "s")
    assert store.list() == []


def test_text_is_stored_as_data_not_as_sql(store):
    nasty = "Robert'); DROP TABLE reviews;--"
    store.create("abc123", [nasty], nasty)
    assert store.get("abc123").summary == nasty
    assert len(store.list()) == 1


def test_two_threads_escalating_the_same_claim_make_one_record(tmp_path):
    path = tmp_path / "reviews.sqlite"
    ReviewStore(path)
    results = []

    def escalate():
        results.append(ReviewStore(path).create("abc123", ["r"], "s")[1])

    threads = [threading.Thread(target=escalate) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 7 + [True]
    assert len(ReviewStore(path).list()) == 1


def test_connections_are_closed_after_each_operation(store, tmp_path):
    store.create("abc123", ["r"], "s")
    store.get("abc123")
    store.list()
    # If a connection were left open on Windows, the file could not be renamed.
    (tmp_path / "runs" / "reviews.sqlite").rename(tmp_path / "runs" / "moved.sqlite")
    assert sqlite3.connect(tmp_path / "runs" / "moved.sqlite").execute("select count(*) from reviews").fetchone() == (1,)
