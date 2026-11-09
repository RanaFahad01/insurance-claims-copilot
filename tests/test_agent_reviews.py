import sqlite3
import threading

import pytest

from app.agent.reviews import MAX_NOTE_CHARS, MAX_REASON_CHARS, MAX_REASONS, MAX_SUMMARY_CHARS, ReviewStore


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


# ---- resolving

def test_resolving_a_pending_review_records_the_decision(store):
    store.create("abc123", ["document_mismatch"], "s", run_id="run-1")
    review, resolved_now = store.resolve("abc123", "rejected", "The invoice is for another person.")
    assert resolved_now is True
    assert (review.status, review.resolution, review.resolution_note) == ("resolved", "rejected", "The invoice is for another person.")
    assert review.resolved_at is not None
    # only the decision fields change
    assert (review.reasons, review.summary, review.run_id, review.created_at) == (
        ["document_mismatch"], "s", "run-1", "2026-11-01T10:00:00+00:00")
    assert store.get("abc123") == review and store.list(status="pending") == [] and store.list(status="resolved") == [review]


def test_a_note_is_optional(store):
    store.create("abc123", ["r"], "s")
    review, _ = store.resolve("abc123", "approved")
    assert review.resolution == "approved" and review.resolution_note is None


def test_a_review_can_only_be_resolved_once_and_the_first_decision_stays(store):
    store.create("abc123", ["r"], "s")
    first, _ = store.resolve("abc123", "approved", "first")
    second, resolved_now = store.resolve("abc123", "rejected", "second")
    assert resolved_now is False and second == first and second.resolution == "approved" and second.resolution_note == "first"


def test_resolving_an_unknown_or_invalid_claim_id_finds_nothing(store):
    assert store.resolve("missing", "approved") == (None, False)
    for bad in ["", "ABC", "../x", "abc123\n", "x'; DROP TABLE reviews; --"]:
        assert store.resolve(bad, "approved") == (None, False)


def test_an_unknown_resolution_is_refused_and_changes_nothing(store):
    store.create("abc123", ["r"], "s")
    with pytest.raises(ValueError):
        store.resolve("abc123", "maybe")
    assert store.get("abc123").status == "pending"


def test_a_long_note_is_limited(store):
    store.create("abc123", ["r"], "s")
    review, _ = store.resolve("abc123", "approved", "n" * 5000)
    assert len(review.resolution_note) == MAX_NOTE_CHARS


def test_a_resolved_claim_that_is_escalated_again_stays_resolved(store):
    store.create("abc123", ["r"], "s")
    store.resolve("abc123", "approved")
    again, created = store.create("abc123", ["new_reason"], "s")
    assert created is False and again.status == "resolved" and again.reasons == ["r"]


def test_two_threads_resolving_the_same_review_give_one_winner(tmp_path):
    path = tmp_path / "reviews.sqlite"
    ReviewStore(path).create("abc123", ["r"], "s")
    outcomes = []

    def resolve(decision):
        outcomes.append((decision, ReviewStore(path).resolve("abc123", decision)[1]))

    threads = [threading.Thread(target=resolve, args=(d,)) for d in ["approved", "rejected"] * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [d for d, won in outcomes if won]
    assert len(winners) == 1 and len(outcomes) == 8
    assert ReviewStore(path).get("abc123").resolution == winners[0]


def test_connections_are_closed_after_each_operation(store, tmp_path):
    store.create("abc123", ["r"], "s")
    store.get("abc123")
    store.list()
    # If a connection were left open on Windows, the file could not be renamed.
    (tmp_path / "runs" / "reviews.sqlite").rename(tmp_path / "runs" / "moved.sqlite")
    assert sqlite3.connect(tmp_path / "runs" / "moved.sqlite").execute("select count(*) from reviews").fetchone() == (1,)
