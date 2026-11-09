"""The queue of claims waiting for a person. SQLite, one row per claim ID.

The primary key makes writing a review idempotent: escalating the same claim twice leaves one record. The first summary
is kept, and reasons that are new are added to a pending record, so a later finding is never lost. A record that
has already been resolved is left alone. A new connection is opened for every operation, so the store is safe to call from API worker threads.
"""
import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

DEFAULT_REVIEW_DB = Path(__file__).resolve().parents[2] / "runs" / "reviews.sqlite"
CLAIM_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_SUMMARY_CHARS = 500
MAX_REASONS = 20
MAX_REASON_CHARS = 60
MAX_NOTE_CHARS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    claim_id        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    reasons         TEXT NOT NULL,
    summary         TEXT NOT NULL,
    run_id          TEXT,
    resolution      TEXT,
    resolved_at     TEXT,
    resolution_note TEXT
)
"""

ReviewStatus = Literal["pending", "resolved"]
Resolution = Literal["approved", "rejected", "needs_more_information"]


class Review(BaseModel):
    claim_id: str
    created_at: str
    status: ReviewStatus
    reasons: list[str]
    summary: str
    run_id: str | None
    resolution: str | None
    resolved_at: str | None
    resolution_note: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, path: Path = DEFAULT_REVIEW_DB, now: Callable[[], str] = _now):
        self.path = Path(path)
        self._now = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        """Commits (or rolls back) and then really closes the connection, which sqlite3's own context manager does not."""
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create(self, claim_id: str, reasons: list[str], summary: str, run_id: str | None = None) -> tuple[Review, bool]:
        """Returns the record and whether this call created it. A second call for the same claim changes nothing."""
        if not CLAIM_ID_PATTERN.fullmatch(claim_id):
            raise ValueError("invalid claim id")
        clean_reasons = [r[:MAX_REASON_CHARS] for r in reasons[:MAX_REASONS]]
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO reviews (claim_id, created_at, status, reasons, summary, run_id) VALUES (?, ?, 'pending', ?, ?, ?)",
                (claim_id, self._now(), json.dumps(clean_reasons), summary[:MAX_SUMMARY_CHARS], run_id),
            )
            created = cursor.rowcount == 1
            row = conn.execute("SELECT * FROM reviews WHERE claim_id = ?", (claim_id,)).fetchone()
            if not created and row["status"] == "pending":
                # A reason found later (for example from a second document) must not be lost. Reasons are only added.
                known = json.loads(row["reasons"])
                merged = (known + [r for r in clean_reasons if r not in known])[:MAX_REASONS]
                if merged != known:
                    conn.execute("UPDATE reviews SET reasons = ? WHERE claim_id = ?", (json.dumps(merged), claim_id))
                    row = conn.execute("SELECT * FROM reviews WHERE claim_id = ?", (claim_id,)).fetchone()
        return self._to_review(row), created

    def resolve(self, claim_id: str, resolution: Resolution, note: str | None = None) -> tuple[Review | None, bool]:
        """Records a person's decision on a pending review. Returns (record, whether this call resolved it).
        (None, False) means there is no such review. (record, False) means it was already resolved and is unchanged.
        The update only matches a pending row, so two simultaneous calls cannot both succeed."""
        if resolution not in get_args(Resolution):
            raise ValueError("invalid resolution")
        if not CLAIM_ID_PATTERN.fullmatch(claim_id):
            return None, False
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reviews SET status = 'resolved', resolution = ?, resolved_at = ?, resolution_note = ? "
                "WHERE claim_id = ? AND status = 'pending'",
                (resolution, self._now(), note[:MAX_NOTE_CHARS] if note else None, claim_id),
            )
            resolved_now = cursor.rowcount == 1
            row = conn.execute("SELECT * FROM reviews WHERE claim_id = ?", (claim_id,)).fetchone()
        return (self._to_review(row) if row else None), resolved_now

    def get(self, claim_id: str) -> Review | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reviews WHERE claim_id = ?", (claim_id,)).fetchone()
        return self._to_review(row) if row else None

    def list(self, status: ReviewStatus | None = None) -> list[Review]:
        query, params = "SELECT * FROM reviews", ()
        if status:
            query, params = query + " WHERE status = ?", (status,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY created_at, claim_id", params).fetchall()
        return [self._to_review(r) for r in rows]

    @staticmethod
    def _to_review(row: sqlite3.Row) -> Review:
        data = dict(row)
        data["reasons"] = json.loads(data["reasons"])
        return Review(**data)

