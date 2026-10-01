import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id              INTEGER PRIMARY KEY,
    -- Normalized Message-ID; NULL when the email has none. Dedup on it is permanent.
    dedup_key       TEXT UNIQUE,
    -- Hash of normalized sender/subject/body. Dedup on it only applies within a time window.
    content_hash    TEXT NOT NULL,
    message_id      TEXT,
    sender          TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','processing','accepted','needs_review','failed')),
    -- Incremented on claim; also serves as the lock token for completing the attempt.
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    locked_at       TEXT,
    last_error      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emails_claim ON emails (status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_emails_content_hash ON emails (content_hash, created_at);

CREATE TABLE IF NOT EXISTS extractions (
    id                     INTEGER PRIMARY KEY,
    email_id               INTEGER NOT NULL REFERENCES emails(id),
    data_json              TEXT NOT NULL,
    field_confidence_json  TEXT NOT NULL,
    validation_errors_json TEXT NOT NULL,
    decision               TEXT NOT NULL CHECK (decision IN ('accepted','needs_review')),
    decision_reasons_json  TEXT NOT NULL,
    model                  TEXT NOT NULL,
    prompt_version         TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    -- Review feedback, stored next to the original extraction.
    review_action          TEXT CHECK (review_action IN ('approve','correct')),
    corrected_data_json    TEXT,
    reviewed_by            TEXT,
    reviewed_at            TEXT
);

CREATE INDEX IF NOT EXISTS idx_extractions_email ON extractions (email_id);

-- One row per processing attempt: the retry history.
CREATE TABLE IF NOT EXISTS attempts (
    id              INTEGER PRIMARY KEY,
    email_id        INTEGER NOT NULL REFERENCES emails(id),
    attempt_no      INTEGER NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    -- NULL while running; 'abandoned' when a stuck job is reclaimed after the lock timeout.
    outcome         TEXT CHECK (outcome IN ('success','retry','failed','abandoned')),
    error           TEXT,
    next_attempt_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_attempts_email ON attempts (email_id);
"""


def to_utc_iso(dt: datetime) -> str:
    """ISO-8601 UTC with fixed format, so string comparison in SQL matches time order.

    Naive datetimes are treated as UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utcnow() -> str:
    return to_utc_iso(datetime.now(timezone.utc))


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)  # autocommit; explicit BEGIN when needed
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE takes the write lock up front, so check-then-write is race-free."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
