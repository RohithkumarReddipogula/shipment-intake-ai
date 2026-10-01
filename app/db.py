import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id              INTEGER PRIMARY KEY,
    dedup_key       TEXT NOT NULL UNIQUE,
    message_id      TEXT,
    sender          TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','processing','accepted','needs_review','failed')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    locked_at       TEXT,
    last_error      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emails_claim ON emails (status, next_attempt_at);

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
"""


def utcnow() -> str:
    """ISO-8601 UTC with fixed format, so string comparison in SQL matches time order."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)  # autocommit; explicit BEGIN when needed
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
