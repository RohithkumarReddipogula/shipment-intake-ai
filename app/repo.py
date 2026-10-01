import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta

from app.db import to_utc_iso, transaction
from app.schemas import EmailIn, ExtractionOutput, FieldConfidence, ShipmentFields
from app.validation import DecisionResult


def message_key(email: EmailIn) -> str | None:
    """Normalized Message-ID, or None if the email has none."""
    if email.message_id and email.message_id.strip("<> \t"):
        return email.message_id.strip("<> \t")
    return None


def content_hash(email: EmailIn) -> str:
    """Hash of sender (case-insensitive), subject (whitespace-collapsed) and body
    (line endings and trailing whitespace normalized)."""
    sender = email.sender.strip().casefold()
    subject = re.sub(r"\s+", " ", email.subject).strip()
    body = "\n".join(line.rstrip() for line in email.body.replace("\r\n", "\n").split("\n")).strip()
    return hashlib.sha256("\x00".join([sender, subject, body]).encode()).hexdigest()


def insert_email(
    conn: sqlite3.Connection, email: EmailIn, dedup_window: timedelta, now: datetime
) -> tuple[int, bool]:
    """Insert unless it's a duplicate. Returns (email_id, created).

    With a Message-ID, a duplicate is any earlier email with that ID. Without one, it's
    an email with the same content hash created within `dedup_window` — so client
    retries are caught but next week's identical repeat booking is a new email.
    """
    key = message_key(email)
    chash = content_hash(email)
    now_iso = to_utc_iso(now)
    with transaction(conn):
        if key is not None:
            existing = conn.execute("SELECT id FROM emails WHERE dedup_key = ?", (key,)).fetchone()
        else:
            existing = conn.execute(
                "SELECT id FROM emails WHERE content_hash = ? AND created_at >= ?"
                " ORDER BY id DESC LIMIT 1",
                (chash, to_utc_iso(now - dedup_window)),
            ).fetchone()
        if existing is not None:
            return existing["id"], False

        row = conn.execute(
            """
            INSERT INTO emails (dedup_key, content_hash, message_id, sender, subject, body,
                                received_at, next_attempt_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (key, chash, email.message_id, email.sender, email.subject, email.body,
             to_utc_iso(email.received_at), now_iso, now_iso, now_iso),
        ).fetchone()
        return row["id"], True


def get_email(conn: sqlite3.Connection, email_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()


def get_latest_extraction(conn: sqlite3.Connection, email_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM extractions WHERE email_id = ? ORDER BY id DESC LIMIT 1", (email_id,)
    ).fetchone()


def get_attempts(conn: sqlite3.Connection, email_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM attempts WHERE email_id = ? ORDER BY attempt_no", (email_id,)
    ).fetchall()


# --- Worker ----------------------------------------------------------------


def claim_next(
    conn: sqlite3.Connection, now: datetime, lock_timeout: timedelta
) -> sqlite3.Row | None:
    """Atomically claim one due job and open an attempt row for it.

    Due means pending with next_attempt_at <= now, or stuck in processing with a lock
    older than lock_timeout (the worker crashed or hung). A reclaimed job's open
    attempt is closed as 'abandoned'. attempts is incremented here, so a job that
    keeps crashing the worker still runs out of attempts.
    """
    now_iso = to_utc_iso(now)
    with transaction(conn):
        job = conn.execute(
            """
            UPDATE emails
            SET status = 'processing', locked_at = :now, attempts = attempts + 1, updated_at = :now
            WHERE id = (
                SELECT id FROM emails
                WHERE (status = 'pending' AND next_attempt_at <= :now)
                   OR (status = 'processing' AND locked_at <= :stale_before)
                ORDER BY next_attempt_at, id
                LIMIT 1
            )
            RETURNING *
            """,
            {"now": now_iso, "stale_before": to_utc_iso(now - lock_timeout)},
        ).fetchone()
        if job is None:
            return None
        conn.execute(
            "UPDATE attempts SET outcome = 'abandoned', finished_at = ?,"
            " error = 'lock timeout: worker did not finish'"
            " WHERE email_id = ? AND finished_at IS NULL",
            (now_iso, job["id"]),
        )
        conn.execute(
            "INSERT INTO attempts (email_id, attempt_no, started_at) VALUES (?, ?, ?)",
            (job["id"], job["attempts"], now_iso),
        )
        return job


class LostLockError(Exception):
    """The job was reclaimed by another worker after our lock timed out."""


def _finish(
    conn: sqlite3.Connection,
    email_id: int,
    attempt_no: int,
    now: datetime,
    *,
    status: str,
    outcome: str,
    error: str | None = None,
    next_attempt_at: datetime | None = None,
) -> None:
    """Update email + attempt row; must run inside a transaction.

    Guarded on (status='processing', attempts=attempt_no): if another worker reclaimed
    the job in the meantime, attempts has moved on and we raise LostLockError.
    """
    now_iso = to_utc_iso(now)
    next_iso = to_utc_iso(next_attempt_at) if next_attempt_at else None
    cur = conn.execute(
        """
        UPDATE emails
        SET status = ?, locked_at = NULL, last_error = ?, updated_at = ?,
            next_attempt_at = COALESCE(?, next_attempt_at)
        WHERE id = ? AND status = 'processing' AND attempts = ?
        """,
        (status, error, now_iso, next_iso, email_id, attempt_no),
    )
    if cur.rowcount == 0:
        raise LostLockError(f"email {email_id} attempt {attempt_no}")
    conn.execute(
        "UPDATE attempts SET outcome = ?, finished_at = ?, error = ?, next_attempt_at = ?"
        " WHERE email_id = ? AND attempt_no = ? AND finished_at IS NULL",
        (outcome, now_iso, error, next_iso, email_id, attempt_no),
    )


def complete_success(
    conn: sqlite3.Connection,
    email_id: int,
    attempt_no: int,
    now: datetime,
    output: ExtractionOutput,
    result: DecisionResult,
    model: str,
    prompt_version: str,
) -> None:
    with transaction(conn):
        _finish(conn, email_id, attempt_no, now, status=result.decision.value, outcome="success")
        conn.execute(
            """
            INSERT INTO extractions (email_id, data_json, field_confidence_json,
                                     validation_errors_json, decision, decision_reasons_json,
                                     model, prompt_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (email_id, output.fields.model_dump_json(), output.confidence.model_dump_json(),
             json.dumps(result.validation_errors), result.decision.value,
             json.dumps(result.reasons), model, prompt_version, to_utc_iso(now)),
        )


def schedule_retry(
    conn: sqlite3.Connection, email_id: int, attempt_no: int, now: datetime,
    error: str, next_attempt_at: datetime,
) -> None:
    with transaction(conn):
        _finish(conn, email_id, attempt_no, now, status="pending", outcome="retry",
                error=error, next_attempt_at=next_attempt_at)


def mark_failed(
    conn: sqlite3.Connection, email_id: int, attempt_no: int, now: datetime, error: str
) -> None:
    with transaction(conn):
        _finish(conn, email_id, attempt_no, now, status="failed", outcome="failed", error=error)


# --- Review ----------------------------------------------------------------

REVIEWABLE = ("needs_review", "failed")


class ReviewConflict(Exception):
    pass


def list_review_queue(
    conn: sqlite3.Connection, reason: str | None = None, limit: int = 100
) -> list[sqlite3.Row]:
    """Emails awaiting review, oldest first, with their latest extraction (if any).

    `reason` filters on an exact reason code, e.g. "dangerous_goods" or "missing:weight".
    """
    return conn.execute(
        """
        SELECT e.id, e.sender, e.subject, e.received_at, e.status, e.last_error,
               x.data_json, x.decision_reasons_json
        FROM emails e
        LEFT JOIN extractions x ON x.id = (
            SELECT MAX(id) FROM extractions WHERE email_id = e.id
        )
        WHERE e.status IN ('needs_review', 'failed')
          AND (:reason IS NULL OR EXISTS (
                SELECT 1 FROM json_each(x.decision_reasons_json) WHERE value = :reason))
        ORDER BY e.created_at, e.id
        LIMIT :limit
        """,
        {"reason": reason, "limit": limit},
    ).fetchall()


def review(
    conn: sqlite3.Connection,
    email_id: int,
    action: str,
    reviewer: str,
    corrected: ShipmentFields | None,
    now: datetime,
) -> None:
    """Record a review next to the original extraction and mark the email accepted.

    The original data_json is never modified; a correction goes in corrected_data_json.
    A failed email has no extraction, so an empty one (all fields null, reason
    extraction_failed) is created first, keeping the original/corrected pair shape.
    Raises LookupError if the email doesn't exist, ReviewConflict if it isn't awaiting review.
    """
    now_iso = to_utc_iso(now)
    with transaction(conn):
        email = get_email(conn, email_id)
        if email is None:
            raise LookupError(email_id)
        if email["status"] not in REVIEWABLE:
            raise ReviewConflict(f"email {email_id} is {email['status']}, not awaiting review")

        extraction = get_latest_extraction(conn, email_id)
        if email["status"] == "failed":
            if action == "approve":
                raise ReviewConflict(
                    f"email {email_id} failed extraction; nothing to approve, use 'correct'"
                )
            empty = ExtractionOutput(fields=ShipmentFields(), confidence=FieldConfidence())
            conn.execute(
                """
                INSERT INTO extractions (email_id, data_json, field_confidence_json,
                                         validation_errors_json, decision, decision_reasons_json,
                                         model, prompt_version, created_at)
                VALUES (?, ?, ?, '[]', 'needs_review', '["extraction_failed"]', 'none', 'none', ?)
                """,
                (email_id, empty.fields.model_dump_json(), empty.confidence.model_dump_json(), now_iso),
            )
            extraction = get_latest_extraction(conn, email_id)

        conn.execute(
            """
            UPDATE extractions
            SET review_action = ?, corrected_data_json = ?, reviewed_by = ?, reviewed_at = ?
            WHERE id = ? AND review_action IS NULL
            """,
            (action, corrected.model_dump_json() if corrected else None, reviewer, now_iso,
             extraction["id"]),
        )
        conn.execute(
            "UPDATE emails SET status = 'accepted', updated_at = ? WHERE id = ?", (now_iso, email_id)
        )
