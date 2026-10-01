import hashlib
import re
import sqlite3

from app.db import to_utc_iso, utcnow
from app.schemas import EmailIn


def dedup_key(email: EmailIn) -> str:
    """Message-ID when present; otherwise a hash of normalized sender, subject and body.

    Prefixed so the two kinds of key can never collide.
    """
    if email.message_id and email.message_id.strip("<> \t"):
        return "mid:" + email.message_id.strip("<> \t")

    sender = email.sender.strip().casefold()
    subject = re.sub(r"\s+", " ", email.subject).strip()
    body = "\n".join(line.rstrip() for line in email.body.replace("\r\n", "\n").split("\n")).strip()
    digest = hashlib.sha256("\x00".join([sender, subject, body]).encode()).hexdigest()
    return "sha256:" + digest


def insert_email(conn: sqlite3.Connection, email: EmailIn) -> tuple[int, bool]:
    """Insert unless already present. Returns (email_id, created)."""
    key = dedup_key(email)
    now = utcnow()
    row = conn.execute(
        """
        INSERT INTO emails (dedup_key, message_id, sender, subject, body, received_at,
                            next_attempt_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dedup_key) DO NOTHING
        RETURNING id
        """,
        (key, email.message_id, email.sender, email.subject, email.body,
         to_utc_iso(email.received_at), now, now, now),
    ).fetchone()
    if row is not None:
        return row["id"], True
    existing = conn.execute("SELECT id FROM emails WHERE dedup_key = ?", (key,)).fetchone()
    return existing["id"], False


def get_email(conn: sqlite3.Connection, email_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()


def get_latest_extraction(conn: sqlite3.Connection, email_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM extractions WHERE email_id = ? ORDER BY id DESC LIMIT 1", (email_id,)
    ).fetchone()
