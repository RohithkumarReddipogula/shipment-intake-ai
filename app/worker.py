"""Background worker: claim a job, extract, validate, decide; retry with backoff on failure."""

import logging
import random
import signal
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from app import repo
from app.config import Settings, get_settings
from app.db import connect, init_db
from app.extraction.base import ExtractionError, Extractor, RetryableExtractionError
from app.schemas import EmailIn
from app.validation import decide

log = logging.getLogger(__name__)


def backoff_delay(
    attempt_no: int, base: float, maximum: float, rand: Callable[[], float] = random.random
) -> timedelta:
    """Exponential backoff with jitter: base * 2^(n-1), capped, then scaled into [50%, 100%]."""
    delay = min(base * 2 ** (attempt_no - 1), maximum)
    return timedelta(seconds=delay * (0.5 + rand() / 2))


def process_one(
    conn: sqlite3.Connection,
    extractor: Extractor,
    settings: Settings,
    now: datetime | None = None,
    rand: Callable[[], float] = random.random,
    clock: Callable[[], datetime] | None = None,
) -> int | None:
    """Process at most one job. Returns its email id, or None if nothing was due.

    Time is read twice: at claim, and after the LLM call, so finished_at and the retry
    backoff are measured from when the call actually ended. Tests pass `now` to freeze
    time, or `clock` to control both readings.
    """
    if clock is None:
        clock = (lambda: now) if now is not None else (lambda: datetime.now(timezone.utc))
    now = clock()
    job = repo.claim_next(conn, now, timedelta(seconds=settings.lock_timeout_seconds))
    if job is None:
        return None

    email_id, attempt_no = job["id"], job["attempts"]
    label = f"email={email_id} attempt={attempt_no}/{settings.max_attempts}"

    try:
        if attempt_no > settings.max_attempts:
            # Only reachable by reclaiming a stuck job: the worker died on every try.
            error = "max attempts exceeded (previous attempts did not finish)"
            repo.mark_failed(conn, email_id, attempt_no, now, error)
            log.error("%s outcome=failed error=%r", label, error)
            return email_id

        email = EmailIn(
            message_id=job["message_id"],
            sender=job["sender"],
            subject=job["subject"],
            body=job["body"],
            received_at=datetime.fromisoformat(job["received_at"]),
        )
        try:
            output = extractor.extract(email)
        except ExtractionError as e:
            retryable = isinstance(e, RetryableExtractionError)
            _handle_failure(conn, settings, email_id, attempt_no, clock(), rand, label,
                            f"{type(e).__name__}: {e}", retryable)
            return email_id
        except Exception as e:
            # Unknown errors are retried: losing an email is worse than a wasted call.
            log.exception("%s unexpected extractor error", label)
            _handle_failure(conn, settings, email_id, attempt_no, clock(), rand, label,
                            f"{type(e).__name__}: {e}", retryable=True)
            return email_id

        result = decide(output, email.received_at.date(), settings.confidence_threshold)
        repo.complete_success(conn, email_id, attempt_no, clock(), output, result,
                              extractor.model, extractor.prompt_version)
        log.info("%s outcome=success decision=%s reasons=%s",
                 label, result.decision.value, result.reasons)
    except repo.LostLockError:
        log.warning("%s lock lost to another worker; result discarded", label)
    return email_id


def _handle_failure(
    conn: sqlite3.Connection,
    settings: Settings,
    email_id: int,
    attempt_no: int,
    now: datetime,
    rand: Callable[[], float],
    label: str,
    error: str,
    retryable: bool,
) -> None:
    if retryable and attempt_no < settings.max_attempts:
        next_at = now + backoff_delay(
            attempt_no, settings.backoff_base_seconds, settings.backoff_max_seconds, rand
        )
        repo.schedule_retry(conn, email_id, attempt_no, now, error, next_at)
        log.warning("%s outcome=retry error=%r next_attempt_at=%s",
                    label, error, next_at.isoformat())
    else:
        repo.mark_failed(conn, email_id, attempt_no, now, error)
        log.error("%s outcome=failed error=%r", label, error)


def run(
    settings: Settings, extractor: Extractor, stop: threading.Event | None = None
) -> None:
    """Poll until `stop` is set. Sleeps only when no job was due."""
    stop = stop or threading.Event()
    conn = connect(settings.db_path)
    init_db(conn)
    log.info("worker started model=%s db=%s", extractor.model, settings.db_path)
    try:
        while not stop.is_set():
            try:
                processed = process_one(conn, extractor, settings)
            except Exception:
                log.exception("worker loop error")
                processed = None
            if processed is None:
                stop.wait(settings.poll_interval_seconds)
    finally:
        conn.close()


def main() -> None:
    from app.extraction.gemini import GeminiExtractor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    extractor = GeminiExtractor.from_settings(settings)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(settings, extractor, stop)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
