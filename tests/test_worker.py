import logging
from datetime import date, datetime, timedelta, timezone

import pytest

from app import repo
from app.extraction.base import ExtractionError, RetryableExtractionError
from app.extraction.fake import FakeExtractor
from app.schemas import EmailIn, ExtractionOutput, FieldConfidence, ShipmentFields
from app.worker import backoff_delay, process_one

T0 = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc)
MAX_RAND = lambda: 1.0  # noqa: E731  - jitter at its max, so delay == base * 2^(n-1)


def good_output(**overrides) -> ExtractionOutput:
    fields = dict(
        request_type="booking", origin="Hamburg", destination="München",
        pickup_date=date(2026, 3, 17), weight_value=1200, weight_unit="kg",
        pallet_count=4, cargo_type="general cargo",
    )
    conf = {k: 0.95 for k in FieldConfidence.model_fields}
    return ExtractionOutput(fields=ShipmentFields(**(fields | overrides)), confidence=FieldConfidence(**conf))


@pytest.fixture
def email_id(conn) -> int:
    email = EmailIn(message_id="m1", sender="dispo@kunde.de", subject="Buchung",
                    body="4 Paletten...", received_at=T0)
    email_id, _ = repo.insert_email(conn, email, timedelta(hours=24), T0)
    return email_id


def status(conn, email_id) -> str:
    return repo.get_email(conn, email_id)["status"]


def history(conn, email_id) -> list[tuple]:
    return [(a["attempt_no"], a["outcome"]) for a in repo.get_attempts(conn, email_id)]


def test_nothing_due_returns_none(conn, settings):
    assert process_one(conn, FakeExtractor(), settings, now=T0) is None


def test_success_accepted(conn, settings, email_id):
    fake = FakeExtractor(good_output())
    assert process_one(conn, fake, settings, now=T0) == email_id

    assert status(conn, email_id) == "accepted"
    extraction = repo.get_latest_extraction(conn, email_id)
    assert extraction["decision"] == "accepted"
    assert extraction["model"] == "fake"
    assert history(conn, email_id) == [(1, "success")]
    # The extractor received the stored email.
    assert fake.calls[0].body == "4 Paletten..."
    assert fake.calls[0].received_at == T0


def test_success_needs_review(conn, settings, email_id):
    process_one(conn, FakeExtractor(good_output(weight_value=None)), settings, now=T0)
    assert status(conn, email_id) == "needs_review"
    assert '"missing:weight"' in repo.get_latest_extraction(conn, email_id)["decision_reasons_json"]


def test_pickup_date_checked_against_received_at_not_now(conn, settings, email_id):
    # Processed a year after the email arrived: the date is still valid relative to received_at.
    later = T0 + timedelta(days=365)
    process_one(conn, FakeExtractor(good_output()), settings, now=later)
    assert status(conn, email_id) == "accepted"


class TestRetries:
    def test_retryable_error_schedules_retry_with_backoff(self, conn, settings, email_id):
        fake = FakeExtractor(RetryableExtractionError("429 rate limited"))
        process_one(conn, fake, settings, now=T0, rand=MAX_RAND)

        row = repo.get_email(conn, email_id)
        assert row["status"] == "pending"
        assert row["last_error"] == "RetryableExtractionError: 429 rate limited"
        next_at = datetime.fromisoformat(row["next_attempt_at"])
        assert next_at == T0 + timedelta(seconds=settings.backoff_base_seconds)

        [attempt] = repo.get_attempts(conn, email_id)
        assert attempt["outcome"] == "retry"
        assert attempt["error"] == "RetryableExtractionError: 429 rate limited"
        assert datetime.fromisoformat(attempt["next_attempt_at"]) == next_at

    def test_not_retried_before_next_attempt_at(self, conn, settings, email_id):
        fake = FakeExtractor(RetryableExtractionError("timeout"), good_output())
        process_one(conn, fake, settings, now=T0, rand=MAX_RAND)
        assert process_one(conn, fake, settings, now=T0 + timedelta(seconds=1)) is None
        assert process_one(conn, fake, settings, now=T0 + timedelta(seconds=2)) == email_id
        assert status(conn, email_id) == "accepted"

    def test_fail_twice_then_succeed_keeps_full_history(self, conn, settings, email_id):
        fake = FakeExtractor(
            RetryableExtractionError("timeout"), RetryableExtractionError("503"), good_output()
        )
        now = T0
        for _ in range(3):
            process_one(conn, fake, settings, now=now)
            now += timedelta(hours=1)

        assert status(conn, email_id) == "accepted"
        assert repo.get_email(conn, email_id)["last_error"] is None
        assert history(conn, email_id) == [(1, "retry"), (2, "retry"), (3, "success")]
        errors = [a["error"] for a in repo.get_attempts(conn, email_id)]
        assert errors == ["RetryableExtractionError: timeout", "RetryableExtractionError: 503", None]

    def test_fails_after_max_attempts(self, conn, settings, email_id):
        fake = FakeExtractor(default=RetryableExtractionError("timeout"))
        now = T0
        for _ in range(settings.max_attempts):
            process_one(conn, fake, settings, now=now)
            now += timedelta(hours=1)

        assert status(conn, email_id) == "failed"
        assert len(fake.calls) == settings.max_attempts
        outcomes = [o for _, o in history(conn, email_id)]
        assert outcomes == ["retry"] * (settings.max_attempts - 1) + ["failed"]
        assert process_one(conn, fake, settings, now=now) is None

    def test_non_retryable_error_fails_immediately(self, conn, settings, email_id):
        process_one(conn, FakeExtractor(ExtractionError("401 invalid API key")), settings, now=T0)
        assert status(conn, email_id) == "failed"
        assert history(conn, email_id) == [(1, "failed")]

    def test_unexpected_exception_is_retried(self, conn, settings, email_id):
        process_one(conn, FakeExtractor(KeyError("boom")), settings, now=T0)
        assert status(conn, email_id) == "pending"
        assert "KeyError" in repo.get_email(conn, email_id)["last_error"]


@pytest.mark.parametrize(
    "attempt_no,rand,expected",
    [(1, 1.0, 2), (2, 1.0, 4), (3, 1.0, 8), (3, 0.0, 4), (20, 1.0, 300), (20, 0.0, 150)],
)
def test_backoff_delay(attempt_no, rand, expected):
    assert backoff_delay(attempt_no, 2.0, 300.0, lambda: rand) == timedelta(seconds=expected)


class TestStuckJobs:
    def crash_mid_attempt(self, conn, settings, now):
        """Claim the job as a worker would, then never finish (simulated crash)."""
        return repo.claim_next(conn, now, timedelta(seconds=settings.lock_timeout_seconds))

    def test_stuck_job_not_reclaimed_before_timeout(self, conn, settings, email_id):
        self.crash_mid_attempt(conn, settings, T0)
        before = T0 + timedelta(seconds=settings.lock_timeout_seconds - 1)
        assert process_one(conn, FakeExtractor(good_output()), settings, now=before) is None
        assert status(conn, email_id) == "processing"

    def test_stuck_job_reclaimed_after_timeout(self, conn, settings, email_id):
        self.crash_mid_attempt(conn, settings, T0)
        after = T0 + timedelta(seconds=settings.lock_timeout_seconds)
        assert process_one(conn, FakeExtractor(good_output()), settings, now=after) == email_id

        assert status(conn, email_id) == "accepted"
        assert history(conn, email_id) == [(1, "abandoned"), (2, "success")]
        assert repo.get_attempts(conn, email_id)[0]["error"] == "lock timeout: worker did not finish"

    def test_slow_worker_cannot_overwrite_after_reclaim(self, conn, settings, email_id):
        slow = self.crash_mid_attempt(conn, settings, T0)
        after = T0 + timedelta(seconds=settings.lock_timeout_seconds)
        process_one(conn, FakeExtractor(good_output()), settings, now=after)

        # The original worker wakes up and tries to report a failure for attempt 1.
        with pytest.raises(repo.LostLockError):
            repo.mark_failed(conn, email_id, slow["attempts"], after, "late")
        assert status(conn, email_id) == "accepted"
        assert history(conn, email_id) == [(1, "abandoned"), (2, "success")]

    def test_job_that_keeps_crashing_eventually_fails(self, conn, settings, email_id):
        now = T0
        timeout = timedelta(seconds=settings.lock_timeout_seconds)
        for _ in range(settings.max_attempts):
            self.crash_mid_attempt(conn, settings, now)
            now += timeout

        fake = FakeExtractor(good_output())
        process_one(conn, fake, settings, now=now)
        assert status(conn, email_id) == "failed"
        assert fake.calls == []  # not sent to the LLM again
        outcomes = [o for _, o in history(conn, email_id)]
        assert outcomes == ["abandoned"] * settings.max_attempts + ["failed"]


def test_each_attempt_is_logged_with_error(conn, settings, email_id, caplog):
    fake = FakeExtractor(RetryableExtractionError("503 unavailable"), good_output())
    with caplog.at_level(logging.INFO, logger="app.worker"):
        process_one(conn, fake, settings, now=T0)
        process_one(conn, fake, settings, now=T0 + timedelta(hours=1))

    lines = [r.getMessage() for r in caplog.records]
    assert "email=1 attempt=1/5 outcome=retry" in lines[0]
    assert "503 unavailable" in lines[0]
    assert "next_attempt_at=" in lines[0]
    assert "email=1 attempt=2/5 outcome=success decision=accepted" in lines[1]


def test_attempt_history_in_api(client, settings):
    from app.db import connect

    email_id = client.post("/emails", json={"sender": "a@b.de", "body": "x"}).json()["id"]
    conn = connect(settings.db_path)
    process_one(conn, FakeExtractor(RetryableExtractionError("timeout")), settings)
    conn.close()

    body = client.get(f"/emails/{email_id}").json()
    assert body["status"] == "pending"
    assert body["attempts"] == 1
    [attempt] = body["attempt_history"]
    assert attempt["outcome"] == "retry"
    assert attempt["error"] == "RetryableExtractionError: timeout"


def test_run_loop_processes_until_stopped(conn, settings, email_id):
    import threading
    import time

    from app.worker import run

    stop = threading.Event()
    fast = settings.model_copy(update={"poll_interval_seconds": 0.01})
    thread = threading.Thread(target=run, args=(fast, FakeExtractor(default=good_output()), stop))
    thread.start()
    deadline = time.monotonic() + 5
    while status(conn, email_id) != "accepted" and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5)

    assert status(conn, email_id) == "accepted"
    assert not thread.is_alive()
