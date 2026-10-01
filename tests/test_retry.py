import logging
from datetime import datetime, timedelta, timezone

import pytest

from app import repo
from app.extraction.base import ExtractionError, RetryableExtractionError
from app.extraction.fake import FakeExtractor
from app.schemas import ExtractionOutput, FieldConfidence, ShipmentFields
from app.worker import process_one

GOOD = ExtractionOutput(
    fields=ShipmentFields(
        request_type="booking", origin="Hamburg", destination="München", pickup_date="2026-03-17",
        weight_value=1200, weight_unit="kg", pallet_count=4, cargo_type="general",
    ),
    confidence=FieldConfidence(**{k: 0.95 for k in FieldConfidence.model_fields}),
)
QUOTA = RetryableExtractionError("Gemini 429 RESOURCE_EXHAUSTED")
MAX_RAND = lambda: 1.0  # noqa: E731


def later(hours: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


@pytest.fixture
def failed_email(client, conn, settings) -> int:
    """An email that used up all its attempts during a quota outage."""
    email_id = client.post("/emails", json={
        "message_id": "m1", "sender": "a@b.de", "subject": "Buchung",
        "body": "4 Paletten", "received_at": "2026-03-10T09:00:00Z",
    }).json()["id"]
    fake = FakeExtractor(default=QUOTA)
    for i in range(settings.max_attempts):
        process_one(conn, fake, settings, now=later(i))
    assert repo.get_email(conn, email_id)["status"] == "failed"
    return email_id


def test_retry_requeues_and_keeps_history(client, conn, settings, failed_email):
    r = client.post(f"/review/{failed_email}/retry")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["attempts"] == settings.max_attempts
    assert len(body["attempt_history"]) == settings.max_attempts
    assert client.get("/review").json() == []  # no longer in the review queue


def test_requeued_email_is_processed_and_numbering_continues(client, conn, settings, failed_email):
    client.post(f"/review/{failed_email}/retry")
    process_one(conn, FakeExtractor(GOOD), settings, now=later(100))

    body = client.get(f"/emails/{failed_email}").json()
    assert body["status"] == "accepted"
    history = [(a["attempt_no"], a["outcome"]) for a in body["attempt_history"]]
    n = settings.max_attempts
    assert history == [(i, "retry") for i in range(1, n)] + [(n, "failed"), (n + 1, "success")]


def test_retry_gives_a_full_fresh_budget(client, conn, settings, failed_email):
    client.post(f"/review/{failed_email}/retry")
    fake = FakeExtractor(default=QUOTA)
    for i in range(settings.max_attempts - 1):
        process_one(conn, fake, settings, now=later(100 + i))
        assert repo.get_email(conn, failed_email)["status"] == "pending"

    process_one(conn, fake, settings, now=later(200))
    email = repo.get_email(conn, failed_email)
    assert email["status"] == "failed"
    assert email["attempts"] == 2 * settings.max_attempts
    assert len(fake.calls) == settings.max_attempts


def test_backoff_restarts_from_base_after_retry(client, conn, settings, failed_email):
    client.post(f"/review/{failed_email}/retry")
    now = later(100)
    process_one(conn, FakeExtractor(QUOTA), settings, now=now, rand=MAX_RAND)
    next_at = datetime.fromisoformat(repo.get_email(conn, failed_email)["next_attempt_at"])
    assert next_at == now + timedelta(seconds=settings.backoff_base_seconds)


def test_log_shows_attempt_against_new_limit(client, conn, settings, failed_email, caplog):
    client.post(f"/review/{failed_email}/retry")
    with caplog.at_level(logging.INFO, logger="app.worker"):
        process_one(conn, FakeExtractor(GOOD), settings, now=later(100))
    n = settings.max_attempts
    assert f"attempt={n + 1}/{2 * n} outcome=success" in caplog.records[0].getMessage()


def test_crash_loop_after_retry_still_runs_out(client, conn, settings, failed_email):
    client.post(f"/review/{failed_email}/retry")
    timeout = timedelta(seconds=settings.lock_timeout_seconds)
    now = later(100)
    for _ in range(settings.max_attempts):
        repo.claim_next(conn, now, timeout)  # worker crashes mid-attempt
        now += timeout

    fake = FakeExtractor(GOOD)
    process_one(conn, fake, settings, now=now)
    assert repo.get_email(conn, failed_email)["status"] == "failed"
    assert fake.calls == []


def test_can_retry_again_after_failing_again(client, conn, settings, failed_email):
    client.post(f"/review/{failed_email}/retry")
    process_one(conn, FakeExtractor(ExtractionError("401")), settings, now=later(100))
    assert repo.get_email(conn, failed_email)["status"] == "failed"

    assert client.post(f"/review/{failed_email}/retry").status_code == 200
    process_one(conn, FakeExtractor(GOOD), settings, now=later(200))
    assert repo.get_email(conn, failed_email)["status"] == "accepted"


@pytest.mark.parametrize("result,status", [(GOOD, "accepted"), (GOOD.model_copy(update={
    "fields": GOOD.fields.model_copy(update={"weight_value": None})}), "needs_review")])
def test_only_failed_emails_can_be_retried(client, conn, settings, result, status):
    email_id = client.post("/emails", json={
        "sender": "a@b.de", "body": "x", "received_at": "2026-03-10T09:00:00Z",
    }).json()["id"]
    process_one(conn, FakeExtractor(result), settings)
    assert repo.get_email(conn, email_id)["status"] == status

    r = client.post(f"/review/{email_id}/retry")
    assert r.status_code == 409
    assert "only failed emails can be retried" in r.json()["detail"]


def test_pending_email_cannot_be_retried(client):
    email_id = client.post("/emails", json={"sender": "a@b.de", "body": "x"}).json()["id"]
    assert client.post(f"/review/{email_id}/retry").status_code == 409


def test_unknown_email_is_404(client):
    assert client.post("/review/999/retry").status_code == 404
