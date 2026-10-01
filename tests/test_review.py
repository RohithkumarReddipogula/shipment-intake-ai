import json
from datetime import date

import pytest

from app import repo
from app.extraction.base import ExtractionError
from app.extraction.fake import FakeExtractor
from app.schemas import ExtractionOutput, FieldConfidence, ShipmentFields
from app.worker import process_one

FIELDS = dict(
    request_type="booking", origin="Hamburg", destination="München",
    pickup_date=date(2026, 3, 17), weight_value=1200, weight_unit="kg",
    pallet_count=4, cargo_type="general",
)
CORRECTED = FIELDS | {"weight_value": 1250, "pickup_date": "2026-03-18"}


def output(**overrides) -> ExtractionOutput:
    conf = {k: 0.95 for k in FieldConfidence.model_fields}
    return ExtractionOutput(fields=ShipmentFields(**(FIELDS | overrides)), confidence=FieldConfidence(**conf))


@pytest.fixture
def submit(client, conn, settings):
    """Post an email and run the worker on it with the given extractor result."""
    counter = iter(range(1, 1000))

    def _submit(result, body="4 Paletten Hamburg -> München"):
        n = next(counter)
        email_id = client.post("/emails", json={
            "message_id": f"m{n}", "sender": "a@b.de", "subject": f"Buchung {n}",
            "body": body, "received_at": "2026-03-10T09:00:00Z",
        }).json()["id"]
        process_one(conn, FakeExtractor(result), settings)
        return email_id

    return _submit


class TestQueue:
    def test_lists_needs_review_and_failed_not_accepted(self, client, submit):
        accepted = submit(output())
        review = submit(output(weight_value=None))
        failed = submit(ExtractionError("401 bad key"))

        items = client.get("/review").json()
        assert [i["id"] for i in items] == [review, failed]
        assert accepted not in [i["id"] for i in items]

        review_item, failed_item = items
        assert review_item["status"] == "needs_review"
        assert review_item["reasons"] == ["missing:weight"]
        assert review_item["fields"]["origin"] == "Hamburg"
        assert failed_item["status"] == "failed"
        assert failed_item["fields"] is None
        assert failed_item["last_error"] == "ExtractionError: 401 bad key"

    def test_filter_by_reason(self, client, submit):
        submit(output(weight_value=None))
        dg = submit(output(), body="4 Paletten Lack, Gefahrgut UN 1263")

        items = client.get("/review", params={"reason": "dangerous_goods"}).json()
        assert [i["id"] for i in items] == [dg]
        assert client.get("/review", params={"reason": "missing:origin"}).json() == []

    def test_limit(self, client, submit):
        for _ in range(3):
            submit(output(weight_value=None))
        assert len(client.get("/review", params={"limit": 2}).json()) == 2


class TestApprove:
    def test_approve_accepts_without_changing_data(self, client, conn, submit):
        email_id = submit(output(), body="Gefahrgut UN 1263")  # needs review: dangerous goods
        before = repo.get_latest_extraction(conn, email_id)["data_json"]

        r = client.post(f"/review/{email_id}", json={"action": "approve", "reviewer": "anna"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "accepted"
        assert body["extraction"]["review_action"] == "approve"
        assert body["extraction"]["reviewed_by"] == "anna"
        assert body["extraction"]["corrected"] is None
        assert body["extraction"]["reasons"] == ["dangerous_goods"]  # why it was reviewed is kept
        assert repo.get_latest_extraction(conn, email_id)["data_json"] == before
        assert client.get("/review").json() == []

    def test_approve_failed_email_is_conflict(self, client, submit):
        email_id = submit(ExtractionError("401"))
        r = client.post(f"/review/{email_id}", json={"action": "approve", "reviewer": "anna"})
        assert r.status_code == 409
        assert "use 'correct'" in r.json()["detail"]


class TestCorrect:
    def test_correction_stored_next_to_original(self, client, conn, submit):
        email_id = submit(output(weight_value=None))
        r = client.post(f"/review/{email_id}", json={
            "action": "correct", "reviewer": "anna", "corrected": CORRECTED,
        })
        assert r.status_code == 200
        extraction = r.json()["extraction"]
        assert r.json()["status"] == "accepted"
        # Original untouched, correction alongside it.
        assert extraction["fields"]["weight_value"] is None
        assert extraction["corrected"]["weight_value"] == 1250
        assert extraction["corrected"]["weight_kg"] == 1250
        assert extraction["corrected"]["pickup_date"] == "2026-03-18"
        assert extraction["review_action"] == "correct"

        # Same row in the DB: one extraction, both versions.
        rows = conn.execute("SELECT * FROM extractions WHERE email_id = ?", (email_id,)).fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0]["data_json"])["weight_value"] is None
        assert json.loads(rows[0]["corrected_data_json"])["weight_value"] == 1250

    def test_correct_failed_email_creates_empty_original(self, client, conn, submit):
        email_id = submit(ExtractionError("401"))
        r = client.post(f"/review/{email_id}", json={
            "action": "correct", "reviewer": "anna", "corrected": CORRECTED,
        })
        assert r.status_code == 200
        extraction = r.json()["extraction"]
        assert extraction["model"] == "none"
        assert extraction["reasons"] == ["extraction_failed"]
        assert all(v is None for v in extraction["fields"].values())
        assert extraction["corrected"]["origin"] == "Hamburg"
        assert r.json()["status"] == "accepted"

    def test_correct_requires_corrected(self, client, submit):
        email_id = submit(output(weight_value=None))
        r = client.post(f"/review/{email_id}", json={"action": "correct", "reviewer": "anna"})
        assert r.status_code == 422

    def test_approve_rejects_corrected(self, client, submit):
        email_id = submit(output(weight_value=None))
        r = client.post(f"/review/{email_id}", json={
            "action": "approve", "reviewer": "anna", "corrected": CORRECTED,
        })
        assert r.status_code == 422

    def test_invalid_corrected_values_rejected(self, client, submit):
        email_id = submit(output(weight_value=None))
        r = client.post(f"/review/{email_id}", json={
            "action": "correct", "reviewer": "anna", "corrected": CORRECTED | {"cargo_type": "furniture"},
        })
        assert r.status_code == 422


class TestConflicts:
    def test_cannot_review_twice(self, client, submit):
        email_id = submit(output(weight_value=None))
        first = client.post(f"/review/{email_id}", json={"action": "approve", "reviewer": "anna"})
        second = client.post(f"/review/{email_id}", json={
            "action": "correct", "reviewer": "ben", "corrected": CORRECTED,
        })
        assert first.status_code == 200
        assert second.status_code == 409
        assert "accepted, not awaiting review" in second.json()["detail"]

    def test_cannot_review_auto_accepted(self, client, submit):
        email_id = submit(output())
        r = client.post(f"/review/{email_id}", json={"action": "approve", "reviewer": "anna"})
        assert r.status_code == 409

    def test_cannot_review_pending(self, client):
        email_id = client.post("/emails", json={"sender": "a@b.de", "body": "x"}).json()["id"]
        r = client.post(f"/review/{email_id}", json={"action": "approve", "reviewer": "anna"})
        assert r.status_code == 409

    def test_unknown_email_is_404(self, client):
        r = client.post("/review/999", json={"action": "approve", "reviewer": "anna"})
        assert r.status_code == 404

    def test_reviewer_required(self, client, submit):
        email_id = submit(output(weight_value=None))
        r = client.post(f"/review/{email_id}", json={"action": "approve", "reviewer": ""})
        assert r.status_code == 422
