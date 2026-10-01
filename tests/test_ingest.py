from datetime import datetime, timezone

from app import repo
from app.schemas import EmailIn

EMAIL = {
    "message_id": "<abc123@forwarder.de>",
    "sender": "dispo@kunde.de",
    "subject": "Anfrage: 4 Paletten Hamburg -> München",
    "body": "Hallo,\nbitte um ein Angebot für 4 Paletten, 1.200 kg, Abholung nächsten Dienstag.\n",
    "received_at": "2026-03-10T09:15:00+01:00",
}


def test_new_email_is_queued(client):
    r = client.post("/emails", json=EMAIL)
    assert r.status_code == 202
    assert r.json() == {"id": 1, "status": "pending", "duplicate": False}


def test_same_message_id_is_not_queued_twice(client, conn):
    first = client.post("/emails", json=EMAIL)
    second = client.post("/emails", json=EMAIL | {"body": "different body, same Message-ID"})
    assert second.status_code == 200
    assert second.json() == {"id": first.json()["id"], "status": "pending", "duplicate": True}
    assert conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0] == 1


def test_without_message_id_dedups_on_content(client):
    no_mid = EMAIL | {"message_id": None}
    first = client.post("/emails", json=no_mid)
    # Whitespace/line-ending noise and sender case don't create a new email.
    resent = no_mid | {
        "sender": "Dispo@Kunde.de",
        "subject": "  Anfrage:  4 Paletten Hamburg -> München ",
        "body": EMAIL["body"].replace("\n", "  \r\n"),
    }
    second = client.post("/emails", json=resent)
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


def test_different_content_without_message_id_is_new(client):
    no_mid = EMAIL | {"message_id": None}
    client.post("/emails", json=no_mid)
    r = client.post("/emails", json=no_mid | {"body": "6 Paletten bitte"})
    assert r.status_code == 202
    assert r.json()["id"] == 2


def test_message_id_angle_brackets_are_ignored():
    a = EmailIn(sender="x@y.de", body="b", message_id="<id@host>")
    b = EmailIn(sender="x@y.de", body="b", message_id="id@host")
    assert repo.dedup_key(a) == repo.dedup_key(b) == "mid:id@host"


def test_received_at_stored_as_utc(client):
    email_id = client.post("/emails", json=EMAIL).json()["id"]
    body = client.get(f"/emails/{email_id}").json()
    assert datetime.fromisoformat(body["received_at"]) == datetime(2026, 3, 10, 8, 15, tzinfo=timezone.utc)


def test_get_email(client):
    email_id = client.post("/emails", json=EMAIL).json()["id"]
    body = client.get(f"/emails/{email_id}").json()
    assert body["status"] == "pending"
    assert body["attempts"] == 0
    assert body["extraction"] is None


def test_get_unknown_email_is_404(client):
    assert client.get("/emails/999").status_code == 404


def test_missing_body_is_rejected(client):
    assert client.post("/emails", json={"sender": "x@y.de"}).status_code == 422
