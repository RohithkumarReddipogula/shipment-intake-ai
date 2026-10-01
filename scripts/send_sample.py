"""Post a sample freight email to the API and poll until it's processed.

    uvicorn app.main:app            # terminal 1
    python -m app.worker            # terminal 2
    python scripts/send_sample.py   # terminal 3
"""

import argparse
import json
import time
import urllib.request
import uuid
from datetime import datetime, timezone

SAMPLE = {
    "sender": "disposition@mueller-maschinenbau.de",
    "subject": "Transportauftrag / booking: 6 EUR-Paletten Hamburg -> Lyon",
    "body": (
        "Hallo zusammen,\n\n"
        "bitte folgenden Transport für uns buchen:\n\n"
        "Abholung: Mueller Maschinenbau GmbH, 21107 Hamburg\n"
        "Zustellung: Atelier Rhône SARL, 69007 Lyon, FR\n"
        "Ladung: 6 EUR-Paletten Maschinenteile, gesamt 2.340 kg\n"
        "Abholtermin: nächsten Dienstag ab 08:00\n\n"
        "Our French colleagues need delivery within 3 days if possible.\n\n"
        "Mit freundlichen Grüßen\nAnna Schmidt\nDisposition"
    ),
}

DONE = {"accepted", "needs_review", "failed"}


def request(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait for processing")
    args = parser.parse_args()

    # A fresh Message-ID per run, so repeated runs aren't deduplicated.
    email = SAMPLE | {
        "message_id": f"<{uuid.uuid4()}@sample.local>",
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    status, created = request("POST", f"{args.base_url}/emails", email)
    print(f"POST /emails -> {status} {created}")

    deadline = time.monotonic() + args.timeout
    last = None
    while time.monotonic() < deadline:
        _, result = request("GET", f"{args.base_url}/emails/{created['id']}")
        if result["status"] != last:
            print(f"status: {result['status']} (attempts: {result['attempts']})")
            last = result["status"]
        if result["status"] in DONE:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        time.sleep(1)
    raise SystemExit(f"timed out after {args.timeout}s; is the worker running?")


if __name__ == "__main__":
    main()
