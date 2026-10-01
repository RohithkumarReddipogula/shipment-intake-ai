# shipment-intake-ai

A small service that reads freight forwarding emails (quote requests and bookings, in
English and German), uses an LLM (Gemini) to extract shipment details, validates them,
and either auto-accepts the email or sends it to a human review queue.

Extracted fields: request type (quote / booking / other), origin, destination, pickup
date, weight (normalized to kg), pallet count, cargo type (fixed categories) and the
original cargo wording.

Stack: FastAPI, SQLite, Pydantic, `google-genai`. No other infrastructure: the job queue
is a table in the same SQLite database.

## Architecture

```
                    POST /emails                          ┌──────────────── SQLite (WAL) ───────────────┐
  mail gateway ───────────────────▶  FastAPI  ──────────▶ │ emails       one row per email + job state  │
                                     (app/main.py)        │ extractions  LLM output, decision, review   │
  reviewer     ◀──── GET /review ───                      │ attempts     one row per processing attempt │
               ──── POST /review/{id} ─▶                  └─────────────────────────────────────────────┘
                                                                 ▲  claim job    │  write result
                                                                 │               ▼
                                     Worker (python -m app.worker, separate process)
                                       1. claim one due job  (atomic UPDATE ... RETURNING)
                                       2. Gemini extracts fields + per-field confidence
                                          (email wrapped in tagged markers, treated as data)
                                       3. rules validate and decide  (app/validation.py)
                                       4. write extraction + status, or schedule a retry
```

Email status lifecycle:

```
 pending ──claim──▶ processing ──▶ accepted                     (all rules pass)
    ▲                   │     └──▶ needs_review ──review──▶ accepted
    └── retry (backoff) ┤
                        └────────▶ failed ──────review──▶ accepted   (correct only)
```

A job stuck in `processing` (worker crashed) is claimed again after `LOCK_TIMEOUT_SECONDS`.

## Project layout

```
app/
  main.py            API: intake, email status, review queue
  worker.py          claim → extract → decide loop, retries, stuck-job recovery
  validation.py      deterministic rules and the accept/review decision
  repo.py            all SQL
  db.py              schema, connection, transactions
  schemas.py         Pydantic models (input, LLM output schema, API responses)
  config.py          settings from environment / .env
  evaluation.py      eval metrics (used by eval/run_eval.py)
  extraction/
    base.py          Extractor protocol + error types
    gemini.py        Gemini extractor and error mapping
    prompt.py        system prompt + email wrapping (PROMPT_VERSION)
    fake.py          scripted fake extractor for tests
eval/
  dataset.jsonl      15 labelled test emails
  run_eval.py        runs the eval against Gemini, saves results
scripts/
  send_sample.py     posts one sample email and follows it through processing
tests/               pytest, no network
```

## Running it

Requires Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env          # then set GEMINI_API_KEY
```

`.env` and `.env.*` (except `.env.example`) are gitignored.

**API and worker** run as two processes and share the SQLite file:

```bash
uvicorn app.main:app          # terminal 1: API on http://127.0.0.1:8000 (docs at /docs)
python -m app.worker          # terminal 2: worker, logs one line per attempt
```

**End-to-end check:** with both running, post a sample mixed German/English booking and
watch it go through:

```bash
python scripts/send_sample.py # terminal 3
```

**Tests:** use a fake extractor and temporary databases; no API key or network needed:

```bash
pytest
```

**Eval:** calls Gemini once per email (15 emails, 2 s apart):

```bash
python eval/run_eval.py
```

### API

| Method | Path | Description |
|---|---|---|
| `POST` | `/emails` | Submit an email. `202` if queued, `200` with `duplicate: true` if already received. |
| `GET` | `/emails/{id}` | Status, latest extraction (with any review), and full attempt history. |
| `GET` | `/review` | Emails in `needs_review` or `failed`, oldest first. `?reason=dangerous_goods` filters by reason code. |
| `POST` | `/review/{id}` | `{"action": "approve", "reviewer": "..."}` or `{"action": "correct", "reviewer": "...", "corrected": {...all fields...}}`. |

### Configuration

Set in `.env` or the environment.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | Required for the worker and the eval. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | |
| `GEMINI_TIMEOUT_SECONDS` | `30` | Per request. |
| `DB_PATH` | `shipment_intake.db` | |
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum per-field confidence for auto-accept. |
| `DEDUP_WINDOW_HOURS` | `24` | Duplicate window for emails without a Message-ID. |
| `MAX_ATTEMPTS` | `8` | Attempts before an email is marked `failed`. |
| `BACKOFF_BASE_SECONDS` / `BACKOFF_MAX_SECONDS` | `2` / `300` | Retry delay: `base * 2^(n-1)`, capped, with jitter. |
| `LOCK_TIMEOUT_SECONDS` | `300` | When a job stuck in `processing` is reclaimed. |
| `POLL_INTERVAL_SECONDS` | `1` | Worker sleep when no job is due. |

## Design decisions

### The AI extracts, the rules decide

The model only fills in fields and a confidence per field. It has no way to accept
anything: the output schema has no decision field. The decision is made by deterministic
code in `app/validation.py`. An email is auto-accepted only if all of these hold:

- every required field is present (pickup date is required for bookings, not for quotes,
  which often don't have a date yet)
- the rules pass: pickup date not before the email's received date and not more than a
  year after it, weight and pallet count positive, weight has a unit, origin ≠ destination,
  request type is a quote or booking
- every present field's confidence is at least `CONFIDENCE_THRESHOLD`, including an
  optional quote pickup date (a wrong date is worse than a missing one)
- it is not dangerous goods (below)

Otherwise it goes to review with stable reason codes (`missing:weight`,
`invalid:pickup_date_before_received`, `low_confidence:origin`, `dangerous_goods`, ...) that
the review queue can filter on and the eval can count.

The model's self-reported confidence is not well calibrated, so it is one check among
several rather than the main one. Dates are checked against the email's received date,
not today, so the same email gives the same result whenever it is processed.

### Dangerous goods are never auto-accepted

Dangerous goods always go to review with reason `dangerous_goods`, however confident the
model is. Two independent checks each trigger it:

1. the model classifies the cargo as `dangerous_goods`, or
2. the email text contains a marker: ADR, Gefahrgut, Gefahrstoff, a UN number, hazardous,
   hazmat, IMDG or "dangerous goods".

The second check is plain code, so the rule doesn't depend on the model. Negated mentions
("kein Gefahrgut", "non-hazardous", "no ADR"), which are common in normal requests, don't
trigger it. ADR and UN numbers are matched in capitals only, so the German abbreviation
"Adr." (Adresse) doesn't trigger it either. The check errs toward review: a false
positive costs one review, a false negative auto-accepts dangerous goods.

### Prompt injection

Email content is untrusted. It is placed between `<<<EMAIL_START tag>>>` and
`<<<EMAIL_END tag>>>` markers with a random tag per call, so text inside the email can't
fake the closing marker. The system prompt says the content is data only and that the
model doesn't make the decision. The real protection is structural: even a model that
obeys an injection can only change field values, and those still go through the rules.
The eval set contains an injection email to measure what happens.

### Idempotency: Message-ID, or a content hash within a time window

- With a Message-ID, an email is a duplicate if that ID was ever seen (unique column).
- Without one, it's a duplicate if the same content hash (sender, subject and body,
  whitespace-normalized) arrived within `DEDUP_WINDOW_HOURS`. That catches client
  retries, which happen within minutes, but still accepts next week's identical repeat
  booking, which is common in freight.

The check and the insert run in one `BEGIN IMMEDIATE` transaction, so two concurrent
identical requests can't both get in.

### Retries and stuck-job recovery

- **Claiming** a job is one atomic `UPDATE ... RETURNING` that sets `processing`,
  increments `attempts` and opens a row in the `attempts` table.
- **Retryable errors** (Gemini 429 and 5xx, timeouts, connection errors, output that
  fails the schema, unexpected exceptions) are retried with exponential backoff and
  jitter. The backoff is measured from when the call ended. After `MAX_ATTEMPTS` the
  email is `failed` and shows up in the review queue.
- **Permanent errors** (400, 401, 403 and other 4xx) fail immediately.
- **Retries happen only in the worker.** The SDK's own retries are off, so every attempt
  is visible in `attempt_history` and the worker log.
- **Stuck jobs:** a job in `processing` longer than `LOCK_TIMEOUT_SECONDS` is claimed
  again and its unfinished attempt is recorded as `abandoned`. A crash still counts as an
  attempt, so an email that crashes the worker every time ends up `failed` instead of
  looping forever.
- **A slow worker can't overwrite a reclaimed job:** completion only succeeds if
  `attempts` still matches the value at claim time. If not, the late result is discarded.

### Review and the feedback loop

`POST /review/{id}` never modifies the original extraction. A correction is stored next
to it on the same row (`corrected_data_json`, `reviewed_by`, `reviewed_at`), so each
reviewed email gives an original/corrected pair that can be turned into a new eval case.
Corrections must be a complete field set, not a patch, so each one is a full label. A
failed email has no extraction, so correcting it first creates an empty one (reason
`extraction_failed`). Reviewing an email that isn't awaiting review returns `409`.

### Other choices

- **Raw `sqlite3`, no ORM:** three tables, and the claim and dedup logic is easier to
  check as plain SQL.
- **Separate worker process:** LLM calls don't block the API's event loop. SQLite in WAL
  mode handles one API process and one worker.
- **Weight:** the model returns the value and unit as written (kg, lbs, t), and code
  converts to kg, so the model doesn't do arithmetic.
- **`PROMPT_VERSION` is stored with every extraction,** so results can be compared per
  prompt version.

## Evaluation results

<!-- EVAL RESULTS PLACEHOLDER: replace this section with the contents of
     eval/results/<model>_<prompt_version>.md after running `python eval/run_eval.py`. -->

> **Placeholder: not run yet.** Paste `eval/results/gemini-3.8-flash_v2.md` here after
> the first complete eval run.

The 15 emails in `eval/dataset.jsonl` cover German, mixed German/English and English
emails, dangerous goods, a prompt injection, missing weight, lbs, a forwarded thread with
a correction, a quote without a date, relative dates ("nächsten Dienstag", "morgen",
"next Tuesday" sent on a Tuesday), a past pickup date, a vague date, a weight given per
pallet and a message that isn't a shipment request.

What `run_eval.py` reports:

- **Per-field accuracy.** Locations match if any accepted name appears in the prediction
  (for example "München" or "Munich" in "80331 München"). Weight is compared in kg with
  ±1 kg tolerance.
- **Auto-accept precision:** of the emails the system auto-accepted, the share with every
  field correct. This is the key number, because a wrong auto-accept costs more than an
  unnecessary review. Wrong ones are listed by id.
- **Review rate,** next to the expected rate for the dataset (6 of 15 should go to review).
- **Decision agreement** with the expected accept/review label.

Each email goes through the same decision code as the worker. The run stops after 3
consecutive API errors, and results are not saved if any email failed to extract, so an
API outage can't show up as bad accuracy.

## Limitations and next steps

- No authentication on the API.
- No schema migrations: schema changes need a fresh database.
- A single SQLite file suits one API process and one worker, not many.
- Failed emails can be corrected but not re-queued for another extraction attempt.
- Turning review corrections into eval cases is manual for now.
- The received date is taken in UTC, so an email sent just after midnight German time
  counts as the previous day for date checks.
