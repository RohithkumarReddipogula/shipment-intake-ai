"""Extraction eval: run each dataset email through extraction + the same decision rules
as the worker, compare against expected values, and compute summary metrics."""

import json
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.extraction.base import ExtractionError, Extractor, RetryableExtractionError
from app.schemas import Decision, EmailIn
from app.validation import decide_email

FIELDS = [
    "request_type", "origin", "destination", "pickup_date", "weight_kg", "pallet_count", "cargo_type",
]
LOCATION_FIELDS = {"origin", "destination"}
WEIGHT_TOLERANCE_KG = 1.0


@dataclass(frozen=True)
class EvalCase:
    id: str
    tags: list[str]
    email: EmailIn
    # Locations are lists of acceptable names ("München", "Munich"); None = must be null.
    expected: dict[str, Any]
    expected_decision: Decision
    # Reasons that must be among the actual reasons (subset check).
    expected_reasons: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(EvalCase(
            id=raw["id"],
            tags=raw["tags"],
            email=EmailIn(**raw["email"]),
            expected=raw["expected"],
            expected_decision=Decision(raw["expected_decision"]),
            expected_reasons=raw.get("expected_reasons", []),
        ))
    return cases


def normalize_location(text: str) -> str:
    text = text.casefold()
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def field_matches(name: str, expected: Any, predicted: Any) -> bool:
    if expected is None or predicted is None:
        return expected is None and predicted is None
    if name in LOCATION_FIELDS:
        got = normalize_location(predicted)
        return any(normalize_location(alt) in got for alt in expected)
    if name == "weight_kg":
        return abs(float(expected) - float(predicted)) <= WEIGHT_TOLERANCE_KG
    return str(expected) == str(predicted)


class EvalAborted(Exception):
    """Too many consecutive extraction errors: the API is down or out of quota, and
    carrying on would only burn more calls and produce meaningless metrics."""


def extract_with_retries(
    extractor: Extractor, email: EmailIn, retries: int, sleep: Callable[[float], None]
):
    """Simple retry for the eval run itself (the worker has its own, persistent retries)."""
    for attempt in range(retries + 1):
        try:
            return extractor.extract(email)
        except RetryableExtractionError:
            if attempt == retries:
                raise
            sleep(2.0 * 2 ** attempt)


def evaluate(
    cases: list[EvalCase],
    extractor: Extractor,
    threshold: float,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    on_case: Callable[[dict], None] | None = None,
    delay: float = 0.0,
    max_consecutive_errors: int = 3,
) -> dict:
    results = []
    consecutive_errors = 0
    for i, case in enumerate(cases):
        if i and delay:
            sleep(delay)
        result: dict[str, Any] = {
            "id": case.id, "tags": case.tags, "expected_decision": case.expected_decision.value,
        }
        try:
            output = extract_with_retries(extractor, case.email, retries, sleep)
        except ExtractionError as e:
            # Counts as review: in production the email would end up in the review queue as failed.
            result |= {"decision": "failed", "reasons": [], "error": f"{type(e).__name__}: {e}",
                       "wrong_fields": {f: {"expected": case.expected.get(f), "got": None} for f in FIELDS},
                       "field_correct": {f: False for f in FIELDS}}
        else:
            decision = decide_email(output, case.email, threshold)
            predicted = output.fields.model_dump(mode="json")
            correct = {f: field_matches(f, case.expected.get(f), predicted.get(f)) for f in FIELDS}
            result |= {
                "decision": decision.decision.value,
                "reasons": decision.reasons,
                "error": None,
                "field_correct": correct,
                "wrong_fields": {
                    f: {"expected": case.expected.get(f), "got": predicted.get(f)}
                    for f in FIELDS if not correct[f]
                },
                "cargo_description": predicted.get("cargo_description"),
            }
        result["all_fields_correct"] = all(result["field_correct"].values())
        result["decision_ok"] = (
            result["decision"] == case.expected_decision.value
            or (result["decision"] == "failed" and case.expected_decision == Decision.NEEDS_REVIEW)
        ) and all(r in result["reasons"] for r in case.expected_reasons)
        results.append(result)
        if on_case:
            on_case(result)
        consecutive_errors = consecutive_errors + 1 if result["error"] else 0
        if consecutive_errors >= max_consecutive_errors:
            raise EvalAborted(
                f"{consecutive_errors} consecutive extraction errors, last: {result['error']}"
            )

    return summarize(results, extractor, threshold)


def summarize(results: list[dict], extractor: Extractor, threshold: float) -> dict:
    n = len(results)
    accepted = [r for r in results if r["decision"] == Decision.ACCEPTED.value]
    wrong_accepts = [r["id"] for r in accepted if not r["all_fields_correct"]]
    return {
        "model": extractor.model,
        "prompt_version": extractor.prompt_version,
        "confidence_threshold": threshold,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": n,
        "field_accuracy": {
            f: sum(r["field_correct"][f] for r in results) / n for f in FIELDS
        },
        "exact_match": sum(r["all_fields_correct"] for r in results) / n,
        "auto_accepted": len(accepted),
        # Of the emails we auto-accepted, the share with every field correct. The key metric:
        # a wrong auto-accept costs more than an unnecessary review.
        "auto_accept_precision": (
            (len(accepted) - len(wrong_accepts)) / len(accepted) if accepted else None
        ),
        "wrong_auto_accepts": wrong_accepts,
        "review_rate": (n - len(accepted)) / n,
        "expected_review_rate": sum(r["expected_decision"] != "accepted" for r in results) / n,
        "decision_agreement": sum(r["decision_ok"] for r in results) / n,
        "decision_mismatches": [r["id"] for r in results if not r["decision_ok"]],
        "errors": [r["id"] for r in results if r["error"]],
        "cases": results,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_text(report: dict) -> str:
    lines = [
        f"Model {report['model']}, prompt {report['prompt_version']}, "
        f"threshold {report['confidence_threshold']}, {report['n']} emails",
        "",
        "Per-field accuracy:",
        *[f"  {f:<14} {_pct(a):>5}" for f, a in report["field_accuracy"].items()],
        "",
        f"All fields correct:     {_pct(report['exact_match'])}",
        f"Auto-accepted:          {report['auto_accepted']}/{report['n']}",
        f"Auto-accept precision:  {_pct(report['auto_accept_precision'])}"
        + (f"  (wrong: {', '.join(report['wrong_auto_accepts'])})" if report["wrong_auto_accepts"] else ""),
        f"Review rate:            {_pct(report['review_rate'])}"
        f"  (expected {_pct(report['expected_review_rate'])})",
        f"Decision agreement:     {_pct(report['decision_agreement'])}"
        + (f"  (mismatches: {', '.join(report['decision_mismatches'])})" if report["decision_mismatches"] else ""),
    ]
    if report["errors"]:
        lines.append(f"Extraction errors:      {', '.join(report['errors'])}")
    wrong = [r for r in report["cases"] if r["wrong_fields"] and not r["error"]]
    if wrong:
        lines += ["", "Wrong fields:"]
        for r in wrong:
            for f, v in r["wrong_fields"].items():
                lines.append(f"  {r['id']:<28} {f:<13} expected {v['expected']!r}, got {v['got']!r}")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    out = [
        f"### Eval results: `{report['model']}`, prompt `{report['prompt_version']}`",
        "",
        f"{report['n']} emails, confidence threshold {report['confidence_threshold']}, "
        f"run {report['run_at']}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| All fields correct | {_pct(report['exact_match'])} |",
        f"| Auto-accepted | {report['auto_accepted']}/{report['n']} |",
        f"| Auto-accept precision | {_pct(report['auto_accept_precision'])} |",
        f"| Review rate (expected) | {_pct(report['review_rate'])} ({_pct(report['expected_review_rate'])}) |",
        f"| Decision agreement | {_pct(report['decision_agreement'])} |",
        "",
        "| Field | Accuracy |",
        "|---|---|",
        *[f"| {f} | {_pct(a)} |" for f, a in report["field_accuracy"].items()],
        "",
        "| Email | Tags | Decision (expected) | Wrong fields |",
        "|---|---|---|---|",
    ]
    for r in report["cases"]:
        wrong = ", ".join(r["wrong_fields"]) or "–"
        decision = r["decision"] + ("" if r["decision_ok"] else f" ({r['expected_decision']}) ✗")
        out.append(f"| {r['id']} | {', '.join(r['tags'])} | {decision} | {wrong} |")
    return "\n".join(out) + "\n"
