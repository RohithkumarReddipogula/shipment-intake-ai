from pathlib import Path

import pytest

from app.evaluation import evaluate, field_matches, load_dataset, render_markdown, render_text
from app.extraction.base import ExtractionError, RetryableExtractionError
from app.extraction.fake import FakeExtractor
from app.schemas import ExtractionOutput, FieldConfidence, ShipmentFields

DATASET = Path(__file__).parent.parent / "eval" / "dataset.jsonl"
CASES = load_dataset(DATASET)
THRESHOLD = 0.8


def oracle_output(case, **overrides) -> ExtractionOutput:
    """What a perfect extractor would return for this case."""
    e = case.expected | overrides
    fields = ShipmentFields(
        request_type=e["request_type"],
        origin=e["origin"][0] if e["origin"] else None,
        destination=e["destination"][0] if e["destination"] else None,
        pickup_date=e["pickup_date"],
        weight_value=e["weight_kg"],
        weight_unit="kg" if e["weight_kg"] is not None else None,
        pallet_count=e["pallet_count"],
        cargo_type=e["cargo_type"],
    )
    return ExtractionOutput(fields=fields, confidence=FieldConfidence(**{k: 1.0 for k in FieldConfidence.model_fields}))


def run(extractor, cases=CASES):
    return evaluate(cases, extractor, THRESHOLD, sleep=lambda _: None)


class TestDataset:
    def test_has_15_unique_cases(self):
        assert len(CASES) == 15
        assert len({c.id for c in CASES}) == 15

    @pytest.mark.parametrize(
        "tag",
        ["german", "mixed_language", "dangerous_goods", "injection", "missing_weight", "lbs",
         "forwarded_thread", "quote_no_date", "relative_date"],
    )
    def test_covers_required_case(self, tag):
        assert any(tag in c.tags for c in CASES)

    def test_expected_decisions_match_our_rules(self):
        # With perfect extraction, our rules must reach each case's expected decision.
        # Catches dataset labels that contradict the validation rules.
        report = run(FakeExtractor(*[oracle_output(c) for c in CASES]))
        assert report["decision_mismatches"] == []


def test_perfect_extractor_scores_100_percent():
    report = run(FakeExtractor(*[oracle_output(c) for c in CASES]))
    assert all(a == 1.0 for a in report["field_accuracy"].values())
    assert report["exact_match"] == 1.0
    assert report["auto_accept_precision"] == 1.0
    assert report["review_rate"] == report["expected_review_rate"] == 6 / 15


def test_wrong_field_on_accepted_email_lowers_precision():
    outputs = [
        oracle_output(c, pallet_count=99) if c.id == "en_booking_lbs" else oracle_output(c) for c in CASES
    ]
    report = run(FakeExtractor(*outputs))
    assert report["wrong_auto_accepts"] == ["en_booking_lbs"]
    assert report["auto_accept_precision"] == 8 / 9
    assert report["field_accuracy"]["pallet_count"] == 14 / 15
    [case] = [r for r in report["cases"] if r["id"] == "en_booking_lbs"]
    assert case["wrong_fields"] == {"pallet_count": {"expected": 2, "got": 99}}


def test_injection_obeyed_is_caught():
    # A model that follows the injection: origin Berlin, weight 500 kg. The email is then
    # complete, so it gets auto-accepted with wrong data, which the eval must report.
    [case] = [c for c in CASES if c.id == "en_prompt_injection"]
    obeyed = oracle_output(case, origin=["Berlin"], weight_kg=500)
    report = run(FakeExtractor(obeyed), cases=[case])
    assert report["wrong_auto_accepts"] == ["en_prompt_injection"]
    assert report["decision_mismatches"] == ["en_prompt_injection"]


def test_retryable_errors_are_retried_then_counted_as_failed():
    case = CASES[0]
    report = run(FakeExtractor(RetryableExtractionError("503"), oracle_output(case)), cases=[case])
    assert report["errors"] == []
    assert report["exact_match"] == 1.0

    report = run(FakeExtractor(ExtractionError("401")), cases=[case])
    assert report["errors"] == [case.id]
    assert report["cases"][0]["decision"] == "failed"
    assert report["review_rate"] == 1.0


@pytest.mark.parametrize(
    "name,expected,predicted,ok",
    [
        ("origin", ["München", "Munich"], "80331 München", True),
        ("origin", ["München", "Munich"], "Muenchen, DE", True),
        ("origin", ["Gdańsk", "Danzig"], "80-001 Gdansk, Poland", True),
        ("origin", ["Hamburg"], "Bremen", False),
        ("origin", None, None, True),
        ("origin", None, "Hamburg", False),
        ("weight_kg", 1995.81, 1995.8, True),
        ("weight_kg", 1995.81, 2000, False),
        ("weight_kg", None, 500, False),
        ("pickup_date", "2026-03-17", "2026-03-17", True),
        ("cargo_type", "machinery", "general", False),
    ],
)
def test_field_matches(name, expected, predicted, ok):
    assert field_matches(name, expected, predicted) is ok


def test_reports_render():
    report = run(FakeExtractor(*[oracle_output(c) for c in CASES]))
    text = render_text(report)
    assert "Auto-accept precision:  100%" in text
    md = render_markdown(report)
    assert "| Auto-accept precision | 100% |" in md
    assert md.count("\n| de_") + md.count("\n| en_") + md.count("\n| mixed_") == 15


def test_aborts_after_consecutive_errors():
    from app.evaluation import EvalAborted

    fake = FakeExtractor(default=RetryableExtractionError("429 quota"))
    with pytest.raises(EvalAborted, match="3 consecutive"):
        evaluate(CASES, fake, THRESHOLD, retries=0, sleep=lambda _: None)
    assert len(fake.calls) == 3


def test_isolated_error_does_not_abort():
    outputs = [ExtractionError("401") if i == 1 else oracle_output(c) for i, c in enumerate(CASES)]
    report = run(FakeExtractor(*outputs))
    assert report["errors"] == [CASES[1].id]


def test_delay_between_emails():
    sleeps = []
    evaluate(CASES[:3], FakeExtractor(*[oracle_output(c) for c in CASES[:3]]), THRESHOLD,
             sleep=sleeps.append, delay=2.0)
    assert sleeps == [2.0, 2.0]


def test_cli_saves_files_named_after_dotted_model(tmp_path, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_eval", DATASET.parent / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    fake = FakeExtractor(*[oracle_output(c) for c in CASES])
    fake.model = "gemini-3.8-flash"
    monkeypatch.setattr(run_eval.GeminiExtractor, "from_settings", lambda _: fake)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--out-dir", str(tmp_path), "--delay", "0"])
    run_eval.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "gemini-3.8-flash_fake-v1.json", "gemini-3.8-flash_fake-v1.md",
    ]


def test_cli_does_not_save_when_emails_failed(tmp_path, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_eval", DATASET.parent / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    outputs = [ExtractionError("401") if i == 1 else oracle_output(c) for i, c in enumerate(CASES)]
    monkeypatch.setattr(run_eval.GeminiExtractor, "from_settings", lambda _: FakeExtractor(*outputs))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--out-dir", str(tmp_path), "--delay", "0"])
    with pytest.raises(SystemExit, match="Nothing saved"):
        run_eval.main()
    assert list(tmp_path.iterdir()) == []
