import pytest

from app.extraction.base import RetryableExtractionError
from app.extraction.fake import FakeExtractor
from app.schemas import EmailIn, ExtractionOutput, FieldConfidence, ShipmentFields

EMAIL = EmailIn(sender="x@y.de", body="b")
OUTPUT = ExtractionOutput(fields=ShipmentFields(origin="Hamburg"), confidence=FieldConfidence())


def test_script_then_default():
    fake = FakeExtractor(RetryableExtractionError("timeout"), OUTPUT, default=OUTPUT)
    with pytest.raises(RetryableExtractionError):
        fake.extract(EMAIL)
    assert fake.extract(EMAIL) is OUTPUT
    assert fake.extract(EMAIL) is OUTPUT
    assert len(fake.calls) == 3


def test_unscripted_call_fails_loudly():
    with pytest.raises(AssertionError):
        FakeExtractor().extract(EMAIL)
