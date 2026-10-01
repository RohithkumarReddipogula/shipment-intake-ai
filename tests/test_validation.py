from datetime import date

import pytest

from app.schemas import Decision, ExtractionOutput, FieldConfidence, ShipmentFields
from app.validation import decide, validate

RECEIVED = date(2026, 3, 10)
THRESHOLD = 0.8


def make_fields(**overrides) -> ShipmentFields:
    base = dict(
        request_type="booking",
        origin="Hamburg",
        destination="München",
        pickup_date=date(2026, 3, 17),
        weight_value=1200,
        weight_unit="kg",
        pallet_count=4,
        cargo_type="general cargo",
    )
    return ShipmentFields(**(base | overrides))


def make_output(confidence: float = 0.95, **overrides) -> ExtractionOutput:
    conf = {k: confidence for k in FieldConfidence.model_fields}
    return ExtractionOutput(fields=make_fields(**overrides), confidence=FieldConfidence(**conf))


def test_complete_confident_extraction_is_accepted():
    result = decide(make_output(), RECEIVED, THRESHOLD)
    assert result.decision == Decision.ACCEPTED
    assert result.reasons == []


@pytest.mark.parametrize(
    "field,reason",
    [
        ("origin", "missing:origin"),
        ("pickup_date", "missing:pickup_date"),
        ("pallet_count", "missing:pallet_count"),
        ("cargo_type", "missing:cargo_type"),
        ("request_type", "missing:request_type"),
    ],
)
def test_missing_required_field_needs_review(field, reason):
    result = decide(make_output(**{field: None}), RECEIVED, THRESHOLD)
    assert result.decision == Decision.NEEDS_REVIEW
    assert reason in result.reasons


def test_missing_weight_reported_once():
    result = decide(make_output(weight_value=None, weight_unit=None), RECEIVED, THRESHOLD)
    assert result.reasons == ["missing:weight"]


def test_weight_without_unit():
    result = decide(make_output(weight_unit=None), RECEIVED, THRESHOLD)
    assert "invalid:weight_unit_missing" in result.reasons
    assert "missing:weight" in result.reasons


def test_lbs_weight_is_valid():
    result = decide(make_output(weight_value=2200, weight_unit="lbs"), RECEIVED, THRESHOLD)
    assert result.decision == Decision.ACCEPTED


class TestPickupDate:
    def test_same_day_as_received_is_valid(self):
        assert validate(make_fields(pickup_date=RECEIVED), RECEIVED) == []

    def test_before_received_is_invalid(self):
        errors = validate(make_fields(pickup_date=date(2026, 3, 9)), RECEIVED)
        assert errors == ["invalid:pickup_date_before_received"]

    def test_compares_against_received_date_not_today(self):
        # Both dates are long past "today"; only their relation matters.
        old_received = date(2020, 1, 1)
        assert validate(make_fields(pickup_date=date(2020, 1, 7)), old_received) == []

    def test_more_than_a_year_out_is_invalid(self):
        errors = validate(make_fields(pickup_date=date(2027, 3, 11)), RECEIVED)
        assert errors == ["invalid:pickup_date_too_far"]


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"weight_value": 0}, "invalid:weight_not_positive"),
        ({"weight_value": -5}, "invalid:weight_not_positive"),
        ({"pallet_count": 0}, "invalid:pallet_count_not_positive"),
        ({"destination": " hamburg "}, "invalid:origin_equals_destination"),
        ({"request_type": "other"}, "invalid:request_type_other"),
    ],
)
def test_rule_violations(overrides, error):
    result = decide(make_output(**overrides), RECEIVED, THRESHOLD)
    assert result.decision == Decision.NEEDS_REVIEW
    assert error in result.validation_errors
    assert error in result.reasons


def test_low_confidence_needs_review():
    output = make_output()
    output.confidence.pickup_date = 0.5
    result = decide(output, RECEIVED, THRESHOLD)
    assert result.decision == Decision.NEEDS_REVIEW
    assert result.reasons == ["low_confidence:pickup_date"]


def test_confidence_at_threshold_is_accepted():
    assert decide(make_output(confidence=THRESHOLD), RECEIVED, THRESHOLD).decision == Decision.ACCEPTED


def test_low_confidence_not_reported_for_missing_field():
    output = make_output(origin=None)
    output.confidence.origin = 0.0
    result = decide(output, RECEIVED, THRESHOLD)
    assert result.reasons == ["missing:origin"]


class TestPickupDateRequiredOnlyForBookings:
    def test_booking_without_pickup_date_needs_review(self):
        result = decide(make_output(request_type="booking", pickup_date=None), RECEIVED, THRESHOLD)
        assert result.decision == Decision.NEEDS_REVIEW
        assert result.reasons == ["missing:pickup_date"]

    def test_quote_without_pickup_date_is_accepted(self):
        result = decide(make_output(request_type="quote", pickup_date=None), RECEIVED, THRESHOLD)
        assert result.decision == Decision.ACCEPTED
        assert result.reasons == []

    def test_quote_with_pickup_date_still_validates_it(self):
        result = decide(
            make_output(request_type="quote", pickup_date=date(2026, 3, 1)), RECEIVED, THRESHOLD
        )
        assert result.reasons == ["invalid:pickup_date_before_received"]

    def test_unknown_request_type_requires_pickup_date(self):
        result = decide(make_output(request_type=None, pickup_date=None), RECEIVED, THRESHOLD)
        assert result.reasons == ["missing:request_type", "missing:pickup_date"]
