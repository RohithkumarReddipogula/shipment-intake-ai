"""Deterministic validation and the accept/review decision.

Reasons are short stable codes ("missing:origin", "invalid:...", "low_confidence:weight")
so they can be counted in the eval and filtered in the review queue.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from app.schemas import Decision, ExtractionOutput, RequestType, ShipmentFields

# Reason/confidence name -> attribute on ShipmentFields. The names match FieldConfidence,
# so "weight" is used everywhere even though the value checked is weight_kg.
CHECKED_FIELDS: dict[str, str] = {
    "request_type": "request_type",
    "origin": "origin",
    "destination": "destination",
    "pickup_date": "pickup_date",
    "weight": "weight_kg",
    "pallet_count": "pallet_count",
    "cargo_type": "cargo_type",
}

# A pickup date this far after the email is almost always a wrong year.
MAX_PICKUP_HORIZON = timedelta(days=365)


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    reasons: list[str]
    validation_errors: list[str]


def required_fields(fields: ShipmentFields) -> list[str]:
    """Quotes often have no fixed pickup date yet; everything else needs one."""
    if fields.request_type == RequestType.QUOTE:
        return [name for name in CHECKED_FIELDS if name != "pickup_date"]
    return list(CHECKED_FIELDS)


def missing_fields(fields: ShipmentFields) -> list[str]:
    return [
        f"missing:{name}"
        for name in required_fields(fields)
        if getattr(fields, CHECKED_FIELDS[name]) is None
    ]


def validate(fields: ShipmentFields, reference_date: date) -> list[str]:
    """Rule violations on the fields that are present.

    reference_date is the email's received date, not today, so results don't
    depend on when the check runs.
    """
    errors: list[str] = []

    if fields.request_type == RequestType.OTHER:
        errors.append("invalid:request_type_other")

    if fields.pickup_date is not None:
        if fields.pickup_date < reference_date:
            errors.append("invalid:pickup_date_before_received")
        elif fields.pickup_date > reference_date + MAX_PICKUP_HORIZON:
            errors.append("invalid:pickup_date_too_far")

    if fields.weight_value is not None and fields.weight_unit is None:
        errors.append("invalid:weight_unit_missing")
    if fields.weight_kg is not None and fields.weight_kg <= 0:
        errors.append("invalid:weight_not_positive")

    if fields.pallet_count is not None and fields.pallet_count <= 0:
        errors.append("invalid:pallet_count_not_positive")

    if (
        fields.origin
        and fields.destination
        and fields.origin.strip().casefold() == fields.destination.strip().casefold()
    ):
        errors.append("invalid:origin_equals_destination")

    return errors


def decide(output: ExtractionOutput, reference_date: date, threshold: float) -> DecisionResult:
    fields = output.fields
    validation_errors = validate(fields, reference_date)

    # Checked for every present field, including an optional quote pickup date:
    # a wrong value is worse than a missing one.
    low_confidence = [
        f"low_confidence:{name}"
        for name, attr in CHECKED_FIELDS.items()
        if getattr(fields, attr) is not None and getattr(output.confidence, name) < threshold
    ]

    reasons = missing_fields(fields) + validation_errors + low_confidence
    decision = Decision.NEEDS_REVIEW if reasons else Decision.ACCEPTED
    return DecisionResult(decision=decision, reasons=reasons, validation_errors=validation_errors)
