from datetime import date, datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

LBS_TO_KG = 0.45359237


class RequestType(StrEnum):
    QUOTE = "quote"
    BOOKING = "booking"
    OTHER = "other"


class WeightUnit(StrEnum):
    KG = "kg"
    LBS = "lbs"
    T = "t"


class EmailStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class Decision(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"


# --- Input ---------------------------------------------------------------


class EmailIn(BaseModel):
    message_id: str | None = None
    sender: str
    subject: str = ""
    body: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- LLM output ----------------------------------------------------------
# The model reports weight as written (value + unit); we convert to kg in code
# rather than trusting the LLM to do arithmetic.


class ShipmentFields(BaseModel):
    request_type: RequestType | None = None
    origin: str | None = None
    destination: str | None = None
    pickup_date: date | None = None
    weight_value: float | None = None
    weight_unit: WeightUnit | None = None
    pallet_count: int | None = None
    cargo_type: str | None = None

    @computed_field
    @property
    def weight_kg(self) -> float | None:
        if self.weight_value is None or self.weight_unit is None:
            return None
        factor = {WeightUnit.KG: 1.0, WeightUnit.LBS: LBS_TO_KG, WeightUnit.T: 1000.0}
        return round(self.weight_value * factor[self.weight_unit], 2)


class FieldConfidence(BaseModel):
    """Model's self-reported confidence per field, 0..1."""

    request_type: float = 0.0
    origin: float = 0.0
    destination: float = 0.0
    pickup_date: float = 0.0
    weight: float = 0.0
    pallet_count: float = 0.0
    cargo_type: float = 0.0


class ExtractionOutput(BaseModel):
    """Schema the extractor must return (also used as Gemini's response schema)."""

    fields: ShipmentFields
    confidence: FieldConfidence


# --- Review --------------------------------------------------------------


class ReviewAction(StrEnum):
    APPROVE = "approve"
    CORRECT = "correct"


class ReviewIn(BaseModel):
    action: ReviewAction
    reviewer: str
    # Required when action == correct: the full corrected field set.
    corrected: ShipmentFields | None = None


# --- API responses -------------------------------------------------------


class EmailCreated(BaseModel):
    id: int
    status: EmailStatus
    duplicate: bool


class ExtractionOut(BaseModel):
    fields: ShipmentFields
    confidence: FieldConfidence
    decision: Decision
    reasons: list[str]
    model: str
    prompt_version: str
    created_at: datetime
    review_action: ReviewAction | None = None
    corrected: ShipmentFields | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class EmailOut(BaseModel):
    id: int
    message_id: str | None
    sender: str
    subject: str
    received_at: datetime
    status: EmailStatus
    attempts: int
    last_error: str | None
    extraction: ExtractionOut | None
