from typing import Protocol

from app.schemas import EmailIn, ExtractionOutput


class ExtractionError(Exception):
    """Extraction failed and retrying won't help (bad API key, rejected request)."""


class RetryableExtractionError(ExtractionError):
    """Transient failure: timeout, rate limit, 5xx, or output that failed schema validation."""


class Extractor(Protocol):
    model: str
    prompt_version: str

    def extract(self, email: EmailIn) -> ExtractionOutput: ...
