from app.schemas import EmailIn, ExtractionOutput


class FakeExtractor:
    """Returns scripted results in order, then `default` once the script runs out.

    Script entries that are exceptions are raised, so "fail twice then succeed" is
    FakeExtractor(RetryableExtractionError("x"), RetryableExtractionError("x"), output).
    """

    model = "fake"
    prompt_version = "fake-v1"

    def __init__(
        self,
        *script: ExtractionOutput | Exception,
        default: ExtractionOutput | Exception | None = None,
    ):
        self._script = list(script)
        self.default = default
        self.calls: list[EmailIn] = []

    def extract(self, email: EmailIn) -> ExtractionOutput:
        self.calls.append(email)
        result = self._script.pop(0) if self._script else self.default
        if result is None:
            raise AssertionError("FakeExtractor called more times than scripted")
        if isinstance(result, Exception):
            raise result
        return result
