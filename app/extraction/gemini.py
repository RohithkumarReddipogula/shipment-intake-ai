import httpx
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from app.config import Settings
from app.extraction.base import ExtractionError, RetryableExtractionError
from app.extraction.prompt import PROMPT_VERSION, SYSTEM_INSTRUCTION, build_user_prompt
from app.schemas import EmailIn, ExtractionOutput

RETRYABLE_STATUS = {429, 500, 503}


def classify_api_error(e: errors.APIError) -> ExtractionError:
    """429/500/503 and other 5xx are retryable; 400/401/403 and other 4xx are not."""
    message = f"Gemini {e.code} {e.status}: {e.message}"
    if e.code in RETRYABLE_STATUS or e.code >= 500:
        return RetryableExtractionError(message)
    return ExtractionError(message)


class GeminiExtractor:
    prompt_version = PROMPT_VERSION

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 30.0, client=None):
        self.model = model
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                # Retries belong to the worker (backoff + attempt history), not the SDK.
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0,
            response_mime_type="application/json",
            response_schema=ExtractionOutput,
            # No tools are used; the SDK enables this by default and logs about it.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiExtractor":
        if settings.gemini_api_key is None:
            raise ExtractionError("GEMINI_API_KEY is not set (add it to .env)")
        return cls(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
        )

    def extract(self, email: EmailIn) -> ExtractionOutput:
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=build_user_prompt(email), config=self._config
            )
        except errors.APIError as e:
            raise classify_api_error(e) from e
        except httpx.TimeoutException as e:
            raise RetryableExtractionError(f"Gemini timeout: {type(e).__name__}") from e
        except httpx.TransportError as e:
            raise RetryableExtractionError(f"Gemini connection error: {e}") from e

        text = response.text
        if not text:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            raise RetryableExtractionError(f"Gemini returned no content (finish_reason={reason})")
        try:
            return ExtractionOutput.model_validate_json(text)
        except ValidationError as e:
            raise RetryableExtractionError(
                f"Gemini output failed schema validation: {e.error_count()} errors"
            ) from e
