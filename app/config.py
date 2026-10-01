from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Optional so tests and the fake extractor run without a key.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"

    db_path: str = "shipment_intake.db"

    # Emails without a Message-ID count as duplicates only if the same content arrived
    # within this window. Catches client retries without dropping repeat bookings.
    dedup_window_hours: float = 24.0

    # Decision: auto-accept only if every required field's confidence is >= this.
    confidence_threshold: float = 0.8

    # Worker / retries
    max_attempts: int = 5
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 300.0
    lock_timeout_seconds: float = 300.0
    poll_interval_seconds: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
