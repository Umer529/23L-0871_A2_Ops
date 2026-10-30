"""Task 4.4 - every setting comes from an environment variable (12-factor style),
so the same image runs locally, in Compose and in the cloud with no code change."""
import os
from dataclasses import dataclass


def _float_or_none(v):
    return float(v) if v not in (None, "") else None


@dataclass(frozen=True)
class Config:
    model_path: str
    threshold: float | None      # None -> use the threshold saved next to the model
    database_url: str
    inference_mode: str          # "local" = model in this process, "remote" = call the cloud model API
    model_api_url: str
    api_key: str
    max_upload_mb: float
    port: int

    @classmethod
    def from_env(cls):
        return cls(
            model_path=os.getenv("MODEL_PATH", "models/champion.keras"),
            threshold=_float_or_none(os.getenv("THRESHOLD")),
            database_url=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
            inference_mode=os.getenv("INFERENCE_MODE", "local").lower(),
            model_api_url=os.getenv("MODEL_API_URL", ""),
            api_key=os.getenv("API_KEY", ""),
            max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "5")),
            port=int(os.getenv("PORT", "5000")),
        )
