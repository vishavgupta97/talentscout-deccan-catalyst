from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the Deccan Catalyst app.

    The project is intentionally small, so settings live in one place and are
    read once at startup. This keeps local setup simple while still giving us
    clean hooks for deployment-time overrides.
    """

    app_name: str = "TalentScout"
    app_env: str = "development"
    database_url: str = f"sqlite:///{BASE_DIR / 'talent_scout.db'}"
    gemini_api_key: str | None = None
    default_model: str = "gemini-2.5-flash"
    dataset_path: str = str(BASE_DIR / "app" / "data" / "candidates.json")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
