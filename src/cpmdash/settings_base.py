# src/cpmdash/settings_base.py
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict  # ← use SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
ENV_FILE_PATH = Path(__file__).resolve().parent / ".env"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
    )

    ENVIRONMENT: Literal["development", "production"] = "development"
    SECRET_KEY: str = "SECRET_KEY"
    ALLOWED_HOSTS: list[str] = ["*"]
    DATABASE_URL: str = "sqlite:///db.sqlite3"
    STATIC_ROOT: str = "staticfiles"
    VERSION: str = "0.1.0"
    CORS_ALLOW_ALL_ORIGINS: bool = True

    @computed_field
    def DEBUG(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    def database_dict(self) -> dict[str, Any]:
        url = urlparse(self.DATABASE_URL)
        if url.scheme.startswith("postgres"):
            return {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": url.path.lstrip("/"),
                "USER": url.username,
                "PASSWORD": url.password,
                "HOST": url.hostname,
                "PORT": str(url.port or ""),
            }
        if url.scheme.startswith("sqlite"):
            # Normalize to a project-local absolute path
            if url.netloc == ":memory:" or url.path.strip("/") == ":memory:":
                name = ":memory:"
            else:
                relative = url.path.lstrip("/") or "db.sqlite3"
                name = str(BASE_DIR / relative)
            return {"ENGINE": "django.db.backends.sqlite3", "NAME": name}
        raise ValueError(f"Unsupported DATABASE_URL scheme: {url.scheme}")


settings = AppSettings()
