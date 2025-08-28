# src/cpmdash/settings_base.py
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from environ import Env

env = Env()
Env.read_env()


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

# Access environment variables
RAILWAY_ENVIRONMENT_NAME = env("RAILWAY_ENVIRONMENT_NAME", default="production")
DJANGO_SECRET_KEY = env("DJANGO_SECRET_KEY")

DATABASE_URL = env("DATABASE_URL")

DJANGO_DEBUG = RAILWAY_ENVIRONMENT_NAME == "development"


def get_database_config() -> dict[str, Any]:
    """Parse DATABASE_URL and return Django database configuration."""
    url = urlparse(str(DATABASE_URL))
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
