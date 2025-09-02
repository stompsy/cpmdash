# src/cpmdash/settings.py
import importlib.util
from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

env = environ.Env()

environ.Env.read_env(BASE_DIR / ".env")

DEBUG = env.bool("DEBUG", default=False)

SECRET_KEY = env("SECRET_KEY", default="dummy-key-for-pre-commit-checks")
ENVIRONMENT = "development"

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost,127.0.0.1"])
CSRF_TRUSTED_ORIGINS = env.list("CSRF_TRUSTED_ORIGINS", default=[])

if ENVIRONMENT == "production":
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

# --- Application Definition ---
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third-party
    "rest_framework",
    "django_filters",
    "drf_spectacular",
    "corsheaders",
    # local
    "apps.core",
    "apps.dashboard",
    "apps.cases",
]

# Add django-browser-reload in development only
if DEBUG and importlib.util.find_spec("django_browser_reload"):
    INSTALLED_APPS.append("django_browser_reload")

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Add django-browser-reload middleware in development only
if DEBUG and importlib.util.find_spec("django_browser_reload"):
    MIDDLEWARE.insert(0, "django_browser_reload.middleware.BrowserReloadMiddleware")

ROOT_URLCONF = "cpmdash.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [SRC_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "cpmdash.wsgi.application"
ASGI_APPLICATION = "cpmdash.asgi.application"


# --- Database ---
DATABASES = {
    "default": env.db("DATABASE_URL", default="sqlite:///db.sqlite3"),
}


# --- Password validation ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# --- Internationalization ---
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# --- Static files (CSS, JavaScript, Images) ---
STATIC_URL = "static/"
STATICFILES_DIRS = [SRC_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# --- Media files (User Uploads) ---
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# --- Default primary key field type ---
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
