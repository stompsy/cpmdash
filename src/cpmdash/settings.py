# src/cpmdash/settings.py
import importlib.util
from pathlib import Path

import environ

# --- Core Paths ---
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

# --- django-environ Setup ---
env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False)
)

# For local development, read .env file. In production, Railway's variables are used.
environ.Env.read_env(BASE_DIR / "src" / "cpmdash" / ".env")

# --- Core Django Settings ---
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env("SECRET_KEY", default="dummy-secret-key-for-type-checking")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env("DEBUG")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])

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

# Add django-browser-reload middleware in development
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
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases
# Uses dj-database-url format: https://github.com/un-esports/dj-database-url
DATABASES = {
    "default": env.db(),
}

# --- Password validation ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- Internationalization ---
# https://docs.djangoproject.com/en/5.0/topics/i18n/
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# --- Static files (CSS, JavaScript, Images) ---
# https://docs.djangoproject.com/en/5.0/howto/static-files/
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [SRC_DIR / "static"]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# --- Media files (User Uploads) ---
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# --- Default primary key field type ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Third-Party App Settings ---
REST_FRAMEWORK = {"DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema"}
SPECTACULAR_SETTINGS = {"TITLE": "cpmdash API", "VERSION": "0.1.0"}
CORS_ALLOW_ALL_ORIGINS = env.bool("CORS_ALLOW_ALL_ORIGINS", default=True)

# --- Production Security Settings ---
# These settings are only active when DEBUG is False.
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    # Trust the CSRF token from your frontend domain
    CSRF_TRUSTED_ORIGINS = env.list("CSRF_TRUSTED_ORIGINS", default=[])
