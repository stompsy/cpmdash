# src/cpmdash/settings.py
import importlib.util

from .settings_base import BASE_DIR, SRC_DIR, settings

SECRET_KEY = settings.SECRET_KEY
DEBUG = settings.DEBUG
ALLOWED_HOSTS = settings.ALLOWED_HOSTS

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

ROOT_URLCONF = "cpmdash.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project-level template directory
        "DIRS": [BASE_DIR / "src" / "templates"],
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

DATABASES = {"default": settings.database_dict()}

# Static files (CSS, JS, images)
STATIC_URL = "/static/"

# Where collectstatic dumps the final files for production
STATIC_ROOT = BASE_DIR / "staticfiles"

# Additional places Django looks for static files at runtime (dev) and for collectstatic (prod)
STATICFILES_DIRS = [
    SRC_DIR / "static",  # project-wide assets
]

STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Media (user uploads) — optional but you’ll want it eventually
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

SPECTACULAR_SETTINGS = {"TITLE": "cpmdash API", "VERSION": settings.VERSION}

CORS_ALLOW_ALL_ORIGINS = settings.CORS_ALLOW_ALL_ORIGINS

if DEBUG and importlib.util.find_spec("django_browser_reload"):
    if "django_browser_reload" not in INSTALLED_APPS:
        INSTALLED_APPS.append("django_browser_reload")
    if "django_browser_reload.middleware.BrowserReloadMiddleware" not in MIDDLEWARE:
        MIDDLEWARE.insert(0, "django_browser_reload.middleware.BrowserReloadMiddleware")

# Production hardening (Railway, etc.)
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 3600
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
    # Use whitenoise to serve immutable manifest files
    WHITENOISE_MAX_AGE = 31536000
