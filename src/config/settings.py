# src/config/settings.py
import importlib.util
import os
from pathlib import Path

import dj_database_url
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

load_dotenv(BASE_DIR / ".env")

DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

SECRET_KEY = os.environ.get("SECRET_KEY", "dummy-key-for-pre-commit-checks")
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")
CSRF_TRUSTED_ORIGINS = os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")
JAWG_ACCESS_TOKEN = os.environ.get("JAWG_ACCESS_TOKEN", "")

INSTALLED_APPS = [
    # Contrib
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    # Project apps
    "apps.core",
    "apps.accounts",
    "apps.dashboard",
    "apps.charts",
    "apps.cases",
    "apps.partials_viewer",
    "apps.blog",
    "apps.tasks",
    # Third-party
    "rest_framework",
    "django_filters",
    "drf_spectacular",
    "corsheaders",
    # Auth
    "django.contrib.sites",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
]

# Dynamically add django-browser-reload only in DEBUG mode
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
    "allauth.account.middleware.AccountMiddleware",
    "apps.core.middleware.GlobalLoginRequiredMiddleware",  # Require login for all pages
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Dynamically add django-browser-reload middleware only in DEBUG mode
if DEBUG and importlib.util.find_spec("django_browser_reload"):
    MIDDLEWARE.insert(0, "django_browser_reload.middleware.BrowserReloadMiddleware")

ROOT_URLCONF = "config.urls"

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
                "apps.tasks.context_processors.tasks_badge",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# Sites framework (required by allauth)
SITE_ID = 1


# --- Database ---
DATABASES = {"default": dj_database_url.config(default="sqlite:///db.sqlite3", conn_max_age=600)}


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
# Use custom user model
AUTH_USER_MODEL = "accounts.User"

# --- Authentication UX ---
LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/accounts/login/"
SESSION_COOKIE_SAMESITE = "Lax"

# django-allauth configuration (email optional, no 2FA)
AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
)
ACCOUNT_LOGIN_METHODS = {"username", "email"}
ACCOUNT_EMAIL_VERIFICATION = "optional"  # can switch to 'mandatory' later
ACCOUNT_SIGNUP_FIELDS = ["username*", "email", "password1*", "password2*"]
ACCOUNT_RATE_LIMITS = {"login_failed": "5/5m"}
# Disable public signup - users must be created by admins
ACCOUNT_ADAPTER = "apps.accounts.adapter.NoSignupAccountAdapter"
ACCOUNT_FORMS = {"signup": "apps.accounts.forms.DisabledSignupForm"}

# Optional: require login for dashboard pages. Enable to gate /dashboard/* behind auth.
LOGIN_REQUIRED = False
# To enable, also add "apps.core.middleware.LoginRequiredForDashboardMiddleware" to MIDDLEWARE.

# --- Email configuration ---
EMAIL_BACKEND = os.environ.get(
    "EMAIL_BACKEND",
    "django.core.mail.backends.console.EmailBackend"
    if DEBUG
    else "django.core.mail.backends.smtp.EmailBackend",
)
DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", "no-reply@cpmdash.local")
