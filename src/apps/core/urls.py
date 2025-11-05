# src/apps/core/urls.py
from django.urls import path

from .views import accessibility, health, healthz, privacy, terms

app_name = "core"

# Expose JSON health check at /api/healthz/ via project include("api/", ...)
# Keep a separate HTML health page at /api/health/ for manual browser checks.
urlpatterns = [
    path("health/", health, name="health_html"),  # Human-friendly HTML page
    path("healthz/", healthz, name="healthz"),  # Machine JSON endpoint
    path("privacy/", privacy, name="privacy"),  # Privacy policy
    path("terms/", terms, name="terms"),  # Terms of service
    path("accessibility/", accessibility, name="accessibility"),  # Accessibility statement
]

__all__ = ["urlpatterns"]
