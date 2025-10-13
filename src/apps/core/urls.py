# src/apps/core/urls.py
from django.urls import path

from .views import health, healthz

# Expose JSON health check at /api/healthz/ via project include("api/", ...)
# Keep a separate HTML health page at /api/health/ for manual browser checks.
urlpatterns = [
    path("health/", health, name="health_html"),  # Human-friendly HTML page
    path("healthz/", healthz, name="healthz"),  # Machine JSON endpoint
]

__all__ = ["urlpatterns"]
