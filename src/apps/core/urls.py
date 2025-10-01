# src/apps/core/urls.py
from django.urls import path

from .views import contact, health, healthz, mission, program_goals

# Expose JSON health check at /api/healthz/ via project include("api/", ...)
# Keep a separate HTML health page at /api/health/ for manual browser checks.
urlpatterns = [
    path("health/", health, name="health_html"),  # Human-friendly HTML page
    path("healthz/", healthz, name="healthz"),  # Machine JSON endpoint
]

# Public-facing About section (consumed at project level with prefix "/about/")
about_urlpatterns = [
    path("", contact, name="index"),
    path("contact/", contact, name="contact"),
    path("mission/", mission, name="mission"),
    path("goals/", program_goals, name="goals"),
]
