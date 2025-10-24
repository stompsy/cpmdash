# src/apps/core/views.py
from datetime import date

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render


def health(_request: HttpRequest) -> HttpResponse:
    return render(_request, "core/health.html", {"status": "ok"})


def healthz(_request: HttpRequest) -> JsonResponse:
    """Lightweight JSON health check used by tests and monitors."""
    return JsonResponse({"status": "ok"})


def overview(request: HttpRequest) -> HttpResponse:
    """Primary landing page for CPM insights overview."""
    updated_on = date(2025, 10, 1)
    context = {
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
        "page_header_read_time": "6 min read",
    }
    return render(request, "core/overview.html", context)
