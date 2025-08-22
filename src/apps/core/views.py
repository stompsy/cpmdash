# src/apps/core/views.py
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render


def health(_request: HttpRequest) -> HttpResponse:
    return render(_request, "core/health.html", {"status": "ok"})


def healthz(_request: HttpRequest) -> JsonResponse:
    """Lightweight JSON health check used by tests and monitors."""
    return JsonResponse({"status": "ok"})
