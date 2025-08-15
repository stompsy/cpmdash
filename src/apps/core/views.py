# src/apps/core/views.py
from django.http import HttpRequest, JsonResponse


def health(_request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok"})
