# src/apps/core/views.py
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render


def health(_request: HttpRequest) -> HttpResponse:
    return render(_request, "core/health.html", {"status": "ok"})


def healthz(_request: HttpRequest) -> JsonResponse:
    """Lightweight JSON health check used by tests and monitors."""
    return JsonResponse({"status": "ok"})


# ----- Public pages: About & Mission -----
def contact(request: HttpRequest) -> HttpResponse:
    """Basic contact page for the Fire Department / CPM office."""
    return render(
        request,
        "core/contact.html",
        {
            "page_title": "Contact Us",
            "about_stats": settings.ABOUT_STATS,
        },
    )


def mission(request: HttpRequest) -> HttpResponse:
    """Mission statement page for the City of Port Angeles Fire Department."""
    return render(
        request,
        "core/mission.html",
        {
            "page_title": "Mission Statement",
            "about_stats": settings.ABOUT_STATS,
        },
    )


def program_goals(request: HttpRequest) -> HttpResponse:
    """Program goals page for the Community Paramedic Program."""
    return render(
        request,
        "core/program_goals.html",
        {
            "page_title": "Program Goals",
            "about_stats": settings.ABOUT_STATS,
        },
    )
