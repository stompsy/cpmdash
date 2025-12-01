# src/apps/core/views.py
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .models import ContactSubmission


def health(_request: HttpRequest) -> HttpResponse:
    return render(_request, "core/health.html", {"status": "ok"})


def healthz(_request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok"})


def overview(request: HttpRequest) -> HttpResponse:
    updated_on = date(2025, 11, 30)
    ecosystem_partners = _load_collaborative_ecosystem()

    context = {
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
        "page_header_read_time": "6 min read",
        "ecosystem_partners": ecosystem_partners,
    }
    return render(request, "core/overview.html", context)


def privacy(request: HttpRequest) -> HttpResponse:
    """Privacy policy page."""
    return render(request, "core/privacy.html")


def terms(request: HttpRequest) -> HttpResponse:
    """Terms of service page."""
    return render(request, "core/terms.html")


def accessibility(request: HttpRequest) -> HttpResponse:
    """Accessibility statement page."""
    return render(request, "core/accessibility.html")


@require_POST
def contact_submit(request: HttpRequest) -> HttpResponse:
    """Handle contact form submission."""
    try:
        ContactSubmission.objects.create(
            first_name=request.POST.get("first-name", ""),
            last_name=request.POST.get("last-name", ""),
            organization=request.POST.get("organization", ""),
            email=request.POST.get("email", ""),
            phone_number=request.POST.get("phone-number", ""),
            message=request.POST.get("message", ""),
        )
        messages.success(request, "Thank you for your message. We will be in touch shortly.")
    except Exception:
        messages.error(request, "There was an error sending your message. Please try again.")

    return redirect("home")


@login_required
def contact_submissions_list(request: HttpRequest) -> HttpResponse:
    """List all contact submissions (Admin only)."""
    if not request.user.is_staff:
        return render(request, "access_denied.html")

    submissions = ContactSubmission.objects.all().order_by("-created_at")
    return render(request, "core/contact_submissions.html", {"submissions": submissions})


def _load_collaborative_ecosystem() -> dict[str, Any]:
    data_path = Path(settings.BASE_DIR) / "src" / "static" / "data" / "collaborative_ecosystem.json"
    try:
        with data_path.open() as source:
            payload: dict[str, Any] = json.load(source)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"categories": [], "metadata": {}}

    categories: list[dict[str, Any]] = []
    for raw_category in payload.get("categories", []):
        agencies = [agency for agency in raw_category.get("agencies", []) if agency.get("name")]
        categories.append(
            {
                "slug": raw_category.get("slug")
                or raw_category.get("name", "").strip().lower().replace(" ", "-")
                or "community-partners",
                "name": raw_category.get("name", "Community Partners"),
                "description": raw_category.get("description", ""),
                "color": raw_category.get("color", "#94a3b8"),
                "agencies": agencies,
            }
        )

    metadata = payload.get("metadata", {})
    return {"categories": categories, "metadata": metadata}
