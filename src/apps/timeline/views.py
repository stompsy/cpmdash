from datetime import date

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .timeline_store import timeline_store


def timeline_view(request: HttpRequest) -> HttpResponse:
    """Render the Community Paramedicine program timeline."""
    entries = timeline_store.get_all()

    updated_on = date(2025, 11, 30)

    context = {
        "title": "PAFD Community Paramedicine Program Timeline",
        "description": "Milestones that have shaped the Port Angeles Fire Department's community paramedicine evolution.",
        "timeline_entries": entries,
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
        "page_header_read_time": "5 min read",
        "is_admin": request.user.is_staff,
    }

    return render(request, "timeline/index.html", context)


@staff_member_required
@require_http_methods(["POST"])
def timeline_create(request: HttpRequest) -> JsonResponse:
    """Create a new timeline entry (Admin only)."""
    import json

    try:
        data = json.loads(request.body)
        entry = timeline_store.create(data)
        return JsonResponse({"success": True, "entry": entry})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)


@staff_member_required
@require_http_methods(["GET"])
def timeline_get(request: HttpRequest, entry_id: int) -> JsonResponse:
    """Get a single timeline entry (Admin only)."""
    entry = timeline_store.get_by_id(entry_id)
    if entry:
        return JsonResponse({"success": True, "entry": entry})
    return JsonResponse({"success": False, "error": "Entry not found"}, status=404)


@staff_member_required
@require_http_methods(["PUT"])
def timeline_update(request: HttpRequest, entry_id: int) -> JsonResponse:
    """Update a timeline entry (Admin only)."""
    import json

    try:
        data = json.loads(request.body)
        entry = timeline_store.update(entry_id, data)
        if entry:
            return JsonResponse({"success": True, "entry": entry})
        return JsonResponse({"success": False, "error": "Entry not found"}, status=404)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)


@staff_member_required
@require_http_methods(["DELETE"])
def timeline_delete(request: HttpRequest, entry_id: int) -> JsonResponse:
    """Delete a timeline entry (Admin only)."""
    success = timeline_store.delete(entry_id)
    if success:
        return JsonResponse({"success": True})
    return JsonResponse({"success": False, "error": "Entry not found"}, status=404)


@staff_member_required
@require_http_methods(["GET"])
def timeline_list_entries(request: HttpRequest) -> HttpResponse:
    """Return all timeline entries as HTML for HTMX swap."""
    entries = timeline_store.get_all()
    context = {"timeline_entries": entries, "is_admin": True}
    return render(request, "timeline/_timeline_entries.html", context)
