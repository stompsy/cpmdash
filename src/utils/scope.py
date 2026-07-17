"""Helpers for interpreting tenant scope filters in chart builders."""

from __future__ import annotations

from typing import Any


def uses_curated_quarter_history(scope_filters: dict[str, Any] | None) -> bool:
    """Return True when curated (Port Angeles / county-total) quarter history applies.

    Curated history covers the hand-tuned historical benchmark years (annotations,
    colored year backgrounds, baseline average line, "did not track" placeholders).
    It only applies to:
      * the unscoped dashboard (county/all total view),
      * an explicit county scope,
      * the Port Angeles agency scope.

    Any other agency-specific scope (e.g. Sequim) is "real data only" and should
    skip the curated decorations entirely.
    """
    if not scope_filters:
        # Unscoped dashboard = county/all total view.
        return True

    if "agency__county_id" in scope_filters:
        # Explicit county scope should keep historical benchmark values.
        return True

    agency_id = scope_filters.get("agency_id")
    agency_ids = scope_filters.get("agency_id__in")
    if isinstance(agency_ids, list) and len(agency_ids) == 1:
        agency_id = agency_ids[0]

    if agency_id is None:
        return False

    try:
        from apps.core.models import Agency

        agency = Agency.objects.filter(id=int(agency_id)).values("slug").first()
    except Exception:
        return False

    return bool(agency and str(agency.get("slug", "")).strip().lower() == "port-angeles")
