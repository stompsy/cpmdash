"""Hargrove Grant reporting is funded for, and limited to, Port Angeles records."""

from __future__ import annotations

from typing import Any

from django.core.exceptions import PermissionDenied

from apps.core.models import Agency

HARGROVE_COUNTY_SLUG = "clallam-county"
HARGROVE_AGENCY_SLUG = "port-angeles"


def get_hargrove_agency() -> Agency:
    return Agency.objects.get(county__slug=HARGROVE_COUNTY_SLUG, slug=HARGROVE_AGENCY_SLUG)


def can_access_hargrove(user: Any) -> bool:
    """Staff and superusers can review the report; other users must belong to Port Angeles."""
    if not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_staff", False) or getattr(user, "is_superuser", False):
        return True
    agency = getattr(user, "agency", None)
    county = getattr(agency, "county", None)
    return (
        getattr(agency, "slug", None) == HARGROVE_AGENCY_SLUG
        and getattr(county, "slug", None) == HARGROVE_COUNTY_SLUG
    )


def require_hargrove_agency(user: Any) -> Agency:
    if not can_access_hargrove(user):
        raise PermissionDenied("Hargrove Grant reporting is limited to Port Angeles.")
    return get_hargrove_agency()
