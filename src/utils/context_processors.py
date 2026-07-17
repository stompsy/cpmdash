from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.http import HttpRequest
from django.urls import reverse

from apps.core.models import Agency

_CHANGELOG = Path(__file__).resolve().parent.parent.parent / "CHANGELOG.md"


@lru_cache(maxsize=1)
def _parse_app_version() -> str:
    """Extract the most recent version tag from CHANGELOG.md."""
    try:
        text = _CHANGELOG.read_text(encoding="utf-8")
        match = re.search(r"##\s+(v[\d]+\.[\d]+\.[\d]+[^\s)]*)\s*\(", text)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "v?"


def app_context(request: HttpRequest) -> dict[str, Any]:  # noqa: C901
    """Inject app_version and current_year into all template contexts."""
    tenant_scope: dict[str, Any] = {
        "tenant_scope_enabled": False,
        "tenant_scope_mode": "agency",
        "tenant_scope_selected_county_id": None,
        "tenant_scope_selected_agency_id": None,
        "tenant_scope_selected_agency_ids": [],
        "tenant_scope_county_name": "",
        "tenant_scope_agencies": [],
        "tenant_scope_query": "",
        "tenant_scope_suffix": "",
    }

    user = getattr(request, "user", None)
    if getattr(user, "is_authenticated", False):
        user_agency = getattr(user, "agency", None)
        if user_agency is not None:
            county = getattr(user_agency, "county", None)
            county_id_raw = getattr(user_agency, "county_id", None)
            agency_id_raw = getattr(user, "agency_id", None)
            county_id = int(county_id_raw) if isinstance(county_id_raw, int) else None
            agency_id = int(agency_id_raw) if isinstance(agency_id_raw, int) else None
            selected_scope = (request.GET.get("scope") or "").strip().lower()
            if selected_scope not in {"agency", "county", "multi"}:
                selected_scope = "agency"

            requested_multi_ids: list[int] = []
            raw_multi_values = request.GET.getlist("agency_ids")
            if not raw_multi_values:
                raw_single = request.GET.get("agency_ids", "")
                if raw_single:
                    raw_multi_values = [raw_single]
            for raw in raw_multi_values:
                for token in str(raw).split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        requested_multi_ids.append(int(token))
                    except (TypeError, ValueError):
                        continue
            requested_multi_ids = list(dict.fromkeys(requested_multi_ids))

            agencies: list[dict[str, Any]] = []
            if county_id is not None:
                agencies = [
                    {"id": int(row["id"]), "name": str(row["name"])}
                    for row in Agency.objects.filter(county_id=county_id)
                    .order_by("name")
                    .values("id", "name")
                ]
            county_entries: list[dict[str, Any]] = []
            if county_id is not None:
                county_entries = [
                    {
                        "id": county_id,
                        "name": getattr(county, "name", ""),
                        "suffix": f"?scope=county&county_id={county_id}",
                        "agencies": [
                            {
                                "id": int(agency_row["id"]),
                                "name": str(agency_row["name"]),
                                "suffix": f"?scope=agency&agency_id={int(agency_row['id'])}",
                            }
                            for agency_row in agencies
                        ],
                    }
                ]
            allowed_agency_ids = {int(a["id"]) for a in agencies}

            selected_agency_id: int | None
            selected_agency_ids: list[int] = []
            if selected_scope == "agency":
                requested_agency_id: int | None
                try:
                    requested_agency_id = int(request.GET.get("agency_id", ""))
                except (TypeError, ValueError):
                    requested_agency_id = agency_id
                if requested_agency_id in allowed_agency_ids:
                    selected_agency_id = requested_agency_id
                else:
                    selected_agency_id = agency_id
                if selected_agency_id is not None:
                    selected_agency_ids = [selected_agency_id]
            elif selected_scope == "multi":
                selected_agency_id = None
                selected_agency_ids = [
                    aid for aid in requested_multi_ids if aid in allowed_agency_ids
                ]
                if not selected_agency_ids and agency_id is not None:
                    selected_agency_ids = [agency_id]
                    selected_scope = "agency"
                    selected_agency_id = agency_id
            else:
                selected_agency_id = None
                selected_agency_ids = [int(a["id"]) for a in agencies]

            selected_county_id = county_id
            if selected_scope == "agency" and selected_agency_id is not None:
                query = f"scope=agency&agency_id={selected_agency_id}"
            elif selected_scope == "multi" and selected_agency_ids:
                query = (
                    f"scope=multi&agency_ids={','.join(str(aid) for aid in selected_agency_ids)}"
                )
            elif selected_county_id is not None:
                query = f"scope=county&county_id={selected_county_id}"
            else:
                query = ""

            tenant_scope = {
                "tenant_scope_enabled": True,
                "tenant_scope_mode": selected_scope,
                "tenant_scope_selected_county_id": selected_county_id,
                "tenant_scope_selected_agency_id": selected_agency_id,
                "tenant_scope_selected_agency_ids": selected_agency_ids,
                "tenant_scope_county_name": getattr(county, "name", ""),
                "tenant_scope_agencies": agencies,
                "tenant_scope_counties": county_entries,
                "tenant_scope_program_sections": [
                    {
                        "label": "Patients",
                        "url": reverse("dashboard:patients"),
                        "page": "patientsOverview",
                    },
                    {
                        "label": "Referrals",
                        "url": reverse("dashboard:referrals"),
                        "page": "referralsOverview",
                    },
                    {
                        "label": "OD Referrals",
                        "url": reverse("dashboard:odreferrals"),
                        "page": "od-referrals-overview",
                    },
                ],
                "tenant_scope_query": query,
                "tenant_scope_suffix": f"?{query}" if query else "",
            }

    return {
        "app_version": _parse_app_version(),
        "current_year": datetime.now().year,
        **tenant_scope,
    }
