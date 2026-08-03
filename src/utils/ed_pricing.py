from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from django.conf import settings

FALLBACK_ED_ENCOUNTER_COST = 1146

ED_VISIT_CODES: dict[str, str] = {
    "99281": "ED visit level 1",
    "99282": "ED visit level 2",
    "99283": "ED visit level 3",
    "99284": "ED visit level 4",
    "99285": "ED visit level 5",
}

COMMON_ED_PROCEDURE_CODES: dict[str, str] = {
    "71045": "Chest X-ray single view",
    "70450": "CT head/brain without contrast",
    "93005": "ECG tracing",
    "80053": "Comprehensive metabolic panel",
    "85025": "CBC with differential",
    "36415": "Venous blood draw",
}

EMERGENCY_KEYWORD_PATTERN = re.compile(
    r"emergency|\bed\b|er\s+visit|trauma|urgent|acute|observation",
    re.IGNORECASE,
)

CODE_DESCRIPTION_HINTS: dict[str, tuple[str, ...]] = {
    "99281": ("emergency", "ed", "visit"),
    "99282": ("emergency", "ed", "visit"),
    "99283": ("emergency", "ed", "visit"),
    "99284": ("emergency", "ed", "visit"),
    "99285": ("emergency", "ed", "visit"),
    "71045": ("chest", "x-ray", "xray", "radiograph"),
    "70450": ("ct", "head", "brain"),
    "93005": ("ecg", "electrocard"),
    "80053": ("metabolic", "panel"),
    "85025": ("cbc", "blood", "differential"),
    "36415": ("blood", "venous", "venipuncture", "draw"),
}


def _hospital_file_paths() -> list[Path]:
    docs_dir = Path(settings.BASE_DIR) / "docs"
    return [
        docs_dir / "911631806_harborview-medical-center_standardcharges.json",
        docs_dir / "916001537_university-of-washington-medical-center_standardcharges.json",
        docs_dir / "916001709_olympic-medical-center_standardcharges.json",
    ]


def _to_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalize_code_information(code_info: Any) -> list[dict[str, Any]]:
    if isinstance(code_info, dict):
        return [code_info]
    if isinstance(code_info, list):
        return [entry for entry in code_info if isinstance(entry, dict)]
    return []


def _pick_procedure_code(code_info: list[dict[str, Any]]) -> tuple[str, str]:
    priority_types = ("HCPCS", "CPT", "RC", "CDM")
    for code_type in priority_types:
        for entry in code_info:
            if str(entry.get("type", "")).strip().upper() != code_type:
                continue
            code = str(entry.get("code", "")).strip()
            if code:
                return code, code_type
    return "", ""


def _is_ed_related(description: str, code: str) -> bool:
    if code in ED_VISIT_CODES:
        return True
    if code in COMMON_ED_PROCEDURE_CODES:
        return True
    return bool(EMERGENCY_KEYWORD_PATTERN.search(description))


def _load_hospital_payload(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as source:
        payload = json.load(source)

    hospital_name = str(payload.get("hospital_name", path.stem)).strip() or path.stem
    updated_on = str(payload.get("last_updated_on", "")).strip()
    charge_rows = payload.get("standard_charge_information", [])
    if not isinstance(charge_rows, list):
        charge_rows = []

    cleaned_rows = [row for row in charge_rows if isinstance(row, dict)]
    return hospital_name, updated_on, cleaned_rows


def _apply_charge_to_aggregate(aggregate: dict[str, Any], charge: dict[str, Any]) -> None:
    gross_value = _to_float(charge.get("gross_charge"))
    if gross_value is None:
        return

    discounted_cash = _to_float(charge.get("discounted_cash"))
    setting = str(charge.get("setting", "")).strip()
    billing_class = str(charge.get("billing_class", "")).strip()

    if aggregate["gross_charge"] is None or gross_value < aggregate["gross_charge"]:
        aggregate["gross_charge"] = gross_value

    if discounted_cash is not None and (
        aggregate["discounted_cash"] is None or discounted_cash < aggregate["discounted_cash"]
    ):
        aggregate["discounted_cash"] = discounted_cash

    if setting:
        aggregate["settings"].add(setting)
    if billing_class:
        aggregate["billing_classes"].add(billing_class)


def _merge_aggregate_row(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    hospital_name: str,
    row: dict[str, Any],
) -> None:
    description = str(row.get("description", "")).strip()
    if not description:
        return

    code_info = _normalize_code_information(row.get("code_information"))
    procedure_code, code_type = _pick_procedure_code(code_info)
    if not procedure_code:
        return

    standard_charges = row.get("standard_charges", [])
    if not isinstance(standard_charges, list) or not standard_charges:
        return

    key = (hospital_name, procedure_code, description)
    aggregate = aggregate_map.setdefault(
        key,
        {
            "facility": hospital_name,
            "procedure_code": procedure_code,
            "procedure_description": description,
            "code_type": code_type,
            "gross_charge": None,
            "discounted_cash": None,
            "settings": set(),
            "billing_classes": set(),
            "is_ed_related": _is_ed_related(description, procedure_code),
        },
    )

    for charge in standard_charges:
        if isinstance(charge, dict):
            _apply_charge_to_aggregate(aggregate, charge)


def _collect_aggregates() -> tuple[
    list[str], list[str], dict[tuple[str, str, str], dict[str, Any]]
]:
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    hospital_names: list[str] = []
    updated_dates: list[str] = []

    for json_path in _hospital_file_paths():
        if not json_path.exists():
            continue

        try:
            hospital_name, updated_on, charge_rows = _load_hospital_payload(json_path)
        except (json.JSONDecodeError, OSError):
            continue

        hospital_names.append(hospital_name)
        if updated_on:
            updated_dates.append(updated_on)

        for row in charge_rows:
            _merge_aggregate_row(aggregate_map, hospital_name, row)

    return hospital_names, updated_dates, aggregate_map


def _materialize_rows(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregate in aggregate_map.values():
        if aggregate["gross_charge"] is None:
            continue
        rows.append(
            {
                "facility": aggregate["facility"],
                "procedure_code": aggregate["procedure_code"],
                "procedure_description": aggregate["procedure_description"],
                "code_type": aggregate["code_type"],
                "gross_charge": aggregate["gross_charge"],
                "discounted_cash": aggregate["discounted_cash"],
                "settings": ", ".join(sorted(aggregate["settings"])),
                "billing_classes": ", ".join(sorted(aggregate["billing_classes"])),
                "is_ed_related": aggregate["is_ed_related"],
            }
        )

    rows.sort(
        key=lambda row: (
            row["facility"],
            row["procedure_code"],
            row["procedure_description"],
        )
    )
    return rows


def _description_matches_code(code: str, description: str) -> bool:
    hints = CODE_DESCRIPTION_HINTS.get(code)
    if not hints:
        return True
    lowered = description.lower()
    return any(hint in lowered for hint in hints)


def _candidate_rank(row: dict[str, Any]) -> tuple[int, int, float]:
    description = str(row["procedure_description"]).strip().lower()
    billing_classes = str(row.get("billing_classes", "")).strip().lower()
    gross_charge = float(row.get("gross_charge") or 0.0)

    if description.startswith("hc "):
        source_rank = 0
    elif "facility" in billing_classes:
        source_rank = 1
    elif description.startswith("pr "):
        source_rank = 3
    else:
        source_rank = 2

    billing_rank = 0 if "facility" in billing_classes else 1
    return (source_rank, billing_rank, -gross_charge)


def _select_preferred_row(code: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [
        row
        for row in candidates
        if _description_matches_code(code, str(row.get("procedure_description", "")))
    ]
    pool = matched or candidates
    return sorted(pool, key=_candidate_rank)[0]


def _build_code_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["procedure_code"], row["facility"])
        grouped.setdefault(key, []).append(row)

    code_lookup: dict[str, dict[str, float]] = {}
    for (code, facility), candidates in grouped.items():
        preferred = _select_preferred_row(code, candidates)
        code_lookup.setdefault(code, {})[facility] = float(preferred["gross_charge"])

    return code_lookup


def _build_ed_level_costs(
    code_lookup: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[float]]:
    ed_level_costs: list[dict[str, Any]] = []
    per_code_averages: list[float] = []

    for code, label in ED_VISIT_CODES.items():
        per_facility = code_lookup.get(code, {})
        if not per_facility:
            continue
        average_cost = mean(per_facility.values())
        per_code_averages.append(average_cost)
        ed_level_costs.append(
            {
                "procedure_code": code,
                "label": label,
                "facility_costs": per_facility,
                "average_gross_charge": average_cost,
            }
        )

    return ed_level_costs, per_code_averages


def _build_common_procedure_costs(code_lookup: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    common_procedure_costs: list[dict[str, Any]] = []
    common_codes = {**ED_VISIT_CODES, **COMMON_ED_PROCEDURE_CODES}

    def _facility_cost(per_facility: dict[str, float], marker: str) -> float | None:
        marker_normalized = marker.lower()
        for facility_name, cost in per_facility.items():
            if marker_normalized in facility_name.lower():
                return cost
        return None

    for code, label in common_codes.items():
        per_facility = code_lookup.get(code, {})
        if not per_facility:
            continue
        common_procedure_costs.append(
            {
                "procedure_code": code,
                "label": label,
                "facility_costs": per_facility,
                "harborview_cost": _facility_cost(per_facility, "harborview medical center"),
                "uwmc_cost": _facility_cost(
                    per_facility, "university of washington medical center"
                ),
                "olympic_cost": _facility_cost(per_facility, "olympic medical center"),
                "average_gross_charge": mean(per_facility.values()),
            }
        )

    common_procedure_costs.sort(key=lambda row: row["procedure_code"])
    return common_procedure_costs


@lru_cache(maxsize=1)
def get_ed_pricing_reference_data() -> dict[str, Any]:
    hospital_names, updated_dates, aggregate_map = _collect_aggregates()
    rows = _materialize_rows(aggregate_map)
    ed_rows = [row for row in rows if row["is_ed_related"]]
    code_lookup = _build_code_lookup(rows)
    ed_level_costs, per_code_averages = _build_ed_level_costs(code_lookup)
    common_procedure_costs = _build_common_procedure_costs(code_lookup)

    blended_ed_average = (
        mean(per_code_averages) if per_code_averages else FALLBACK_ED_ENCOUNTER_COST
    )

    return {
        "facilities": sorted(set(hospital_names)),
        "last_updated_on": max(updated_dates) if updated_dates else "",
        "all_rows": rows,
        "ed_rows": ed_rows,
        "ed_level_costs": ed_level_costs,
        "common_procedure_costs": common_procedure_costs,
        "ed_encounter_average": blended_ed_average,
        "ed_encounter_average_rounded": int(round(blended_ed_average)),
    }


def get_ed_encounter_unit_cost() -> int:
    data = get_ed_pricing_reference_data()
    unit_cost = data.get("ed_encounter_average_rounded", FALLBACK_ED_ENCOUNTER_COST)
    if not isinstance(unit_cost, int):
        return FALLBACK_ED_ENCOUNTER_COST
    return unit_cost
