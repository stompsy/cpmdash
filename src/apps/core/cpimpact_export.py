"""De-identified export of cpmdash data for import into CP Impact.

Everything that leaves cpmdash through this module is scrubbed on the way out,
whatever is stored in the database:

* No addresses, coordinates, ZIP codes, districts, or jail dates.
* No dates at all. Each row gets a reporting year and quarter instead, and
  encounters get a per-patient sequence number and a coarse gap bucket so the
  importer can still tell first contacts from repeats.
* Ages become CP Impact's age bands (exact ages over 89 are identifiers).
* Patients are keyed by their cpmdash row id ("P123"). The crosswalk back to a
  person stays in cpmdash, the same way agencies keep their own patient list.
* Text values seen fewer than ``min_count`` times in a column are replaced with
  "(rare)", so a stray name typed into a category field can't slip out.
* Sex, race, marital status and ZIP are left out: CP Impact doesn't collect
  them, and in a small town they make people easier to pick out.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import IO, Any

from django.db.models import QuerySet

from apps.core.models import Encounters, ODReferrals, Patients, Referrals

RARE = "(rare)"

DROPPED = {
    "patients": [
        "address",
        "latitude",
        "longitude",
        "zip_code",
        "sex",
        "race",
        "marital_status",
        "created_date (exported as first_year, first_quarter)",
        "modified_date (exported as last_year, last_quarter)",
        "age (exported as a band)",
    ],
    "encounters": ["encounter_date (exported as year, quarter, seq, gap)"],
    "referrals": ["sex", "zipcode", "date_received (exported as year, quarter)", "age (band)"],
    "od_referrals": [
        "patient_sex",
        "patient_race",
        "patient_zipcode",
        "od_address",
        "lat",
        "long",
        "od_district",
        "jail_start_1",
        "jail_end_1",
        "jail_start_2",
        "jail_end_2",
        "od_date (exported as year, quarter)",
        "patient_age (band)",
    ],
}


# --- small transforms ---------------------------------------------------------


def age_band(age: int | None) -> str:
    """Match CP Impact's AgeBand values."""
    if age is None or age < 0:
        return "unknown"
    if age < 18:
        return "lt18"
    if age < 35:
        return "18_34"
    if age < 50:
        return "35_49"
    if age < 65:
        return "50_64"
    if age < 80:
        return "65_79"
    return "80p"


def year_quarter(value: date | datetime | None) -> tuple[str, str]:
    if value is None:
        return "", ""
    return str(value.year), str((value.month - 1) // 3 + 1)


def gap_bucket(prev: date | None, cur: date | None) -> str:
    if prev is None:
        return "first"
    if cur is None:
        return "unknown"
    days = (cur - prev).days
    if days <= 0:
        return "same_day"
    if days <= 7:
        return "1_7d"
    if days <= 30:
        return "8_30d"
    if days <= 90:
        return "31_90d"
    return "91d_plus"


def key(prefix: str, value: Any) -> str:
    return f"{prefix}{value}" if value not in (None, "") else ""


def as_date(value: date | datetime | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    return value


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def bool_cell(value: bool | None) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


# --- table definitions --------------------------------------------------------


@dataclass
class Table:
    name: str
    columns: list[str]
    text_columns: list[str]  # get the rare-value scrub and a codebook entry
    rows: list[dict[str, str]] = field(default_factory=list)


ENC_TEXT = ["encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"]
DIV_TEXT = ["diversion_type_cat1", "diversion_type_cat2"]
SHARED_BOOLS = ["med_manage", "med_script", "pcp_connect", "survey_willing"]

PATIENT_BOOLS = [
    "behavioral_health",
    "sud",
    "aud",
    "bh_sus",
    "sud_sus",
    "aud_sus",
    "is_high_utilizer",
    "case_management",
    "safety_plan",
    "three_c_client",
    "flyer",
]

OD_TEXT = [
    "patient_insurance",
    "living_situation",
    "delay_in_referral",
    "cpm_notification",
    "cpm_disposition",
    "referral_agency",
    "referral_source",
    "engagement_location",
    "suspected_drug",
    "cpr_administered",
    "police_ita",
    "disposition",
    "transport_to_location",
    "transported_by",
    "contact_level_rediscovery",
    "contact_level_reflections",
    "contact_level_pbh",
    "contact_level_other",
    "bup_not_indicated_reason",
    "bup_already_prescribed",
    "overdose_recent",
    *DIV_TEXT,
    *ENC_TEXT,
]
OD_NUMBERS = [
    "number_of_nonems_onscene",
    "number_of_ems_onscene",
    "number_of_peers_onscene",
    "number_of_police_onscene",
    "narcan_doses_prior_to_ems",
    "narcan_prior_to_ems_dosage",
    "narcan_doses_by_ems",
    "narcan_by_ems_dosage",
    "leave_behind_narcan_amount",
    "persons_trained",
    "referral_rediscovery",
    "referral_reflections",
    "referral_pbh",
    "referral_other",
    "accepted_rediscovery",
    "accepted_reflections",
    "accepted_pbh",
    "accepted_other",
    "is_bup_indicated",
    "bup_admin",
    "client_agrees_to_mat",
]
OD_BOOLS = ["narcan_given", "referral_to_sud_agency", *SHARED_BOOLS]

# Output header renames (house style: "law enforcement", not "police").
RENAME = {
    "number_of_police_onscene": "number_of_law_enforcement_onscene",
    "police_ita": "law_enforcement_ita",
}


def _out(name: str) -> str:
    return RENAME.get(name, name)


def _filtered(
    qs: QuerySet[Any],
    agencies: Sequence[str] | None,
    date_field: str | None,
    since_year: int | None,
) -> QuerySet[Any]:
    if agencies:
        qs = qs.filter(agency__slug__in=agencies)
    if since_year and date_field:
        qs = qs.filter(**{f"{date_field}__year__gte": since_year})
    return qs


def build_patients(agencies: Sequence[str] | None) -> Table:
    t = Table(
        "patients",
        [
            "patient_key",
            "agency",
            "age_band",
            "insurance",
            "pcp_agency",
            "veteran_status",
            "first_year",
            "first_quarter",
            "last_year",
            "last_quarter",
            *PATIENT_BOOLS,
        ],
        ["insurance", "pcp_agency", "veteran_status"],
    )
    fields = [
        "id",
        "agency__slug",
        "age",
        "insurance",
        "pcp_agency",
        "veteran_status",
        "created_date",
        "modified_date",
        *PATIENT_BOOLS,
    ]
    for r in _filtered(Patients.objects.order_by("id"), agencies, None, None).values(*fields):
        # When the record was created (first seen) and last changed, as year and quarter only.
        # Missing created dates fall back to modified, as cpmdash's patients-served counts do.
        fy, fq = year_quarter(r["created_date"] or r["modified_date"])
        ly, lq = year_quarter(r["modified_date"] or r["created_date"])
        row = {
            "patient_key": key("P", r["id"]),
            "agency": r["agency__slug"],
            "age_band": age_band(r["age"]),
            "insurance": clean_text(r["insurance"]),
            "pcp_agency": clean_text(r["pcp_agency"]),
            "veteran_status": clean_text(r["veteran_status"]),
            "first_year": fy,
            "first_quarter": fq,
            "last_year": ly,
            "last_quarter": lq,
        }
        row.update({b: bool_cell(r[b]) for b in PATIENT_BOOLS})
        t.rows.append(row)
    return t


def build_encounters(agencies: Sequence[str] | None, since_year: int | None) -> Table:
    t = Table(
        "encounters",
        [
            "encounter_key",
            "patient_key",
            "referral_key",
            "port_referral_key",
            "agency",
            "year",
            "quarter",
            "seq",
            "gap_from_prev",
            "pcp_agency",
            *ENC_TEXT,
            *DIV_TEXT,
            *SHARED_BOOLS,
        ],
        ["pcp_agency", *ENC_TEXT, *DIV_TEXT],
    )
    fields = [
        "ID",
        "patient_ID",
        "referral_ID",
        "port_referral_ID",
        "agency__slug",
        "encounter_date",
        "pcp_agency",
        *ENC_TEXT,
        *DIV_TEXT,
        *SHARED_BOOLS,
    ]
    # Sequence and gaps are computed over the patient's whole history, then
    # the year filter is applied, so seq stays true even for partial exports.
    qs = _filtered(Encounters.objects.all(), agencies, None, None)
    rows = list(qs.order_by("patient_ID", "encounter_date", "ID").values(*fields))
    last: dict[Any, tuple[int, date | None]] = {}
    for r in rows:
        pid = r["patient_ID"]
        cur = as_date(r["encounter_date"])
        if pid is None:
            seq, gap = "", "unknown"
        else:
            n, prev = last.get(pid, (0, None))
            gap = gap_bucket(prev if n else None, cur)
            seq = str(n + 1)
            last[pid] = (n + 1, cur if cur is not None else prev)
        y, q = year_quarter(cur)
        if since_year and (not y or int(y) < since_year):
            continue
        row = {
            "encounter_key": key("E", r["ID"]),
            "patient_key": key("P", pid),
            "referral_key": key("R", r["referral_ID"]),
            "port_referral_key": key("OD", r["port_referral_ID"]),
            "agency": r["agency__slug"],
            "year": y,
            "quarter": q,
            "seq": seq,
            "gap_from_prev": gap,
        }
        row.update({c: clean_text(r[c]) for c in ["pcp_agency", *ENC_TEXT, *DIV_TEXT]})
        row.update({b: bool_cell(r[b]) for b in SHARED_BOOLS})
        t.rows.append(row)
    t.rows.sort(key=lambda x: (x["year"], x["quarter"], x["agency"], x["encounter_key"]))
    return t


def build_referrals(agencies: Sequence[str] | None, since_year: int | None) -> Table:
    refs = [f"referral_{i}" for i in range(1, 6)]
    text = ["referral_agency", *ENC_TEXT, "referral_closed_reason", "insurance", *refs, *DIV_TEXT]
    t = Table(
        "referrals",
        [
            "referral_key",
            "patient_key",
            "agency",
            "year",
            "quarter",
            "age_band",
            *text,
            *SHARED_BOOLS,
        ],
        text,
    )
    fields = ["ID", "patient_ID", "agency__slug", "date_received", "age", *text, *SHARED_BOOLS]
    qs = _filtered(
        Referrals.objects.order_by("date_received", "ID"), agencies, "date_received", since_year
    )
    for r in qs.values(*fields):
        y, q = year_quarter(r["date_received"])
        row = {
            "referral_key": key("R", r["ID"]),
            "patient_key": key("P", r["patient_ID"]),
            "agency": r["agency__slug"],
            "year": y,
            "quarter": q,
            "age_band": age_band(r["age"]),
        }
        row.update({c: clean_text(r[c]) for c in text})
        row.update({b: bool_cell(r[b]) for b in SHARED_BOOLS})
        t.rows.append(row)
    return t


def build_od_referrals(agencies: Sequence[str] | None, since_year: int | None) -> Table:
    t = Table(
        "od_referrals",
        [
            "od_key",
            "patient_key",
            "agency",
            "year",
            "quarter",
            "age_band",
            *[_out(c) for c in OD_TEXT],
            *[_out(c) for c in OD_NUMBERS],
            *OD_BOOLS,
        ],
        [_out(c) for c in OD_TEXT],
    )
    fields = [
        "ID",
        "patient_id",
        "agency__slug",
        "od_date",
        "patient_age",
        *OD_TEXT,
        *OD_NUMBERS,
        *OD_BOOLS,
    ]
    qs = _filtered(ODReferrals.objects.order_by("od_date", "ID"), agencies, "od_date", since_year)
    for r in qs.values(*fields):
        y, q = year_quarter(r["od_date"])
        row = {
            "od_key": key("OD", r["ID"]),
            "patient_key": key("P", r["patient_id"]),
            "agency": r["agency__slug"],
            "year": y,
            "quarter": q,
            "age_band": age_band(r["patient_age"]),
        }
        row.update({_out(c): clean_text(r[c]) for c in OD_TEXT})
        row.update({_out(c): "" if r[c] is None else str(r[c]) for c in OD_NUMBERS})
        row.update({b: bool_cell(r[b]) for b in OD_BOOLS})
        t.rows.append(row)
    return t


# --- scrub, codebook, write ---------------------------------------------------


def scrub_rare(table: Table, min_count: int) -> dict[str, int]:
    """Replace values seen fewer than ``min_count`` times. Returns cells replaced per column."""
    replaced: dict[str, int] = {}
    if min_count <= 1:
        return replaced
    for col in table.text_columns:
        counts = Counter(r[col] for r in table.rows if r[col])
        rare = {v for v, n in counts.items() if n < min_count}
        if not rare:
            continue
        n = 0
        for r in table.rows:
            if r[col] in rare:
                r[col] = RARE
                n += 1
        replaced[col] = n
    return replaced


def codebook(tables: Iterable[Table]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for t in tables:
        skip = {"agency"} | {c for c in t.columns if c.endswith("_key")}
        for col in t.columns:
            if col in skip:
                continue
            counts = Counter(r[col] for r in t.rows)
            for value, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                out.append(
                    {"table": t.name, "column": col, "value": value or "(blank)", "count": str(n)}
                )
    return out


def _csv(columns: Sequence[str], rows: Iterable[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


README = """cpmdash export for CP Impact
============================

De-identified on export. Not included: names, addresses, coordinates, ZIP codes,
districts, any dates, jail dates, sex, race or marital status. Ages are bands.
Text values that appeared fewer than {min_count} times in a column read "(rare)".

Files
  patients.csv      one row per patient (patient_key = P + cpmdash row id);
                    first/last year and quarter = record created / last changed
  encounters.csv    one row per encounter; seq = patient's nth encounter,
                    gap_from_prev = time since their previous encounter
                    (first, same_day, 1_7d, 8_30d, 31_90d, 91d_plus)
  referrals.csv     one row per referral
  od_referrals.csv  one row per overdose referral
  codebook.csv      every value in every column with its count
  manifest.json     row counts, filters and what was dropped

The P/E/R/OD keys only mean something inside your own cpmdash database.
"""


def build_export(
    agencies: Sequence[str] | None = None, since_year: int | None = None, min_count: int = 3
) -> tuple[list[Table], dict[str, Any]]:
    tables = [
        build_patients(agencies),
        build_encounters(agencies, since_year),
        build_referrals(agencies, since_year),
        build_od_referrals(agencies, since_year),
    ]
    replaced = {t.name: scrub_rare(t, min_count) for t in tables}
    years: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for t in tables:
        for r in t.rows:
            if r.get("year"):
                years[t.name][f"{r['year']}-Q{r['quarter']}"] += 1
    manifest = {
        "source": "cpmdash",
        "format_version": 1,
        "generated": date.today().isoformat(),
        "filters": {"agencies": list(agencies or []), "since_year": since_year},
        "min_count": min_count,
        "rows": {t.name: len(t.rows) for t in tables},
        "rows_by_quarter": {k: dict(sorted(v.items())) for k, v in years.items()},
        "columns": {t.name: t.columns for t in tables},
        "dropped_fields": DROPPED,
        "renamed_columns": RENAME,
        "rare_values_replaced": replaced,
    }
    return tables, manifest


def write_zip(path: str | IO[bytes], tables: list[Table], manifest: dict[str, Any]) -> None:
    """Write to a file path, or to an open binary stream (the admin download uses BytesIO)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for t in tables:
            zf.writestr(f"{t.name}.csv", _csv(t.columns, t.rows))
        zf.writestr("codebook.csv", _csv(["table", "column", "value", "count"], codebook(tables)))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("README.txt", README.format(min_count=manifest["min_count"]))


__all__ = ["build_export", "write_zip", "age_band", "gap_bucket", "year_quarter"]
