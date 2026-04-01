"""
ETL Service for Data Import
============================
Extracted from the load_csv_data management command. This module contains all
the data cleaning, transformation, and normalization logic needed to convert
raw PowerApps/SharePoint CSV exports into production-ready DataFrames.

The service is used by:
  - The ``load_csv_data`` management command (CLI workflow)
  - The ``data_import`` web views (admin upload workflow)

Each ``clean_*`` method returns a ``CleaningResult`` dataclass containing the
cleaned DataFrame, a list of log messages emitted during processing, and counts
of warnings/errors per row for use in the review UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import IO

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Insurance values to normalize — checked in priority order.
INSURANCE_PRIORITY = [
    "Dual Medicaid/Medicare",
    "Indian Health Service",
    "Medicaid",
    "Medicare",
    "Tricare",
    "Private",
    "Uninsured",
]

RACE_CLEANUP: dict[str, str] = {
    "": "Not disclosed",
    "Select Race": "Not disclosed",
    "Unknown": "Not disclosed",
    "Other": "Not disclosed",
    "Latino": "Hispanic or Latino",
}

SEX_CLEANUP: dict[str, str] = {
    "": "Not disclosed",
    "Select Sex": "Not disclosed",
    "0": "Not disclosed",
}

OD_ZIP_CLEANUP: dict[str, str] = {
    "Experiencing Homelessness, no current ZIP Code": "Homeless/Transient",
    "Non-Clallam County ZIP Code": "",
    "Unknown": "",
    "Other": "",
    "": "",
}

DRUG_NAME_MAP: dict[str, str] = {
    "Opiate - unk/other": "Opiate/opioid (Unknown)",
    "Sedative - unk/other": "Sedative (Unknown)",
    "Stimulant - unk/other": "Stimulant (Unknown)",
    "Prescription drug - other/unk": "Prescription drug",
}

# Sentinel date for missing jail dates in OD referrals
_JAIL_SENTINEL = "2000-01-01"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CleaningResult:
    """Return value from each ``clean_*`` method."""

    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    row_warnings: dict[int, list[str]] = field(default_factory=dict)
    row_errors: dict[int, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------
class DataCleaningService:
    """Stateless service that cleans raw CSV data into production-ready DataFrames.

    Each ``clean_*`` method accepts either a file path or a file-like object
    (for uploaded files that haven't been saved to disk yet).
    """

    # ------------------------------------------------------------------
    # Public cleaning methods
    # ------------------------------------------------------------------
    def clean_patients(self, source: Path | IO[bytes] | IO[str]) -> CleaningResult:
        log: list[str] = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading patients source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # --- Merge duplicate fields ------------------------------------
        df["_ins_merged"] = df.apply(self._merge_insurance, axis=1)
        df["insurance_clean"] = df["_ins_merged"].apply(self._normalize_insurance)

        # Sex: prefer patient_sex, fall back to Sex
        df["sex_clean"] = df.apply(
            lambda r: _coalesce(r.get("patient_sex", ""), r.get("Sex", "")), axis=1
        )
        df["sex_clean"] = df["sex_clean"].map(SEX_CLEANUP).fillna(df["sex_clean"])

        # Zipcode: prefer Zip Code, fall back to zipcode
        df["zip_clean"] = df.apply(
            lambda r: _coalesce(r.get("Zip Code", ""), r.get("zipcode", "")), axis=1
        )

        # PCP Agency: prefer pcp_agency, fall back to pcp_agency_1
        df["pcp_clean"] = df.apply(
            lambda r: _coalesce(r.get("pcp_agency", ""), r.get("pcp_agency_1", "")), axis=1
        )
        df["pcp_clean"] = df["pcp_clean"].replace({"": "Not disclosed", "None": "Not disclosed"})

        # Race cleanup
        df["race_clean"] = df["race"].map(RACE_CLEANUP).fillna(df["race"])

        # Boolean fields
        df["sud_clean"] = df["SUD"].apply(_parse_bool_or_none)
        df["bh_clean"] = df["behavioral_health"].apply(_parse_bool_or_none)
        df["aud_clean"] = df["aud"].apply(_parse_bool_or_none)
        df["3c_clean"] = df["3C Client"].apply(_parse_bool_or_none)

        # Dates
        df["created_clean"] = df["Created"].apply(_parse_date)
        df["modified_clean"] = df["Modified"].apply(_parse_date)

        # Build output frame
        out = pd.DataFrame(
            {
                "id": df["ID"].astype(int),
                "age": pd.to_numeric(df["Age"], errors="coerce"),
                "insurance": df["insurance_clean"],
                "pcp_agency": df["pcp_clean"],
                "race": df["race_clean"],
                "sex": df["sex_clean"],
                "sud": df["sud_clean"],
                "zip_code": df["zip_clean"],
                "created_date": df["created_clean"],
                "modified_date": df["modified_clean"],
                "marital_status": df["marital_status"].replace("", "Not disclosed"),
                "veteran_status": df["veteran_status"].replace("", "Not disclosed"),
                "behavioral_health": df["bh_clean"],
                "aud": df["aud_clean"],
                "latitude": "",
                "longitude": "",
                "three_c_client": df["3c_clean"],
            }
        )

        # Validate rows
        for i, (_idx, row) in enumerate(out.iterrows()):
            warnings: list[str] = []
            errors: list[str] = []
            if pd.isna(row["age"]) or row["age"] == "":
                warnings.append("Missing age")
            if row["insurance"] == "Not disclosed":
                warnings.append("Insurance not disclosed")
            if row["zip_code"] == "":
                warnings.append("Missing zip code")
            if errors:
                row_errors[i] = errors
            if warnings:
                row_warnings[i] = warnings

        log.append(f"  → {len(out)} patient records cleaned")
        log.append(
            f"  → {len(row_warnings)} rows with warnings, {len(row_errors)} rows with errors"
        )
        return CleaningResult(df=out, log=log, row_warnings=row_warnings, row_errors=row_errors)

    def clean_referrals(self, source: Path | IO[bytes] | IO[str]) -> CleaningResult:
        log: list[str] = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading referrals source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # Sex: prefer RefPatientSex, fall back to PatientSex
        df["sex_clean"] = df.apply(
            lambda r: _coalesce(r.get("RefPatientSex", ""), r.get("PatientSex", "")), axis=1
        )
        df["sex_clean"] = df["sex_clean"].map(SEX_CLEANUP).fillna(df["sex_clean"])

        # Date received
        df["date_clean"] = df["date_received"].apply(_parse_date)

        # Referral type: JSON array → referral_1 through referral_5
        ref_cols = df["referral_type"].apply(_split_referral_type)
        ref_df = pd.DataFrame(ref_cols.tolist(), columns=[f"referral_{i}" for i in range(1, 6)])

        # Insurance
        df["insurance_clean"] = df["PatientInsurance"].apply(
            lambda v: v.strip() if v.strip() else "Not disclosed"
        )

        # Empty string fields → "No data"
        for col in [
            "referral_agency",
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
            "referral_closed_reason",
        ]:
            df[col] = df[col].replace("", "No data")

        df["zip_clean"] = df["PatientZipcode"].replace("", "No data")

        out = pd.DataFrame(
            {
                "ID": df["ID"].astype(int),
                "patient_ID": df["patient_ID"].astype(int),
                "sex": df["sex_clean"],
                "age": pd.to_numeric(df["PatientAge"], errors="coerce"),
                "date_received": df["date_clean"],
                "referral_agency": df["referral_agency"],
                "encounter_type_cat1": df["encounter_type_cat1"],
                "encounter_type_cat2": df["encounter_type_cat2"],
                "encounter_type_cat3": df["encounter_type_cat3"],
                "referral_closed_reason": df["referral_closed_reason"],
                "zipcode": df["zip_clean"],
                "insurance": df["insurance_clean"],
                "referral_1": ref_df["referral_1"],
                "referral_2": ref_df["referral_2"],
                "referral_3": ref_df["referral_3"],
                "referral_4": ref_df["referral_4"],
                "referral_5": ref_df["referral_5"],
            }
        )

        # Validate
        for i, (_idx, row) in enumerate(out.iterrows()):
            warnings = []
            if row["date_received"] == "":
                warnings.append("Missing date_received")
            if row["referral_agency"] == "No data":
                warnings.append("Missing referral_agency")
            if warnings:
                row_warnings[i] = warnings

        log.append(f"  → {len(out)} referral records cleaned")
        log.append(
            f"  → {len(row_warnings)} rows with warnings, {len(row_errors)} rows with errors"
        )
        return CleaningResult(df=out, log=log, row_warnings=row_warnings, row_errors=row_errors)

    def clean_odreferrals(self, source: Path | IO[bytes] | IO[str]) -> CleaningResult:
        log: list[str] = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading OD referrals source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # Date + time → combined datetime
        df["od_datetime"] = df.apply(self._combine_od_datetime, axis=1)

        # Zipcode cleanup
        df["zip_clean"] = df["patient_zipcode"].map(OD_ZIP_CLEANUP).fillna(df["patient_zipcode"])

        # Suspected drug: JSON array → comma-separated
        df["drug_clean"] = df["suspected_drug"].apply(_clean_suspected_drug)

        # Boolean fields → int
        bool_fields = [
            "referral_to_sud_agency",
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
        for col in bool_fields:
            df[f"{col}_int"] = df[col].apply(_bool_to_int)

        df["narcan_int"] = df["narcan_given"].apply(_bool_to_int)

        # Numeric fields
        int_fields = [
            "number_of_nonems_onscene",
            "number_of_ems_onscene",
            "number_of_peers_onscene",
            "number_of_police_onscene",
            "narcan_doses_prior_to_ems",
            "narcan_doses_by_ems",
            "leave_behind_narcan_amount",
        ]
        for col in int_fields:
            df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        float_fields = ["narcan_prior_to_ems_dosage", "narcan_by_ems_dosage", "persons_trained"]
        for col in float_fields:
            df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Contact level fields
        for col in [
            "contact_level_rediscovery",
            "contact_level_reflections",
            "contact_level_pbh",
            "contact_level_other",
        ]:
            df[col] = df[col].replace("", "No data")

        out = pd.DataFrame(
            {
                "ID": df["ID"].astype(int),
                "patient_id": df["patient_id"].astype(int),
                "patient_sex": df["patient_sex"],
                "patient_race": "",
                "patient_age": pd.to_numeric(df["patient_age"], errors="coerce"),
                "patient_zipcode": df["zip_clean"],
                "patient_insurance": df["patient_insurance"].replace("", "Not disclosed"),
                "living_situation": df["living_situation"],
                "od_date": df["od_datetime"],
                "delay_in_referral": df["delay_in_referral"],
                "cpm_notification": df["cpm_notification"],
                "cpm_disposition": df["cpm_disposition"],
                "referral_agency": df["referral_agency"],
                "referral_source": df["referral_source"],
                "od_address": df["od_address"],
                "lat": "",
                "long": "",
                "engagement_location": df["engagement_location"],
                "number_of_nonems_onscene": df["number_of_nonems_onscene_num"],
                "number_of_ems_onscene": df["number_of_ems_onscene_num"],
                "number_of_peers_onscene": df["number_of_peers_onscene_num"],
                "number_of_police_onscene": df["number_of_police_onscene_num"],
                "suspected_drug": df["drug_clean"],
                "cpr_administered": df["cpr_administered"],
                "police_ita": df["police_ita"],
                "disposition": df["disposition"],
                "transport_to_location": df["transport_to_location"],
                "transported_by": df["transported_by"],
                "narcan_given": df["narcan_int"],
                "narcan_doses_prior_to_ems": df["narcan_doses_prior_to_ems_num"],
                "narcan_prior_to_ems_dosage": df["narcan_prior_to_ems_dosage_num"],
                "narcan_doses_by_ems": df["narcan_doses_by_ems_num"],
                "narcan_by_ems_dosage": df["narcan_by_ems_dosage_num"],
                "leave_behind_narcan_amount": df["leave_behind_narcan_amount_num"],
                "persons_trained": df["persons_trained_num"],
                "referral_to_sud_agency": df["referral_to_sud_agency_int"],
                "referral_rediscovery": df["referral_rediscovery_int"],
                "referral_reflections": df["referral_reflections_int"],
                "referral_pbh": df["referral_pbh_int"],
                "referral_other": df["referral_other_int"],
                "contact_level_rediscovery": df["contact_level_rediscovery"],
                "contact_level_reflections": df["contact_level_reflections"],
                "contact_level_pbh": df["contact_level_pbh"],
                "contact_level_other": df["contact_level_other"],
                "accepted_rediscovery": df["accepted_rediscovery_int"],
                "accepted_reflections": df["accepted_reflections_int"],
                "accepted_pbh": df["accepted_pbh_int"],
                "accepted_other": df["accepted_other_int"],
                "is_bup_indicated": df["is_bup_indicated_int"],
                "bup_not_indicated_reason": df["bup_not_indicated_reason"],
                "bup_already_prescribed": df["bup_already_prescribed"],
                "bup_admin": df["bup_admin_int"],
                "client_agrees_to_mat": df["client_agrees_to_mat_int"],
                "overdose_recent": df["overdose_recent"].replace("", "No data"),
                "jail_start_1": _JAIL_SENTINEL,
                "jail_end_1": _JAIL_SENTINEL,
                "jail_start_2": _JAIL_SENTINEL,
                "jail_end_2": _JAIL_SENTINEL,
            }
        )

        # Validate
        for i, (_idx, row) in enumerate(out.iterrows()):
            warnings = []
            if row["od_date"] == "":
                warnings.append("Missing od_date")
            if row["patient_sex"] == "":
                warnings.append("Missing patient_sex")
            if warnings:
                row_warnings[i] = warnings

        log.append(f"  → {len(out)} OD referral records cleaned")
        log.append(
            f"  → {len(row_warnings)} rows with warnings, {len(row_errors)} rows with errors"
        )
        return CleaningResult(df=out, log=log, row_warnings=row_warnings, row_errors=row_errors)

    def clean_encounters(self, source: Path | IO[bytes] | IO[str]) -> CleaningResult:
        log: list[str] = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading encounters source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # Filter out rows with empty patient_ID
        before = len(df)
        df = df[df["patient_ID"].str.strip() != ""]
        dropped = before - len(df)
        if dropped:
            log.append(f"  ⚠ Dropped {dropped} rows with empty patient_ID")

        # Date
        df["date_clean"] = df["encounter_date"].apply(_parse_date)

        # Empty fields → "No data"
        for col in [
            "pcp_agency",
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
        ]:
            df[col] = df[col].replace("", "No data")

        # port_referral_ID: empty → 0
        df["port_ref_clean"] = (
            pd.to_numeric(df["port_referral_ID"], errors="coerce").fillna(0).astype(int)
        )

        out = pd.DataFrame(
            {
                "ID": df["ID"].astype(int),
                "referral_ID": pd.to_numeric(df["referral_ID"], errors="coerce"),
                "port_referral_ID": df["port_ref_clean"],
                "patient_ID": pd.to_numeric(df["patient_ID"], errors="coerce"),
                "encounter_date": df["date_clean"],
                "pcp_agency": df["pcp_agency"],
                "encounter_type_cat1": df["encounter_type_cat1"],
                "encounter_type_cat2": df["encounter_type_cat2"],
                "encounter_type_cat3": df["encounter_type_cat3"],
            }
        )

        # Validate
        for i, (_idx, row) in enumerate(out.iterrows()):
            warnings = []
            if row["encounter_date"] == "":
                warnings.append("Missing encounter_date")
            if pd.isna(row["patient_ID"]):
                warnings.append("Missing patient_ID")
            if warnings:
                row_warnings[i] = warnings

        log.append(f"  → {len(out)} encounter records cleaned")
        log.append(
            f"  → {len(row_warnings)} rows with warnings, {len(row_errors)} rows with errors"
        )
        return CleaningResult(df=out, log=log, row_warnings=row_warnings, row_errors=row_errors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_csv(source: Path | IO[bytes] | IO[str]) -> pd.DataFrame:
        """Read a CSV from a file path or file-like object."""
        if isinstance(source, Path):
            return pd.read_csv(source, dtype=str).fillna("")
        # For file-like objects (Django UploadedFile), read content
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return pd.read_csv(StringIO(str(content)), dtype=str).fillna("")

    @staticmethod
    def _merge_insurance(row: pd.Series) -> str:
        upper = str(row.get("Insurance", "")).strip()
        lower = str(row.get("insurance", "")).strip()

        if upper and upper != "[]":
            return upper
        if lower and lower != "[]":
            try:
                items = json.loads(lower)
                if isinstance(items, list):
                    return ",".join(items)
            except (json.JSONDecodeError, TypeError):
                pass
            return lower
        return ""

    @staticmethod
    def _normalize_insurance(raw: str) -> str:
        if not raw or raw == "[]":
            return "Not disclosed"
        if raw.strip().lower() in ("unknown", "other", "select insurance"):
            return "Not disclosed"
        for candidate in INSURANCE_PRIORITY:
            if candidate.lower() in raw.lower():
                return candidate
        if raw.strip():
            return raw.strip()
        return "Not disclosed"

    @staticmethod
    def _combine_od_datetime(row: pd.Series) -> str:
        raw_date = str(row.get("od_date", "")).strip()
        raw_time = str(row.get("overdose_time", "")).strip()

        if not raw_date:
            return ""
        try:
            dt = datetime.strptime(raw_date.split(" ")[0], "%m/%d/%Y")
        except ValueError:
            try:
                dt = datetime.strptime(raw_date.split(" ")[0], "%Y-%m-%d")
            except ValueError:
                return ""

        if raw_time and len(raw_time) >= 3:
            raw_time = raw_time.zfill(4)
            try:
                hour = int(raw_time[:2])
                minute = int(raw_time[2:4])
                dt = dt.replace(hour=hour, minute=minute)
            except ValueError:
                pass

        return dt.strftime("%Y-%m-%d %H:%M:%S+00")


# ======================================================================
# Module-level utility functions
# ======================================================================
def _coalesce(*values: str) -> str:
    """Return the first non-empty string value."""
    for v in values:
        v = str(v).strip()
        if v and v != "nan":
            return v
    return ""


def _parse_date(raw: str) -> str:
    """Parse various date formats into YYYY-MM-DD or empty string."""
    raw = str(raw).strip()
    if not raw or raw == "nan":
        return ""
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _parse_bool_or_none(raw: str) -> str:
    raw = str(raw).strip().lower()
    if raw in ("true", "1", "yes"):
        return "True"
    if raw in ("false", "0", "no"):
        return "False"
    return ""


def _bool_to_int(raw: str) -> int:
    return 1 if str(raw).strip().lower() in ("true", "1", "yes") else 0


def _split_referral_type(raw: str) -> list[str]:
    """Parse a JSON array of referral types into exactly 5 slots."""
    items: list[str] = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed if str(x).strip()]
        except (json.JSONDecodeError, TypeError):
            items = [x.strip() for x in raw.split(",") if x.strip()]
    while len(items) < 5:
        items.append("No data")
    return items[:5]


def _clean_suspected_drug(raw: str) -> str:
    """Parse JSON array of drug names, normalize, return comma-separated."""
    if not raw:
        return ""
    items: list[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed]
    except (json.JSONDecodeError, TypeError):
        items = [raw.strip()]
    cleaned = [DRUG_NAME_MAP.get(drug, drug) for drug in items]
    return ", ".join(cleaned)
