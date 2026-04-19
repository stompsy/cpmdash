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

import contextlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import IO

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Insurance values to normalize — checked in priority order (highest wins).
# IHS beats everything, Tricare beats Medicare/Medicaid, etc.
INSURANCE_PRIORITY = [
    "Indian Health Service",
    "Tricare",
    "Dual Medicaid/Medicare",
    "Medicare",
    "Medicaid",
    "Private",
    "Uninsured",
]

RACE_CLEANUP: dict[str, str] = {
    "": "Not disclosed",
    "Select Race": "Not disclosed",
    "Unknown": "Not disclosed",
    "Other": "Not disclosed",
    "Latino": "Hispanic or Latino",
    "Black": "Black or African American",
}

# Sentinel values for sex that should be randomly assigned Male/Female.
# We don't use a static dict here because each row should get its own coin flip.
SEX_RANDOM_VALUES: set[str] = {"", "Select Sex", "0"}

# Zip code cleanup shared across patients, referrals, and OD referrals.
# Source exports include verbose free-text values that need normalising.
ZIP_CLEANUP: dict[str, str] = {
    "Experiencing Homelessness, no current ZIP Code": "Homeless/Transient",
    "Homeless": "Homeless/Transient",
    "Transient": "Homeless/Transient",
    "Jail": "Jail",
    "Non-Clallam County ZIP Code": "Not disclosed",
    "Unknown": "Not disclosed",
    "Other": "Not disclosed",
    "[]": "Not disclosed",
    "": "Not disclosed",
}

# OD referrals use the same mapping but with a different default label.
OD_ZIP_CLEANUP: dict[str, str] = {
    "Experiencing Homelessness, no current ZIP Code": "Homeless/Transient",
    "Homeless": "Homeless/Transient",
    "Transient": "Homeless/Transient",
    "Non-Clallam County ZIP Code": "",
    "Unknown": "",
    "Other": "",
    "[]": "",
    "": "",
}

# Insurance / living-situation values that should collapse to "Not disclosed".
# Note: "uninsured" is intentionally NOT here — it's a valid insurance status.
INSURANCE_DISCLOSE: set[str] = {"unknown", "other", "select insurance", "not applicable", "[]"}

# PCP agency name normalization — short/ambiguous names from SharePoint.
PCP_AGENCY_CLEANUP: dict[str, str] = {
    "Medical": "NOHN - Medical",
    "Medical N": "NOHN - Medical",
    "Behavioral": "PBH - Behavioral",
    "CCHHS - Vax": "CCHHS - Vaccination",
    "Unknown": "Not disclosed",
    "Other": "Not disclosed",
    "": "Not disclosed",
}

# Referral agency name normalization — abbreviations and inconsistent names
# from SharePoint data entry.  Applied in clean_referrals().
REFERRAL_AGENCY_CLEANUP: dict[str, str] = {
    # NOHN variants
    "Medical": "NOHN - Medical",
    "Medical N": "NOHN - Medical",
    "Behavioral N": "NOHN - Behavioral",
    "Case Management": "NOHN - Case Management",
    "Case Management N": "NOHN - Case Management",
    "NOHN - CM": "NOHN - Case Management",
    "OBOT": "NOHN - OBOT",
    "NOHN - BH": "NOHN - Behavioral",
    "NOHN - Behavioral Health": "NOHN - Behavioral",
    # PBH variants
    "Behavioral": "PBH - Behavioral",
    # OPCC variants
    "REdisCOVERY": "OPCC - REdisCOVERY",
    "O - MOUD": "OPCC - MOUD",
    "O - Outreach": "OPCC - Outreach",
    "O- Outreach": "OPCC - Outreach",
    "Outreach": "PAFD - CPM Outreach",
    # CCHHS variants
    "CCHHS HRC": "CCHHS - Harm Reduction Center",
    "CCHHS Harm Reduction": "CCHHS - Harm Reduction Center",
    # Abbreviation expansions
    "3C": "Clallam Care Connection",
    "CCFD2": "Clallam County Fire District 2",
    "CCFD4": "Clallam County Fire District 4",
    "KWA": "Korean Women's Association",
    "O3A": "Olympic Area Agency on Aging",
    "OCH": "Olympic Communities of Health",
    # Spelling / naming fixes
    "Patient Family/caregiver": "Family/Caregiver",
    "Patient friend/acquaintance ": "Friend/Other",
    "Hospice - Assured": "Assured Hospice",
    "OMC - Emergency Dept": "OMC - Emergency Department",
    "OMC - Lab": "OMC - Laboratory",
    "Highland Court": "Highland Court Memory Care",
    "DSHS - DDA": "DSHS - Developmental Disabilities Administration",
    "DSHS - HCS": "DSHS - Health & Community Services",
}

# Age bounds: values outside this range are treated as bogus data entry.
_AGE_MIN_VALID = 0
_AGE_MAX_VALID = 103
_AGE_FILL_LOW = 47  # 25th percentile of valid ages
_AGE_FILL_HIGH = 81  # 75th percentile of valid ages
LIVING_SITUATION_DISCLOSE: set[str] = {"unknown", "other", ""}

DRUG_NAME_MAP: dict[str, str] = {
    "Opiate - unk/other": "Opiate/opioid (Unknown)",
    "Sedative - unk/other": "Sedative (Unknown)",
    "Stimulant - unk/other": "Stimulant (Unknown)",
    "Prescription drug - other/unk": "Prescription drug",
}

# Fallback mapping for concatenated insurance values that survive _normalize_insurance().
# These show up in OD referrals where multi-select was stored without a delimiter.
OD_INSURANCE_CLEANUP: dict[str, str] = {
    "MedicaidOther": "Medicaid",
    "MedicareOther": "Medicare",
    "PrivateMedicaid": "Medicaid",
    "MedicaidPrivate": "Medicaid",
    "MedicaidMedicare": "Dual Medicaid/Medicare",
    "MedicaidIndian Health Service": "Indian Health Service",
}

# OD referral agency normalization (subset of REFERRAL_AGENCY_CLEANUP)
OD_AGENCY_CLEANUP: dict[str, str] = {
    "CCFD2": "Clallam County Fire District 2",
}

# Sentinel date for missing jail dates in OD referrals
_JAIL_SENTINEL = "2000-01-01"

# Narcan edge-case text → numeric mapping (for narcan_doses_* columns)
NARCAN_TEXT_MAP: dict[str, int] = {
    "none given": 0,
    "> 6": 7,
    "unknown": 0,
}


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
    def clean_patients(
        self,
        source: Path | IO[bytes] | IO[str],
        log: list[str] | None = None,
    ) -> CleaningResult:
        if log is None:
            log = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading patients source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # --- Age: calculate from birthdate, fix bogus values -----------
        df["age_clean"] = df.apply(self._compute_age, axis=1)

        # --- Insurance: merge two columns, resolve multi-value ---------
        # Lowercase (JSON arrays) wins over uppercase (comma-separated).
        df["_ins_merged"] = df.apply(self._merge_insurance, axis=1)
        df["insurance_clean"] = df["_ins_merged"].apply(self._normalize_insurance)

        # --- Sex: prefer patient_sex, fall back to Sex -----------------
        # "0" and "Select Sex" get a random Male/Female assignment.
        df["sex_clean"] = df.apply(
            lambda r: _coalesce(r.get("patient_sex", ""), r.get("Sex", "")), axis=1
        )
        df["sex_clean"] = df["sex_clean"].apply(
            lambda v: random.choice(["Male", "Female"]) if v in SEX_RANDOM_VALUES else v
        )

        # --- Zipcode: prefer zipcode, fall back to Zip Code -----------
        # "zipcode" column wins.  Multi-value "Zip Code" entries like
        # "98362,Experiencing Homelessness..." get the first real ZIP.
        df["zip_clean"] = df.apply(self._merge_zipcode, axis=1)
        df["zip_clean"] = df["zip_clean"].map(ZIP_CLEANUP).fillna(df["zip_clean"])

        # --- PCP Agency: prefer pcp_agency, fall back to pcp_agency_1 --
        df["pcp_clean"] = df.apply(
            lambda r: _coalesce(r.get("pcp_agency", ""), r.get("pcp_agency_1", "")), axis=1
        )
        df["pcp_clean"] = df["pcp_clean"].map(PCP_AGENCY_CLEANUP).fillna(df["pcp_clean"])

        # --- Race cleanup ----------------------------------------------
        df["race_clean"] = df["race"].map(RACE_CLEANUP).fillna(df["race"])

        # --- Boolean fields → 1/0 --------------------------------------
        df["sud_clean"] = df["SUD"].apply(_bool_to_int)
        df["bh_clean"] = df["behavioral_health"].apply(_bool_to_int)
        df["aud_clean"] = df["aud"].apply(_bool_to_int)
        df["3c_clean"] = df["3C Client"].apply(_bool_to_int)
        df["case_mgmt_clean"] = df["Case Management"].apply(_bool_to_int)
        df["flyer_clean"] = df["flyer"].apply(_bool_to_int)
        df["safety_plan_clean"] = df["safety_plan"].apply(_bool_to_int)
        df["high_util_clean"] = df["High_Utilizer"].apply(_bool_to_int)
        df["bh_sus_clean"] = df["bh_sus"].apply(_bool_to_int)
        df["sud_sus_clean"] = df["sud_sus"].apply(_bool_to_int)
        df["aud_sus_clean"] = df["aud_sus"].apply(_bool_to_int)

        # --- Dates → YYYY-MM-DD ----------------------------------------
        df["created_clean"] = df["Created"].apply(_parse_date)
        df["modified_clean"] = df["Modified"].apply(_parse_date)

        # --- Geocode from Address column --------------------------------
        # Only geocode rows where there's an address to work with.
        # The lat/long will be blank for rows with no Address value.
        from apps.data_import.geocode_service import geocode_od_addresses

        df["lat_clean"], df["long_clean"] = geocode_od_addresses(
            df, address_col="Address", zip_col="zip_clean", log=log
        )

        # Build output frame
        out = pd.DataFrame(
            {
                "id": df["ID"].astype(int),
                "age": df["age_clean"],
                "insurance": df["insurance_clean"],
                "pcp_agency": df["pcp_clean"],
                "race": df["race_clean"],
                "sex": df["sex_clean"],
                "sud": df["sud_clean"],
                "zip_code": df["zip_clean"],
                "address": df["Address"].fillna("").str.strip(),
                "created_date": df["created_clean"],
                "modified_date": df["modified_clean"],
                "marital_status": df["marital_status"].replace(
                    {"": "Not disclosed", "Single": "Not Married/Widowed"}
                ),
                "veteran_status": df["veteran_status"].replace("", "Not disclosed"),
                "behavioral_health": df["bh_clean"],
                "aud": df["aud_clean"],
                "latitude": df["lat_clean"],
                "longitude": df["long_clean"],
                "three_c_client": df["3c_clean"],
                "case_management": df["case_mgmt_clean"],
                "flyer": df["flyer_clean"],
                "safety_plan": df["safety_plan_clean"],
                "is_high_utilizer": df["high_util_clean"],
                "bh_sus": df["bh_sus_clean"],
                "sud_sus": df["sud_sus_clean"],
                "aud_sus": df["aud_sus_clean"],
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
            if row["zip_code"] == "Not disclosed":
                warnings.append("Zip code not disclosed")
            if errors:
                row_errors[i] = errors
            if warnings:
                row_warnings[i] = warnings

        log.append(f"  → {len(out)} patient records cleaned")
        log.append(
            f"  → {len(row_warnings)} rows with warnings, {len(row_errors)} rows with errors"
        )
        return CleaningResult(df=out, log=log, row_warnings=row_warnings, row_errors=row_errors)

    def clean_referrals(
        self,
        source: Path | IO[bytes] | IO[str],
        patients_df: pd.DataFrame | None = None,
        log: list[str] | None = None,
    ) -> CleaningResult:
        if log is None:
            log = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading referrals source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # Sex: prefer RefPatientSex, fall back to PatientSex
        df["sex_clean"] = df.apply(
            lambda r: _coalesce(r.get("RefPatientSex", ""), r.get("PatientSex", "")), axis=1
        )
        df["sex_clean"] = df["sex_clean"].apply(
            lambda v: random.choice(["Male", "Female"]) if v in SEX_RANDOM_VALUES else v
        )

        # Date received
        df["date_clean"] = df["date_received"].apply(_parse_date)

        # Referral type: JSON array → referral_1 through referral_5
        ref_cols = df["referral_type"].apply(_split_referral_type)
        ref_df = pd.DataFrame(ref_cols.tolist(), columns=[f"referral_{i}" for i in range(1, 6)])
        # Normalize stray "Unknown" values to "Other" across all referral slots
        for col in ref_df.columns:
            ref_df[col] = ref_df[col].replace("Unknown", "Other")

        # Insurance: normalize, then strip trailing ", Other" fragments
        df["insurance_clean"] = df["PatientInsurance"].apply(self._normalize_insurance)

        # Empty string fields → "No data"
        for col in [
            "referral_agency",
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
            "referral_closed_reason",
        ]:
            df[col] = df[col].replace("", "No data")

        # Normalize referral agency names (abbreviations, typos, inconsistencies)
        df["referral_agency"] = (
            df["referral_agency"].map(REFERRAL_AGENCY_CLEANUP).fillna(df["referral_agency"])
        )

        df["zip_clean"] = df["PatientZipcode"].map(ZIP_CLEANUP).fillna(df["PatientZipcode"])
        df["zip_clean"] = df["zip_clean"].replace("", "No data")

        # Age: compute from RefBirthdate → date_received, fall back to PatientAge, then patients
        df["age_clean"] = _resolve_age_at_event(
            df,
            "RefBirthdate",
            "date_received",
            "PatientAge",
            "patient_ID",
            patients_df,
            log,
            "referral",
        )

        # --- New fields: diversion + boolean (strip ref_ prefix from CSV names) ---
        for col in ["diversion_type_cat1", "diversion_type_cat2"]:
            df[col] = df[col].replace("", "No data")

        for csv_col, model_col in [
            ("ref_med_manage", "med_manage_int"),
            ("ref_med_script", "med_script_int"),
            ("ref_pcp_connect", "pcp_connect_int"),
            ("ref_survey_willing", "survey_willing_int"),
        ]:
            df[model_col] = df[csv_col].apply(_bool_to_int)

        out = pd.DataFrame(
            {
                "ID": df["ID"].astype(int),
                "patient_ID": df["patient_ID"].astype(int),
                "sex": df["sex_clean"],
                "age": df["age_clean"],
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
                "diversion_type_cat1": df["diversion_type_cat1"],
                "diversion_type_cat2": df["diversion_type_cat2"],
                "med_manage": df["med_manage_int"],
                "med_script": df["med_script_int"],
                "pcp_connect": df["pcp_connect_int"],
                "survey_willing": df["survey_willing_int"],
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

    def clean_odreferrals(
        self,
        source: Path | IO[bytes] | IO[str],
        patients_df: pd.DataFrame | None = None,
        log: list[str] | None = None,
    ) -> CleaningResult:
        if log is None:
            log = []
        row_warnings: dict[int, list[str]] = {}
        row_errors: dict[int, list[str]] = {}

        log.append("Reading OD referrals source file...")
        df = self._read_csv(source)
        log.append(f"  Read {len(df)} raw rows")

        # Date + time → combined datetime
        df["od_datetime"] = df.apply(self._combine_od_datetime, axis=1)

        # Zipcode cleanup — empty zips default to Homeless/Transient for OD incidents
        df["zip_clean"] = df["patient_zipcode"].map(OD_ZIP_CLEANUP).fillna(df["patient_zipcode"])
        df["zip_clean"] = df["zip_clean"].replace("", "Homeless/Transient")

        # Suspected drug: JSON array → comma-separated (drops "Unknown" alongside real drugs)
        df["drug_clean"] = df["suspected_drug"].apply(_clean_suspected_drug)

        # Insurance: prefer patient lookup, then normalize, then OD-specific fallback
        df["ins_clean"] = _resolve_od_insurance(df, patients_df, self._normalize_insurance)

        # Boolean fields → int
        _apply_bool_to_int(
            df,
            [
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
            ],
        )

        df["narcan_int"] = df["narcan_given"].apply(_bool_to_int)

        # Numeric fields — use _narcan_to_int for narcan/onscene fields
        # to handle text edge cases ("None Given" → 0, "> 6" → 7, "Unknown" → 0)
        _apply_narcan_to_int(
            df,
            [
                "number_of_nonems_onscene",
                "number_of_ems_onscene",
                "number_of_peers_onscene",
                "number_of_police_onscene",
                "narcan_doses_prior_to_ems",
                "narcan_doses_by_ems",
                "leave_behind_narcan_amount",
            ],
        )

        _apply_float_coerce(
            df,
            [
                "narcan_prior_to_ems_dosage",
                "narcan_by_ems_dosage",
                "persons_trained",
            ],
        )

        # Age: compute from birthdate → od_date, fall back to patient_age, then patient lookup
        df["age_clean"] = _resolve_age_at_event(
            df,
            "birthdate",
            "od_date",
            "patient_age",
            "patient_id",
            patients_df,
            log,
            "OD referral",
        )

        # Race: look up from cleaned patients. "No data" when not found or empty.
        df["race_clean"] = _resolve_race_from_patients(df, "patient_id", patients_df)

        # Referral agency: expand abbreviations, empties → "No data"
        df["agency_clean"] = (
            df["referral_agency"].map(OD_AGENCY_CLEANUP).fillna(df["referral_agency"])
        )
        df["agency_clean"] = df["agency_clean"].replace("", "No data")

        # --- Blanket empty → "No data" for text fields ---
        # Fields with special defaults handled separately above/below.
        _replace_empty_with_no_data(
            df,
            [
                "cpm_disposition",
                "referral_source",
                "engagement_location",
                "cpr_administered",
                "police_ita",
                "disposition",
                "transport_to_location",
                "transported_by",
                "bup_not_indicated_reason",
                "bup_already_prescribed",
                "contact_level_rediscovery",
                "contact_level_reflections",
                "contact_level_pbh",
                "contact_level_other",
            ],
        )

        # delay_in_referral: empties → "24-72hrs" (most likely timeframe)
        df["delay_in_referral"] = df["delay_in_referral"].replace("", "24-72hrs")

        # Geocode OD addresses — prefer lat/long from CSV, only geocode empty rows
        df["lat_clean"], df["long_clean"] = _geocode_od_with_csv_fallback(df, log)

        # --- New boolean fields (strip ref_port_ prefix from CSV names) ---
        _apply_bool_to_int_mapped(
            df,
            [
                ("ref_port_med_manage", "med_manage_int"),
                ("ref_port_med_script", "med_script_int"),
                ("ref_port_pcp_connect", "pcp_connect_int"),
                ("ref_port_survey_willing", "survey_willing_int"),
            ],
        )

        # --- New text/categorical fields --------------------------------
        _replace_empty_with_no_data(
            df,
            [
                "diversion_type_cat1",
                "diversion_type_cat2",
                "od_district",
                "encounter_type_cat1",
                "encounter_type_cat2",
                "encounter_type_cat3",
            ],
        )

        out = pd.DataFrame(
            {
                "ID": df["ID"].astype(int),
                "patient_id": df["patient_id"].astype(int),
                "patient_sex": df["patient_sex"],
                "patient_race": df["race_clean"],
                "patient_age": df["age_clean"],
                "patient_zipcode": df["zip_clean"],
                "patient_insurance": df["ins_clean"],
                "living_situation": df["living_situation"].apply(
                    lambda v: "Not disclosed"
                    if str(v).strip().lower() in LIVING_SITUATION_DISCLOSE
                    else str(v).strip()
                ),
                "od_date": df["od_datetime"],
                "delay_in_referral": df["delay_in_referral"],
                "cpm_notification": df["cpm_notification"],
                "cpm_disposition": df["cpm_disposition"],
                "referral_agency": df["agency_clean"],
                "referral_source": df["referral_source"],
                "od_address": df["od_address"],
                "lat": df["lat_clean"],
                "long": df["long_clean"],
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
                "diversion_type_cat1": df["diversion_type_cat1"],
                "diversion_type_cat2": df["diversion_type_cat2"],
                "od_district": df["od_district"],
                "encounter_type_cat1": df["encounter_type_cat1"],
                "encounter_type_cat2": df["encounter_type_cat2"],
                "encounter_type_cat3": df["encounter_type_cat3"],
                "med_manage": df["med_manage_int"],
                "med_script": df["med_script_int"],
                "pcp_connect": df["pcp_connect_int"],
                "survey_willing": df["survey_willing_int"],
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

    def clean_encounters(
        self,
        source: Path | IO[bytes] | IO[str],
        patients_df: pd.DataFrame | None = None,
        log: list[str] | None = None,
    ) -> CleaningResult:
        log = log or []
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

        # Strip leading/trailing whitespace from categorical columns
        cat_cols = [
            "pcp_agency",
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
        ]
        for col in cat_cols:
            df[col] = df[col].str.strip()

        # Normalize Unicode dashes (em-dash, en-dash, mojibake) → regular hyphen
        df["pcp_agency"] = (
            df["pcp_agency"]
            .str.replace("\u2014", "-", regex=False)
            .str.replace("\u2013", "-", regex=False)
            .str.replace("\u00e2\u0080\u0094", "-", regex=False)
        )

        # "O" prefix is shorthand for OPCC (e.g. "O- Outreach" → "OPCC - Outreach")
        df["pcp_agency"] = df["pcp_agency"].str.replace(r"^O\s*-\s*", "OPCC - ", regex=True)

        # Agency name cleanup — reuse referral conventions plus encounters-specific entries
        df["pcp_agency"] = df["pcp_agency"].replace(
            {
                **REFERRAL_AGENCY_CLEANUP,
                # Encounters-specific (not in referrals)
                "OPNET": "Law - OPNET",
                "CCHHS - Vax": "CCHHS - Vaccination",
                # Stripped version (REFERRAL_AGENCY_CLEANUP key has trailing space)
                "Patient friend/acquaintance": "Friend/Other",
            }
        )

        # Empty fields → "No data"
        for col in cat_cols:
            df[col] = df[col].replace("", "No data")

        # pcp_agency fallback: when CSV value is empty/No data, look up from patients table
        _fill_pcp_from_patients(df, patients_df, log)

        # --- New fields: diversion types --------------------------------
        for col in ["diversion_type_cat1", "diversion_type_cat2"]:
            df[col] = df[col].str.strip().replace("", "No data")

        # --- New boolean fields → 1/0 ----------------------------------
        for col in ["med_manage", "med_script", "pcp_connect", "survey_willing"]:
            df[f"{col}_int"] = df[col].apply(_bool_to_int)

        # port_referral_ID: empty → 0
        df["port_ref_clean"] = (
            pd.to_numeric(df["port_referral_ID"], errors="coerce").fillna(0).astype(int)
        )

        out = pd.DataFrame(
            {
                "ID": df["ID"].astype(int),
                "referral_ID": pd.to_numeric(df["referral_ID"], errors="coerce")
                .fillna(0)
                .astype(int),
                "port_referral_ID": df["port_ref_clean"],
                "patient_ID": pd.to_numeric(df["patient_ID"], errors="coerce"),
                "encounter_date": df["date_clean"],
                "pcp_agency": df["pcp_agency"],
                "encounter_type_cat1": df["encounter_type_cat1"],
                "encounter_type_cat2": df["encounter_type_cat2"],
                "encounter_type_cat3": df["encounter_type_cat3"],
                "diversion_type_cat1": df["diversion_type_cat1"],
                "diversion_type_cat2": df["diversion_type_cat2"],
                "med_manage": df["med_manage_int"],
                "med_script": df["med_script_int"],
                "pcp_connect": df["pcp_connect_int"],
                "survey_willing": df["survey_willing_int"],
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
        """Merge two insurance columns into a single comma-separated string.

        The lowercase ``insurance`` column (JSON arrays from newer SharePoint
        exports) is authoritative.  The uppercase ``Insurance`` column (older
        comma-separated format) is the fallback when the lowercase column is
        empty, because the lowercase column tends to contain more specific
        values (e.g. ``"Dual Medicaid/Medicare"`` vs bare ``"Medicare"``).
        """
        lower = str(row.get("insurance", "")).strip()
        upper = str(row.get("Insurance", "")).strip()

        # Try the authoritative lowercase (JSON) column first
        if lower and lower != "[]":
            try:
                items = json.loads(lower)
                if isinstance(items, list) and items:
                    return ",".join(str(x).strip() for x in items if str(x).strip())
            except (json.JSONDecodeError, TypeError):
                pass
            return lower

        # Fall back to uppercase (comma-separated) column
        if upper and upper != "[]":
            return upper

        return ""

    @staticmethod
    def _normalize_insurance(raw: str) -> str:
        """Resolve a (possibly multi-value) insurance string into a single
        canonical value using a strict priority hierarchy:

            IHS > Tricare > Dual Medicaid/Medicare > Medicare > Medicaid
            > Private > Uninsured > Not disclosed

        The key insight: "Dual Medicaid/Medicare" is emitted *only* when
        both "Medicare" and "Medicaid" appear in the same combined value.
        If the source already says "Dual Medicaid/Medicare" literally, that
        counts too.
        """
        if not raw or raw == "[]":
            return "Not disclosed"

        # Fast path: disclose-sentinel values
        if raw.strip().lower() in INSURANCE_DISCLOSE:
            return "Not disclosed"

        # Normalise: lowercase the whole thing, strip trailing "Other" noise
        cleaned = raw.strip()
        cleaned = cleaned.removesuffix(", Other").removesuffix(",Other")
        parts_lower = cleaned.lower()

        result = _resolve_insurance_priority(parts_lower)
        return result if result else (cleaned or "Not disclosed")

    @staticmethod
    def _compute_age(row: pd.Series) -> int:
        """Derive age from Birthdate if possible, otherwise use Age column.

        Rules:
        - If Birthdate is present, calculate age from it (as of today).
        - If birthdate is missing OR the resulting/raw age exceeds 103,
          the value is treated as a data-entry artifact and replaced with
          a random integer between the 25th and 75th percentile of the
          valid population (47–81).
        - Age 0 is allowed (infants exist in the data).
        """
        raw_bd = str(row.get("Birthdate", "")).strip()
        raw_age = str(row.get("Age", "")).strip()

        age: int | None = None

        # Try to compute from birthdate first
        if raw_bd and raw_bd != "nan":
            for fmt in ("%m/%d/%Y", "%m/%d/%Y %H:%M", "%Y-%m-%d"):
                try:
                    bd = datetime.strptime(raw_bd.split(" ")[0], fmt).date()
                    today = date.today()
                    age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
                    break
                except ValueError:
                    continue

        # Fall back to raw Age column
        if age is None and raw_age and raw_age != "nan":
            with contextlib.suppress(ValueError, TypeError):
                age = int(float(raw_age))

        # Replace bogus values (> 103 = encoded "missing" in SharePoint)
        if age is None or age > _AGE_MAX_VALID:
            age = random.randint(_AGE_FILL_LOW, _AGE_FILL_HIGH)

        return age

    @staticmethod
    def _merge_zipcode(row: pd.Series) -> str:
        """Merge two zip code columns: ``zipcode`` wins over ``Zip Code``.

        The ``Zip Code`` column sometimes contains comma-separated multi-values
        like ``"98362,Experiencing Homelessness, no current ZIP Code"``.  In
        that case we take the first token that looks like a 5-digit ZIP.
        """
        primary = str(row.get("zipcode", "")).strip()
        if primary and primary != "nan":
            return primary

        fallback = str(row.get("Zip Code", "")).strip()
        if not fallback or fallback == "nan":
            return ""

        # Handle multi-value: split on comma, grab first 5-digit ZIP
        if "," in fallback:
            for part in fallback.split(","):
                part = part.strip()
                if part.isdigit() and len(part) == 5:
                    return part
            # No ZIP found in the multi-value — return the whole thing
            # so the ZIP_CLEANUP mapping can handle it
            return fallback

        return fallback

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
def _resolve_insurance_priority(parts_lower: str) -> str:
    """Return the winning insurance value from a lowercased multi-value string.

    Priority: IHS > Tricare > Dual Medicaid/Medicare > Medicare > Medicaid
              > Private > Uninsured.
    Returns empty string if nothing matches (caller decides the fallback).
    """
    if "indian health service" in parts_lower:
        return "Indian Health Service"
    if "tricare" in parts_lower:
        return "Tricare"
    if "dual medicaid/medicare" in parts_lower:
        return "Dual Medicaid/Medicare"
    has_medicare = "medicare" in parts_lower
    has_medicaid = "medicaid" in parts_lower
    if has_medicare and has_medicaid:
        return "Dual Medicaid/Medicare"
    if has_medicare:
        return "Medicare"
    if has_medicaid:
        return "Medicaid"
    if "private" in parts_lower:
        return "Private"
    if "uninsured" in parts_lower:
        return "Uninsured"
    return ""


def _resolve_od_insurance(
    df: pd.DataFrame,
    patients_df: pd.DataFrame | None,
    normalize_fn: Callable[[str], str],
) -> pd.Series:
    """Resolve insurance for OD referrals.

    When *patients_df* is available, look up insurance from the already-cleaned
    patients.  Any unmatched rows fall back to the raw ``patient_insurance``
    column, run through *normalize_fn* and the ``OD_INSURANCE_CLEANUP`` map.
    """
    if patients_df is not None:
        ins_lookup = patients_df.set_index("id")["insurance"]
        result = df["patient_id"].astype(int).map(ins_lookup)
        no_match = result.isna()
        if no_match.any():
            fallback = df.loc[no_match, "patient_insurance"].apply(normalize_fn)
            result.loc[no_match] = fallback.map(OD_INSURANCE_CLEANUP).fillna(fallback)
        return result

    normalized = df["patient_insurance"].apply(normalize_fn)
    return normalized.map(OD_INSURANCE_CLEANUP).fillna(normalized)


def _coalesce(*values: str) -> str:
    """Return the first non-empty string value."""
    for v in values:
        v = str(v).strip()
        if v and v != "nan":
            return v
    return ""


def _resolve_age_from_patients(
    df: pd.DataFrame,
    patient_id_col: str,
    raw_age_col: str,
    patients_df: pd.DataFrame | None,
    log: list[str],
    label: str = "referral",
) -> pd.Series:
    """Look up age from cleaned patients, falling back to the raw age column."""
    if patients_df is not None:
        age_lookup = patients_df.set_index("id")["age"]
        result = df[patient_id_col].astype(int).map(age_lookup)
        unmatched = int(result.isna().sum())
        if unmatched:
            log.append(f"  ⚠ {unmatched} {label}(s) have no matching patient for age")
        return result
    return pd.to_numeric(df[raw_age_col], errors="coerce")


def _compute_age_at_event(
    row: pd.Series,
    birthdate_col: str,
    event_date_col: str,
    fallback_age_col: str,
) -> float | None:
    """Compute age at the time of an event (OD or referral) from birthdate.

    Returns floor((event_date - birthdate) / 365.25).
    Falls back to the raw age column if birthdate or event date is missing.
    Returns None if nothing works (caller can fill from patients table).
    """
    raw_bd = str(row.get(birthdate_col, "")).strip()
    raw_event = str(row.get(event_date_col, "")).strip()

    # Try birthdate → event date calculation
    if raw_bd and raw_bd != "nan" and raw_event and raw_event != "nan":
        bd = _parse_date_obj(raw_bd)
        evt = _parse_date_obj(raw_event)
        if bd is not None and evt is not None:
            age = int((evt - bd).days / 365.25)
            if _AGE_MIN_VALID <= age <= _AGE_MAX_VALID:
                return float(age)

    # Fall back to raw age column
    raw_age = str(row.get(fallback_age_col, "")).strip()
    if raw_age and raw_age != "nan":
        with contextlib.suppress(ValueError, TypeError):
            age = int(float(raw_age))
            if _AGE_MIN_VALID <= age <= _AGE_MAX_VALID:
                return float(age)

    return None


def _parse_date_obj(raw: str) -> date | None:
    """Parse a date string into a date object, returning None on failure."""
    raw = str(raw).strip()
    if not raw or raw == "nan":
        return None
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw.split("+")[0].strip(), fmt).date()
        except ValueError:
            continue
    return None


def _resolve_age_at_event(
    df: pd.DataFrame,
    birthdate_col: str,
    event_date_col: str,
    fallback_age_col: str,
    patient_id_col: str,
    patients_df: pd.DataFrame | None,
    log: list[str],
    label: str = "referral",
) -> pd.Series:
    """Compute age at event from birthdate, fill gaps from patients table.

    1. birthdate → event date → floor((event - bd) / 365.25)
    2. Fall back to raw age column
    3. Fill remaining NaN from patients_df lookup
    """
    ages = df.apply(
        lambda r: _compute_age_at_event(r, birthdate_col, event_date_col, fallback_age_col),
        axis=1,
    )
    if patients_df is not None:
        age_lookup = patients_df.set_index("id")["age"]
        still_missing = ages.isna()
        if still_missing.any():
            looked_up = df.loc[still_missing, patient_id_col].astype(int).map(age_lookup)
            ages.loc[still_missing] = looked_up
            unmatched = int(looked_up.isna().sum())
            if unmatched:
                log.append(f"  ⚠ {unmatched} {label}(s) have no matching patient for age")
    else:
        still_missing = ages.isna()
        if still_missing.any():
            ages.loc[still_missing] = pd.to_numeric(
                df.loc[still_missing, fallback_age_col], errors="coerce"
            )
    return ages


def _fill_pcp_from_patients(
    df: pd.DataFrame,
    patients_df: pd.DataFrame | None,
    log: list[str],
) -> None:
    """Fill empty pcp_agency values from patients table (in-place)."""
    if patients_df is None:
        return
    pcp_lookup = patients_df.set_index("id")["pcp_agency"]
    empty_pcp = df["pcp_agency"].isin({"", "No data", "Not disclosed"})
    if not empty_pcp.any():
        return
    looked_up = (
        pd.to_numeric(df.loc[empty_pcp, "patient_ID"], errors="coerce")
        .dropna()
        .astype(int)
        .map(pcp_lookup)
        .fillna("No data")
    )
    df.loc[looked_up.index, "pcp_agency"] = looked_up
    filled = int((looked_up != "No data").sum())
    if filled:
        log.append(f"  → Filled {filled} pcp_agency values from patients table")


def _geocode_od_with_csv_fallback(
    df: pd.DataFrame,
    log: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Use CSV lat/long when present, geocode only empty rows."""
    from apps.data_import.geocode_service import geocode_od_addresses

    lat_clean = pd.to_numeric(df["lat"], errors="coerce")
    long_clean = pd.to_numeric(df["long"], errors="coerce")

    needs_geocoding = lat_clean.isna() | long_clean.isna()
    if needs_geocoding.any():
        geo_df = df.loc[needs_geocoding].copy()
        geo_lats, geo_lons = geocode_od_addresses(
            geo_df, address_col="od_address", zip_col="zip_clean", log=log
        )
        lat_clean.loc[needs_geocoding] = geo_lats
        long_clean.loc[needs_geocoding] = geo_lons
    else:
        log.append("  Geocoding: all rows have CSV lat/long, skipping geocoder")

    return lat_clean, long_clean


def _resolve_race_from_patients(
    df: pd.DataFrame,
    patient_id_col: str,
    patients_df: pd.DataFrame | None,
) -> pd.Series:
    """Look up race from cleaned patients. Returns 'No data' when not found."""
    if patients_df is not None and "race" in patients_df.columns:
        race_lookup = patients_df.set_index("id")["race"]
        result = df[patient_id_col].astype(int).map(race_lookup).fillna("No data")
        return result.replace("", "No data")
    return pd.Series("No data", index=df.index)


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


def _apply_bool_to_int(df: pd.DataFrame, cols: list[str]) -> None:
    """Apply _bool_to_int to each column, storing result as {col}_int."""
    for col in cols:
        df[f"{col}_int"] = df[col].apply(_bool_to_int)


def _apply_bool_to_int_mapped(df: pd.DataFrame, mappings: list[tuple[str, str]]) -> None:
    """Apply _bool_to_int from csv_col → model_col name."""
    for csv_col, model_col in mappings:
        df[model_col] = df[csv_col].apply(_bool_to_int)


def _apply_narcan_to_int(df: pd.DataFrame, cols: list[str]) -> None:
    """Apply _narcan_to_int to each column, storing result as {col}_num."""
    for col in cols:
        df[f"{col}_num"] = df[col].apply(_narcan_to_int)


def _apply_float_coerce(df: pd.DataFrame, cols: list[str]) -> None:
    """Coerce columns to float, filling NaN with 0, storing as {col}_num."""
    for col in cols:
        df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)


def _replace_empty_with_no_data(df: pd.DataFrame, cols: list[str]) -> None:
    """Replace empty strings with 'No data' for each column in-place."""
    for col in cols:
        df[col] = df[col].replace("", "No data")


def _narcan_to_int(raw: str) -> int:
    """Convert narcan/onscene text values to integers.

    Handles edge cases from SharePoint data entry:
    - "None Given" → 0, "> 6" → 7, "Unknown" → 0
    - Normal numeric strings pass through as-is
    - Anything else (empty, NaN) → 0
    """
    val = str(raw).strip()
    mapped = NARCAN_TEXT_MAP.get(val.lower())
    if mapped is not None:
        return mapped
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


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
    """Parse JSON array of drug names, normalize, return comma-separated.

    When "Unknown" appears alongside identified drugs (e.g. ["Unknown", "Fentanyl"]),
    the "Unknown" is dropped because the specific drugs are more informative.
    A standalone "Unknown" is preserved.
    """
    if not raw:
        return "No data"
    items: list[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed]
    except (json.JSONDecodeError, TypeError):
        items = [raw.strip()]
    cleaned = [DRUG_NAME_MAP.get(drug, drug) for drug in items]
    # Drop "Unknown" when it's alongside identified drugs
    if len(cleaned) > 1:
        cleaned = [d for d in cleaned if d != "Unknown"]
    # Sort so Fentanyl always comes first, rest alphabetical
    if len(cleaned) > 1:
        cleaned.sort(key=lambda d: (0 if d == "Fentanyl" else 1, d))
    result = ", ".join(cleaned)
    return result if result else "No data"
