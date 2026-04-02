"""Tests for the ETL cleaning service (etl_service.py).

These cover the individual helper functions (insurance normalization, zip cleanup,
boolean parsing, etc.) and the full clean_* pipeline methods using small in-memory
CSV fixtures.
"""

from __future__ import annotations

import csv
import io

import pytest

from apps.data_import.etl_service import (
    INSURANCE_DISCLOSE,
    INSURANCE_PRIORITY,
    ZIP_CLEANUP,
    CleaningResult,
    DataCleaningService,
    _bool_to_int,
    _clean_suspected_drug,
    _coalesce,
    _parse_bool_or_none,
    _parse_date,
    _split_referral_type,
)


# ======================================================================
# Helper function tests
# ======================================================================
class TestCoalesce:
    def test_first_nonempty(self) -> None:
        assert _coalesce("", "hello", "world") == "hello"

    def test_all_empty(self) -> None:
        assert _coalesce("", "", "") == ""

    def test_nan_skipped(self) -> None:
        assert _coalesce("nan", "valid") == "valid"

    def test_strips_whitespace(self) -> None:
        assert _coalesce("  ", " val ") == "val"


class TestParseDate:
    def test_mmddyyyy(self) -> None:
        assert _parse_date("01/15/2024") == "2024-01-15"

    def test_mmddyyyy_with_time(self) -> None:
        assert _parse_date("03/22/2023 14:30") == "2023-03-22"

    def test_iso_format(self) -> None:
        assert _parse_date("2024-06-01") == "2024-06-01"

    def test_empty(self) -> None:
        assert _parse_date("") == ""

    def test_nan(self) -> None:
        assert _parse_date("nan") == ""

    def test_garbage(self) -> None:
        assert _parse_date("not-a-date") == ""


class TestParseBoolOrNone:
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes"])
    def test_true(self, value: str) -> None:
        assert _parse_bool_or_none(value) == "True"

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "No"])
    def test_false(self, value: str) -> None:
        assert _parse_bool_or_none(value) == "False"

    @pytest.mark.parametrize("value", ["", "nan", "maybe", "N/A"])
    def test_none(self, value: str) -> None:
        assert _parse_bool_or_none(value) == ""


class TestBoolToInt:
    def test_truthy(self) -> None:
        assert _bool_to_int("true") == 1
        assert _bool_to_int("1") == 1
        assert _bool_to_int("Yes") == 1

    def test_falsy(self) -> None:
        assert _bool_to_int("false") == 0
        assert _bool_to_int("") == 0
        assert _bool_to_int("0") == 0


class TestSplitReferralType:
    def test_json_array(self) -> None:
        result = _split_referral_type('["AL", "SNF"]')
        assert result[0] == "AL"
        assert result[1] == "SNF"
        assert result[2] == "No data"
        assert len(result) == 5

    def test_empty(self) -> None:
        result = _split_referral_type("")
        assert result == ["No data"] * 5

    def test_csv_fallback(self) -> None:
        result = _split_referral_type("AL, SNF")
        assert result[0] == "AL"
        assert result[1] == "SNF"

    def test_truncated_to_five(self) -> None:
        result = _split_referral_type('["a","b","c","d","e","f","g"]')
        assert len(result) == 5


class TestCleanSuspectedDrug:
    def test_json_array(self) -> None:
        result = _clean_suspected_drug('["Fentanyl", "Heroin"]')
        assert result == "Fentanyl, Heroin"

    def test_known_mapping(self) -> None:
        result = _clean_suspected_drug('["Opiate - unk/other"]')
        assert result == "Opiate/opioid (Unknown)"

    def test_empty(self) -> None:
        assert _clean_suspected_drug("") == "No data"

    def test_plain_string_fallback(self) -> None:
        result = _clean_suspected_drug("Fentanyl")
        assert result == "Fentanyl"


# ======================================================================
# Insurance normalization tests
# ======================================================================
class TestNormalizeInsurance:
    def setup_method(self) -> None:
        self.svc = DataCleaningService()

    def test_empty(self) -> None:
        assert self.svc._normalize_insurance("") == "Not disclosed"

    def test_brackets(self) -> None:
        assert self.svc._normalize_insurance("[]") == "Not disclosed"

    @pytest.mark.parametrize("val", list(INSURANCE_DISCLOSE))
    def test_disclose_values(self, val: str) -> None:
        assert self.svc._normalize_insurance(val) == "Not disclosed"

    def test_strips_trailing_other(self) -> None:
        assert self.svc._normalize_insurance("Medicare, Other") == "Medicare"
        assert self.svc._normalize_insurance("Medicaid,Other") == "Medicaid"

    @pytest.mark.parametrize(
        "name", [p for p in INSURANCE_PRIORITY if p.lower() not in INSURANCE_DISCLOSE]
    )
    def test_priority_match(self, name: str) -> None:
        result = self.svc._normalize_insurance(name)
        assert result == name

    def test_uninsured_stays(self) -> None:
        """Uninsured is a valid insurance status — NOT collapsed to Not disclosed."""
        assert self.svc._normalize_insurance("Uninsured") == "Uninsured"

    def test_dual_priority(self) -> None:
        result = self.svc._normalize_insurance("Dual Medicaid/Medicare")
        assert result == "Dual Medicaid/Medicare"

    def test_unknown_value_passthrough(self) -> None:
        result = self.svc._normalize_insurance("ACME Insurance Co")
        assert result == "ACME Insurance Co"


# ======================================================================
# Zip cleanup tests
# ======================================================================
class TestZipCleanup:
    def test_homeless_variants(self) -> None:
        assert ZIP_CLEANUP["Experiencing Homelessness, no current ZIP Code"] == "Homeless/Transient"
        assert ZIP_CLEANUP["Homeless"] == "Homeless/Transient"
        assert ZIP_CLEANUP["Transient"] == "Homeless/Transient"

    def test_empty_bracket(self) -> None:
        assert ZIP_CLEANUP["[]"] == "Not disclosed"
        assert ZIP_CLEANUP[""] == "Not disclosed"

    def test_non_clallam(self) -> None:
        assert ZIP_CLEANUP["Non-Clallam County ZIP Code"] == "Not disclosed"


# ======================================================================
# Full pipeline tests with small CSV fixtures
# ======================================================================
def _make_csv(headers: list[str], rows: list[list[str]]) -> io.BytesIO:
    """Build an in-memory CSV file from headers and rows, with proper quoting."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return io.BytesIO(buf.getvalue().encode("utf-8"))


class TestCleanPatients:
    """Integration test for clean_patients with a minimal CSV."""

    def _make_patient_csv(self, overrides: dict[str, str] | None = None) -> io.BytesIO:
        headers = [
            "ID",
            "Birthdate",
            "Age",
            "Insurance",
            "insurance",
            "pcp_agency",
            "pcp_agency_1",
            "race",
            "patient_sex",
            "Sex",
            "SUD",
            "behavioral_health",
            "aud",
            "3C Client",
            "Zip Code",
            "zipcode",
            "Created",
            "Modified",
            "marital_status",
            "veteran_status",
        ]
        defaults = {
            "ID": "1",
            "Birthdate": "",
            "Age": "42",
            "Insurance": "Medicare",
            "insurance": "",
            "pcp_agency": "NOHN",
            "pcp_agency_1": "",
            "race": "White",
            "patient_sex": "Male",
            "Sex": "",
            "SUD": "true",
            "behavioral_health": "false",
            "aud": "",
            "3C Client": "true",
            "Zip Code": "98362",
            "zipcode": "",
            "Created": "01/15/2024",
            "Modified": "02/20/2024",
            "marital_status": "Single",
            "veteran_status": "No",
        }
        if overrides:
            defaults.update(overrides)
        row = [defaults[h] for h in headers]
        return _make_csv(headers, [row])

    def test_basic_cleaning(self) -> None:
        svc = DataCleaningService()
        result = svc.clean_patients(self._make_patient_csv())
        assert isinstance(result, CleaningResult)
        assert len(result.df) == 1
        row = result.df.iloc[0]
        assert row["id"] == 1
        assert row["age"] == 42
        assert row["insurance"] == "Medicare"
        assert row["sud"] == 1
        assert row["behavioral_health"] == 0
        assert row["aud"] == 0
        assert row["three_c_client"] == 1
        assert row["zip_code"] == "98362"
        assert row["created_date"] == "2024-01-15"

    def test_homeless_zip_normalized(self) -> None:
        svc = DataCleaningService()
        csv = self._make_patient_csv({"Zip Code": "Experiencing Homelessness, no current ZIP Code"})
        result = svc.clean_patients(csv)
        assert result.df.iloc[0]["zip_code"] == "Homeless/Transient"

    def test_bracket_zip_normalized(self) -> None:
        svc = DataCleaningService()
        csv = self._make_patient_csv({"Zip Code": "[]"})
        result = svc.clean_patients(csv)
        assert result.df.iloc[0]["zip_code"] == "Not disclosed"

    def test_insurance_strip_other(self) -> None:
        svc = DataCleaningService()
        csv = self._make_patient_csv({"Insurance": "Medicare, Other"})
        result = svc.clean_patients(csv)
        assert result.df.iloc[0]["insurance"] == "Medicare"

    def test_race_cleanup(self) -> None:
        svc = DataCleaningService()
        csv = self._make_patient_csv({"race": "Select Race"})
        result = svc.clean_patients(csv)
        assert result.df.iloc[0]["race"] == "Not disclosed"

    def test_sex_cleanup(self) -> None:
        svc = DataCleaningService()
        csv = self._make_patient_csv({"patient_sex": "Select Sex", "Sex": ""})
        result = svc.clean_patients(csv)
        assert result.df.iloc[0]["sex"] in ("Male", "Female")


class TestCleanReferrals:
    """Integration test for clean_referrals with a minimal CSV."""

    def _make_referral_csv(self, overrides: dict[str, str] | None = None) -> io.BytesIO:
        headers = [
            "ID",
            "patient_ID",
            "RefPatientSex",
            "PatientSex",
            "PatientAge",
            "date_received",
            "referral_agency",
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
            "referral_closed_reason",
            "PatientZipcode",
            "PatientInsurance",
            "referral_type",
        ]
        defaults = {
            "ID": "100",
            "patient_ID": "1",
            "RefPatientSex": "Male",
            "PatientSex": "",
            "PatientAge": "42",
            "date_received": "01/15/2024",
            "referral_agency": "NOHN",
            "encounter_type_cat1": "Clinical",
            "encounter_type_cat2": "Follow-up",
            "encounter_type_cat3": "Phone",
            "referral_closed_reason": "Completed",
            "PatientZipcode": "98362",
            "PatientInsurance": "Medicaid",
            "referral_type": '["AL","SNF"]',
        }
        if overrides:
            defaults.update(overrides)
        row = [defaults[h] for h in headers]
        return _make_csv(headers, [row])

    def test_basic_cleaning(self) -> None:
        svc = DataCleaningService()
        result = svc.clean_referrals(self._make_referral_csv())
        assert len(result.df) == 1
        row = result.df.iloc[0]
        assert row["ID"] == 100
        assert row["insurance"] == "Medicaid"
        assert row["referral_1"] == "AL"
        assert row["referral_2"] == "SNF"

    def test_empty_agency_becomes_no_data(self) -> None:
        svc = DataCleaningService()
        csv = self._make_referral_csv({"referral_agency": ""})
        result = svc.clean_referrals(csv)
        assert result.df.iloc[0]["referral_agency"] == "No data"

    def test_zip_homeless_normalized(self) -> None:
        svc = DataCleaningService()
        csv = self._make_referral_csv(
            {"PatientZipcode": "Experiencing Homelessness, no current ZIP Code"}
        )
        result = svc.clean_referrals(csv)
        assert result.df.iloc[0]["zipcode"] == "Homeless/Transient"

    def test_insurance_other_stripped(self) -> None:
        svc = DataCleaningService()
        csv = self._make_referral_csv({"PatientInsurance": "Medicaid, Other"})
        result = svc.clean_referrals(csv)
        assert result.df.iloc[0]["insurance"] == "Medicaid"


class TestCleanEncounters:
    """Integration test for clean_encounters with a minimal CSV."""

    def _make_encounters_csv(self, overrides: dict[str, str] | None = None) -> io.BytesIO:
        headers = [
            "ID",
            "referral_ID",
            "port_referral_ID",
            "patient_ID",
            "encounter_date",
            "pcp_agency",
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
        ]
        defaults = {
            "ID": "500",
            "referral_ID": "100",
            "port_referral_ID": "",
            "patient_ID": "1",
            "encounter_date": "03/10/2024",
            "pcp_agency": "NOHN",
            "encounter_type_cat1": "Clinical",
            "encounter_type_cat2": "Follow-up",
            "encounter_type_cat3": "Phone",
        }
        if overrides:
            defaults.update(overrides)
        row = [defaults[h] for h in headers]
        return _make_csv(headers, [row])

    def test_basic_cleaning(self) -> None:
        svc = DataCleaningService()
        result = svc.clean_encounters(self._make_encounters_csv())
        assert len(result.df) == 1
        row = result.df.iloc[0]
        assert row["ID"] == 500
        assert row["referral_ID"] == 100
        assert row["encounter_date"] == "2024-03-10"

    def test_referral_id_as_int(self) -> None:
        """referral_ID must be integer-typed, not float."""
        svc = DataCleaningService()
        csv = self._make_encounters_csv({"referral_ID": "51"})
        result = svc.clean_encounters(csv)
        val = result.df.iloc[0]["referral_ID"]
        assert int(val) == 51
        # Must not have a decimal point in the string representation
        assert "." not in str(val)

    def test_empty_patient_id_dropped(self) -> None:
        svc = DataCleaningService()
        csv = self._make_encounters_csv({"patient_ID": ""})
        result = svc.clean_encounters(csv)
        assert len(result.df) == 0

    def test_empty_fields_become_no_data(self) -> None:
        svc = DataCleaningService()
        csv = self._make_encounters_csv({"encounter_type_cat3": ""})
        result = svc.clean_encounters(csv)
        assert result.df.iloc[0]["encounter_type_cat3"] == "No data"
