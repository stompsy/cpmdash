"""Tests for change detection and commit logic in the data_import views.

These test the _detect_field_changes, _classify_row, _staging_to_core_kwargs,
and _do_commit functions using real database models.
"""

from __future__ import annotations

from typing import Any

import pytest

from apps.core.models import Patients
from apps.data_import.models import (
    DataImportBatch,
    RowStatus,
    StagingPatient,
)
from apps.data_import.views import (
    _build_staging_kwargs,
    _detect_field_changes,
    _do_commit,
    _staging_to_core_kwargs,
)

pytestmark = pytest.mark.django_db


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture()
def batch(django_user_model: Any) -> DataImportBatch:
    user = django_user_model.objects.create_user(username="tester", password="testing123")
    return DataImportBatch.objects.create(
        created_by=user,
        status=DataImportBatch.Status.REVIEW,
    )


@pytest.fixture()
def existing_patient() -> Patients:
    return Patients.objects.create(
        id=1,
        age=42,
        insurance="Medicare",
        pcp_agency="NOHN",
        race="White",
        sex="Male",
        zip_code="98362",
        marital_status="Single",
        veteran_status="No",
    )


# ======================================================================
# _detect_field_changes tests
# ======================================================================
class TestDetectFieldChanges:
    def test_no_changes(self) -> None:
        compare_fields = ["age", "insurance", "race"]
        staging_kwargs = {"age": 42, "insurance": "Medicare", "race": "White"}
        prod_row = {"age": 42, "insurance": "Medicare", "race": "White"}
        diffs = _detect_field_changes(StagingPatient, compare_fields, staging_kwargs, prod_row)
        assert diffs == []

    def test_detects_age_change(self) -> None:
        compare_fields = ["age", "insurance"]
        staging_kwargs = {"age": 43, "insurance": "Medicare"}
        prod_row = {"age": 42, "insurance": "Medicare"}
        diffs = _detect_field_changes(StagingPatient, compare_fields, staging_kwargs, prod_row)
        assert len(diffs) == 1
        assert "age: 42 → 43" in diffs[0]

    def test_none_and_empty_equivalent_for_strings(self) -> None:
        compare_fields = ["insurance"]
        staging_kwargs = {"insurance": ""}
        prod_row = {"insurance": None}
        diffs = _detect_field_changes(StagingPatient, compare_fields, staging_kwargs, prod_row)
        assert diffs == []

    def test_whitespace_stripped_for_comparison(self) -> None:
        compare_fields = ["insurance"]
        staging_kwargs = {"insurance": " Medicare "}
        prod_row = {"insurance": "Medicare"}
        diffs = _detect_field_changes(StagingPatient, compare_fields, staging_kwargs, prod_row)
        assert diffs == []

    def test_multiple_changes(self) -> None:
        compare_fields = ["age", "insurance", "race"]
        staging_kwargs = {"age": 43, "insurance": "Medicaid", "race": "White"}
        prod_row = {"age": 42, "insurance": "Medicare", "race": "White"}
        diffs = _detect_field_changes(StagingPatient, compare_fields, staging_kwargs, prod_row)
        assert len(diffs) == 2


# ======================================================================
# _build_staging_kwargs tests
# ======================================================================
class TestBuildStagingKwargs:
    def test_builds_correct_kwargs(self, batch: DataImportBatch) -> None:
        record = {"id": "1", "age": "42", "insurance": "Medicare", "race": "White"}
        kwargs = _build_staging_kwargs("patients", record, "id", batch, RowStatus.NEW, [])
        assert kwargs["source_id"] == 1
        assert kwargs["batch"] == batch
        assert kwargs["row_status"] == RowStatus.NEW
        assert kwargs["age"] == 42
        assert kwargs["insurance"] == "Medicare"

    def test_validation_notes_joined(self, batch: DataImportBatch) -> None:
        record = {"id": "1"}
        kwargs = _build_staging_kwargs(
            "patients", record, "id", batch, RowStatus.WARNING, ["note1", "note2"]
        )
        assert kwargs["validation_notes"] == "note1; note2"


# ======================================================================
# _staging_to_core_kwargs tests
# ======================================================================
class TestStagingToCoreKwargs:
    def test_maps_staging_to_core(self, batch: DataImportBatch) -> None:
        staging = StagingPatient.objects.create(
            batch=batch,
            source_id=99,
            age=50,
            insurance="Medicaid",
            race="White",
            sex="Female",
            zip_code="98363",
        )
        exclude = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
        core_fields = {f.name for f in Patients._meta.get_fields() if hasattr(f, "name")}
        kwargs = _staging_to_core_kwargs(StagingPatient, staging, "id", exclude, core_fields)
        assert kwargs["id"] == 99
        assert kwargs["age"] == 50
        assert kwargs["insurance"] == "Medicaid"
        assert "batch" not in kwargs
        assert "row_status" not in kwargs


# ======================================================================
# _do_commit tests
# ======================================================================
class TestDoCommit:
    def test_inserts_new_rows(self, batch: DataImportBatch) -> None:
        StagingPatient.objects.create(
            batch=batch,
            source_id=999,
            row_status=RowStatus.NEW,
            age=30,
            insurance="Medicaid",
            race="White",
            sex="Male",
            zip_code="98362",
        )
        commit_summary = {"patients": {"new": 1, "changed": 0}}
        _do_commit(batch, commit_summary)

        assert Patients.objects.filter(id=999).exists()
        patient = Patients.objects.get(id=999)
        assert patient.age == 30
        assert patient.insurance == "Medicaid"
        assert batch.status == DataImportBatch.Status.COMMITTED

    def test_updates_changed_rows(self, batch: DataImportBatch, existing_patient: Patients) -> None:
        StagingPatient.objects.create(
            batch=batch,
            source_id=1,
            row_status=RowStatus.CHANGED,
            age=43,
            insurance="Medicaid",
            race="White",
            sex="Male",
            zip_code="98362",
        )
        commit_summary = {"patients": {"new": 0, "changed": 1}}
        _do_commit(batch, commit_summary)

        existing_patient.refresh_from_db()
        assert existing_patient.age == 43
        assert existing_patient.insurance == "Medicaid"

    def test_skips_existing_rows(self, batch: DataImportBatch) -> None:
        StagingPatient.objects.create(
            batch=batch,
            source_id=1,
            row_status=RowStatus.EXISTING,
            age=42,
            insurance="Medicare",
            race="White",
            sex="Male",
            zip_code="98362",
        )
        commit_summary = {"patients": {"new": 0, "changed": 0}}
        _do_commit(batch, commit_summary)
        # Should not crash, and batch should be committed
        assert batch.status == DataImportBatch.Status.COMMITTED

    def test_commit_summary_recorded(self, batch: DataImportBatch) -> None:
        StagingPatient.objects.create(
            batch=batch,
            source_id=888,
            row_status=RowStatus.NEW,
            age=25,
            insurance="Uninsured",
        )
        commit_summary = {"patients": {"new": 1, "changed": 0}}
        _do_commit(batch, commit_summary)
        assert batch.committed_patients == 1
