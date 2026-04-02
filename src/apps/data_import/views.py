from __future__ import annotations

import json
import traceback
from collections.abc import Iterator
from typing import Any, cast

import pandas as pd
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import AbstractUser
from django.db import models as dm
from django.db import transaction
from django.http import HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.core.models import Encounters, ODReferrals, Patients, Referrals

from .etl_service import DataCleaningService
from .forms import DataUploadForm
from .models import (
    DataImportBatch,
    DataImportFile,
    ProcessingLog,
    RowStatus,
    StagingEncounter,
    StagingODReferral,
    StagingPatient,
    StagingReferral,
)

# ---------------------------------------------------------------------------
# Schema definitions for the review UI
# ---------------------------------------------------------------------------
SCHEMA_INFO = {
    "patients": {
        "model_name": "Patients",
        "fields": [
            ("source_id", "ID", "Integer", "Primary key from source"),
            ("age", "Age", "Integer", "Patient age"),
            ("insurance", "Insurance", "Text (50)", "Primary insurance type"),
            ("pcp_agency", "PCP Agency", "Text (50)", "Primary care provider"),
            ("race", "Race", "Text (50)", "Patient race"),
            ("sex", "Sex", "Text (10)", "Patient sex"),
            ("sud", "SUD", "Boolean", "Substance use disorder"),
            ("behavioral_health", "Behavioral Health", "Boolean", ""),
            ("zip_code", "Zip Code", "Text (50)", ""),
            ("created_date", "Created Date", "Date", "Record creation date"),
            ("modified_date", "Modified Date", "Date", "Last modification"),
            ("marital_status", "Marital Status", "Text (50)", ""),
            ("veteran_status", "Veteran Status", "Text (50)", ""),
            ("aud", "AUD", "Boolean", "Alcohol use disorder"),
            ("three_c_client", "3C Client", "Boolean", ""),
        ],
    },
    "referrals": {
        "model_name": "Referrals",
        "fields": [
            ("source_id", "ID", "Integer", "Primary key from source"),
            ("patient_ID", "Patient ID", "Integer", ""),
            ("sex", "Sex", "Text (10)", ""),
            ("age", "Age", "Integer", ""),
            ("date_received", "Date Received", "Date", ""),
            ("referral_agency", "Referral Agency", "Text (50)", ""),
            ("encounter_type_cat1", "Encounter Cat 1", "Text (50)", ""),
            ("encounter_type_cat2", "Encounter Cat 2", "Text (50)", ""),
            ("encounter_type_cat3", "Encounter Cat 3", "Text (50)", ""),
            ("referral_closed_reason", "Closed Reason", "Text (50)", ""),
            ("zipcode", "Zipcode", "Text (50)", ""),
            ("insurance", "Insurance", "Text (50)", ""),
            ("referral_1", "Referral 1", "Text (50)", ""),
            ("referral_2", "Referral 2", "Text (50)", ""),
            ("referral_3", "Referral 3", "Text (50)", ""),
            ("referral_4", "Referral 4", "Text (50)", ""),
            ("referral_5", "Referral 5", "Text (50)", ""),
        ],
    },
    "odreferrals": {
        "model_name": "OD Referrals",
        "fields": [
            ("source_id", "ID", "Integer", "Primary key from source"),
            ("patient_id", "Patient ID", "Integer", ""),
            ("patient_sex", "Sex", "Text (20)", ""),
            ("patient_race", "Race", "Text (30)", ""),
            ("patient_age", "Age", "Integer", ""),
            ("patient_zipcode", "Zipcode", "Text (20)", ""),
            ("patient_insurance", "Insurance", "Text (50)", ""),
            ("od_date", "OD Date/Time", "DateTime", ""),
            ("disposition", "Disposition", "Text (50)", ""),
            ("suspected_drug", "Suspected Drug", "Text (50)", ""),
            ("narcan_given", "Narcan Given", "Boolean", ""),
        ],
    },
    "encounters": {
        "model_name": "Encounters",
        "fields": [
            ("source_id", "ID", "Integer", "Primary key from source"),
            ("referral_ID", "Referral ID", "Integer", ""),
            ("port_referral_ID", "Port Referral ID", "Integer", ""),
            ("patient_ID", "Patient ID", "Integer", ""),
            ("encounter_date", "Encounter Date", "Date", ""),
            ("pcp_agency", "PCP Agency", "Text (50)", ""),
            ("encounter_type_cat1", "Encounter Cat 1", "Text (50)", ""),
            ("encounter_type_cat2", "Encounter Cat 2", "Text (50)", ""),
            ("encounter_type_cat3", "Encounter Cat 3", "Text (50)", ""),
        ],
    },
}

# Map file types to staging models and core models
STAGING_MODELS = {
    "patients": StagingPatient,
    "referrals": StagingReferral,
    "odreferrals": StagingODReferral,
    "encounters": StagingEncounter,
}

CORE_MODELS: dict[str, type[dm.Model]] = {
    "patients": Patients,
    "referrals": Referrals,
    "odreferrals": ODReferrals,
    "encounters": Encounters,
}

# PK field name in the core models (used for delta detection)
CORE_PK_FIELDS = {
    "patients": "id",
    "referrals": "ID",
    "odreferrals": "ID",
    "encounters": "ID",
}

# Maps cleaned DataFrame column → staging model field, plus the ID column name
ID_COLUMN = {
    "patients": "id",
    "referrals": "ID",
    "odreferrals": "ID",
    "encounters": "ID",
}


# ======================================================================
# History view
# ======================================================================
@staff_member_required
def batch_list(request: HttpRequest) -> HttpResponse:
    batches = DataImportBatch.objects.select_related("created_by").all()
    return render(request, "data_import/history.html", {"batches": batches})


# ======================================================================
# Upload view
# ======================================================================
@staff_member_required
def upload(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            batch = DataImportBatch.objects.create(
                created_by=cast(AbstractUser, request.user),
                notes=form.cleaned_data.get("notes", ""),
                status=DataImportBatch.Status.UPLOADING,
            )

            uploaded = form.cleaned_data["file"]
            file_type = form.cleaned_data["file_type"]
            DataImportFile.objects.create(
                batch=batch,
                file_type=file_type,
                file=uploaded,
                original_filename=uploaded.name or "unknown",
            )

            return redirect("data_import:process", batch_id=batch.pk)
    else:
        form = DataUploadForm()

    return render(request, "data_import/upload.html", {"form": form})


# ======================================================================
# Processing view + SSE stream
# ======================================================================
@staff_member_required
def process_view(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    return render(request, "data_import/process.html", {"batch": batch})


def _process_single_file(
    import_file: DataImportFile,
    batch: DataImportBatch,
    service: DataCleaningService,
) -> Iterator[tuple[str, str]]:
    """Process a single uploaded file: clean, delta-detect, stage. Yields (event, data) tuples."""
    file_type_key: str = {
        DataImportFile.FileType.PATIENTS: "patients",
        DataImportFile.FileType.REFERRALS: "referrals",
        DataImportFile.FileType.ODREFERRALS: "odreferrals",
        DataImportFile.FileType.ENCOUNTERS: "encounters",
    }.get(import_file.file_type, "patients")

    clean_fn = {
        "patients": service.clean_patients,
        "referrals": service.clean_referrals,
        "odreferrals": service.clean_odreferrals,
        "encounters": service.clean_encounters,
    }[file_type_key]

    yield ("log", f"\n▶ Processing {import_file.original_filename}...")

    import_file.file.seek(0)
    result = clean_fn(import_file.file)

    for line in result.log:
        yield ("log", f"  {line}")

    import_file.row_count = len(result.df)
    import_file.save(update_fields=["row_count"])

    yield ("log", f"  Comparing against production {file_type_key}...")

    core_model = CORE_MODELS[file_type_key]
    pk_field = CORE_PK_FIELDS[file_type_key]
    existing_pks = set(core_model.objects.values_list(pk_field, flat=True))

    id_col = ID_COLUMN[file_type_key]
    staging_model = STAGING_MODELS[file_type_key]
    staging_model.objects.filter(batch=batch).delete()

    # Pre-fetch existing production rows for field-level change detection.
    # We load them into a dict keyed by PK so we can compare field-by-field.
    exclude_staging = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
    compare_fields = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_staging
    ]
    existing_rows: dict[int, dict[str, Any]] = {}
    if existing_pks:
        core_qs = core_model.objects.filter(**{f"{pk_field}__in": existing_pks})
        for obj in core_qs.iterator():
            pk_val = getattr(obj, pk_field)
            existing_rows[pk_val] = {fname: getattr(obj, fname, None) for fname in compare_fields}

    new_count = 0
    existing_count = 0
    changed_count = 0
    warning_count = 0
    raw_records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in result.df.to_dict("records")
    ]

    staging_instances = []
    for i, record in enumerate(raw_records):
        source_id = int(record[id_col])

        row_status, notes_parts = _classify_row(
            source_id,
            i,
            record,
            id_col,
            file_type_key,
            existing_pks,
            existing_rows,
            staging_model,
            compare_fields,
            batch,
            result,
        )

        if row_status == RowStatus.NEW:
            new_count += 1
        elif row_status == RowStatus.CHANGED:
            changed_count += 1
        elif row_status == RowStatus.EXISTING:
            existing_count += 1
        elif row_status == RowStatus.WARNING:
            warning_count += 1

        staging_kwargs = _build_staging_kwargs(
            file_type_key, record, id_col, batch, row_status, notes_parts
        )
        staging_instances.append(staging_model(**staging_kwargs))

    staging_model.objects.bulk_create(staging_instances, batch_size=500)

    summary = (
        f"  ✓ {new_count} new, {changed_count} changed, {existing_count} unchanged, "
        f"{warning_count} warnings — staged for review"
    )
    yield ("log", summary)


@staff_member_required
def process_stream(request: HttpRequest, batch_id: int) -> StreamingHttpResponse:
    """SSE endpoint that runs ETL processing and streams log output."""
    batch = get_object_or_404(DataImportBatch, pk=batch_id)

    def event_stream() -> Iterator[str]:
        try:
            yield _sse("status", "Processing started...")
            batch.status = DataImportBatch.Status.PROCESSING
            batch.save(update_fields=["status"])

            service = DataCleaningService()
            log_lines: list[str] = []

            for import_file in DataImportFile.objects.filter(batch=batch):
                for event, data in _process_single_file(import_file, batch, service):
                    yield _sse(event, data)
                    log_lines.append(data)

            # Save processing log
            ProcessingLog.objects.create(
                batch=batch,
                content="\n".join(log_lines),
            )

            batch.status = DataImportBatch.Status.REVIEW
            batch.save(update_fields=["status"])
            yield _sse("log", "\n✓ Processing complete. Ready for review.")
            yield _sse("done", "complete")

        except Exception as e:
            batch.status = DataImportBatch.Status.FAILED
            batch.save(update_fields=["status"])
            yield _sse("error", f"Processing failed: {e}")
            yield _sse("log", traceback.format_exc())

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# ======================================================================
# Review view
# ======================================================================
@staff_member_required
def review(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    active_tab = request.GET.get("tab", "patients")

    # Gather summary stats for each dataset
    datasets: dict[str, dict[str, Any]] = {}
    for key, model in STAGING_MODELS.items():
        qs = model.objects.filter(batch=batch)
        total = qs.count()
        if total > 0:
            datasets[key] = {
                "total": total,
                "new": qs.filter(row_status=RowStatus.NEW).count(),
                "existing": qs.filter(row_status=RowStatus.EXISTING).count(),
                "changed": qs.filter(row_status=RowStatus.CHANGED).count(),
                "warning": qs.filter(row_status=RowStatus.WARNING).count(),
                "error": qs.filter(row_status=RowStatus.ERROR).count(),
                "schema": SCHEMA_INFO.get(key, {}),
            }

    # Has errors that block commit?
    has_errors = any(d["error"] > 0 for d in datasets.values())

    return render(
        request,
        "data_import/review.html",
        {
            "batch": batch,
            "datasets": datasets,
            "active_tab": active_tab,
            "has_errors": has_errors,
        },
    )


@staff_member_required
def review_table(request: HttpRequest, batch_id: int, dataset: str) -> HttpResponse:
    """HTMX partial: paginated data table for a dataset."""
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    staging_model = STAGING_MODELS.get(dataset)
    if not staging_model:
        return HttpResponse("Invalid dataset", status=400)

    # Filters
    status_filter = request.GET.get("status", "")
    filter_empty = status_filter == "empty"

    qs = staging_model.objects.filter(batch=batch)

    if status_filter and not filter_empty:
        qs = qs.filter(row_status=status_filter)

    # Get field names for the table header (exclude batch, row_status, etc.)
    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes"}
    field_names = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    ]

    # For "has empty fields" filter, build Q objects for string/nullable fields
    if filter_empty:
        from django.db.models import Q

        empty_q = Q()
        for fname in field_names:
            field_obj = staging_model._meta.get_field(fname)
            if isinstance(field_obj, dm.CharField | dm.TextField):
                empty_q |= Q(**{fname: ""})
            if field_obj.null:
                empty_q |= Q(**{f"{fname}__isnull": True})
        qs = qs.filter(empty_q)

    # Sorting
    sort_by = request.GET.get("sort", "")
    sort_dir = request.GET.get("dir", "asc")
    if sort_by and sort_by in field_names:
        order_prefix = "-" if sort_dir == "desc" else ""
        qs = qs.order_by(f"{order_prefix}{sort_by}")
    else:
        qs = qs.order_by("source_id")
        sort_by = ""
        sort_dir = "asc"

    # Pagination
    page = int(request.GET.get("page", 1))
    per_page = 50
    total = qs.count()
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))  # clamp to valid range
    rows = list(qs[(page - 1) * per_page : page * per_page])

    # Build page number list with ellipsis gaps
    page_numbers = _build_page_numbers(page, total_pages)

    # Build row data with per-cell metadata
    row_data = []
    for row in rows:
        row_data.append(_build_row_context(row, field_names))

    schema = SCHEMA_INFO.get(dataset, {})

    return render(
        request,
        "data_import/_data_table.html",
        {
            "batch": batch,
            "dataset": dataset,
            "field_names": field_names,
            "rows": row_data,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "status_filter": status_filter,
            "sort_by": sort_by,
            "sort_dir": sort_dir,
            "page_numbers": page_numbers,
            "schema": schema,
        },
    )


# ======================================================================
# Inline row editing
# ======================================================================
@staff_member_required
def row_edit(request: HttpRequest, batch_id: int, dataset: str, row_pk: int) -> HttpResponse:
    """Return an inline edit form for a single staging row."""
    staging_model = STAGING_MODELS.get(dataset)
    if not staging_model:
        return HttpResponse("Invalid dataset", status=400)

    row = get_object_or_404(staging_model, pk=row_pk, batch_id=batch_id)

    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes"}
    field_names = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    ]

    cells_dict: dict[str, Any] = {}
    for fname in field_names:
        cells_dict[fname] = getattr(row, fname, "")

    return render(
        request,
        "data_import/_row_edit.html",
        {
            "batch": {"pk": batch_id},
            "dataset": dataset,
            "row": {"pk": row.pk, "status": row.row_status, "cells_dict": cells_dict},
            "field_names": field_names,
        },
    )


@require_POST
@staff_member_required
def row_update(request: HttpRequest, batch_id: int, dataset: str, row_pk: int) -> HttpResponse:
    """Save inline edits for a staging row."""
    staging_model = STAGING_MODELS.get(dataset)
    if not staging_model:
        return HttpResponse("Invalid dataset", status=400)

    row = get_object_or_404(staging_model, pk=row_pk, batch_id=batch_id)

    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes"}
    field_names = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    ]

    for fname in field_names:
        if fname in request.POST:
            field_obj = staging_model._meta.get_field(fname)
            raw_value = request.POST[fname]
            try:
                if isinstance(field_obj, dm.IntegerField):
                    setattr(row, fname, int(raw_value) if raw_value else None)
                elif isinstance(field_obj, dm.FloatField):
                    setattr(row, fname, float(raw_value) if raw_value else None)
                elif isinstance(field_obj, dm.BooleanField):
                    setattr(
                        row, fname, raw_value.lower() in ("true", "1", "yes") if raw_value else None
                    )
                elif isinstance(field_obj, dm.DateField | dm.DateTimeField):
                    setattr(row, fname, raw_value if raw_value else None)
                else:
                    setattr(row, fname, raw_value)
            except (ValueError, TypeError):
                setattr(row, fname, raw_value)

    row.save()

    return render(
        request,
        "data_import/_row_display.html",
        {
            "batch": {"pk": batch_id},
            "dataset": dataset,
            "row": _build_row_context(row, field_names),
            "field_names": field_names,
        },
    )


@staff_member_required
def row_cancel(request: HttpRequest, batch_id: int, dataset: str, row_pk: int) -> HttpResponse:
    """Cancel editing and return the display row."""
    staging_model = STAGING_MODELS.get(dataset)
    if not staging_model:
        return HttpResponse("Invalid dataset", status=400)

    row = get_object_or_404(staging_model, pk=row_pk, batch_id=batch_id)

    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes"}
    field_names = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    ]

    return render(
        request,
        "data_import/_row_display.html",
        {
            "batch": {"pk": batch_id},
            "dataset": dataset,
            "row": _build_row_context(row, field_names),
            "field_names": field_names,
        },
    )


# ======================================================================
# Single-cell inline editing
# ======================================================================
@require_POST
@staff_member_required
def cell_update(
    request: HttpRequest, batch_id: int, dataset: str, row_pk: int, field_name: str
) -> HttpResponse:
    """Save a single cell value (HTMX inline edit). Returns just the updated cell HTML."""
    staging_model = STAGING_MODELS.get(dataset)
    if not staging_model:
        return HttpResponse("Invalid dataset", status=400)

    row = get_object_or_404(staging_model, pk=row_pk, batch_id=batch_id)

    # Validate field name
    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
    valid_fields = {
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    }
    if field_name not in valid_fields:
        return HttpResponse("Invalid field", status=400)

    raw_value = request.POST.get("value", "")
    field_obj = staging_model._meta.get_field(field_name)

    try:
        if isinstance(field_obj, dm.IntegerField):
            setattr(row, field_name, int(raw_value) if raw_value else None)
        elif isinstance(field_obj, dm.FloatField):
            setattr(row, field_name, float(raw_value) if raw_value else None)
        elif isinstance(field_obj, dm.BooleanField):
            setattr(
                row,
                field_name,
                raw_value.lower() in ("true", "1", "yes") if raw_value else None,
            )
        elif isinstance(field_obj, dm.DateField | dm.DateTimeField):
            setattr(row, field_name, raw_value if raw_value else None)
        else:
            setattr(row, field_name, raw_value)
    except (ValueError, TypeError):
        setattr(row, field_name, raw_value)

    row.save()

    # Return the cell content HTML
    val = getattr(row, field_name, "")
    is_empty = val is None or str(val).strip() == ""
    is_placeholder = str(val).strip() in PLACEHOLDER_VALUES

    return render(
        request,
        "data_import/_cell.html",
        {
            "cell": {
                "name": field_name,
                "value": val,
                "is_empty": is_empty,
                "is_placeholder": is_placeholder,
            },
            "batch_pk": batch_id,
            "dataset": dataset,
            "row_pk": row_pk,
        },
    )


# ======================================================================
# Production reference data
# ======================================================================
@staff_member_required
def production_reference(request: HttpRequest, batch_id: int) -> HttpResponse:
    """HTMX partial: show production DB stats — last record ID, most recent dates, counts."""
    ref_data: dict[str, dict[str, Any]] = {}

    date_fields = {
        "patients": "created_date",
        "referrals": "date_received",
        "odreferrals": "od_date",
        "encounters": "encounter_date",
    }

    for key, core_model in CORE_MODELS.items():
        pk_field = CORE_PK_FIELDS[key]
        total = core_model.objects.count()
        if total == 0:
            ref_data[key] = {"total": 0, "max_id": None, "latest_date": None}
            continue

        max_id = core_model.objects.aggregate(max_id=dm.Max(pk_field))["max_id"]
        date_field = date_fields.get(key)
        latest_date = None
        if date_field:
            latest_date = core_model.objects.aggregate(latest=dm.Max(date_field))["latest"]

        ref_data[key] = {
            "total": total,
            "max_id": max_id,
            "latest_date": latest_date,
        }

    return render(
        request,
        "data_import/_production_ref.html",
        {"ref_data": ref_data, "batch_id": batch_id},
    )


# ======================================================================
# Batch operations
# ======================================================================
@require_POST
@staff_member_required
def batch_update_field(request: HttpRequest, batch_id: int, dataset: str) -> HttpResponse:
    """Apply a field value change to selected rows."""
    staging_model = STAGING_MODELS.get(dataset)
    if not staging_model:
        return HttpResponse("Invalid dataset", status=400)

    row_ids = request.POST.getlist("row_ids")
    field_name = request.POST.get("field_name", "")
    field_value = request.POST.get("field_value", "")

    if not row_ids or not field_name:
        messages.error(request, "Select rows and specify a field to update.")
        return redirect("data_import:review", batch_id=batch_id)

    # Validate field
    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
    valid_fields = {
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    }
    if field_name not in valid_fields:
        messages.error(request, f"Invalid field: {field_name}")
        return redirect("data_import:review", batch_id=batch_id)

    updated = staging_model.objects.filter(
        batch_id=batch_id, pk__in=[int(x) for x in row_ids]
    ).update(**{field_name: field_value})

    messages.success(request, f"Updated {updated} rows: {field_name} = '{field_value}'")

    if request.headers.get("HX-Request"):
        return HttpResponse(
            f'<div class="text-emerald-400 text-sm py-2">✓ Updated {updated} rows</div>'
        )
    return redirect("data_import:review", batch_id=batch_id)


# ======================================================================
# Commit view
# ======================================================================
@staff_member_required
def commit_view(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(DataImportBatch, pk=batch_id)

    if batch.status != DataImportBatch.Status.REVIEW:
        messages.error(request, "This batch is not ready for commit.")
        return redirect("data_import:review", batch_id=batch_id)

    # Collect new + changed counts
    commit_summary: dict[str, dict[str, int]] = {}
    has_errors = False
    for key, model in STAGING_MODELS.items():
        qs = model.objects.filter(batch=batch)
        if qs.exists():
            error_count = qs.filter(row_status=RowStatus.ERROR).count()
            if error_count > 0:
                has_errors = True
            commit_summary[key] = {
                "new": qs.filter(row_status=RowStatus.NEW).count(),
                "changed": qs.filter(row_status=RowStatus.CHANGED).count(),
            }

    if request.method == "POST":
        if has_errors:
            messages.error(request, "Cannot commit — there are rows with errors. Fix them first.")
            return redirect("data_import:review", batch_id=batch_id)

        confirm_text = request.POST.get("confirm_text", "")
        if confirm_text != "COMMIT":
            messages.error(request, "Type COMMIT to confirm.")
            return render(
                request,
                "data_import/commit.html",
                {
                    "batch": batch,
                    "commit_summary": commit_summary,
                    "has_errors": has_errors,
                },
            )

        # Execute the commit
        try:
            _do_commit(batch, commit_summary)
            messages.success(request, "Data committed to production successfully!")
            return redirect("data_import:post_commit", batch_id=batch_id)
        except Exception as e:
            messages.error(request, f"Commit failed: {e}")
            batch.status = DataImportBatch.Status.FAILED
            batch.save(update_fields=["status"])
            return redirect("data_import:review", batch_id=batch_id)

    return render(
        request,
        "data_import/commit.html",
        {
            "batch": batch,
            "commit_summary": commit_summary,
            "has_errors": has_errors,
        },
    )


@staff_member_required
def post_commit(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    return render(request, "data_import/post_commit.html", {"batch": batch})


@require_POST
@staff_member_required
def purge_staging(request: HttpRequest, batch_id: int) -> HttpResponse:
    """Purge staging data for a committed batch."""
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    for model in STAGING_MODELS.values():
        model.objects.filter(batch=batch).delete()
    messages.success(request, "Staging data purged.")
    return redirect("data_import:batch_list")


@require_POST
@staff_member_required
def batch_delete(request: HttpRequest, batch_id: int) -> HttpResponse:
    """Delete a batch and all its staging data + uploaded files."""
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    label = f"Import #{batch.pk}"

    # Delete staging rows
    for model in STAGING_MODELS.values():
        model.objects.filter(batch=batch).delete()

    # Delete uploaded files from disk
    for import_file in DataImportFile.objects.filter(batch=batch):
        if import_file.file:
            import_file.file.delete(save=False)
        import_file.delete()

    # Delete associated logs
    ProcessingLog.objects.filter(batch=batch).delete()

    batch.delete()
    messages.success(request, f"{label} deleted.")

    if request.headers.get("HX-Request"):
        return HttpResponse("")
    return redirect("data_import:batch_list")


# ======================================================================
# Log management
# ======================================================================
@staff_member_required
def log_list(request: HttpRequest) -> HttpResponse:
    logs = ProcessingLog.objects.select_related("batch").all()
    return render(request, "data_import/logs.html", {"logs": logs})


@require_POST
@staff_member_required
def log_delete(request: HttpRequest, log_id: int) -> HttpResponse:
    log = get_object_or_404(ProcessingLog, pk=log_id)
    log.delete()
    messages.success(request, "Log deleted.")
    if request.headers.get("HX-Request"):
        return HttpResponse("")
    return redirect("data_import:log_list")


# ======================================================================
# Internal helpers
# ======================================================================

# Values that indicate empty/suspicious data for cell highlighting
PLACEHOLDER_VALUES = {"Not disclosed", "Unknown", "None", "N/A", "n/a", "Select Race"}


def _build_row_context(row: Any, field_names: list[str]) -> dict[str, Any]:
    """Build a unified row context dict with per-cell metadata for templates."""
    cells_list: list[dict[str, Any]] = []
    cells_dict: dict[str, Any] = {}
    empty_count = 0
    for fname in field_names:
        val = getattr(row, fname, "")
        cells_dict[fname] = val
        is_empty = val is None or str(val).strip() == ""
        is_placeholder = str(val).strip() in PLACEHOLDER_VALUES
        if is_empty:
            empty_count += 1
        cells_list.append(
            {
                "name": fname,
                "value": val,
                "is_empty": is_empty,
                "is_placeholder": is_placeholder,
            }
        )
    return {
        "pk": row.pk,
        "status": row.row_status,
        "validation_notes": row.validation_notes,
        "cells": cells_list,
        "cells_dict": cells_dict,
        "empty_count": empty_count,
    }


def _sse(event: str, data: str) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_page_numbers(current: int, total: int) -> list[int | str]:
    """Build a page number list with '...' ellipsis for gaps.

    Always shows first/last page, pages around the current page, and uses
    '...' to represent skipped ranges.  Keeps the pagination bar compact
    even for hundreds of pages.
    """
    if total <= 9:
        return list(range(1, total + 1))

    pages: list[int | str] = []
    # Always include first 2 pages, last 2 pages, and a window around current
    include = {1, 2, total - 1, total}
    for p in range(max(1, current - 2), min(total, current + 2) + 1):
        include.add(p)

    prev = 0
    for p in sorted(include):
        if p - prev > 1:
            pages.append("...")
        pages.append(p)
        prev = p

    return pages


def _detect_field_changes(
    staging_model: type[dm.Model],
    compare_fields: list[str],
    staging_kwargs: dict[str, Any],
    prod_row: dict[str, Any],
) -> list[str]:
    """Compare incoming staged values against production row, return human-readable diffs.

    Only fields that differ produce an entry like ``"age: 64 → 65"``.  Fields
    that are identical or where both sides are empty are skipped.
    """
    diffs: list[str] = []
    for fname in compare_fields:
        new_val = staging_kwargs.get(fname)
        old_val = prod_row.get(fname)

        # Normalize for comparison: treat None and "" as equivalent for strings
        field_obj = staging_model._meta.get_field(fname)
        if isinstance(field_obj, dm.CharField | dm.TextField):
            new_cmp = str(new_val).strip() if new_val is not None else ""
            old_cmp = str(old_val).strip() if old_val is not None else ""
        else:
            new_cmp = new_val
            old_cmp = old_val

        if new_cmp != old_cmp:
            diffs.append(f"{fname}: {old_val} → {new_val}")
    return diffs


def _classify_row(
    source_id: int,
    row_index: int,
    record: dict[str, Any],
    id_col: str,
    file_type: str,
    existing_pks: set[Any],
    existing_rows: dict[int, dict[str, Any]],
    staging_model: type[dm.Model],
    compare_fields: list[str],
    batch: DataImportBatch,
    result: Any,
) -> tuple[str, list[str]]:
    """Determine row status and build validation notes for a single record."""
    notes_parts: list[str] = []
    diffs: list[str] = []

    if source_id in existing_pks:
        staging_kwargs_preview = _build_staging_kwargs(
            file_type, record, id_col, batch, RowStatus.EXISTING, []
        )
        prod_row = existing_rows.get(source_id, {})
        diffs = _detect_field_changes(
            staging_model, compare_fields, staging_kwargs_preview, prod_row
        )
        row_status = RowStatus.CHANGED if diffs else RowStatus.EXISTING
    elif row_index in result.row_errors:
        row_status = RowStatus.ERROR
    elif row_index in result.row_warnings:
        row_status = RowStatus.WARNING
    else:
        row_status = RowStatus.NEW

    if diffs:
        notes_parts.append("Changed: " + "; ".join(diffs))
    if row_index in result.row_warnings:
        notes_parts.extend(result.row_warnings[row_index])
    if row_index in result.row_errors:
        notes_parts.extend(result.row_errors[row_index])

    return row_status, notes_parts


def _build_staging_kwargs(
    file_type: str,
    record: dict[str, Any],
    id_col: str,
    batch: DataImportBatch,
    row_status: str,
    notes_parts: list[str],
) -> dict[str, Any]:
    """Build kwargs dict for creating a staging model instance from a cleaned record."""
    kwargs: dict[str, Any] = {
        "batch": batch,
        "row_status": row_status,
        "validation_notes": "; ".join(notes_parts) if notes_parts else "",
        "source_id": int(record[id_col]),
    }

    # Copy all fields from the record, skipping the ID column (goes to source_id)
    staging_model = STAGING_MODELS[file_type]
    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
    model_field_names = {
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    }

    for col_name, value in record.items():
        if col_name == id_col:
            continue
        # Map the cleaned DF column to the staging model field
        field_name = col_name
        # Handle three_c_client mapping
        if col_name == "three_c_client" and "three_c_client" in model_field_names:
            field_name = "three_c_client"

        if field_name in model_field_names:
            # Handle type coercion
            field_obj = staging_model._meta.get_field(field_name)
            kwargs[field_name] = _coerce_value(field_obj, value)

    return kwargs


def _coerce_value(field: Any, value: Any) -> Any:
    """Coerce a value from the DataFrame to match the Django model field type."""
    if pd.isna(value) or value == "":
        return "" if isinstance(field, dm.CharField | dm.TextField) else None
    return _coerce_non_empty(field, value)


def _coerce_non_empty(field: Any, value: Any) -> Any:
    """Coerce a non-empty, non-NaN value to match the Django field type."""
    if isinstance(field, dm.BooleanField):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        return False if s in ("false", "0", "no") else None

    if isinstance(field, dm.IntegerField):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    if isinstance(field, dm.FloatField):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    return value


@transaction.atomic
def _do_commit(batch: DataImportBatch, commit_summary: dict[str, dict[str, int]]) -> None:
    """Commit new and changed staging rows to production tables inside a transaction."""
    log_lines = [f"Commit started at {timezone.now():%Y-%m-%d %H:%M:%S}"]

    total_new = 0
    total_updated = 0

    for file_type, counts in commit_summary.items():
        new_count = counts.get("new", 0)
        changed_count = counts.get("changed", 0)

        if new_count == 0 and changed_count == 0:
            log_lines.append(f"  {file_type}: 0 new, 0 changed — skipping")
            continue

        staging_model = STAGING_MODELS[file_type]
        core_model = CORE_MODELS[file_type]
        pk_field = CORE_PK_FIELDS[file_type]

        exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
        core_field_names = {f.name for f in core_model._meta.get_fields() if hasattr(f, "name")}

        # --- Insert NEW rows ---
        if new_count > 0:
            new_rows = staging_model.objects.filter(batch=batch, row_status=RowStatus.NEW)
            instances = []
            for staging_row in new_rows:
                kwargs = _staging_to_core_kwargs(
                    staging_model, staging_row, pk_field, exclude_fields, core_field_names
                )
                instances.append(core_model(**kwargs))
            core_model.objects.bulk_create(instances, batch_size=500)
            total_new += len(instances)
            log_lines.append(f"  {file_type}: inserted {len(instances)} new rows")

        # --- Update CHANGED rows ---
        if changed_count > 0:
            changed_rows = staging_model.objects.filter(batch=batch, row_status=RowStatus.CHANGED)
            updated = 0
            for staging_row in changed_rows:
                kwargs = _staging_to_core_kwargs(
                    staging_model, staging_row, pk_field, exclude_fields, core_field_names
                )
                pk_val = kwargs.pop(pk_field)
                core_model.objects.filter(**{pk_field: pk_val}).update(**kwargs)
                updated += 1
            total_updated += updated
            log_lines.append(f"  {file_type}: updated {updated} changed rows")

    # Update batch metadata
    batch.status = DataImportBatch.Status.COMMITTED
    batch.committed_at = timezone.now()
    batch.committed_patients = commit_summary.get("patients", {}).get(
        "new", 0
    ) + commit_summary.get("patients", {}).get("changed", 0)
    batch.committed_referrals = commit_summary.get("referrals", {}).get(
        "new", 0
    ) + commit_summary.get("referrals", {}).get("changed", 0)
    batch.committed_odreferrals = commit_summary.get("odreferrals", {}).get(
        "new", 0
    ) + commit_summary.get("odreferrals", {}).get("changed", 0)
    batch.committed_encounters = commit_summary.get("encounters", {}).get(
        "new", 0
    ) + commit_summary.get("encounters", {}).get("changed", 0)
    batch.save()

    log_lines.append(f"  Total: {total_new} inserted, {total_updated} updated")
    ProcessingLog.objects.create(batch=batch, content="\n".join(log_lines))


def _staging_to_core_kwargs(
    staging_model: type[dm.Model],
    staging_row: Any,
    pk_field: str,
    exclude_fields: set[str],
    core_field_names: set[str],
) -> dict[str, Any]:
    """Build a dict of kwargs for creating/updating a core model instance from a staging row."""
    kwargs: dict[str, Any] = {pk_field: staging_row.source_id}
    staging_field_names = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    ]
    for fname in staging_field_names:
        if fname in core_field_names:
            kwargs[fname] = getattr(staging_row, fname)
    return kwargs
