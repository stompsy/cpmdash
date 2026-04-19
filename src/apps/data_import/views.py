from __future__ import annotations

import io
import json
import queue
import threading
import time
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

from .etl_service import CleaningResult, DataCleaningService
from .forms import DataUploadForm, DataUploadToBatchForm
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
            ("address", "Address", "Text (255)", "Street address for geocoding"),
            ("created_date", "Created Date", "Date", "Record creation date"),
            ("modified_date", "Modified Date", "Date", "Last modification"),
            ("marital_status", "Marital Status", "Text (50)", ""),
            ("veteran_status", "Veteran Status", "Text (50)", ""),
            ("aud", "AUD", "Boolean", "Alcohol use disorder"),
            ("three_c_client", "3C Client", "Boolean", ""),
            ("latitude", "Latitude", "Float", "Geocoded latitude"),
            ("longitude", "Longitude", "Float", "Geocoded longitude"),
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

# Fields to skip during change detection — these exist in the staging model
# but the CSV never supplies them, so comparing against production values
# would flag every existing row as "changed" (e.g. lat/long are populated
# in prod via geocoding but the raw CSV doesn't contain them).
CHANGE_DETECT_SKIP: dict[str, set[str]] = {
    "patients": {"latitude", "longitude", "address"},
    "referrals": set(),
    "odreferrals": {"lat", "long"},
    "encounters": set(),
}


# Clallam County zip-code centroids (approximate) for geocoding patients
# that have a zip_code but no lat/long.  Avoids external API calls —
# centroid-level precision is sufficient for mapping and analytics.
#
# "Homeless/Transient" and similar non-zip values map to the Serenity House
# location at 2321 W 18th St, Port Angeles — the CPM program's home base.
_SERENITY_HOUSE = (48.1224, -123.4912)

ZIP_CENTROIDS: dict[str, tuple[float, float]] = {
    "98305": (47.9073, -124.6353),  # La Push
    "98324": (48.0734, -123.1168),  # Gardiner
    "98326": (48.1572, -123.8481),  # Joyce
    "98331": (47.9498, -124.3505),  # Forks
    "98343": (48.1889, -124.2467),  # Pysht
    "98357": (48.3651, -124.6248),  # Neah Bay
    "98362": (48.1181, -123.4307),  # Port Angeles
    "98363": (48.0976, -123.7403),  # Port Angeles (rural)
    "98381": (48.2653, -124.3923),  # Sekiu / Clallam Bay
    "98382": (48.0795, -123.1018),  # Sequim
    "98386": (47.8225, -122.8268),  # Quilcene (Jefferson Co. — USGS GNIS ref)
    "98339": (48.0311, -122.8103),  # Port Hadlock (Jefferson Co.)
    "98368": (48.1170, -122.7604),  # Port Townsend (Jefferson Co.)
    "Homeless": _SERENITY_HOUSE,
    "Homeless/Transient": _SERENITY_HOUSE,
    "Transient": _SERENITY_HOUSE,
    "Not disclosed": _SERENITY_HOUSE,
}


# ======================================================================
# History view
# ======================================================================
@staff_member_required
def batch_list(request: HttpRequest) -> HttpResponse:
    batches = DataImportBatch.objects.select_related("created_by").all()
    return render(request, "data_import/history.html", {"batches": batches})


# ======================================================================
# Upload view — new batch (patients first, always)
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
            DataImportFile.objects.create(
                batch=batch,
                file_type=DataImportFile.FileType.PATIENTS,
                file=uploaded,
                original_filename=uploaded.name or "unknown",
            )

            return redirect("data_import:process", batch_id=batch.pk)
    else:
        form = DataUploadForm()

    return render(request, "data_import/upload.html", {"form": form})


# ======================================================================
# Upload view — add dataset to existing batch
# ======================================================================
@staff_member_required
def upload_to_batch(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(DataImportBatch, pk=batch_id)

    # Which file types are already uploaded for this batch?
    existing_types = set(
        DataImportFile.objects.filter(batch=batch).values_list("file_type", flat=True)
    )
    # Remaining choices (exclude already-uploaded types)
    remaining_choices = [
        (val, label) for val, label in DataImportFile.FileType.choices if val not in existing_types
    ]

    if not remaining_choices:
        messages.info(request, "All dataset types have already been uploaded for this batch.")
        return redirect("data_import:review", batch_id=batch.pk)

    if request.method == "POST":
        form = DataUploadToBatchForm(
            request.POST, request.FILES, file_type_choices=remaining_choices
        )
        if form.is_valid():
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
        form = DataUploadToBatchForm(file_type_choices=remaining_choices)

    return render(
        request,
        "data_import/upload.html",
        {
            "form": form,
            "batch": batch,
            "existing_types": existing_types,
        },
    )


# ======================================================================
# Processing view + SSE stream
# ======================================================================
@staff_member_required
def process_view(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(DataImportBatch, pk=batch_id)
    return render(request, "data_import/process.html", {"batch": batch})


class _StreamingLog(list):
    """A list subclass that pushes each appended item to a ``queue.Queue``.

    Pass an instance of this as the ``log`` parameter to any ``clean_*()``
    method.  While the clean function runs in a background thread, the SSE
    generator on the main thread drains the queue and yields messages to the
    browser in real-time — no more frozen console during geocoding.
    """

    def __init__(self, q: queue.Queue[str | None]) -> None:
        super().__init__()
        self._queue = q

    def append(self, item: str) -> None:  # type: ignore[override]
        super().append(item)
        self._queue.put(item)


def _process_single_file(
    import_file: DataImportFile,
    batch: DataImportBatch,
    service: DataCleaningService,
    patients_df: pd.DataFrame | None = None,
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

    # Read file into memory for thread safety — Django FieldFile
    # objects aren't safe to share across threads.
    import_file.file.seek(0)
    file_bytes = import_file.file.read()
    buf = io.BytesIO(file_bytes)

    # --------------- threaded clean with real-time log streaming -------
    # StreamingLog is a list subclass that pushes each append() to a
    # queue, allowing the main SSE generator thread to yield messages
    # to the browser while the clean function is still running.
    q: queue.Queue[str | None] = queue.Queue()
    streaming_log = _StreamingLog(q)
    result_holder: list[CleaningResult | None] = [None]
    error_holder: list[BaseException | None] = [None]

    def _run_clean() -> None:
        try:
            if file_type_key in ("referrals", "odreferrals", "encounters"):
                result_holder[0] = clean_fn(buf, patients_df=patients_df, log=streaming_log)
            else:
                result_holder[0] = clean_fn(buf, log=streaming_log)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            q.put(None)  # sentinel — tells the reader the thread is done

    t_start = time.monotonic()
    thread = threading.Thread(target=_run_clean, daemon=True)
    thread.start()

    # Drain the queue in real-time, yielding SSE messages as they arrive.
    # Use a short timeout so we can emit keepalive comments between
    # messages — Gunicorn sync workers can't send heartbeats while
    # blocked on a lock, so an indefinite q.get() causes WORKER TIMEOUT
    # kills during slow geocoding (Nominatim: 1 req/sec × 200+ addrs).
    while True:
        try:
            msg = q.get(timeout=5)
        except queue.Empty:
            # Yield an SSE comment to keep the connection alive.
            # Comments (lines starting with ':') are ignored by the
            # browser's EventSource but reset Gunicorn's idle timer.
            yield ("keepalive", "")
            continue
        if msg is None:
            break
        yield ("log", f"  {msg}")

    thread.join()

    if error_holder[0]:
        raise error_holder[0]  # propagate to process_stream's try/except

    result = result_holder[0]
    assert result is not None  # noqa: S101

    elapsed_clean = time.monotonic() - t_start
    yield ("log", f"  Cleaning finished ({elapsed_clean:.1f}s)")

    # --------------- delta-detection & staging -------------------------
    import_file.row_count = len(result.df)
    import_file.save(update_fields=["row_count"])

    yield ("log", f"  ⏳ Comparing against production {file_type_key}...")

    yield from _stage_records(file_type_key, result, batch)


def _save_file_log(
    batch: DataImportBatch,
    import_file: DataImportFile,
    log_lines: list[str],
) -> None:
    """Extract summary and persist a per-file processing log."""
    summary = ""
    for line in reversed(log_lines):
        if "✓" in line and ("new" in line or "staged" in line):
            summary = line.strip()
            break
    ProcessingLog.objects.create(
        batch=batch,
        file_type=import_file.file_type,
        original_filename=import_file.original_filename,
        content="\n".join(log_lines),
        summary=summary,
    )


def _stage_records(
    file_type_key: str,
    result: CleaningResult,
    batch: DataImportBatch,
) -> Iterator[tuple[str, str]]:
    """Delta-detect against production and create staging rows. Yields progress."""
    core_model = CORE_MODELS[file_type_key]
    pk_field = CORE_PK_FIELDS[file_type_key]
    existing_pks = set(core_model.objects.values_list(pk_field, flat=True))

    id_col = ID_COLUMN[file_type_key]
    staging_model = STAGING_MODELS[file_type_key]
    staging_model.objects.filter(batch=batch).delete()

    # Pre-fetch existing production rows for field-level change detection.
    exclude_staging = {"id", "batch", "batch_id", "row_status", "validation_notes", "source_id"}
    skip_fields = CHANGE_DETECT_SKIP.get(file_type_key, set())
    compare_fields = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_staging and f.name not in skip_fields
    ]
    fetch_fields = compare_fields + [f for f in skip_fields]
    existing_rows: dict[int, dict[str, Any]] = {}
    if existing_pks:
        core_qs = core_model.objects.filter(**{f"{pk_field}__in": existing_pks})
        for obj in core_qs.iterator():
            pk_val = getattr(obj, pk_field)
            existing_rows[pk_val] = {fname: getattr(obj, fname, None) for fname in fetch_fields}

    new_count = 0
    existing_count = 0
    changed_count = 0
    warning_count = 0
    raw_records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in result.df.to_dict("records")
    ]

    staging_instances = []
    total_records = len(raw_records)
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

        if i in result.row_warnings:
            warning_count += 1

        staging_kwargs = _build_staging_kwargs(
            file_type_key, record, id_col, batch, row_status, notes_parts
        )
        _backfill_skip_fields(staging_kwargs, source_id, existing_rows, skip_fields)
        staging_instances.append(staging_model(**staging_kwargs))

        # Yield progress every 100 rows for larger datasets
        if total_records > 100 and (i + 1) % 100 == 0:
            yield ("log", f"  ⏳ Staging: {i + 1}/{total_records} records...")

    staging_model.objects.bulk_create(staging_instances, batch_size=500)

    # For patients, immediately backfill missing coords from zip-code centroids.
    # This runs against ALL rows in the batch — not just whatever page the user
    # happens to view in the review UI — so no records slip through un-geocoded.
    if file_type_key == "patients":
        all_patient_rows = list(staging_model.objects.filter(batch=batch))
        _geocode_patient_rows(all_patient_rows)

    summary = (
        f"  ✓ {new_count} new, {changed_count} changed, {existing_count} unchanged, "
        f"{warning_count} warnings — staged for review"
    )
    yield ("log", summary)


def _stream_file_events(
    import_file: DataImportFile,
    batch: DataImportBatch,
    service: DataCleaningService,
    patients_df: pd.DataFrame | None,
) -> Iterator[str]:
    """Process one file and yield SSE strings, including keepalive comments."""
    log_lines: list[str] = []
    for event, data in _process_single_file(import_file, batch, service, patients_df=patients_df):
        if event == "keepalive":
            yield ": keepalive\n\n"
            continue
        yield _sse(event, data)
        log_lines.append(data)
    _save_file_log(batch, import_file, log_lines)


@staff_member_required
def process_stream(request: HttpRequest, batch_id: int) -> StreamingHttpResponse:
    """SSE endpoint that runs ETL processing and streams log output."""
    batch = get_object_or_404(DataImportBatch, pk=batch_id)

    def event_stream() -> Iterator[str]:
        try:
            yield _sse("status", "Processing started...")
            t_total = time.monotonic()
            batch.status = DataImportBatch.Status.PROCESSING
            batch.save(update_fields=["status"])

            service = DataCleaningService()

            unprocessed = DataImportFile.objects.filter(batch=batch, processed=False)
            if not unprocessed.exists():
                yield _sse("log", "⚠ No new files to process.")
            else:
                # Sort so patients are processed first — other datasets depend on them
                # for age lookups, pcp_agency fallback, etc.
                _FILE_TYPE_ORDER = {
                    DataImportFile.FileType.PATIENTS: 0,
                    DataImportFile.FileType.REFERRALS: 1,
                    DataImportFile.FileType.ODREFERRALS: 2,
                    DataImportFile.FileType.ENCOUNTERS: 3,
                }
                sorted_files = sorted(
                    unprocessed, key=lambda f: _FILE_TYPE_ORDER.get(f.file_type, 99)
                )

                # Build patients_df from production data (will be replaced
                # if a patients file is in this batch)
                patients_df: pd.DataFrame | None = None
                if Patients.objects.exists():
                    patients_df = pd.DataFrame.from_records(
                        Patients.objects.values("id", "age", "insurance", "pcp_agency")
                    )
                else:
                    # Check if any non-patients files are in the batch without patients
                    has_patients_file = any(
                        f.file_type == DataImportFile.FileType.PATIENTS for f in sorted_files
                    )
                    non_patient_files = [
                        f for f in sorted_files if f.file_type != DataImportFile.FileType.PATIENTS
                    ]
                    if non_patient_files and not has_patients_file:
                        yield _sse(
                            "log",
                            "⚠ No patients data in production or batch — "
                            "age/pcp lookups will be unavailable",
                        )

                for import_file in sorted_files:
                    yield from _stream_file_events(import_file, batch, service, patients_df)

                    # If we just processed patients, capture the cleaned DataFrame
                    # for use by subsequent files in this batch
                    if import_file.file_type == DataImportFile.FileType.PATIENTS:
                        staging_qs = StagingPatient.objects.filter(batch=batch).values(
                            "source_id", "age", "insurance", "pcp_agency"
                        )
                        if staging_qs.exists():
                            patients_df = pd.DataFrame.from_records(staging_qs)
                            patients_df = patients_df.rename(columns={"source_id": "id"})

                    import_file.processed = True
                    import_file.save(update_fields=["processed"])

            batch.status = DataImportBatch.Status.REVIEW
            batch.save(update_fields=["status"])
            elapsed = time.monotonic() - t_total
            yield _sse("log", f"\n✓ Processing complete. Ready for review. ({elapsed:.1f}s)")
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
                "warning": qs.exclude(validation_notes="")
                .exclude(row_status=RowStatus.ERROR)
                .count(),
                "error": qs.filter(row_status=RowStatus.ERROR).count(),
                "schema": SCHEMA_INFO.get(key, {}),
            }

    # Has errors that block commit?
    has_errors = any(d["error"] > 0 for d in datasets.values())

    # What file types still have room to add?
    existing_file_types = set(
        DataImportFile.objects.filter(batch=batch).values_list("file_type", flat=True)
    )
    all_file_types = {val for val, _ in DataImportFile.FileType.choices}
    can_add_datasets = bool(all_file_types - existing_file_types)

    return render(
        request,
        "data_import/review.html",
        {
            "batch": batch,
            "datasets": datasets,
            "active_tab": active_tab,
            "has_errors": has_errors,
            "can_add_datasets": can_add_datasets,
        },
    )


def _narrow_changed_fields(
    field_names: list[str],
    row_data: list[dict[str, Any]],
    status_filter: str,
) -> list[str]:
    """When viewing changed rows, narrow to only fields with actual changes."""
    if status_filter != "changed" or not row_data:
        return list(field_names)
    visible = {"source_id"}
    for rd in row_data:
        visible.update(rd.get("changed_fields", set()))
    return [f for f in field_names if f in visible]


def _apply_empty_filter(qs: Any, staging_model: type[dm.Model], field_names: list[str]) -> Any:
    """Filter queryset to rows that have at least one empty/null field."""
    from django.db.models import Q

    empty_q = Q()
    for fname in field_names:
        field_obj = staging_model._meta.get_field(fname)
        if isinstance(field_obj, dm.CharField | dm.TextField):
            empty_q |= Q(**{fname: ""})
        if field_obj.null:
            empty_q |= Q(**{f"{fname}__isnull": True})
    return qs.filter(empty_q)


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
        if status_filter == "warning":
            qs = qs.exclude(validation_notes="").exclude(row_status=RowStatus.ERROR)
        else:
            qs = qs.filter(row_status=status_filter)

    # Get field names for the table header (exclude batch, row_status, etc.)
    exclude_fields = {"id", "batch", "batch_id", "row_status", "validation_notes"}
    field_names = [
        f.name
        for f in staging_model._meta.get_fields()
        if hasattr(f, "name") and f.name not in exclude_fields
    ]

    # For "has empty fields" filter
    if filter_empty:
        qs = _apply_empty_filter(qs, staging_model, field_names)

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

    # Geocode patients with missing lat/lng from zip code centroids
    if dataset == "patients":
        _geocode_patient_rows(rows)

    # Fetch production data for the current page so we can show old→new diffs
    source_ids = [row.source_id for row in rows]
    prod_map = _fetch_prod_rows(dataset, source_ids)

    # Build row data with per-cell metadata (including production comparison)
    row_data = []
    for row in rows:
        prod_row = prod_map.get(row.source_id)
        row_data.append(_build_row_context(row, field_names, prod_row))

    # When viewing "changed" rows, narrow columns to only fields with changes
    # plus source_id for identity. This declutters the table dramatically.
    display_fields = _narrow_changed_fields(field_names, row_data, status_filter)

    schema = SCHEMA_INFO.get(dataset, {})

    return render(
        request,
        "data_import/_data_table.html",
        {
            "batch": batch,
            "dataset": dataset,
            "field_names": display_fields,
            "all_field_names": field_names,
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

    prod_row = _get_single_prod_row(dataset, row.source_id)
    return render(
        request,
        "data_import/_row_display.html",
        {
            "batch": {"pk": batch_id},
            "dataset": dataset,
            "row": _build_row_context(row, field_names, prod_row or None),
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

    prod_row = _get_single_prod_row(dataset, row.source_id)
    return render(
        request,
        "data_import/_row_display.html",
        {
            "batch": {"pk": batch_id},
            "dataset": dataset,
            "row": _build_row_context(row, field_names, prod_row or None),
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
    # Group logs by batch for accordion display
    batches_seen: dict[int | None, dict[str, Any]] = {}
    for log in logs:
        batch_pk = log.batch_id
        if batch_pk not in batches_seen:
            batches_seen[batch_pk] = {
                "batch": log.batch,
                "logs": [],
                "created_at": log.created_at,
            }
        batches_seen[batch_pk]["logs"].append(log)
    batch_groups = sorted(batches_seen.values(), key=lambda g: g["created_at"], reverse=True)
    return render(request, "data_import/logs.html", {"batch_groups": batch_groups})


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


def _geocode_patient_rows(rows: list[Any]) -> None:
    """Backfill lat/lng on StagingPatient rows that have a zip_code but no coords.

    Uses the static ZIP_CENTROIDS lookup so there are zero network calls.
    Rows that gain coordinates are bulk-updated in a single query.
    Appends a 'Geocoded:' note to validation_notes so the UI can highlight
    the affected cells.
    """
    # Zip values that should be reclassified to "Homeless/Transient" when
    # they get the Serenity House fallback — these indicate the patient has
    # no known address and is effectively homeless for mapping purposes.
    _reclassify_to_homeless = {"Not disclosed", "No data", "Unknown", "None", "N/A"}

    to_update: list[Any] = []
    for row in rows:
        if row.latitude is not None or row.longitude is not None:
            continue
        zc = str(getattr(row, "zip_code", "") or "").strip()
        if zc in ZIP_CENTROIDS:
            lat, lng = ZIP_CENTROIDS[zc]
            row.latitude = lat
            row.longitude = lng
            geo_note = f"Geocoded: latitude: → {lat}; longitude: → {lng}"
            # Reclassify ambiguous zip values to Homeless/Transient
            if zc in _reclassify_to_homeless:
                row.zip_code = "Homeless/Transient"
                geo_note += f"; zip_code: {zc} → Homeless/Transient"
            existing = (row.validation_notes or "").strip()
            row.validation_notes = f"{existing}; {geo_note}" if existing else geo_note
            to_update.append(row)
    if to_update:
        StagingPatient.objects.bulk_update(
            to_update, ["latitude", "longitude", "zip_code", "validation_notes"]
        )


PLACEHOLDER_VALUES = {"Not disclosed", "Unknown", "None", "N/A", "n/a", "Select Race"}


def _build_row_context(
    row: Any, field_names: list[str], prod_values: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a unified row context dict with per-cell metadata for templates.

    ``prod_values`` is an optional dict of the *production* row's field values
    (keyed by staging field names, i.e. core PK is mapped to ``source_id``).
    When present, each cell gets an ``old_value`` entry so the template can
    render an old → new diff for changed fields.
    """
    # Parse changed field names from validation_notes.
    # Supported prefixes: "Changed: field: old → new; ..." and
    # "Geocoded: field: → val; ...".  Both use semicolon-delimited
    # entries where the field name precedes the first colon.
    changed_fields: set[str] = set()
    notes = getattr(row, "validation_notes", "") or ""
    for prefix in ("Changed: ", "Geocoded: "):
        if prefix not in notes:
            continue
        segment = notes.split(prefix, 1)[1]
        # Stop at the next prefix or end-of-string
        for other in ("Changed: ", "Geocoded: "):
            if other != prefix and other in segment:
                segment = segment.split(other, 1)[0]
        for chunk in segment.split(";"):
            chunk = chunk.strip()
            if ": " in chunk:
                changed_fields.add(chunk.split(": ", 1)[0].strip())

    cells_list: list[dict[str, Any]] = []
    cells_dict: dict[str, Any] = {}
    cells_by_name: dict[str, dict[str, Any]] = {}
    empty_count = 0
    for fname in field_names:
        val = getattr(row, fname, "")
        cells_dict[fname] = val
        is_empty = val is None or str(val).strip() == ""
        is_placeholder = str(val).strip() in PLACEHOLDER_VALUES
        is_changed = fname in changed_fields

        # Grab old production value when available
        old_value = prod_values.get(fname) if prod_values else None

        if is_empty:
            empty_count += 1
        cell_info = {
            "name": fname,
            "value": val,
            "old_value": old_value,
            "is_empty": is_empty,
            "is_placeholder": is_placeholder,
            "is_changed": is_changed,
        }
        cells_list.append(cell_info)
        cells_by_name[fname] = cell_info
    return {
        "pk": row.pk,
        "status": row.row_status,
        "validation_notes": row.validation_notes,
        "cells": cells_list,
        "cells_dict": cells_dict,
        "cells_by_name": cells_by_name,
        "empty_count": empty_count,
        "changed_fields": changed_fields,
    }


def _fetch_prod_rows(dataset: str, source_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Fetch production rows for a list of source_ids and return them keyed by source_id.

    The returned dicts use *staging* field names (i.e. the core PK field is
    mapped to ``source_id``), so callers can pass them straight into
    ``_build_row_context`` without any key translation.
    """
    core_model = CORE_MODELS.get(dataset)
    pk_field = CORE_PK_FIELDS.get(dataset, "id")
    if not core_model or not source_ids:
        return {}

    prod_qs = core_model.objects.filter(**{f"{pk_field}__in": source_ids}).values()
    prod_map: dict[int, dict[str, Any]] = {}
    for row in prod_qs:
        sid = row.pop(pk_field, None)
        if sid is not None:
            row["source_id"] = sid
            prod_map[sid] = row
    return prod_map


def _get_single_prod_row(dataset: str, source_id: int) -> dict[str, Any]:
    """Fetch a single production row for a given source_id, mapped to staging field names."""
    rows = _fetch_prod_rows(dataset, [source_id])
    return rows.get(source_id, {})


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


def _backfill_skip_fields(
    staging_kwargs: dict[str, Any],
    source_id: int,
    existing_rows: dict[int, dict[str, Any]],
    skip_fields: set[str],
) -> None:
    """Copy skip_fields (e.g. lat/long) from production into staging kwargs.

    The CSV doesn't supply these fields, but production has them (from
    geocoding etc.).  Back-filling lets users see existing values in the
    review table without triggering false change-detection.
    """
    if not skip_fields or source_id not in existing_rows:
        return
    prod_row = existing_rows[source_id]
    for sf in skip_fields:
        if sf in prod_row and prod_row[sf] is not None and not staging_kwargs.get(sf):
            staging_kwargs[sf] = prod_row[sf]


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

    # Final safety net: backfill any staging patients still missing coords.
    # This catches edge cases where staging data was manually edited after
    # the initial staging backfill, or rows from older import batches.
    if "patients" in commit_summary:
        patient_staging = STAGING_MODELS["patients"]
        all_pts = list(patient_staging.objects.filter(batch=batch))
        _geocode_patient_rows(all_pts)

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
