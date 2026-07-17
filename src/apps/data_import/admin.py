from typing import Any

from django.contrib import admin
from django.http import HttpRequest

from .models import DataImportBatch, DataImportFile, ProcessingLog


class DataImportFileInline(admin.TabularInline):
    model = DataImportFile
    extra = 0
    readonly_fields = ("uploaded_at",)


@admin.register(DataImportBatch)
class DataImportBatchAdmin(admin.ModelAdmin):
    list_display = ("__str__", "status", "created_by", "created_at")
    list_filter = ("status",)
    inlines = [DataImportFileInline]

    def get_queryset(self, request: HttpRequest) -> Any:
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        agency_id = getattr(request.user, "agency_id", None)
        if not agency_id:
            return qs.none()
        return qs.filter(agency_id=agency_id)


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = ("__str__", "created_at")
    list_filter = ("batch",)

    def get_queryset(self, request: HttpRequest) -> Any:
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        agency_id = getattr(request.user, "agency_id", None)
        if not agency_id:
            return qs.none()
        return qs.filter(batch__agency_id=agency_id)
