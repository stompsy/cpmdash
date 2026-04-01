from django.contrib import admin

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


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = ("__str__", "created_at")
    list_filter = ("batch",)
