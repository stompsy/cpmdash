from typing import Any

from django.contrib import admin

from .models import HargroveMetricOverride


@admin.register(HargroveMetricOverride)
class HargroveMetricOverrideAdmin(admin.ModelAdmin):
    list_display = ("year", "quarter", "metric_key", "value", "notes", "updated_by", "updated_at")
    list_filter = ("year", "quarter")
    search_fields = ("metric_key", "value", "notes")
    ordering = ("year", "quarter", "metric_key")
    readonly_fields = ("updated_at",)
    actions = ["clear_values"]

    @admin.action(description="Clear value overrides (revert to computed values)")
    def clear_values(self, request: Any, queryset: Any) -> None:
        updated = queryset.update(value="")
        self.message_user(request, f"{updated} override(s) cleared.")
