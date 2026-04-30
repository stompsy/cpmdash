from django.conf import settings
from django.db import models


class HargroveMetricOverride(models.Model):
    """User-entered overrides for dynamically computed Hargrove Grant metric rows.

    The combination of (year, quarter, metric_key) uniquely identifies a row.
    ``metric_key`` is the display-normalized metric name produced by
    ``_hargrove_display_metric()`` — i.e. what the user sees in the table.

    Only rows rendered with ``editable=True`` (year >= 2026) are saveable via
    the UI; the model itself has no year restriction so historical data can be
    seeded manually through the admin if needed.
    """

    year = models.PositiveSmallIntegerField()
    quarter = models.PositiveSmallIntegerField()
    metric_id = models.CharField(max_length=20, blank=True)
    metric_key = models.CharField(max_length=500)
    value = models.CharField(max_length=500, blank=True)
    notes = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )

    class Meta:
        unique_together = [("year", "quarter", "metric_key")]
        ordering = ["year", "quarter", "metric_key"]

    def __str__(self) -> str:
        return f"Q{self.quarter} {self.year} \u2014 {self.metric_key[:80]}"
