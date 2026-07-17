from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.dashboard.models import HargroveMetricOverride
from apps.dashboard.views import (
    _build_historical_hargrove_metrics,
    _hargrove_narrative_metric_key,
    _hargrove_override_metric_key,
)


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


class Command(BaseCommand):
    help = "Import Hargrove historical JSON metrics/narratives into HargroveMetricOverride."

    def add_arguments(self, parser) -> None:  # type: ignore[no-untyped-def]
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing DB values for matching year/quarter/metric_key rows.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        overwrite = bool(options.get("overwrite"))

        metrics_path = (
            Path(settings.BASE_DIR) / "src" / "static" / "data" / "HARGROVE_GRANT_METRICS.json"
        )
        narratives_path = (
            Path(settings.BASE_DIR) / "src" / "static" / "data" / "hargrove_grant_narratives.json"
        )

        metrics_data: dict[str, Any] = {}
        narratives_data: dict[str, Any] = {}

        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as fp:
                loaded = json.load(fp)
                if isinstance(loaded, dict):
                    metrics_data = loaded

        if narratives_path.exists():
            with narratives_path.open(encoding="utf-8") as fp:
                loaded = json.load(fp)
                if isinstance(loaded, dict):
                    narratives_data = loaded

        created = updated = skipped = 0

        created, updated, skipped = self._import_metrics_json(
            metrics_data,
            overwrite=overwrite,
            created=created,
            updated=updated,
            skipped=skipped,
        )
        created, updated, skipped = self._import_narratives_json(
            narratives_data,
            overwrite=overwrite,
            created=created,
            updated=updated,
            skipped=skipped,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Import complete. created={created}, updated={updated}, skipped={skipped}"
            )
        )

    def _import_metrics_json(
        self,
        metrics_data: dict[str, Any],
        *,
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        for year_str, year_metrics in metrics_data.items():
            year = _safe_int(year_str)
            if year <= 0 or not isinstance(year_metrics, dict):
                continue

            for q_str in ("1", "2", "3", "4"):
                if q_str not in year_metrics:
                    continue
                quarter = _safe_int(q_str)
                sections = _build_historical_hargrove_metrics(year_metrics, q_str)
                created, updated, skipped = self._import_sections(
                    year=year,
                    quarter=quarter,
                    sections=sections,
                    overwrite=overwrite,
                    created=created,
                    updated=updated,
                    skipped=skipped,
                )
        return created, updated, skipped

    def _import_sections(
        self,
        *,
        year: int,
        quarter: int,
        sections: list[dict[str, object]],
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        for section in sections:
            section_type = section.get("type")
            if section_type == "table":
                rows_obj = section.get("rows")
                rows = rows_obj if isinstance(rows_obj, list) else []
                created, updated, skipped = self._import_table_rows(
                    year=year,
                    quarter=quarter,
                    rows=rows,
                    overwrite=overwrite,
                    created=created,
                    updated=updated,
                    skipped=skipped,
                )
            elif section_type == "narrative":
                items_obj = section.get("questions_responses")
                items = items_obj if isinstance(items_obj, list) else []
                created, updated, skipped = self._import_narrative_items(
                    year=year,
                    quarter=quarter,
                    items=items,
                    overwrite=overwrite,
                    created=created,
                    updated=updated,
                    skipped=skipped,
                )
        return created, updated, skipped

    def _import_table_rows(
        self,
        *,
        year: int,
        quarter: int,
        rows: list[object],
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            metric_key = _hargrove_override_metric_key(str(row.get("metric") or ""))
            if not metric_key:
                continue
            defaults = {
                "metric_id": str(row.get("id") or "").strip(),
                "value": str(row.get("value") or "").strip(),
                "notes": str(row.get("notes") or "").strip(),
            }
            created, updated, skipped = self._upsert_row(
                year=year,
                quarter=quarter,
                metric_key=metric_key,
                defaults=defaults,
                overwrite=overwrite,
                created=created,
                updated=updated,
                skipped=skipped,
            )
        return created, updated, skipped

    def _import_narrative_items(
        self,
        *,
        year: int,
        quarter: int,
        items: list[object],
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            response = str(item.get("response") or "").strip()
            if not response:
                continue
            metric_key = _hargrove_override_metric_key(_hargrove_narrative_metric_key(idx))
            defaults = {"metric_id": "", "value": response, "notes": ""}
            created, updated, skipped = self._upsert_row(
                year=year,
                quarter=quarter,
                metric_key=metric_key,
                defaults=defaults,
                overwrite=overwrite,
                created=created,
                updated=updated,
                skipped=skipped,
            )
        return created, updated, skipped

    def _import_narratives_json(
        self,
        narratives_data: dict[str, Any],
        *,
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        for year_str, q_data in narratives_data.items():
            year = _safe_int(year_str)
            if year <= 0 or not isinstance(q_data, dict):
                continue
            for q_str in ("1", "2", "3", "4"):
                responses_obj = q_data.get(q_str)
                responses: list[str] = (
                    [r.strip() for r in responses_obj if isinstance(r, str) and r.strip()]
                    if isinstance(responses_obj, list)
                    else []
                )
                if not responses:
                    continue
                quarter = _safe_int(q_str)
                created, updated, skipped = self._import_narrative_responses(
                    year=year,
                    quarter=quarter,
                    responses=responses,
                    overwrite=overwrite,
                    created=created,
                    updated=updated,
                    skipped=skipped,
                )
        return created, updated, skipped

    def _import_narrative_responses(
        self,
        *,
        year: int,
        quarter: int,
        responses: Iterable[str],
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        for idx, response in enumerate(responses, start=1):
            metric_key = _hargrove_override_metric_key(_hargrove_narrative_metric_key(idx))
            defaults = {"metric_id": "", "value": response, "notes": ""}
            created, updated, skipped = self._upsert_row(
                year=year,
                quarter=quarter,
                metric_key=metric_key,
                defaults=defaults,
                overwrite=overwrite,
                created=created,
                updated=updated,
                skipped=skipped,
            )
        return created, updated, skipped

    def _upsert_row(
        self,
        *,
        year: int,
        quarter: int,
        metric_key: str,
        defaults: dict[str, str],
        overwrite: bool,
        created: int,
        updated: int,
        skipped: int,
    ) -> tuple[int, int, int]:
        obj, was_created = HargroveMetricOverride.objects.get_or_create(
            year=year,
            quarter=quarter,
            metric_key=metric_key,
            defaults=defaults,
        )

        if was_created:
            return created + 1, updated, skipped

        if not overwrite:
            return created, updated, skipped + 1

        changed = False
        for field, value in defaults.items():
            current_value = getattr(obj, field, "") or ""
            if current_value != value:
                setattr(obj, field, value)
                changed = True
        if changed:
            obj.save(update_fields=["metric_id", "value", "notes", "updated_at"])
            return created, updated + 1, skipped
        return created, updated, skipped + 1
