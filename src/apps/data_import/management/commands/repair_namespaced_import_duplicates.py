from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction

from apps.core.models import Agency, Encounters, ODReferrals, Patients, Referrals

NAMESPACE_WIDTH = 10_000_000


class Command(BaseCommand):
    help = (
        "Repair duplicate rows created by namespaced import IDs by merging "
        "namespaced rows into legacy raw-ID rows per agency."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Apply changes. Without this flag the command runs in dry-run mode.",
        )
        parser.add_argument(
            "--agency-id",
            type=int,
            default=None,
            help="Limit repair to a single agency ID.",
        )

    def handle(self, *_args: Any, **options: Any) -> None:
        apply_changes = bool(options["apply"])
        agency_id = options.get("agency_id")

        agencies = Agency.objects.all().order_by("id")
        if agency_id is not None:
            agencies = agencies.filter(id=agency_id)

        if not agencies.exists():
            self.stdout.write(self.style.WARNING("No agencies matched the request."))
            return

        mode_label = "APPLY" if apply_changes else "DRY-RUN"
        self.stdout.write(f"Mode: {mode_label}")

        with transaction.atomic():
            for agency in agencies:
                self._repair_agency(agency, apply_changes)

            if not apply_changes:
                transaction.set_rollback(True)

    def _repair_agency(self, agency: Agency, apply_changes: bool) -> None:
        self.stdout.write(f"\nAgency {agency.id} ({agency.slug})")

        patient_map, patient_merged = self._merge_namespaced_rows(
            model=Patients,
            pk_field="id",
            agency=agency,
            apply_changes=apply_changes,
        )
        referral_map, referral_merged = self._merge_namespaced_rows(
            model=Referrals,
            pk_field="ID",
            agency=agency,
            apply_changes=apply_changes,
        )
        _, encounter_merged = self._merge_namespaced_rows(
            model=Encounters,
            pk_field="ID",
            agency=agency,
            apply_changes=apply_changes,
        )
        _, od_merged = self._merge_namespaced_rows(
            model=ODReferrals,
            pk_field="ID",
            agency=agency,
            apply_changes=apply_changes,
        )

        referral_patient_updates = self._rewrite_foreign_keys(
            model=Referrals,
            field_name="patient_ID",
            mapping=patient_map,
            agency=agency,
            apply_changes=apply_changes,
        )
        od_patient_updates = self._rewrite_foreign_keys(
            model=ODReferrals,
            field_name="patient_id",
            mapping=patient_map,
            agency=agency,
            apply_changes=apply_changes,
        )
        enc_patient_updates = self._rewrite_foreign_keys(
            model=Encounters,
            field_name="patient_ID",
            mapping=patient_map,
            agency=agency,
            apply_changes=apply_changes,
        )
        enc_ref_updates = self._rewrite_foreign_keys(
            model=Encounters,
            field_name="referral_ID",
            mapping=referral_map,
            agency=agency,
            apply_changes=apply_changes,
        )
        enc_port_ref_updates = self._rewrite_foreign_keys(
            model=Encounters,
            field_name="port_referral_ID",
            mapping=referral_map,
            agency=agency,
            apply_changes=apply_changes,
        )

        self.stdout.write(
            "  merged rows: "
            f"patients={patient_merged}, referrals={referral_merged}, "
            f"encounters={encounter_merged}, odreferrals={od_merged}"
        )
        self.stdout.write(
            "  foreign-key rewrites: "
            f"referrals.patient_ID={referral_patient_updates}, "
            f"odreferrals.patient_id={od_patient_updates}, "
            f"encounters.patient_ID={enc_patient_updates}, "
            f"encounters.referral_ID={enc_ref_updates}, "
            f"encounters.port_referral_ID={enc_port_ref_updates}"
        )

    def _merge_namespaced_rows(
        self,
        model: type[Any],
        pk_field: str,
        agency: Agency,
        apply_changes: bool,
    ) -> tuple[dict[int, int], int]:
        lower = agency.id * NAMESPACE_WIDTH
        upper = (agency.id + 1) * NAMESPACE_WIDTH

        namespaced_qs = model.objects.filter(
            agency=agency,
            **{f"{pk_field}__gte": lower, f"{pk_field}__lt": upper},
        ).order_by(pk_field)

        field_names = [f.name for f in model._meta.fields if f.name != pk_field]

        mapping: dict[int, int] = {}
        merged = 0
        for ns_row in namespaced_qs.iterator(chunk_size=500):
            ns_id = int(getattr(ns_row, pk_field))
            raw_id = ns_id % NAMESPACE_WIDTH
            if raw_id <= 0:
                continue

            raw_row = model.objects.filter(agency=agency, **{pk_field: raw_id}).first()
            if raw_row is None:
                continue

            mapping[ns_id] = raw_id
            merged += 1

            if not apply_changes:
                continue

            for field_name in field_names:
                setattr(raw_row, field_name, getattr(ns_row, field_name))
            raw_row.save(update_fields=field_names)
            ns_row.delete()

        return mapping, merged

    def _rewrite_foreign_keys(
        self,
        model: type[Any],
        field_name: str,
        mapping: dict[int, int],
        agency: Agency,
        apply_changes: bool,
    ) -> int:
        if not mapping:
            return 0

        updates = 0
        for old_ids in _chunked(mapping.keys(), 1000):
            qs = model.objects.filter(agency=agency, **{f"{field_name}__in": old_ids})
            rows = list(qs)
            updates += len(rows)

            if not apply_changes or not rows:
                continue

            for row in rows:
                old_val = int(getattr(row, field_name))
                new_val = mapping.get(old_val)
                if new_val is None:
                    continue
                setattr(row, field_name, new_val)
            model.objects.bulk_update(rows, [field_name], batch_size=500)

        return updates


def _chunked(values: Iterable[int], size: int) -> Iterable[list[int]]:
    chunk: list[int] = []
    for value in values:
        chunk.append(int(value))
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
