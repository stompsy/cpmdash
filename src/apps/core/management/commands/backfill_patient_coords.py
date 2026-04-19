"""One-time management command to backfill missing lat/long on production Patient records.

Uses the same ZIP_CENTROIDS lookup as the data-import review UI so results
are consistent.  Patients whose zip_code is "Not disclosed" (or similar
non-address values) get reclassified to "Homeless/Transient" and mapped to
the Serenity House location.

Usage:
    python src/manage.py backfill_patient_coords          # dry-run (default)
    python src/manage.py backfill_patient_coords --apply   # actually write changes
"""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand

from apps.core.models import Patients
from apps.data_import.views import ZIP_CENTROIDS

# Zip values that should be reclassified to "Homeless/Transient"
_RECLASSIFY = {"Not disclosed", "No data", "Unknown", "None", "N/A"}


class Command(BaseCommand):
    help = "Backfill missing lat/long on Patient records using zip-code centroids."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="Actually update the database (default is dry-run).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        apply = options["apply"]
        missing = Patients.objects.filter(latitude__isnull=True, longitude__isnull=True)

        self.stdout.write(f"Found {missing.count()} patients with missing coordinates.\n")

        to_update = []
        for p in missing:
            zc = (p.zip_code or "").strip()
            if zc not in ZIP_CENTROIDS:
                self.stdout.write(
                    self.style.WARNING(f"  SKIP ID={p.id} zip_code={zc!r} — no centroid")
                )
                continue

            lat, lng = ZIP_CENTROIDS[zc]
            p.latitude = lat
            p.longitude = lng

            if zc in _RECLASSIFY:
                old_zc = zc
                p.zip_code = "Homeless/Transient"
                self.stdout.write(
                    f"  ID={p.id}: ({lat}, {lng}) zip: {old_zc!r} → 'Homeless/Transient'"
                )
            else:
                self.stdout.write(f"  ID={p.id}: ({lat}, {lng}) zip={zc}")

            to_update.append(p)

        self.stdout.write(f"\n{len(to_update)} patients to update.")

        if not apply:
            self.stdout.write(
                self.style.WARNING("\nDRY RUN — no changes made. Use --apply to commit.")
            )
            return

        Patients.objects.bulk_update(to_update, ["latitude", "longitude", "zip_code"])
        self.stdout.write(self.style.SUCCESS(f"\nUpdated {len(to_update)} patient records."))
