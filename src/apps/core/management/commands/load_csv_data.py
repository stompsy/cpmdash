"""
Django management command: load_csv_data
========================================
Reads the raw PowerApps/SharePoint CSV exports from assets/, cleans and
transforms them to match the production schema, writes the cleaned CSVs
(core_*.csv) back to assets/, and bulk-loads the data into whichever
database Django is currently pointed at (sqlite in dev, PostgreSQL in prod).

This command delegates all ETL logic to ``apps.data_import.etl_service``.

Usage:
    # Full pipeline: clean → export cleaned CSVs → load into DB
    python src/manage.py load_csv_data

    # Clean + export CSVs only (no DB writes)
    python src/manage.py load_csv_data --dry-run

    # Only process specific tables
    python src/manage.py load_csv_data --only patients referrals

    # Skip the "wipe existing rows" safety prompt
    python src/manage.py load_csv_data --no-input
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import models as dm

from apps.core.models import Encounters, ODReferrals, Patients, Referrals
from apps.data_import.etl_service import DataCleaningService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parents[5] / "assets"

ALL_TABLES = ("patients", "referrals", "odreferrals", "encounters")

# Source file map: table name → raw CSV filename in assets/
SOURCE_MAP = {
    "patients": "patients.csv",
    "referrals": "referrals.csv",
    "odreferrals": "referrals_port.csv",
    "encounters": "encounters.csv",
}

# Output file map: table name → cleaned CSV filename
OUTPUT_MAP = {
    "patients": "core_patients.csv",
    "referrals": "core_referrals.csv",
    "odreferrals": "core_odreferrals.csv",
    "encounters": "core_encounters.csv",
}

# Model map: table name → Django model class
MODEL_MAP: dict[str, type[dm.Model]] = {
    "patients": Patients,
    "referrals": Referrals,
    "odreferrals": ODReferrals,
    "encounters": Encounters,
}


class Command(BaseCommand):
    help = "Clean raw CSV exports and load data into the database."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Clean and export CSVs but skip database writes.",
        )
        parser.add_argument(
            "--only",
            nargs="+",
            choices=ALL_TABLES,
            help="Process only the specified table(s).",
        )
        parser.add_argument(
            "--no-input",
            action="store_true",
            help="Skip the confirmation prompt before wiping existing data.",
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def handle(self, *args: Any, **options: Any) -> None:
        tables = options["only"] or list(ALL_TABLES)
        dry_run: bool = options["dry_run"]
        no_input: bool = options["no_input"]

        self.stdout.write(self.style.NOTICE(f"Assets directory: {ASSETS_DIR}"))
        self.stdout.write(self.style.NOTICE(f"Tables to process: {', '.join(tables)}"))

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — no database writes."))

        service = DataCleaningService()
        results: dict[str, pd.DataFrame] = {}

        # Map table names → service methods
        clean_methods = {
            "patients": service.clean_patients,
            "referrals": service.clean_referrals,
            "odreferrals": service.clean_odreferrals,
            "encounters": service.clean_encounters,
        }

        for table in tables:
            src = ASSETS_DIR / SOURCE_MAP[table]
            if not src.exists():
                raise CommandError(f"Missing source file: {src}")

            self.stdout.write(self.style.HTTP_INFO(f"Cleaning {table}..."))
            result = clean_methods[table](src)

            # Print log lines
            for line in result.log:
                self.stdout.write(f"  {line}")

            results[table] = result.df

        # --- Write cleaned CSVs -----------------------------------------
        for table, df in results.items():
            out_path = ASSETS_DIR / OUTPUT_MAP[table]
            df.to_csv(out_path, index=False)
            self.stdout.write(self.style.SUCCESS(f"Wrote {len(df)} rows → {out_path.name}"))

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run complete. No DB changes."))
            return

        # --- Confirm before wiping existing data ------------------------
        if not no_input:
            self.stdout.write(
                self.style.WARNING(
                    "\nThis will DELETE all existing rows in the target tables "
                    "and replace them with the cleaned data."
                )
            )
            confirm = input("Type 'yes' to continue: ")
            if confirm.lower() != "yes":
                raise CommandError("Aborted by user.")

        # --- Load into database -----------------------------------------
        for table, df in results.items():
            model = MODEL_MAP[table]
            self._load_into_db(model, df, table)

        self.stdout.write(self.style.SUCCESS("\nAll done. Data loaded successfully."))

    # ==================================================================
    # Database loading
    # ==================================================================
    def _load_into_db(self, model: type[dm.Model], df: pd.DataFrame, table_name: str) -> None:
        """Wipe existing rows and bulk-insert the cleaned DataFrame."""
        self.stdout.write(f"  Loading {table_name} into database...")

        deleted_count, _ = model.objects.all().delete()
        if deleted_count:
            self.stdout.write(f"    Deleted {deleted_count} existing rows")

        # Build a set of CharField/TextField names so we keep "" instead of None.
        string_fields = {
            f.name for f in model._meta.get_fields() if isinstance(f, dm.CharField | dm.TextField)
        }

        records = df.to_dict("records")
        instances = []

        for record in records:
            cleaned = {}
            for key, value in record.items():
                if pd.isna(value) or value == "":
                    cleaned[key] = "" if key in string_fields else None
                else:
                    cleaned[key] = value
            instances.append(model(**cleaned))

        model.objects.bulk_create(instances, batch_size=500)
        self.stdout.write(
            self.style.SUCCESS(f"    Inserted {len(instances)} rows into {model.__name__}")
        )
