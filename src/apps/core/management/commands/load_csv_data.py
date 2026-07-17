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
        # Always process in dependency order (patients → referrals → odreferrals →
        # encounters), regardless of the order the user passed to --only. Downstream
        # datasets read committed patient data for age-at-event and PCP fallbacks, so
        # the sequence matters.
        selected = set(options["only"] or ALL_TABLES)
        tables = [t for t in ALL_TABLES if t in selected]
        dry_run: bool = options["dry_run"]
        no_input: bool = options["no_input"]

        self.stdout.write(self.style.NOTICE(f"Assets directory: {ASSETS_DIR}"))
        self.stdout.write(self.style.NOTICE(f"Tables to process: {', '.join(tables)}"))

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — no database writes."))

        service = DataCleaningService()

        # Map table names → service methods. Only clean_patients lacks a patients_df
        # parameter; the downstream cleaners all accept one.
        clean_methods = {
            "patients": service.clean_patients,
            "referrals": service.clean_referrals,
            "odreferrals": service.clean_odreferrals,
            "encounters": service.clean_encounters,
        }

        # --- Confirm before wiping existing data ------------------------
        # The prompt moves up front because we now commit each table to the DB as soon
        # as it's cleaned, rather than batching all DB writes at the end.
        if not dry_run and not no_input:
            self.stdout.write(
                self.style.WARNING(
                    "\nThis will DELETE all existing rows in the target tables "
                    "and replace them with the cleaned data."
                )
            )
            confirm = input("Type 'yes' to continue: ")
            if confirm.lower() != "yes":
                raise CommandError("Aborted by user.")

        # Patient lookup frame passed to downstream cleaners. If patients aren't part of
        # this run, seed it from production so referrals/encounters can still resolve
        # ages and PCP agencies against existing data.
        patients_df: pd.DataFrame | None = None
        if "patients" not in tables:
            patients_df = self._patients_df_from_db()

        # --- Process each table: clean → write CSV → commit -------------
        for table in tables:
            df = self._process_table(table, clean_methods[table], patients_df, dry_run)

            # Refresh the patient lookup frame once patients are committed so the
            # referral/encounter cleaners pull from freshly committed production data.
            # During a dry run there are no DB writes, so fall back to the in-memory
            # cleaned frame (it already carries id/age/insurance/pcp_agency).
            if table == "patients":
                patients_df = df if dry_run else self._patients_df_from_db()

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run complete. No DB changes."))
            return

        self.stdout.write(self.style.SUCCESS("\nAll done. Data loaded successfully."))

    def _process_table(
        self,
        table: str,
        clean_method: Any,
        patients_df: pd.DataFrame | None,
        dry_run: bool,
    ) -> pd.DataFrame:
        """Clean a single table, write its CSV, and commit it to the DB.

        Returns the cleaned DataFrame so the caller can reuse it (e.g. seeding the
        patient lookup frame during a dry run).
        """
        src = ASSETS_DIR / SOURCE_MAP[table]
        if not src.exists():
            raise CommandError(f"Missing source file: {src}")

        self.stdout.write(self.style.HTTP_INFO(f"Cleaning {table}..."))
        # Only clean_patients lacks a patients_df parameter; the downstream cleaners
        # all accept one for age/PCP lookups.
        if table == "patients":
            result = clean_method(src)
        else:
            result = clean_method(src, patients_df=patients_df)

        for line in result.log:
            self.stdout.write(f"  {line}")

        df = result.df

        # Write the cleaned CSV alongside the raw export.
        out_path = ASSETS_DIR / OUTPUT_MAP[table]
        df.to_csv(out_path, index=False)
        self.stdout.write(self.style.SUCCESS(f"Wrote {len(df)} rows → {out_path.name}"))

        # Commit this table before moving on so downstream datasets see it as
        # production data.
        if not dry_run:
            self._load_into_db(MODEL_MAP[table], df, table)

        return df

    def _patients_df_from_db(self) -> pd.DataFrame | None:
        """Build a patient lookup frame from production for downstream cleaners.

        Returns ``None`` when no patients exist yet so callers can treat the lookup as
        unavailable instead of passing an empty frame around.
        """
        records = list(Patients.objects.values("id", "age", "insurance", "pcp_agency"))
        if not records:
            return None
        return pd.DataFrame.from_records(records)

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
