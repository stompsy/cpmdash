"""Management command to geocode patient addresses using Census + Nominatim.

Reads the patient CSV to get street addresses, purges stale zip_centroid
entries from the geocode cache, re-geocodes through the full 4-phase
pipeline, and updates both the Patients model and the cache.

Usage:
    python src/manage.py geocode_patient_addresses                # dry-run
    python src/manage.py geocode_patient_addresses --apply        # update DB
    python src/manage.py geocode_patient_addresses --apply --zip 98382  # only Sequim
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand

from apps.core.models import Patients
from apps.data_import.geocode_service import (
    _cache_key,
    _is_non_address,
    _load_cache,
    _phase_census,
    _phase_nominatim,
    _phase_zip_centroid,
    _save_cache,
)

_CSV_PATH = Path("assets/patients.csv")


class Command(BaseCommand):
    help = (
        "Geocode patient street addresses via Census + Nominatim, "
        "replacing stale zip-centroid coordinates."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="Actually update the database (default is dry-run).",
        )
        parser.add_argument(
            "--zip",
            type=str,
            default="",
            help="Limit to a specific zip code (e.g. 98382).",
        )
        parser.add_argument(
            "--purge-centroids",
            action="store_true",
            default=True,
            help="Remove zip_centroid entries from cache before geocoding (default: True).",
        )
        parser.add_argument(
            "--no-purge-centroids",
            action="store_false",
            dest="purge_centroids",
            help="Keep existing zip_centroid cache entries (skip purge).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        apply = options["apply"]
        zip_filter = options["zip"].strip()
        purge = options["purge_centroids"]

        if not apply:
            self.stdout.write(self.style.WARNING("DRY RUN — pass --apply to update the DB.\n"))

        df = self._load_csv(zip_filter)
        if df is None:
            return

        cache = self._purge_cache(zip_filter) if purge else _load_cache()
        lats, lons = self._geocode(df, cache)
        updates = self._build_updates(df, lats, lons)
        self._preview_and_apply(updates, apply)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_csv(self, zip_filter: str) -> pd.DataFrame | None:
        """Read patient CSV file and filter to geocodable rows."""
        if not _CSV_PATH.exists():
            self.stderr.write(self.style.ERROR(f"CSV not found: {_CSV_PATH}"))
            return None

        df = pd.read_csv(_CSV_PATH, dtype=str).fillna("")
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

        id_col = "ID" if "ID" in df.columns else df.columns[0]
        addr_col = "Address" if "Address" in df.columns else None

        if addr_col is None:
            self.stderr.write(self.style.ERROR("No 'Address' column found in CSV."))
            return None

        df["_id"] = df[id_col].astype(int)
        df["_addr"] = df[addr_col].str.strip()

        # Zip resolution: prefer "zipcode" (specific), fall back to
        # "Zip Code" (multi-select, may contain "[]" or comma lists).
        # This mirrors the ETL's _merge_zipcode() logic.
        from apps.data_import.etl_service import ZIP_CLEANUP

        if "zipcode" in df.columns:
            df["_zip"] = df["zipcode"].str.strip()
            if "Zip Code" in df.columns:
                # Back-fill from "Zip Code" where "zipcode" is empty
                fallback = df["Zip Code"].str.strip()
                empty = (df["_zip"] == "") | (df["_zip"] == "nan")
                df.loc[empty, "_zip"] = fallback[empty]
        elif "Zip Code" in df.columns:
            df["_zip"] = df["Zip Code"].str.strip()
        else:
            df["_zip"] = ""

        # Normalise CRM artifacts like "[]", verbose labels, etc.
        df["_zip"] = df["_zip"].map(ZIP_CLEANUP).fillna(df["_zip"])

        if zip_filter:
            df = df[df["_zip"] == zip_filter]
            self.stdout.write(f"Filtered to zip {zip_filter}: {len(df)} rows\n")

        has_address = df["_addr"].apply(lambda a: bool(a) and not _is_non_address(a))
        df = df[has_address].copy()
        self.stdout.write(f"Patients with geocodable addresses: {len(df)}\n")

        if df.empty:
            self.stdout.write("Nothing to geocode.\n")
            return None
        return df

    def _purge_cache(self, zip_filter: str) -> dict[str, Any]:
        """Remove zip_centroid entries from cache and return fresh copy."""
        cache = _load_cache()
        centroid_keys = [k for k, v in cache.items() if v.get("method") == "zip_centroid"]
        if zip_filter:
            centroid_keys = [k for k in centroid_keys if k.endswith(f"||{zip_filter.lower()}")]
        for k in centroid_keys:
            del cache[k]
        self.stdout.write(f"Purged {len(centroid_keys)} stale zip_centroid cache entries.\n")
        _save_cache(cache)
        return _load_cache()

    def _geocode(self, df: pd.DataFrame, cache: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
        """Run 4-phase geocoding pipeline on addresses."""
        lats = pd.Series([float("nan")] * len(df), index=df.index)
        lons = pd.Series([float("nan")] * len(df), index=df.index)

        todo: dict[str, tuple[str, str, list[Any]]] = {}
        cached_count = 0
        for idx, row in df.iterrows():
            addr = row["_addr"]
            zc = row["_zip"]
            ck = _cache_key(addr, zc)
            if ck in cache and cache[ck].get("method") != "zip_centroid":
                entry = cache[ck]
                if entry.get("lat") is not None:
                    lats[idx] = entry["lat"]
                    lons[idx] = entry["lon"]
                cached_count += 1
            else:
                if ck not in todo:
                    todo[ck] = (addr, zc, [])
                todo[ck][2].append(idx)

        self.stdout.write(f"Cache hits: {cached_count}, to geocode: {len(todo)}\n")
        if not todo:
            self.stdout.write("All addresses already geocoded. Nothing to do.\n")
            return lats, lons

        self.stdout.write("Running Census batch geocoder...\n")
        census_hits = _phase_census(todo, cache, lats, lons)
        self.stdout.write(f"  Census: {census_hits} hits, {len(todo)} remaining\n")

        if todo:
            log: list[str] = []
            self.stdout.write(f"Running Nominatim ({len(todo)} addresses)...\n")
            nom_hits = _phase_nominatim(todo, cache, lats, lons, log=log)
            self.stdout.write(f"  Nominatim: {nom_hits} hits, {len(todo)} remaining\n")

        if todo:
            centroid_hits = _phase_zip_centroid(todo, cache, lats, lons)
            self.stdout.write(f"  Zip centroids (fallback): {centroid_hits}\n")

        _save_cache(cache)
        self.stdout.write(self.style.SUCCESS("Cache saved.\n"))
        return lats, lons

    def _build_updates(
        self, df: pd.DataFrame, lats: pd.Series, lons: pd.Series
    ) -> list[tuple[int, float, float, str]]:
        """Build list of (patient_id, lat, lon, address) tuples to update."""
        updates: list[tuple[int, float, float, str]] = []
        for idx, row in df.iterrows():
            lat = lats[idx]
            lon = lons[idx]
            if pd.isna(lat) or pd.isna(lon):
                continue
            updates.append((int(row["_id"]), lat, lon, row["_addr"]))
        self.stdout.write(f"\nPatients to update: {len(updates)}\n")
        return updates

    def _preview_and_apply(self, updates: list[tuple[int, float, float, str]], apply: bool) -> None:
        """Show diff preview and optionally apply updates."""
        upgraded = 0
        unchanged = 0
        for pid, lat, lon, addr in updates:
            try:
                p = Patients.objects.get(pk=pid)
            except Patients.DoesNotExist:
                continue
            moved = (
                p.latitude is None
                or p.longitude is None
                or abs(p.latitude - lat) > 0.0001
                or abs(p.longitude - lon) > 0.0001
            )
            if moved:
                upgraded += 1
                if upgraded <= 20:
                    self.stdout.write(
                        f"  ID={pid}: ({p.latitude}, {p.longitude})"
                        f" -> ({lat:.6f}, {lon:.6f})  {addr[:40]}"
                    )
            else:
                unchanged += 1

        if upgraded > 20:
            self.stdout.write(f"  ... and {upgraded - 20} more")
        self.stdout.write(f"\n  Upgraded: {upgraded}, Unchanged: {unchanged}\n")

        if apply and upgraded > 0:
            batch: list[Patients] = []
            for pid, lat, lon, addr in updates:
                try:
                    p = Patients.objects.get(pk=pid)
                except Patients.DoesNotExist:
                    continue
                p.latitude = lat
                p.longitude = lon
                p.address = addr
                batch.append(p)
            Patients.objects.bulk_update(batch, ["latitude", "longitude", "address"])
            self.stdout.write(self.style.SUCCESS(f"Updated {len(batch)} patients.\n"))
        elif apply:
            self.stdout.write("No patients needed updating.\n")
