"""
Geocoding Service for Data Import
===================================
Geocodes OD referral addresses using a 4-phase fallback strategy with
persistent caching. Only NEW addresses (not already in the cache) incur
API calls — everything else is an instant lookup.

Phases:
  1. Cache lookup (instant)
  2. US Census Batch Geocoder (free, no API key, batches of 500)
  3. Nominatim / OpenStreetMap (free, 1 req/sec rate limit)
  4. Zip-code centroid fallback (approximate)

The cache lives at ``assets/geocode_cache.json`` and is shared with the
standalone ``scripts/geocode_call_volume.py`` tool.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # project root
CACHE_PATH = BASE_DIR / "assets" / "geocode_cache.json"

# ---------------------------------------------------------------------------
# Olympic Peninsula bounding box — sanity filter for wrong geocoding results.
# Covers Clallam + NE Jefferson County (Port Townsend, Quilcene, etc.)
# because referrals sometimes come from neighbouring counties.
# ---------------------------------------------------------------------------
_LAT_MIN, _LAT_MAX = 47.75, 48.40
_LON_MIN, _LON_MAX = -124.80, -122.65

# Zip codes that aren't geocodable
_SKIP_ZIPS = {"Homeless", "Homeless/Transient", "Not disclosed", "No data", ""}

# Address values that are clearly not street addresses
_NON_ADDRESS_PATTERNS = [
    "homeless",
    "transient",
    "not disclosed",
    "general delivery",
    "unhoused",
    "unknown",
    "n/a",
    "none",
    "shelter",
    "no data",
]

# Cities in the service area for sweep fallback
# (Clallam County + neighbouring Jefferson County communities)
_CLALLAM_CITIES = [
    "Port Angeles",
    "Sequim",
    "Forks",
    "Joyce",
    "Neah Bay",
    "Sekiu",
    "La Push",
    "Gardiner",
    "Clallam Bay",
    "Port Townsend",
    "Port Hadlock",
    "Quilcene",
]

# Zip-to-city mapping (Clallam + Jefferson County)
_ZIP_TO_CITY: dict[str, str] = {
    "98305": "La Push",
    "98324": "Gardiner",
    "98326": "Joyce",
    "98331": "Forks",
    "98339": "Port Hadlock",
    "98343": "Pysht",
    "98357": "Neah Bay",
    "98362": "Port Angeles",
    "98363": "Port Angeles",
    "98368": "Port Townsend",
    "98381": "Sekiu",
    "98382": "Sequim",
    "98386": "Quilcene",
}

# Approximate centroids — last-resort fallback.
# Aligned with ZIP_CENTROIDS in data_import/views.py — keep them in sync!
_ZIP_CENTROIDS: dict[str, tuple[float, float]] = {
    "98305": (47.9073, -124.6353),  # La Push
    "98324": (48.0734, -123.1168),  # Gardiner
    "98326": (48.1572, -123.8481),  # Joyce
    "98331": (47.9498, -124.3505),  # Forks
    "98343": (48.1889, -124.2467),  # Pysht
    "98357": (48.3651, -124.6248),  # Neah Bay
    "98362": (48.1181, -123.4307),  # Port Angeles
    "98363": (48.0976, -123.7403),  # Port Angeles (rural)
    "98381": (48.2653, -124.3923),  # Sekiu / Clallam Bay
    "98382": (48.0795, -123.1018),  # Sequim
    "98386": (47.8225, -122.8268),  # Quilcene (Jefferson Co. — USGS GNIS ref)
    "98339": (48.0311, -122.8103),  # Port Hadlock (Jefferson Co.)
    "98368": (48.1170, -122.7604),  # Port Townsend (Jefferson Co.)
}

_DEFAULT_STATE = "WA"
_DEFAULT_COUNTY = "Clallam County"
_DEFAULT_CITY = "Port Angeles"
# OD responses are local — when zip is non-geocodable, assume Port Angeles
_DEFAULT_OD_ZIP = "98362"

# API endpoints
_CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_DELAY = 1.1  # seconds between requests


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------
def _cache_key(address: str, zipcode: str) -> str:
    return f"{address.strip().lower()}||{zipcode.strip().lower()}"


def _load_cache() -> _CacheDict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: _CacheDict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------
def _normalize_address(address: str) -> str:
    """Strip apartment/unit suffixes that break Census geocoder."""
    addr = address.strip()
    addr = re.sub(r",?\s*#\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*apt\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*unit\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*suite\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*ste\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*space\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*lot\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",?\s*room\.?\s*\d+.*$", "", addr, flags=re.IGNORECASE)
    return addr.strip().rstrip(",")


def _is_non_address(address: str) -> bool:
    lower = address.strip().lower()
    return any(p in lower for p in _NON_ADDRESS_PATTERNS)


def _has_location_context(address: str) -> bool:
    """Return True if address already contains a city/state suffix."""
    lower = address.strip().lower()
    if re.search(r",\s*wa\s*(?:\d{5})?\s*$", lower):
        return True
    city_names = [
        "port angeles",
        "sequim",
        "forks",
        "neah bay",
        "joyce",
        "sekiu",
        "la push",
        "gardiner",
        "clallam bay",
        "pysht",
        "port orchard",
        "bremerton",
        "silverdale",
        "poulsbo",
        "seattle",
        "tacoma",
        "olympia",
        "port townsend",
    ]
    return any(city in lower for city in city_names)


def _build_full_address(address: str, zipcode: str) -> str:
    norm = _normalize_address(address)
    if _has_location_context(norm):
        return norm
    parts = [norm]
    if zipcode not in _SKIP_ZIPS:
        city = _ZIP_TO_CITY.get(zipcode, "")
        if city:
            parts.append(f"{city}, {_DEFAULT_STATE} {zipcode}")
        else:
            parts.append(f"{_DEFAULT_STATE} {zipcode}")
    else:
        parts.append(f"{_DEFAULT_CITY}, {_DEFAULT_STATE} {_DEFAULT_OD_ZIP}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Phase 2: US Census Batch Geocoder
# ---------------------------------------------------------------------------
def _geocode_census_batch(
    addresses: list[tuple[str, str, str]],
) -> dict[str, tuple[float, float] | None]:
    """Geocode a batch via Census Batch API. Returns {uid: (lat, lon) | None}."""
    results: dict[str, tuple[float, float] | None] = {}

    buf = io.StringIO()
    writer = csv.writer(buf)
    for uid, street, zipcode in addresses:
        norm = _normalize_address(street)
        zc = zipcode if zipcode not in _SKIP_ZIPS else _DEFAULT_OD_ZIP
        city = ""
        city_match = re.match(r"^(.+?),\s*([A-Za-z ]+?),?\s*(?:wa)?\s*$", norm, re.IGNORECASE)
        if city_match and _has_location_context(norm):
            norm = city_match.group(1).strip()
            city = city_match.group(2).strip()
        elif zc in _ZIP_TO_CITY:
            city = _ZIP_TO_CITY[zc]
        writer.writerow([uid, norm, city, _DEFAULT_STATE, zc])

    csv_data = buf.getvalue().encode()
    boundary = "----GeocodeFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="benchmark"\r\n\r\n'
        "Public_AR_Current\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="addressFile"; filename="addresses.csv"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode()
    body += csv_data
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        _CENSUS_BATCH_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
            response_text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return results

    reader = csv.reader(io.StringIO(response_text))
    for row in reader:
        if len(row) < 6:
            continue
        uid = row[0].strip().strip('"')
        match_status = row[2].strip().strip('"')
        coords_raw = row[5].strip().strip('"') if len(row) > 5 else ""

        if match_status == "Match" and coords_raw:
            try:
                lon_str, lat_str = coords_raw.split(",")
                lat, lon = float(lat_str.strip()), float(lon_str.strip())
                if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
                    results[uid] = (lat, lon)
                else:
                    results[uid] = None
            except (ValueError, IndexError):
                results[uid] = None
        else:
            results[uid] = None

    return results


# ---------------------------------------------------------------------------
# Phase 3: Nominatim / OpenStreetMap
# ---------------------------------------------------------------------------
def _nominatim_query(query: str, use_clallam_bbox: bool = True) -> tuple[float, float] | None:
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "limit": "1",
        "countrycodes": "us",
    }
    if use_clallam_bbox:
        params["viewbox"] = f"{_LON_MIN},{_LAT_MIN},{_LON_MAX},{_LAT_MAX}"
        params["bounded"] = "0"

    url = f"{_NOMINATIM_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "COResponse-Geocoder/1.0 (healthcare research project)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    if not data:
        return None

    lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
    if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
        return (lat, lon)
    return None


def _geocode_with_city_sweep(address: str) -> tuple[float, float, str] | None:
    """Try pairing an address with known Clallam County cities."""
    norm = _normalize_address(address)
    if _is_non_address(address):
        return None

    # Strip trailing question marks and parenthetical notes
    clean = re.sub(r"\?.*$", "", norm).strip()
    clean = re.sub(r"\s*\(.*\)\s*$", "", clean).strip()
    if not clean:
        return None

    for city in _CLALLAM_CITIES:
        query = f"{clean}, {city}, WA"
        result = _nominatim_query(query)
        if result:
            return (result[0], result[1], f"city_sweep({city})")
        time.sleep(_NOMINATIM_DELAY)

    return None


# ---------------------------------------------------------------------------
# Internal phase runners — each processes the ``todo`` dict in-place, updating
# ``cache``, ``lats``, and ``lons``.
# ---------------------------------------------------------------------------
_CacheDict = dict[str, dict[str, float | str | None]]
_TodoMap = dict[str, tuple[str, str, list[int]]]  # cache_key → (addr, zip, row_indices)


def _phase_cache_lookup(
    df: pd.DataFrame,
    address_col: str,
    zip_col: str,
    cache: _CacheDict,
    lats: pd.Series,
    lons: pd.Series,
) -> tuple[_TodoMap, int, int]:
    """Phase 1: look up every row in the cache. Return (todo, cached, skipped)."""
    todo: _TodoMap = {}
    cached_count = 0
    skip_count = 0

    for idx, row in df.iterrows():
        addr = str(row[address_col]).strip()
        zc = str(row[zip_col]).strip()

        if not addr or _is_non_address(addr):
            skip_count += 1
            continue

        ck = _cache_key(addr, zc)
        if ck in cache:
            entry = cache[ck]
            # Treat zip_centroid entries as retryable — these are real
            # street addresses that previously fell through to centroid
            # fallback.  Re-attempt Census/Nominatim on each import so
            # we can upgrade them to precise coordinates.
            if entry.get("method") == "zip_centroid":
                if ck not in todo:
                    todo[ck] = (addr, zc, [])
                todo[ck][2].append(idx)
                # Use centroid coords as interim value in case retry
                # still fails — _phase_zip_centroid will overwrite.
                if entry.get("lat") is not None and entry.get("lon") is not None:
                    lats[idx] = entry["lat"]
                    lons[idx] = entry["lon"]
            elif entry.get("lat") is not None and entry.get("lon") is not None:
                lats[idx] = entry["lat"]
                lons[idx] = entry["lon"]
                cached_count += 1
            else:
                cached_count += 1
        else:
            if ck not in todo:
                todo[ck] = (addr, zc, [])
            todo[ck][2].append(idx)

    return todo, cached_count, skip_count


def _phase_census(
    todo: _TodoMap,
    cache: _CacheDict,
    lats: pd.Series,
    lons: pd.Series,
) -> int:
    """Phase 2: Census Batch API. Returns number of hits."""
    batch = [(ck, todo[ck][0], todo[ck][1]) for ck in todo]
    results = _geocode_census_batch(batch)
    hits = 0
    for ck in list(todo.keys()):
        result = results.get(ck)
        if result is not None:
            lat, lon = result
            cache[ck] = {"lat": lat, "lon": lon}
            for idx in todo[ck][2]:
                lats[idx] = lat
                lons[idx] = lon
            del todo[ck]
            hits += 1
    return hits


def _phase_nominatim(
    todo: _TodoMap,
    cache: _CacheDict,
    lats: pd.Series,
    lons: pd.Series,
    log: list[str] | None = None,
) -> int:
    """Phase 3: Nominatim direct + city sweep. Returns number of hits."""
    hits = 0
    total = len(todo)
    for i, ck in enumerate(list(todo.keys())):
        addr, zc, indices = todo[ck]
        if log is not None:
            log.append(f"  ⏳ Nominatim {i + 1}/{total}: {addr[:50]}...")
        full = _build_full_address(addr, zc)
        result = _nominatim_query(full)

        if result is None and zc in _SKIP_ZIPS:
            if log is not None:
                log.append(f"  ⏳ Nominatim {i + 1}/{total}: {addr[:50]}... city sweep")
            sweep = _geocode_with_city_sweep(addr)
            if sweep:
                result = (sweep[0], sweep[1])
                cache[ck] = {"lat": sweep[0], "lon": sweep[1], "method": sweep[2]}
        else:
            time.sleep(_NOMINATIM_DELAY)

        if result is not None:
            lat, lon = result
            if ck not in cache:
                cache[ck] = {"lat": lat, "lon": lon}
            for idx in indices:
                lats[idx] = lat
                lons[idx] = lon
            del todo[ck]
            hits += 1
        else:
            time.sleep(_NOMINATIM_DELAY)
    return hits


def _phase_zip_centroid(
    todo: _TodoMap,
    cache: _CacheDict,
    lats: pd.Series,
    lons: pd.Series,
) -> int:
    """Phase 4: zip-code centroid fallback. Returns number of hits."""
    hits = 0
    for ck in list(todo.keys()):
        _addr, zc, indices = todo[ck]
        if zc in _ZIP_CENTROIDS:
            lat, lon = _ZIP_CENTROIDS[zc]
            cache[ck] = {"lat": lat, "lon": lon, "method": "zip_centroid"}
            for idx in indices:
                lats[idx] = lat
                lons[idx] = lon
            hits += 1
        else:
            cache[ck] = {"lat": None, "lon": None}
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def geocode_od_addresses(
    df: pd.DataFrame,
    address_col: str = "od_address",
    zip_col: str = "patient_zipcode",
    log: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Geocode OD referral addresses, returning (lat_series, lon_series).

    Uses a 4-phase strategy:
      1. Cache lookup
      2. US Census Batch Geocoder
      3. Nominatim / OpenStreetMap with city sweep
      4. Zip-code centroid fallback

    Updates the shared ``assets/geocode_cache.json`` with any new results.
    Non-geocodable addresses (empty, homeless, etc.) get NaN.
    """
    if log is None:
        log = []

    cache = _load_cache()
    lats = pd.Series([float("nan")] * len(df), index=df.index)
    lons = pd.Series([float("nan")] * len(df), index=df.index)

    todo, cached, skipped = _phase_cache_lookup(df, address_col, zip_col, cache, lats, lons)
    log.append(f"  Geocoding: {cached} cached, {skipped} skipped, {len(todo)} to geocode")

    if not todo:
        return lats, lons

    log.append("  ⏳ Running Census batch geocoder...")
    t0 = time.time()
    census_hits = _phase_census(todo, cache, lats, lons)
    log.append(f"  Census geocoder: {census_hits} hits ({time.time() - t0:.1f}s)")

    if todo:
        log.append(f"  ⏳ Starting Nominatim lookups ({len(todo)} addresses)...")
        t0 = time.time()
        nom_hits = _phase_nominatim(todo, cache, lats, lons, log=log)
        log.append(f"  Nominatim: {nom_hits} hits ({time.time() - t0:.1f}s)")

    if todo:
        centroid_hits = _phase_zip_centroid(todo, cache, lats, lons)
        if centroid_hits:
            log.append(f"  Zip centroids: {centroid_hits} hits")

    still_missing = sum(1 for ck in todo if cache.get(ck, {}).get("lat") is None)
    if still_missing:
        log.append(f"  ⚠ {still_missing} address(es) could not be geocoded")

    _save_cache(cache)
    return lats, lons
