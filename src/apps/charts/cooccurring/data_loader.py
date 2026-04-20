"""
Load and cache the de-identified OD/SUD/AUD co-occurring disorder dataset.

The JSON lives in src/static/data/od_sud_aud.json and was extracted from the
original od-sud-aud.xlsx with all PII (names, patient IDs) stripped out.  We
cache a single load because the file is static and re-reading on every request
would be wasteful.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parents[3] / "static" / "data" / "od_sud_aud.json"


@lru_cache(maxsize=1)
def load_od_sud_aud_data() -> dict[str, Any]:
    """Return the full parsed JSON structure, cached after first read."""
    with open(_DATA_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_opioid_patients() -> list[dict[str, Any]]:
    return load_od_sud_aud_data()["opioid_od_patients"]


def get_sud_patients() -> list[dict[str, Any]]:
    return load_od_sud_aud_data()["sud_patients"]


def get_summary_stats() -> dict[str, str]:
    return load_od_sud_aud_data()["summary_stats"]
