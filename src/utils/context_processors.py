from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.http import HttpRequest

_CHANGELOG = Path(__file__).resolve().parent.parent.parent / "CHANGELOG.md"


@lru_cache(maxsize=1)
def _parse_app_version() -> str:
    """Extract the most recent version tag from CHANGELOG.md."""
    try:
        text = _CHANGELOG.read_text(encoding="utf-8")
        match = re.search(r"##\s+(v[\d]+\.[\d]+\.[\d]+[^\s)]*)\s*\(", text)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "v?"


def app_context(request: HttpRequest) -> dict[str, Any]:
    """Inject app_version and current_year into all template contexts."""
    return {
        "app_version": _parse_app_version(),
        "current_year": datetime.now().year,
    }
