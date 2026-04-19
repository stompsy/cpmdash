from __future__ import annotations

from typing import Any

from django import template

register = template.Library()


@register.filter
def lookup(d: dict[str, Any] | None, key: str) -> Any:
    """Look up a key in a dictionary. Usage: {{ mydict|lookup:keyvar }}"""
    if d is None:
        return ""
    return d.get(key, "")


@register.filter
def split_changes(notes: str) -> list[str]:
    """Split validation notes like 'Changed: f1: a → b; f2: c → d' into a list."""
    if not notes:
        return []
    text = notes
    if text.startswith("Changed: "):
        text = text[len("Changed: ") :]
    return [part.strip() for part in text.split(";") if part.strip()]


@register.filter
def grouped_notes(notes: str) -> list[dict[str, Any]]:
    """Parse validation_notes into grouped sections for structured tooltip display.

    Returns a list of dicts like:
        [
            {"label": "Changes", "color": "sky",   "items": ["age: 64 → 65", ...]},
            {"label": "Geocoded", "color": "emerald", "items": ["latitude: → 48.12", ...]},
            {"label": "Warnings", "color": "amber", "items": ["Some warning"]},
        ]

    The notes string is a semicolon-delimited mess that can contain entries from
    different sources.  Known prefixes (``Changed: ``, ``Geocoded: ``) get their
    own sections; everything else falls into a catch-all bucket.
    """
    if not notes:
        return []

    # Split the raw notes on known prefixes, preserving the prefix
    import re

    # Tokenize: split on boundaries just before a known prefix
    parts = re.split(r";\s*(?=Changed: |Geocoded: )", notes)

    changes: list[str] = []
    geocoded: list[str] = []
    other: list[str] = []

    for part in parts:
        part = part.strip().rstrip(";").strip()
        if not part:
            continue
        if part.startswith("Changed: "):
            body = part[len("Changed: ") :]
            changes.extend(c.strip() for c in body.split(";") if c.strip())
        elif part.startswith("Geocoded: "):
            body = part[len("Geocoded: ") :]
            geocoded.extend(c.strip() for c in body.split(";") if c.strip())
        else:
            # Could be standalone warnings/errors separated by semicolons
            other.extend(c.strip() for c in part.split(";") if c.strip())

    groups: list[dict[str, Any]] = []
    if changes:
        groups.append({"label": "Changes", "color": "sky", "items": changes})
    if geocoded:
        groups.append({"label": "Geocoded", "color": "emerald", "items": geocoded})
    if other:
        groups.append({"label": "Notes", "color": "amber", "items": other})
    return groups


@register.filter
def header_label(field_name: str, schema: dict[str, Any] | None) -> str:
    """Return human-readable header for a field using SCHEMA_INFO, or title-case the name."""
    if schema and "fields" in schema:
        for entry in schema["fields"]:
            if entry[0] == field_name:
                return entry[1]
    # Fallback: convert snake_case to Title Case
    return field_name.replace("_", " ").title()


@register.simple_tag
def table_url(base_url: str, **kwargs: Any) -> str:
    """Build a query string URL preserving sort/filter/page params.

    Usage: {% table_url base_url page=1 sort='age' dir='asc' status='new' %}
    """
    params = {k: v for k, v in kwargs.items() if v}
    if not params:
        return base_url
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base_url}?{qs}"
