"""
Timeline data storage and management.
Uses JSON file in static/data/timeline/ for persistence.
"""

import json
from pathlib import Path
from typing import Any

from django.conf import settings


class TimelineStore:
    """Manages timeline entries stored in JSON format."""

    def __init__(self) -> None:
        self.data_file = (
            Path(settings.BASE_DIR) / "src" / "static" / "data" / "timeline" / "entries.json"
        )
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        """Read timeline entries from JSON file."""
        if not self.data_file.exists():
            return []
        with open(self.data_file, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, entries: list[dict[str, Any]]) -> None:
        """Write timeline entries to JSON file."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def get_all(self) -> list[dict[str, Any]]:
        """Get all timeline entries, sorted by year."""
        entries = self._read()
        return sorted(entries, key=lambda x: x.get("year", ""))

    def get_by_id(self, entry_id: int) -> dict[str, Any] | None:
        """Get a single timeline entry by ID."""
        entries = self._read()
        for entry in entries:
            if entry.get("id") == entry_id:
                return entry
        return None

    def create(self, entry_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new timeline entry."""
        entries = self._read()

        # Generate new ID
        max_id = max((e.get("id", 0) for e in entries), default=0)
        entry_data["id"] = max_id + 1

        # Ensure proper structure
        entry_data.setdefault("bullets", [])
        entry_data.setdefault("sections", [])

        entries.append(entry_data)
        self._write(entries)
        return entry_data

    def update(self, entry_id: int, entry_data: dict[str, Any]) -> dict[str, Any] | None:
        """Update an existing timeline entry."""
        entries = self._read()

        for i, entry in enumerate(entries):
            if entry.get("id") == entry_id:
                entry_data["id"] = entry_id  # Preserve ID
                entry_data.setdefault("bullets", [])
                entry_data.setdefault("sections", [])
                entries[i] = entry_data
                self._write(entries)
                return entry_data

        return None

    def delete(self, entry_id: int) -> bool:
        """Delete a timeline entry."""
        entries = self._read()
        original_length = len(entries)

        entries = [e for e in entries if e.get("id") != entry_id]

        if len(entries) < original_length:
            self._write(entries)
            return True
        return False


# Singleton instance
timeline_store = TimelineStore()
