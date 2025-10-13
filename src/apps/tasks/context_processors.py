from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from .models import Task


def tasks_badge(request: HttpRequest) -> dict[str, Any]:
    # Show global count of incomplete tasks
    count = Task.objects.filter(is_completed=False).count()
    return {"incomplete_tasks_count": count}
