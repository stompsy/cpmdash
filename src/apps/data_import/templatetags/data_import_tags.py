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
