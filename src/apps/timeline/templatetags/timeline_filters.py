"""Template filters for timeline app."""

import json
from typing import Any

from django import template
from django.core.serializers.json import DjangoJSONEncoder

register = template.Library()


@register.filter(name="tojson")
def tojson(value: Any) -> str:
    """Convert a Python object to JSON string for use in templates."""
    return json.dumps(value, cls=DjangoJSONEncoder)
