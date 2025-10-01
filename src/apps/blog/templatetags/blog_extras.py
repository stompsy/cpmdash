from __future__ import annotations

import math

import markdown as md
from django import template

register = template.Library()


@register.filter
def markdown(value: str) -> str:
    return md.markdown(value or "", extensions=["extra", "codehilite", "toc"])  # nosec B308


@register.filter
def reading_time(value: str, wpm: int = 200) -> str:
    """Estimate reading time in minutes from raw markdown/text.

    Args:
        value: The content string (markdown or plain text).
        wpm: Words per minute baseline (default 200).
    Returns:
        A human friendly string like "3 min read".
    """
    if not value:
        return "< 1 min read"
    # Rough word split; markdown symbols don't affect magnitude much.
    words = [w for w in value.split() if w.strip()]  # simple tokenization
    minutes = max(1, math.ceil(len(words) / max(100, wpm)))  # guard against unrealistic low wpm
    return f"{minutes} min read"


@register.filter
def word_count(value: str) -> int:
    """Return the integer word count of the provided markdown/plain text.

    Simple whitespace split is sufficient for an approximate count; we ignore
    markdown syntax nuances for performance and simplicity.
    """
    if not value:
        return 0
    return len([w for w in value.split() if w.strip()])
