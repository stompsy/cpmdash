from typing import Any

import pytest
from django.test import RequestFactory

from utils.plotly import (
    create_theme_aware_chart,
    get_color_palette,
    get_plotly_theme,
    get_theme_colors,
    style_plotly_layout,
)

pytestmark = pytest.mark.django_db


class DummyFig:
    def __init__(self) -> None:
        self.layout_updates: dict[str, Any] = {}
        self._config = None

    def update_layout(self, **kwargs: Any) -> None:  # pragma: no cover - simple setter
        self.layout_updates.update(kwargs)


def dummy_chart(theme: str = "dark") -> str:
    # Return simple identifiable string so wrapper works
    return f"<div data-theme='{theme}'></div>"


def test_get_theme_colors_variants() -> None:
    dark = get_theme_colors("dark")
    light = get_theme_colors("light")
    assert dark["axis_font_color"] != light["axis_font_color"]
    assert "hover_bg" in dark


def test_style_plotly_layout_basic() -> None:
    fig = DummyFig()
    out = style_plotly_layout(fig, theme="dark", x_title="X", y_title="Y")
    assert out is fig
    assert fig._config and fig._config.get("responsive") is True
    assert "xaxis" in fig.layout_updates


def test_create_theme_aware_chart_uses_theme(rf: RequestFactory) -> None:
    req = rf.get("/")
    html = create_theme_aware_chart(dummy_chart, req)
    assert "data-theme='dark'" in html


def test_get_plotly_theme_and_palette() -> None:
    ptheme = get_plotly_theme("dark")
    palette = get_color_palette("dark")
    assert "font" in ptheme
    assert "primary" in palette
