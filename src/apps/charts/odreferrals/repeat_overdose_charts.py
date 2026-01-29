from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS


def _to_rate(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _marker_symbols(rows: list[dict[str, Any]], censor_key: str, y_key: str) -> list[str]:
    symbols: list[str] = []
    for row in rows:
        y_val = _to_rate(row.get(y_key))
        if y_val is None:
            symbols.append("circle")
            continue
        censored = bool(row.get(censor_key))
        symbols.append("circle-open" if censored else "circle")
    return symbols


def build_chart_repeat_overdose_quarterly_trend(
    *,
    theme: str,
    quarterly_rows: list[dict[str, Any]],
) -> str:
    """Build a trend chart for quarter-over-quarter repeat overdose rates.

    Expects each row to include numeric keys:
    - repeat_pct_value
    - repeat_30d_pct_value
    - repeat_90d_pct_value
    - repeat_180d_pct_value
    - repeat_365d_pct_value

    And censor flags:
    - censored_30d
    - censored_90d
    - censored_180d
    - censored_365d

    This chart is intentionally built from the same precomputed rows used by the table
    so the numbers never drift.
    """
    if not quarterly_rows:
        return ""

    x = [str(r.get("label", "")) for r in quarterly_rows]
    y_within = [_to_rate(r.get("repeat_pct_value")) for r in quarterly_rows]
    y_30 = [_to_rate(r.get("repeat_30d_pct_value")) for r in quarterly_rows]
    y_90 = [_to_rate(r.get("repeat_90d_pct_value")) for r in quarterly_rows]
    y_180 = [_to_rate(r.get("repeat_180d_pct_value")) for r in quarterly_rows]
    y_365 = [_to_rate(r.get("repeat_365d_pct_value")) for r in quarterly_rows]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_within,
            mode="lines+markers",
            name="Repeat (same quarter)",
            line=dict(color=CHART_COLORS_VIBRANT[0], width=3),
            marker=dict(color=CHART_COLORS_VIBRANT[0], size=8),
            hovertemplate="<b>%{x}</b><br>Repeat (same quarter): %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_30,
            mode="lines+markers",
            name="Repeat ≤30 days (next event)",
            line=dict(color=CHART_COLORS_VIBRANT[1], width=3),
            marker=dict(
                color=CHART_COLORS_VIBRANT[1],
                size=8,
                symbol=_marker_symbols(quarterly_rows, "censored_30d", "repeat_30d_pct_value"),
            ),
            hovertemplate="<b>%{x}</b><br>Repeat ≤30 days: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_90,
            mode="lines+markers",
            name="Repeat ≤90 days (next event)",
            line=dict(color=CHART_COLORS_VIBRANT[2], width=3),
            marker=dict(
                color=CHART_COLORS_VIBRANT[2],
                size=8,
                symbol=_marker_symbols(quarterly_rows, "censored_90d", "repeat_90d_pct_value"),
            ),
            hovertemplate="<b>%{x}</b><br>Repeat ≤90 days: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_180,
            mode="lines+markers",
            name="Repeat ≤180 days (next event)",
            line=dict(color=CHART_COLORS_VIBRANT[3], width=3),
            marker=dict(
                color=CHART_COLORS_VIBRANT[3],
                size=8,
                symbol=_marker_symbols(quarterly_rows, "censored_180d", "repeat_180d_pct_value"),
            ),
            hovertemplate="<b>%{x}</b><br>Repeat ≤180 days: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_365,
            mode="lines+markers",
            name="Repeat ≤365 days (next event)",
            line=dict(color=CHART_COLORS_VIBRANT[4], width=3),
            marker=dict(
                color=CHART_COLORS_VIBRANT[4],
                size=8,
                symbol=_marker_symbols(quarterly_rows, "censored_365d", "repeat_365d_pct_value"),
            ),
            hovertemplate="<b>%{x}</b><br>Repeat ≤365 days: %{y:.1f}%<extra></extra>",
        )
    )

    # Overall trend line derived from the 365-day series (simple least-squares fit).
    points: list[tuple[float, float]] = [
        (float(i), float(y)) for i, y in enumerate(y_365) if y is not None
    ]
    if len(points) >= 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((xi - x_mean) ** 2 for xi in xs)
        if denom > 0:
            slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in points) / denom
            intercept = y_mean - slope * x_mean
            y_fit = [slope * float(i) + intercept for i in range(len(x))]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_fit,
                    mode="lines",
                    name="Overall trend (from 365d)",
                    line=dict(color=TAILWIND_COLORS["slate-500"], width=4, dash="dash"),
                    hoverinfo="skip",
                )
            )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        x_title="Quarter",
        y_title="Repeat rate (%)",
        show_legend=True,
    )

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(t=70),
    )

    # Use `to_html` so Plotly.js is emitted *before* the init script.
    return pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )
