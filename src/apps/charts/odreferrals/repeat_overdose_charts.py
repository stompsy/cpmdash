from __future__ import annotations

import math
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


def _confirmed_series(
    rows: list[dict[str, Any]],
    y_key: str,
    censor_key: str,
) -> list[float | None]:
    """Return the y-series with censored quarters set to None.

    A None value tells Plotly to break the line at that point, so the
    rendered curve only spans quarters whose follow-up window has fully
    elapsed. This removes the optical illusion that recent quarters with
    immature follow-up are showing a real \"drop\" in repeat rates.
    """
    out: list[float | None] = []
    for row in rows:
        if bool(row.get(censor_key)):
            out.append(None)
            continue
        out.append(_to_rate(row.get(y_key)))
    return out


def _first_censored_index(rows: list[dict[str, Any]], censor_key: str) -> int | None:
    for i, row in enumerate(rows):
        if bool(row.get(censor_key)):
            return i
    return None


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

    # Each window's series is restricted to fully-observed quarters. Censored
    # quarters become None, which Plotly renders as a gap -- the line simply
    # stops where the data stops being trustworthy. The within-quarter series
    # is never censored (the quarter's own window closes when the quarter
    # ends), so it stays as the long-running anchor trace.
    y_within = [_to_rate(r.get("repeat_pct_value")) for r in quarterly_rows]
    y_30 = _confirmed_series(quarterly_rows, "repeat_30d_pct_value", "censored_30d")
    y_90 = _confirmed_series(quarterly_rows, "repeat_90d_pct_value", "censored_90d")
    y_180 = _confirmed_series(quarterly_rows, "repeat_180d_pct_value", "censored_180d")
    y_365 = _confirmed_series(quarterly_rows, "repeat_365d_pct_value", "censored_365d")

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

    # Linear trend overlay -- fit on the within-quarter series only. That
    # series is the only one with no right-censoring (each quarter's own
    # window closes when the quarter ends), so an OLS line here isn't
    # tainted by quarters with immature follow-up. We fit only on quarters
    # with a real numeric value, then draw the line across the full x range
    # so the trend reads visually even if a stray quarter is missing.
    fit_pts = [(i, v) for i, v in enumerate(y_within) if v is not None]
    if len(fit_pts) >= 3:
        n = len(fit_pts)
        sx = sum(i for i, _ in fit_pts)
        sy = sum(v for _, v in fit_pts)
        sxx = sum(i * i for i, _ in fit_pts)
        sxy = sum(i * v for i, v in fit_pts)
        denom = n * sxx - sx * sx
        if denom != 0:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            trend_y = [intercept + slope * i for i in range(len(x))]
            direction = "down" if slope < 0 else ("up" if slope > 0 else "flat")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=trend_y,
                    mode="lines",
                    name=f"Trend (same quarter, {direction})",
                    line=dict(
                        color=TAILWIND_COLORS["slate-400"],
                        width=2,
                        dash="dash",
                    ),
                    hovertemplate=("<b>%{x}</b><br>Trend (same-quarter): %{y:.1f}%<extra></extra>"),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_30,
            mode="lines+markers",
            name="Repeat ≤30 days (next event)",
            line=dict(color=CHART_COLORS_VIBRANT[1], width=3),
            marker=dict(color=CHART_COLORS_VIBRANT[1], size=8),
            connectgaps=False,
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
            marker=dict(color=CHART_COLORS_VIBRANT[2], size=8),
            connectgaps=False,
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
            marker=dict(color=CHART_COLORS_VIBRANT[3], size=8),
            connectgaps=False,
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
            marker=dict(color=CHART_COLORS_VIBRANT[4], size=8),
            connectgaps=False,
            hovertemplate="<b>%{x}</b><br>Repeat ≤365 days: %{y:.1f}%<extra></extra>",
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


# ---------------------------------------------------------------------------
# Kaplan-Meier cumulative incidence of repeat overdose
# ---------------------------------------------------------------------------


def _km_cumulative_incidence(
    durations: list[int],
    events: list[int],
) -> tuple[list[int], list[float], list[float], list[float], list[int]]:
    """Compute Kaplan-Meier cumulative incidence with Greenwood-derived 95% CI.

    Why this exists:
        Fixed-window rates ("% with repeat within 90 days") only make sense for
        patients who have actually been observed for at least 90 days. Patients
        we've only seen for 20 days are tossed out (or, worse, silently treated
        as non-repeaters). Kaplan-Meier handles this correctly by letting each
        patient contribute follow-up time for as long as we observed them and
        formally "censoring" them at the end of their observation. The result
        is a single curve that says: "of patients followed for at least t days,
        what fraction had a repeat OD by day t?"

    Args:
        durations: For each patient, days from index OD to either their next OD
            (if they had one) or the data cutoff (if they didn't).
        events: 1 if a repeat OD was observed for that patient, 0 if censored.

    Returns:
        (time_points, cumulative_incidence, ci_lower, ci_upper, n_at_risk_at_t)
        as Python lists, suitable for direct plotting. Both incidence and CI
        bounds are returned as fractions (0.0-1.0), not percents.
    """
    if not durations or len(durations) != len(events):
        return [], [], [], [], []  # type: ignore[return-value]

    # Sort observations by duration so we can sweep forward through unique
    # event times. The classic KM step function only changes at event times,
    # not at censoring times.
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    sorted_durations = [durations[i] for i in order]
    sorted_events = [events[i] for i in order]

    n_total = len(sorted_durations)
    times: list[int] = []
    surv: list[float] = []
    var_sum = 0.0  # Greenwood's running variance sum of (d / (n * (n - d)))
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    at_risk: list[int] = []

    # Always anchor the curve at t=0 with S=1 so the chart starts cleanly at
    # 0% cumulative incidence on day 0.
    times.append(0)
    surv.append(1.0)
    ci_lo.append(0.0)
    ci_hi.append(0.0)
    at_risk.append(n_total)

    s_running = 1.0

    # Iterate over unique event times only. For each, count how many patients
    # are still "at risk" (duration >= t) and how many had the event at t.
    i = 0
    while i < n_total:
        t = sorted_durations[i]
        # Number at risk at time t = patients with duration >= t.
        n_at_risk = n_total - i
        d = 0
        # Tally events and censorings at this exact time.
        j = i
        while j < n_total and sorted_durations[j] == t:
            if sorted_events[j] == 1:
                d += 1
            j += 1
        # Only emit a step when an event actually happens at t (KM convention).
        if d > 0 and n_at_risk > 0:
            s_running *= 1.0 - d / n_at_risk
            # Greenwood's formula for variance of S(t).
            denom = n_at_risk * (n_at_risk - d)
            if denom > 0:
                var_sum += d / denom
            se = s_running * math.sqrt(var_sum) if var_sum > 0 else 0.0
            cum_inc = 1.0 - s_running
            # Symmetric CI around incidence on the survival scale, then flip.
            lo = max(0.0, cum_inc - 1.96 * se)
            hi = min(1.0, cum_inc + 1.96 * se)
            times.append(t)
            surv.append(s_running)
            ci_lo.append(lo)
            ci_hi.append(hi)
            at_risk.append(n_at_risk)
        i = j

    incidence = [1.0 - s for s in surv]
    return times, incidence, ci_lo, ci_hi, at_risk


def build_chart_repeat_overdose_km(
    *,
    theme: str,
    durations: list[int],
    events: list[int],
    horizon_days: int = 365,
) -> str:
    """Cumulative-incidence (Kaplan-Meier) curve for time-to-repeat-overdose.

    The curve answers: "by day t after a patient's first observed overdose,
    what fraction have had a second one, given how long we've actually been
    able to watch each patient?" Unlike the per-quarter chart, this is not
    biased by left-truncation or right-censoring -- each patient contributes
    exactly the follow-up time we have for them, no more.

    A shaded band shows the Greenwood 95% confidence interval. With a small
    cohort the band will be wide; that's the math being honest.
    """
    if not durations:
        return ""

    times, incidence, ci_lo, ci_hi, at_risk = _km_cumulative_incidence(durations, events)
    if not times:
        return ""

    # Clip the curve at the requested horizon for readability. Most of the
    # action happens in the first year; tails beyond that are sparse and
    # noisy.
    clipped: list[tuple[int, float, float, float, int]] = [
        (t, inc, lo, hi, ar)
        for t, inc, lo, hi, ar in zip(times, incidence, ci_lo, ci_hi, at_risk, strict=True)
        if t <= horizon_days
    ]
    if not clipped:
        return ""

    t_vals = [row[0] for row in clipped]
    inc_pct = [row[1] * 100 for row in clipped]
    lo_pct = [row[2] * 100 for row in clipped]
    hi_pct = [row[3] * 100 for row in clipped]
    n_at_risk = [row[4] for row in clipped]

    fig = go.Figure()

    # CI band -- two traces, one filled to the next, so the area between
    # ci_upper and ci_lower shades. Has to be added before the main line so
    # the line draws on top.
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=hi_pct,
            mode="lines",
            line=dict(width=0, shape="hv"),
            hoverinfo="skip",
            showlegend=False,
            name="CI upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=lo_pct,
            mode="lines",
            line=dict(width=0, shape="hv"),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.15)",  # indigo-500 @ 15%
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        )
    )

    # Main step curve. shape='hv' gives the proper KM staircase look.
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=inc_pct,
            mode="lines",
            line=dict(color=CHART_COLORS_VIBRANT[0], width=3, shape="hv"),
            customdata=list(zip(n_at_risk, lo_pct, hi_pct, strict=True)),
            hovertemplate=(
                "Day %{x}<br>"
                "Cumulative repeat-OD rate: <b>%{y:.1f}%</b><br>"
                "95%% CI: %{customdata[1]:.1f}%% – %{customdata[2]:.1f}%%<br>"
                "Patients still at risk: %{customdata[0]}<extra></extra>"
            ),
            name="Cumulative repeat rate",
        )
    )

    # Reference markers at clinically meaningful follow-up windows.
    for marker_day in (30, 90, 180, 365):
        if marker_day <= horizon_days:
            fig.add_vline(
                x=marker_day,
                line=dict(color=TAILWIND_COLORS["slate-400"], width=1, dash="dot"),
                opacity=0.6,
                annotation_text=f"{marker_day}d",
                annotation_position="top",
                annotation=dict(
                    font=dict(size=10, color=TAILWIND_COLORS["slate-500"]),
                ),
            )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        x_title="Days since patient's first observed overdose",
        y_title="Cumulative repeat rate (%)",
        show_legend=False,
    )

    fig.update_xaxes(range=[0, horizon_days])
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(margin=dict(t=40))

    return pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


# ---------------------------------------------------------------------------
# Cohort-stratified Kaplan-Meier (one curve per index year)
# ---------------------------------------------------------------------------


def build_chart_repeat_overdose_km_by_cohort(
    *,
    theme: str,
    cohorts: list[tuple[str, list[int], list[int]]],
    horizon_days: int = 365,
) -> str:
    """Stratified KM curves -- one curve per index-year cohort.

    Why this exists:
        The single-curve KM tells you "what fraction of patients have a
        repeat by day t." The stratified version answers the question that
        actually matters for program evaluation: "are patients we picked up
        more recently doing better than patients we picked up earlier?"
        If the 2025 curve sits below the 2024 curve sits below the 2023
        curve, that's improvement -- and it's improvement that's already
        been adjusted for follow-up time, so the 2025 curve isn't penalized
        for having less data behind it.

    Args:
        cohorts: list of (label, durations, events) tuples. Each tuple is a
            cohort; the function plots one KM curve per cohort. Empty
            cohorts are skipped silently.
        horizon_days: x-axis cap. Curves are clipped here.
    """
    # Filter out empty cohorts up front so we don't draw an empty legend
    # entry or a flat line at 0%.
    real = [(lbl, d, e) for (lbl, d, e) in cohorts if d]
    if not real:
        return ""

    fig = go.Figure()

    # We only have so many "vibrant" colors and the cohorts are ordered
    # oldest -> newest, so the eye naturally tracks the progression. A
    # secondary visual signal (line dash) reinforces the ordering even for
    # colorblind viewers.
    palette = CHART_COLORS_VIBRANT
    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash"]

    for idx, (label, durations, events) in enumerate(real):
        times, incidence, ci_lo, ci_hi, at_risk = _km_cumulative_incidence(durations, events)
        if not times:
            continue

        clipped = [
            (t, inc, lo, hi, ar)
            for t, inc, lo, hi, ar in zip(times, incidence, ci_lo, ci_hi, at_risk, strict=True)
            if t <= horizon_days
        ]
        if not clipped:
            continue

        t_vals = [row[0] for row in clipped]
        inc_pct = [row[1] * 100 for row in clipped]
        lo_pct = [row[2] * 100 for row in clipped]
        hi_pct = [row[3] * 100 for row in clipped]
        n_at_risk = [row[4] for row in clipped]

        color = palette[idx % len(palette)]
        dash = dash_cycle[idx % len(dash_cycle)]
        cohort_n = len(durations)

        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=inc_pct,
                mode="lines",
                line=dict(color=color, width=3, shape="hv", dash=dash),
                name=f"{label} (n={cohort_n})",
                customdata=list(zip(n_at_risk, lo_pct, hi_pct, strict=True)),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Day %{x}<br>"
                    "Cumulative repeat-OD rate: <b>%{y:.1f}%</b><br>"
                    "95%% CI: %{customdata[1]:.1f}%% – %{customdata[2]:.1f}%%<br>"
                    "Patients still at risk: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    # Reference markers at clinically meaningful follow-up windows.
    for marker_day in (30, 90, 180, 365):
        if marker_day <= horizon_days:
            fig.add_vline(
                x=marker_day,
                line=dict(color=TAILWIND_COLORS["slate-400"], width=1, dash="dot"),
                opacity=0.6,
                annotation_text=f"{marker_day}d",
                annotation_position="top",
                annotation=dict(
                    font=dict(size=10, color=TAILWIND_COLORS["slate-500"]),
                ),
            )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=420,
        x_title="Days since patient's first observed overdose",
        y_title="Cumulative repeat rate (%)",
        show_legend=True,
    )

    fig.update_xaxes(range=[0, horizon_days])
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

    return pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )
