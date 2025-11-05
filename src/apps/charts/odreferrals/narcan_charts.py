"""
Narcan administration analysis charts for OD referrals.

This module visualizes the evolving landscape of Narcan administration in overdose response,
highlighting the shift from EMS-first to bystander-first intervention and its clinical impact.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.db.models import Avg

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_narcan_administration_grid(theme: str) -> str:
    """
    Build visual grid showing maximum Narcan doses and dosages comparing
    bystander vs EMS administration patterns.

    This chart tells the story of how widespread naloxone distribution has changed
    overdose response dynamics - bystanders now often give higher doses before
    EMS arrival, leaving patients in precipitated withdrawal and less cooperative.

    Returns:
        HTML string containing the Plotly visualization
    """
    # Query all relevant narcan fields
    queryset = ODReferrals.objects.all().values(
        "narcan_given",
        "narcan_doses_prior_to_ems",
        "narcan_prior_to_ems_dosage",
        "narcan_doses_by_ems",
        "narcan_by_ems_dosage",
    )

    df = pd.DataFrame.from_records(list(queryset))

    # Clean data
    df = df[df["narcan_given"] == True]  # noqa: E712

    # Fill NaN values with 0 for calculations
    df["narcan_doses_prior_to_ems"] = df["narcan_doses_prior_to_ems"].fillna(0)
    df["narcan_prior_to_ems_dosage"] = df["narcan_prior_to_ems_dosage"].fillna(0.0)
    df["narcan_doses_by_ems"] = df["narcan_doses_by_ems"].fillna(0)
    df["narcan_by_ems_dosage"] = df["narcan_by_ems_dosage"].fillna(0.0)

    # Calculate total doses for each group
    bystander_total_doses = int(df["narcan_doses_prior_to_ems"].sum())
    ems_total_doses = int(df["narcan_doses_by_ems"].sum())

    # Calculate average doses per incident
    bystander_avg_doses = df[df["narcan_doses_prior_to_ems"] > 0][
        "narcan_doses_prior_to_ems"
    ].mean()
    ems_avg_doses = df[df["narcan_doses_by_ems"] > 0]["narcan_doses_by_ems"].mean()

    # Calculate average dosages
    bystander_avg_dosage = df[df["narcan_prior_to_ems_dosage"] > 0][
        "narcan_prior_to_ems_dosage"
    ].mean()
    ems_avg_dosage = df[df["narcan_by_ems_dosage"] > 0]["narcan_by_ems_dosage"].mean()

    # Count incidents where bystanders gave more doses than EMS
    bystander_led = len(df[df["narcan_doses_prior_to_ems"] > df["narcan_doses_by_ems"]])
    ems_led = len(df[df["narcan_doses_by_ems"] > df["narcan_doses_prior_to_ems"]])
    equal = len(df[df["narcan_doses_by_ems"] == df["narcan_doses_prior_to_ems"]])

    # Create figure with subplots
    fig = go.Figure()

    # Create bar chart comparing totals
    categories = [
        "Bystander<br>Total Doses",
        "EMS<br>Total Doses",
        "Bystander<br>Avg Doses",
        "EMS<br>Avg Doses",
        "Bystander<br>Avg Dosage (mg)",
        "EMS<br>Avg Dosage (mg)",
    ]

    values = [
        bystander_total_doses,
        ems_total_doses,
        bystander_avg_doses if not pd.isna(bystander_avg_doses) else 0,
        ems_avg_doses if not pd.isna(ems_avg_doses) else 0,
        bystander_avg_dosage if not pd.isna(bystander_avg_dosage) else 0,
        ems_avg_dosage if not pd.isna(ems_avg_dosage) else 0,
    ]

    colors = [
        CHART_COLORS_VIBRANT[0],  # Bystander
        CHART_COLORS_VIBRANT[1],  # EMS
        CHART_COLORS_VIBRANT[0],  # Bystander
        CHART_COLORS_VIBRANT[1],  # EMS
        CHART_COLORS_VIBRANT[0],  # Bystander
        CHART_COLORS_VIBRANT[1],  # EMS
    ]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" if i >= 2 else f"{v:,}" for i, v in enumerate(values)],
            textposition="outside",
            hovertemplate=("<b>%{x}</b><br>Value: %{y:.1f}<br><extra></extra>"),
        )
    )

    # Add annotation about the paradigm shift
    shift_pct = (
        (bystander_led / (bystander_led + ems_led + equal) * 100)
        if (bystander_led + ems_led + equal) > 0
        else 0
    )

    annotations = [
        dict(
            text=(
                f"<b>Paradigm Shift:</b> In {shift_pct:.0f}% of cases, bystanders administered more Narcan "
                f"doses than EMS<br>"
                f"<b>Clinical Impact:</b> Patients often awake in precipitated withdrawal before EMS arrival, "
                f"reducing cooperation & transport rates"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.15,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(size=11),
            align="center",
        )
    ]

    # Style the layout
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=500,
        x_title="",
        y_title="Count / Dosage",
        show_legend=False,
    )

    # Add annotations
    fig.update_layout(annotations=annotations, margin=dict(t=100))

    # Return HTML
    from plotly.offline import plot

    return plot(
        fig,
        include_plotlyjs=False,
        output_type="div",
        config={"displayModeBar": False},
    )


def build_narcan_response_timeline(theme: str) -> str:
    """
    Build timeline chart showing the evolution of who administers Narcan first.

    This chart visualizes the temporal shift from EMS-first to bystander-first
    Narcan administration by total dosage, helping illustrate when the paradigm changed.

    Returns:
        HTML string containing the Plotly visualization
    """
    queryset = ODReferrals.objects.exclude(od_date__isnull=True).values(
        "od_date",
        "narcan_given",
        "narcan_prior_to_ems_dosage",
        "narcan_by_ems_dosage",
    )

    df = pd.DataFrame.from_records(list(queryset))

    if df.empty or "od_date" not in df.columns:
        return ""

    # Convert dates
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date"])
    df = df[df["narcan_given"] == True]  # noqa: E712

    # Sort by date for chronological display
    df = df.sort_values("od_date")

    # Separate into bystander and EMS datasets (only include non-zero dosages)
    df_bystander = df[df["narcan_prior_to_ems_dosage"] > 0].copy()
    df_ems = df[df["narcan_by_ems_dosage"] > 0].copy()

    # Create scatter chart showing individual overdose dosages over time
    fig = go.Figure()

    # Add bystander dosage scatter with trend line
    if not df_bystander.empty:
        fig.add_trace(
            go.Scatter(
                x=df_bystander["od_date"],
                y=df_bystander["narcan_prior_to_ems_dosage"],
                mode="markers",
                name="Bystander Dosage",
                marker=dict(
                    color=CHART_COLORS_VIBRANT[0],
                    size=8,
                    opacity=0.6,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=(
                    "<b>Bystander Dosage</b><br>"
                    "Date: %{x|%b %d, %Y}<br>"
                    "Dosage (mg): %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Add bystander trend line using polynomial regression
        if len(df_bystander) >= 3:  # Need at least 3 points for trend
            # Convert dates to numeric for regression
            x_numeric = np.array(
                (df_bystander["od_date"] - df_bystander["od_date"].min()).dt.days.values
            )
            y_values = np.array(df_bystander["narcan_prior_to_ems_dosage"].values)

            # Fit polynomial (degree 2 for smooth curve)
            degree = min(2, len(df_bystander) - 1)
            coeffs = np.polyfit(x_numeric, y_values, degree)
            poly = np.poly1d(coeffs)

            # Generate smooth curve
            x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 100)
            y_smooth = poly(x_smooth)

            # Convert back to dates
            date_smooth = df_bystander["od_date"].min() + pd.to_timedelta(x_smooth, unit="D")

            fig.add_trace(
                go.Scatter(
                    x=date_smooth,
                    y=y_smooth,
                    mode="lines",
                    name="Bystander Trend",
                    line=dict(color=CHART_COLORS_VIBRANT[0], width=2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    # Add EMS dosage scatter with trend line
    if not df_ems.empty:
        fig.add_trace(
            go.Scatter(
                x=df_ems["od_date"],
                y=df_ems["narcan_by_ems_dosage"],
                mode="markers",
                name="EMS Dosage",
                marker=dict(
                    color=CHART_COLORS_VIBRANT[1],
                    size=8,
                    opacity=0.6,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=(
                    "<b>EMS Dosage</b><br>Date: %{x|%b %d, %Y}<br>Dosage (mg): %{y:.1f}<br><extra></extra>"
                ),
            )
        )

        # Add EMS trend line using polynomial regression
        if len(df_ems) >= 3:  # Need at least 3 points for trend
            # Convert dates to numeric for regression
            x_numeric = np.array((df_ems["od_date"] - df_ems["od_date"].min()).dt.days.values)
            y_values = np.array(df_ems["narcan_by_ems_dosage"].values)

            # Fit polynomial (degree 2 for smooth curve)
            degree = min(2, len(df_ems) - 1)
            coeffs = np.polyfit(x_numeric, y_values, degree)
            poly = np.poly1d(coeffs)

            # Generate smooth curve
            x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 100)
            y_smooth = poly(x_smooth)

            # Convert back to dates
            date_smooth = df_ems["od_date"].min() + pd.to_timedelta(x_smooth, unit="D")

            fig.add_trace(
                go.Scatter(
                    x=date_smooth,
                    y=y_smooth,
                    mode="lines",
                    name="EMS Trend",
                    line=dict(color=CHART_COLORS_VIBRANT[1], width=2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    # Style the layout
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=400,
        x_title="Date",
        y_title="Narcan Dosage per Overdose (mg)",
        show_legend=True,
    )

    # Return HTML
    from plotly.offline import plot

    return plot(
        fig,
        include_plotlyjs=False,
        output_type="div",
        config={"displayModeBar": False},
    )


def build_narcan_stats() -> list[dict[str, str]]:
    """
    Build quick stats about Narcan administration patterns.

    Returns:
        List of stat dictionaries with label, value, and description
    """
    # Total ODs where Narcan was given
    narcan_given_count = ODReferrals.objects.filter(narcan_given=True).count()
    total_count = ODReferrals.objects.count()
    narcan_rate = (narcan_given_count / total_count * 100) if total_count > 0 else 0

    # Average doses
    bystander_avg = (
        ODReferrals.objects.filter(narcan_given=True, narcan_doses_prior_to_ems__gt=0).aggregate(
            avg=Avg("narcan_doses_prior_to_ems")
        )["avg"]
        or 0
    )

    ems_avg = (
        ODReferrals.objects.filter(narcan_given=True, narcan_doses_by_ems__gt=0).aggregate(
            avg=Avg("narcan_doses_by_ems")
        )["avg"]
        or 0
    )

    # Cases where bystander gave more - calculate from raw data for simplicity
    queryset_compare = ODReferrals.objects.filter(narcan_given=True).values(
        "narcan_doses_prior_to_ems", "narcan_doses_by_ems"
    )
    df_compare = pd.DataFrame.from_records(list(queryset_compare))
    df_compare["narcan_doses_prior_to_ems"] = df_compare["narcan_doses_prior_to_ems"].fillna(0)
    df_compare["narcan_doses_by_ems"] = df_compare["narcan_doses_by_ems"].fillna(0)
    bystander_led = len(
        df_compare[df_compare["narcan_doses_prior_to_ems"] > df_compare["narcan_doses_by_ems"]]
    )

    bystander_led_pct = (bystander_led / narcan_given_count * 100) if narcan_given_count > 0 else 0

    return [
        {
            "label": "Narcan administered",
            "value": f"{narcan_given_count:,}",
            "description": f"{narcan_rate:.0f}% of all overdose referrals",
        },
        {
            "label": "Bystander-first cases",
            "value": f"{bystander_led:,}",
            "description": f"{bystander_led_pct:.0f}% had bystanders give more doses than EMS",
        },
        {
            "label": "Avg bystander doses",
            "value": f"{bystander_avg:.1f}",
            "description": "Per incident with bystander administration",
        },
        {
            "label": "Avg EMS doses",
            "value": f"{ems_avg:.1f}",
            "description": "Per incident with EMS administration",
        },
    ]
