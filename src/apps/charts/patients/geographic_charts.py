"""
Geographic visualization charts for patient demographics.
- PCP Agency coverage map showing partnership distribution
- ZIP Code heat map showing patient density and trends
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.plotly import style_plotly_layout

from ...core.models import Patients


def build_pcp_agency_coverage_map(theme: str) -> str:
    """
    Geographic visualization of primary care agency partnerships.

    Shows:
    - Bubble map of PCP agencies by geographic location (if coordinates available)
    - Alternative: Enhanced bar chart with partnership metrics
    - Color-coded by patient volume
    - Shows partnership strength and coverage areas
    """
    # Get PCP agency data
    patients_qs = Patients.objects.all().values("pcp_agency", "zip_code")
    patients_data = list(patients_qs)
    df = pd.DataFrame.from_records(patients_data) if patients_data else pd.DataFrame()

    if df.empty:
        return "<p>No primary care agency data available</p>"

    # Clean PCP agency names
    df["pcp_agency"] = df["pcp_agency"].fillna("Unknown").astype(str).str.strip()
    df = df[df["pcp_agency"] != ""]

    if df.empty:
        return "<p>No primary care agency assignments found</p>"

    # Count patients by agency
    agency_counts = df.groupby("pcp_agency").size().reset_index(name="patient_count")
    agency_counts = agency_counts.sort_values("patient_count", ascending=True)

    # Calculate percentages
    total_patients = agency_counts["patient_count"].sum()
    agency_counts["percentage"] = (agency_counts["patient_count"] / total_patients * 100).round(1)

    # Get top agencies
    top_n = 12
    top_agencies = agency_counts.tail(top_n) if len(agency_counts) > top_n else agency_counts

    # Create horizontal bar chart with enhanced styling
    fig = go.Figure()

    # Use gradient colors based on count
    max_count = top_agencies["patient_count"].max()
    colors = []
    for count in top_agencies["patient_count"]:
        # Color intensity based on patient volume
        if count / max_count > 0.7:
            colors.append(CHART_COLORS_VIBRANT[3])  # Emerald - strong partnership
        elif count / max_count > 0.4:
            colors.append(CHART_COLORS_VIBRANT[1])  # Cyan - moderate partnership
        elif count / max_count > 0.2:
            colors.append(CHART_COLORS_VIBRANT[4])  # Amber - developing partnership
        else:
            colors.append(CHART_COLORS_VIBRANT[2])  # Rose - emerging partnership

    fig.add_trace(
        go.Bar(
            y=top_agencies["pcp_agency"],
            x=top_agencies["percentage"],  # Use percentage instead of count
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=[f"{pct}%" for pct in top_agencies["percentage"]],
            textposition="outside",
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]:.1f}%<br><extra></extra>"),
            customdata=top_agencies[["percentage"]],
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=max(400, len(top_agencies) * 35),
        x_title="Percentage",
        y_title="Primary Care Agency",
        margin={"t": 60, "l": 200, "r": 80, "b": 60},
    )

    fig.update_layout(
        title=dict(
            text="Primary Care Partnership Strength",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(size=11),
        ticklabelposition="outside left",
        ticklabelstandoff=20,  # Add 10px gap between labels and plot area
    )

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


def build_zip_code_heat_map(theme: str) -> str:
    """
    ZIP code heat map showing patient density and trends over time.

    Shows:
    - Horizontal bars by ZIP code
    - Color intensity = patient count
    - Sorted by count (highest at top)
    - Shows geographic concentration of service delivery
    """
    # Get ZIP code data with created_date for trends
    patients_qs = Patients.objects.all().values("zip_code", "created_date")
    patients_data = list(patients_qs)
    df = pd.DataFrame.from_records(patients_data) if patients_data else pd.DataFrame()

    if df.empty:
        return "<p>No ZIP code data available</p>"

    # Clean ZIP codes
    df["zip_code"] = df["zip_code"].fillna("Unknown").astype(str).str.strip()
    df = df[df["zip_code"] != ""]
    df = df[df["zip_code"] != "Unknown"]

    if df.empty:
        return "<p>No valid ZIP codes found</p>"

    # Parse dates
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")

    # Count patients by ZIP
    zip_counts = df.groupby("zip_code").size().reset_index(name="total_count")
    zip_counts = zip_counts.sort_values("total_count", ascending=True)

    # Calculate recent vs. older patients (trend indicator)
    if "created_date" in df.columns and not df["created_date"].isna().all():
        # Get cutoff date (6 months ago or median date)
        median_date = df["created_date"].median()
        recent_counts = (
            df[df["created_date"] >= median_date]
            .groupby("zip_code")
            .size()
            .reset_index(name="recent_count")
        )
        zip_counts = zip_counts.merge(recent_counts, on="zip_code", how="left")
        zip_counts["recent_count"] = zip_counts["recent_count"].fillna(0).astype(int)
        zip_counts["growth_rate"] = (
            zip_counts["recent_count"] / zip_counts["total_count"] * 100
        ).round(1)
    else:
        zip_counts["recent_count"] = 0
        zip_counts["growth_rate"] = 0

    # Get top ZIP codes
    top_n = 15
    top_zips = zip_counts.tail(top_n) if len(zip_counts) > top_n else zip_counts

    # Calculate percentages
    total_patients = zip_counts["total_count"].sum()
    top_zips["percentage"] = (top_zips["total_count"] / total_patients * 100).round(1)

    # Create horizontal bar chart with color coding
    fig = go.Figure()

    # Color bars based on growth rate
    colors = []
    for growth in top_zips["growth_rate"]:
        if growth > 60:
            colors.append(CHART_COLORS_VIBRANT[3])  # Emerald - growing fast
        elif growth > 40:
            colors.append(CHART_COLORS_VIBRANT[1])  # Cyan - steady growth
        elif growth > 20:
            colors.append(CHART_COLORS_VIBRANT[4])  # Amber - stable
        else:
            colors.append(CHART_COLORS_VIBRANT[6])  # Pink - declining/stable

    fig.add_trace(
        go.Bar(
            y=top_zips["zip_code"],
            x=top_zips["percentage"],  # Use percentage instead of count
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=[f"{pct:.1f}%" for pct in top_zips["percentage"]],
            textposition="outside",
            hovertemplate=("<b>ZIP: %{y}</b><br>%{x:.1f}%<br><extra></extra>"),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=max(400, len(top_zips) * 32),
        x_title="Percentage",
        y_title="ZIP Code",
        margin={"t": 60, "l": 80, "r": 80, "b": 60},
    )

    fig.update_layout(
        title=dict(
            text="Patient Distribution by ZIP Code",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12))

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)
