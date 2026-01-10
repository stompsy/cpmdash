import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.plotly import get_theme_colors, style_plotly_layout

from ...core.models import Patients

# Standardized color palette matching Veteran Care Coordination chart
PATIENT_CHART_COLORS = [
    CHART_COLORS_VIBRANT[0],  # Violet
    CHART_COLORS_VIBRANT[1],  # Cyan
    CHART_COLORS_VIBRANT[3],  # Emerald
    CHART_COLORS_VIBRANT[4],  # Amber
    CHART_COLORS_VIBRANT[5],  # Blue
    CHART_COLORS_VIBRANT[2],  # Rose
    CHART_COLORS_VIBRANT[7],  # Teal
    CHART_COLORS_VIBRANT[6],  # Pink
]


def build_patients_age_by_sex_boxplot(theme: str) -> str:
    qs = Patients.objects.all().values("age", "sex")
    df = pd.DataFrame.from_records(list(qs))
    if df.empty:
        return ""
    df = df.dropna(subset=["age", "sex"]).copy()
    # Filter out undesired labels
    df["sex"] = df["sex"].astype(str).str.strip()
    df = df[~df["sex"].str.lower().isin({"not disclosed", "single"})]
    # Order by count
    order = df["sex"].value_counts().sort_values(ascending=False).index.tolist()
    fig = px.box(
        df,
        x="sex",
        y="age",
        color="sex",
        points="all",
        category_orders={"sex": order},
        color_discrete_sequence=PATIENT_CHART_COLORS,  # Use standardized colors
    )
    fig = style_plotly_layout(fig, theme=theme, x_title="Sex", y_title="Age", scroll_zoom=False)

    # Add darkened horizontal gridlines
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        layer="below traces",
    )
    fig.update_xaxes(showgrid=False)

    return plot(
        fig,
        output_type="div",
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "toggleSpikelines",
            ],
        },
    )


def _build_patients_age_by_race_boxplot(
    *,
    theme: str,
    df: pd.DataFrame,
    include_missing: bool,
) -> str:
    if df.empty:
        return ""

    missing_label = "Missing"
    missing_tokens = {"unknown", "not disclosed", "no data", "na", "none", "nan"}

    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"]).copy()
    if df.empty:
        return ""

    df["race"] = df["race"].fillna("").astype(str).str.strip()
    race_lower = df["race"].str.lower()
    missing_mask = df["race"].eq("") | race_lower.isin(missing_tokens)
    if include_missing:
        df.loc[missing_mask, "race"] = missing_label
    else:
        df = df[~missing_mask].copy()
    if df.empty:
        return ""

    # Wrap long race labels for better display
    race_label_mapping = {
        "American Indian or Alaska Native": "American Indian or<br>Alaska Native",
        "Black or African American": "Black or<br>African American",
        "Hispanic or Latino": "Hispanic or<br>Latino",
        "Hawaiian or Other Pacific Islander": "Hawaiian or Other<br>Pacific Islander",
    }
    df["race"] = df["race"].replace(race_label_mapping)

    # Order by count (descending)
    order = df["race"].value_counts().sort_values(ascending=False).index.tolist()

    # We keep the visual traces the same (box + jittered points), but split hover behavior:
    # - Points: show age only
    # - Box glyphs: show min/q1/median/q3/max (no race; it's already on the axis)
    points_fig = px.box(
        df,
        y="race",
        x="age",
        color="race",
        points="all",
        orientation="h",
        category_orders={"race": order},
        color_discrete_sequence=PATIENT_CHART_COLORS,
    )

    box_fig = px.box(
        df,
        y="race",
        x="age",
        color="race",
        points=False,
        orientation="h",
        category_orders={"race": order},
        color_discrete_sequence=PATIENT_CHART_COLORS,
    )

    colors = get_theme_colors(theme)

    # Configure points layer: keep jittered points, hide box visuals, hover shows age only.
    points_fig.update_traces(
        hoveron="points",
        hovertemplate="Age: %{x}<extra></extra>",
        showlegend=False,
        line=dict(width=0),
        fillcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor=colors["hover_bg"],
            bordercolor=colors["hover_border"],
            font=dict(color=colors["hover_font"], family="Arial, sans-serif", size=14),
            align="left",
            namelength=-1,
        ),
    )

    # Configure box layer: box-only, hover shows stats only.
    box_fig.update_traces(
        hoveron="boxes",
        hovertemplate=(
            "Min: %{lowerfence:.0f}<br>"
            "Q1: %{q1:.0f}<br>"
            "Median: %{median:.0f}<br>"
            "Q3: %{q3:.0f}<br>"
            "Max: %{upperfence:.0f}"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor=colors["hover_bg"],
            bordercolor=colors["hover_border"],
            font=dict(color=colors["hover_font"], family="Arial, sans-serif", size=14),
            align="left",
            namelength=-1,
        ),
    )

    # Combine traces: points first (behind), then boxes (on top).
    fig = go.Figure()
    fig.add_traces(points_fig.data)
    fig.add_traces(box_fig.data)

    fig = style_plotly_layout(
        fig,
        theme=theme,
        x_title="Age",
        y_title="Race",
        scroll_zoom=False,
        margin={
            "t": 40,
            "l": 20,
            "r": 20,
            "b": 60,
        },
    )

    # Configure x-axis to constrain gridlines to plot area
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        layer="below traces",  # Ensure grid is behind data
    )

    # Configure y-axis with minimal gap between labels and axis
    fig.update_yaxes(
        showgrid=False,  # No horizontal gridlines for categorical y-axis
        automargin=True,  # Let Plotly calculate needed space for labels
        ticklabelstandoff=2,  # Minimal gap (~1/8 inch at 96 DPI) between labels and axis
        categoryorder="array",
        categoryarray=list(reversed(order)),
    )

    return plot(
        fig,
        output_type="div",
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "toggleSpikelines",
            ],
        },
    )


def build_patients_age_by_race_boxplot(theme: str, *, include_missing: bool = False) -> str:
    qs = Patients.objects.all().values("age", "race")
    df = pd.DataFrame.from_records(list(qs))
    return _build_patients_age_by_race_boxplot(theme=theme, df=df, include_missing=include_missing)
