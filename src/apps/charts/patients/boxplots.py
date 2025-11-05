import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.plotly import style_plotly_layout

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


def build_patients_age_by_race_boxplot(theme: str) -> str:
    qs = Patients.objects.all().values("age", "race")
    df = pd.DataFrame.from_records(list(qs))
    if df.empty:
        return ""
    df = df.dropna(subset=["age", "race"]).copy()
    # Filter out undesired labels
    df["race"] = df["race"].astype(str).str.strip()
    df = df[~df["race"].str.lower().isin({"not disclosed", "single"})]

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

    # Use horizontal orientation (swap x and y) for better label visibility on narrow screens
    fig = px.box(
        df,
        y="race",  # Race on y-axis (vertical labels have more room)
        x="age",  # Age on x-axis
        color="race",
        points="all",
        orientation="h",  # Horizontal boxplots
        category_orders={"race": order},
        color_discrete_sequence=PATIENT_CHART_COLORS,  # Use standardized colors
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        x_title="Age",  # Swapped labels
        y_title="Race",
        scroll_zoom=False,
        margin={
            "t": 40,
            "l": 20,
            "r": 20,
            "b": 60,
        },  # Minimal margins, let automargin and ticklabelstandoff control spacing
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
