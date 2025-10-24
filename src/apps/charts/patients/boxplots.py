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
    return plot(
        fig,
        output_type="div",
        config=getattr(fig, "_config", {"responsive": True, "displaylogo": False}),
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
    # Order by count
    order = df["race"].value_counts().sort_values(ascending=False).index.tolist()
    fig = px.box(
        df,
        x="race",
        y="age",
        color="race",
        points="all",
        category_orders={"race": order},
        color_discrete_sequence=PATIENT_CHART_COLORS,  # Use standardized colors
    )
    fig = style_plotly_layout(fig, theme=theme, x_title="Race", y_title="Age", scroll_zoom=False)
    return plot(
        fig,
        output_type="div",
        config=getattr(fig, "_config", {"responsive": True, "displaylogo": False}),
    )
