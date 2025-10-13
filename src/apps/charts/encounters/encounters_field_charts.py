import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Encounters

COLOR_SEQUENCE = [
    TAILWIND_COLORS["indigo-600"],
    TAILWIND_COLORS["blue-500"],
    TAILWIND_COLORS["cyan-500"],
    TAILWIND_COLORS["teal-500"],
    TAILWIND_COLORS["emerald-500"],
    TAILWIND_COLORS["green-500"],
    TAILWIND_COLORS["yellow-500"],
    TAILWIND_COLORS["amber-500"],
    TAILWIND_COLORS["orange-500"],
    TAILWIND_COLORS["red-500"],
    TAILWIND_COLORS["pink-500"],
    TAILWIND_COLORS["purple-500"],
]

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _shorten_label(text: str, max_len: int = 28) -> str:
    try:
        s = str(text)
    except Exception:
        s = ""
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def _clean_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})
    s = s[~s.str.lower().isin({"not disclosed", "no data", "no-data", "no_data"})]
    return s


def _build_donut_chart(vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str) -> str:
    fig = px.pie(
        vc_df,
        names=label_col,
        values=value_col,
        hole=0.55,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value} (%{percent})<extra></extra>",
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title=None,
        margin={"t": 30, "l": 24, "r": 24, "b": 10},
    )
    fig.update_layout(showlegend=False)
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})


def _build_treemap_chart(vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str) -> str:
    fig = px.treemap(
        vc_df,
        path=[label_col],
        values=value_col,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(hovertemplate="%{label}<br>Count: %{value}<extra></extra>")
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title=None,
        margin={"t": 30, "l": 10, "r": 10, "b": 10},
    )
    fig.update_layout(showlegend=False)
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})


def _build_horizontal_bar(vc_df: pd.DataFrame, field: str, theme: str) -> str:
    vc_df["label_short"] = vc_df[field].astype(str).apply(lambda v: _shorten_label(v, 32))
    vc_df["full_label"] = vc_df[field].astype(str)
    fig = px.bar(
        vc_df,
        x="count",
        y="label_short",
        orientation="h",
        text="count",
        color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]],
        custom_data=["full_label"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_traces(hovertemplate="%{customdata[0]}<br>Count: %{x}<extra></extra>")
    fig.update_yaxes(autorange="reversed")
    y_title = "Primary Care Agency" if field == "pcp_agency" else field.replace("_", " ").title()
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title="Count",
        y_title=y_title,
        margin={"t": 30, "l": 140, "r": 10, "b": 40},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        showticklabels=True,
        title=y_title,
        ticklabelposition="outside",
        automargin=True,
    )
    fig.update_layout(bargap=0.1)
    return plot(
        fig,
        output_type="div",
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
        },
    )


def _build_chart_for_field(df: pd.DataFrame, field: str, theme: str) -> str | None:
    if field not in df.columns:
        return None
    s = _clean_series(df[field])
    if s.empty:
        return None
    vc_full = s.value_counts()
    top_n = 8
    vc_top = vc_full.head(top_n)
    other_count = int(vc_full.iloc[top_n:].sum()) if vc_full.size > top_n else 0
    vc_df = vc_top.reset_index()
    vc_df.columns = [field, "count"]
    if other_count > 0:
        vc_df = pd.concat(
            [vc_df, pd.DataFrame([{field: "Other", "count": other_count}])],
            ignore_index=True,
        )

    few_categories = int(vc_full.size) <= 3

    if field == "pcp_agency" and not few_categories:
        vc2 = vc_df.rename(columns={field: "label"})
        return _build_treemap_chart(vc2, "label", "count", theme)
    if field.startswith("encounter_type_cat") and not few_categories:
        vc2 = vc_df.rename(columns={field: "label"})
        return _build_donut_chart(vc2, "label", "count", theme)
    return _build_horizontal_bar(vc_df, field, theme)


def _build_monthly_chart(df: pd.DataFrame, theme: str) -> str | None:
    if "encounter_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["encounter_date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return None
    monthly = (
        dates.dt.to_period("M")
        .value_counts()
        .sort_index()
        .rename_axis("month")
        .reset_index(name="count")
    )
    monthly["label"] = monthly["month"].dt.to_timestamp().dt.strftime("%b %Y")
    fig = go.Figure(
        data=[
            go.Bar(
                x=monthly["label"],
                y=monthly["count"],
                text=monthly["count"],
                textposition="outside",
                marker_color=TAILWIND_COLORS["indigo-600"],
            )
        ]
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title="Encounter count",
        margin={"t": 30, "l": 10, "r": 10, "b": 60},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        showticklabels=True,
        title="Encounter count",
        ticklabelposition="outside",
        automargin=True,
    )
    fig.update_layout(bargap=0.1)
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})


def _build_weekday_chart(df: pd.DataFrame, theme: str) -> str | None:
    if "encounter_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["encounter_date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return None
    weekdays = dates.dt.day_name()
    weekday_counts = (
        weekdays.value_counts()
        .reindex(WEEKDAY_ORDER, fill_value=0)
        .rename_axis("weekday")
        .reset_index(name="count")
    )
    fig = px.bar(
        weekday_counts,
        x="weekday",
        y="count",
        text="count",
        color="weekday",
        color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]] * len(WEEKDAY_ORDER),
    )
    fig.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title="Encounter count",
        margin={"t": 30, "l": 10, "r": 10, "b": 40},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        showticklabels=True,
        title="Encounter count",
        ticklabelposition="outside",
        automargin=True,
    )
    fig.update_layout(bargap=0.1)
    return plot(
        fig,
        output_type="div",
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
        },
    )


def build_encounters_field_charts(theme: str) -> dict[str, str]:
    qs = Encounters.objects.all().values(
        "encounter_date",
        "pcp_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
    )
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    charts: dict[str, str] = {}

    if df.empty:
        return charts

    monthly_chart = _build_monthly_chart(df, theme)
    if monthly_chart:
        charts["encounters_counts_monthly"] = monthly_chart

    weekday_chart = _build_weekday_chart(df, theme)
    if weekday_chart:
        charts["encounters_counts_weekday"] = weekday_chart

    for field in [
        "pcp_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
    ]:
        chart = _build_chart_for_field(df, field, theme)
        if chart:
            charts[field] = chart

    return charts
