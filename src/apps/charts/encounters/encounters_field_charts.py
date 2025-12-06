from collections.abc import Collection

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns, count_share_text, rolling_average
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Encounters

# Use professional vibrant color palette for all charts
COLOR_SEQUENCE = CHART_COLORS_VIBRANT

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
    vc_df = add_share_columns(vc_df, value_col)
    vc_df["share_pct_rounded"] = vc_df["share_pct"].fillna(0.0).round(1)
    fig = px.pie(
        vc_df,
        names=label_col,
        values=value_col,
        hole=0.55,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=vc_df["share_pct_rounded"].to_numpy(),
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="%{label}<br>%{customdata[0]:.1f}%",
        hovertemplate="%{label}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
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


def _build_treemap_chart(vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str) -> str:
    vc_df = add_share_columns(vc_df, value_col)
    fig = px.treemap(
        vc_df,
        path=[label_col],
        values=value_col,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=[vc_df["share_pct"].round(1)],
    )
    fig.update_traces(
        hovertemplate="%{label}<br>Count: %{value}<br>Share: %{customdata[0]:.1f}%<extra></extra>"
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title=None,
        margin={"t": 30, "l": 10, "r": 10, "b": 10},
    )
    fig.update_layout(showlegend=False)
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


def _build_hierarchical_encounter_treemap(df: pd.DataFrame, theme: str) -> str | None:
    """Build a hierarchical treemap showing Cat1 → Cat2 → Cat3 structure."""
    # Clean and prepare data
    df_clean = df[["encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"]].copy()

    # Fill NaN and clean values
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("").astype(str).str.strip()
        df_clean[col] = df_clean[col].replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})

    # Filter out unwanted values
    excluded = {"not disclosed", "no data", "no-data", "no_data", ""}
    for col in df_clean.columns:
        df_clean = df_clean[~df_clean[col].str.lower().isin(excluded)]

    if df_clean.empty:
        return None

    # Create hierarchical structure: All → Cat1 → Cat2 → Cat3
    df_clean["count"] = 1
    df_clean["root"] = "All Encounters"

    # Build the path
    treemap_df = (
        df_clean.groupby(
            ["root", "encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    if treemap_df.empty:
        return None

    # Use explicit color value for text
    text_color = "#0f172a" if theme == "light" else "#f8fafc"

    fig = px.treemap(
        treemap_df,
        path=["root", "encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"],
        values="count",
        color="encounter_type_cat1",  # Color by Cat1 for consistency
        color_discrete_sequence=COLOR_SEQUENCE,
    )

    fig.update_traces(
        textposition="top left",
        textfont=dict(size=12, family="Roboto, sans-serif", color=text_color),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        marker=dict(line=dict(width=1)),
        root_color="rgba(0,0,0,0)",
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=500,  # Taller for hierarchical visualization
        x_title=None,
        y_title=None,
        margin={"t": 10, "l": 10, "r": 10, "b": 10},
    )
    fig.update_layout(showlegend=False)

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


def _build_horizontal_bar(vc_df: pd.DataFrame, field: str, theme: str) -> str:
    vc_df["label_short"] = vc_df[field].astype(str).apply(lambda v: _shorten_label(v, 32))
    vc_df["full_label"] = vc_df[field].astype(str)
    vc_df = add_share_columns(vc_df, "count")
    text_values = [count_share_text(row["count"], row["share_pct"]) for _, row in vc_df.iterrows()]
    fig = px.bar(
        vc_df,
        x="count",
        y="label_short",
        orientation="h",
        text=text_values,
        color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]],
        custom_data=vc_df[["full_label", "share_pct"]],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_traces(
        hovertemplate="%{customdata[0]}<br>Count: %{x}<br>Share: %{customdata[1]:.1f}%<extra></extra>"
    )
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
        dates.dt.tz_localize(None)
        .dt.to_period("M")
        .value_counts()
        .sort_index()
        .rename_axis("month")
        .reset_index(name="count")
    )
    monthly["label"] = monthly["month"].dt.to_timestamp().dt.strftime("%b %Y")
    monthly = add_share_columns(monthly, "count")
    monthly["text_label"] = [
        count_share_text(row["count"], row["share_pct"]) for _, row in monthly.iterrows()
    ]
    rolling_counts = rolling_average(monthly["count"], window=3)
    fig = go.Figure(
        data=[
            go.Bar(
                x=monthly["label"],
                y=monthly["count"],
                text=monthly["text_label"],
                textposition="outside",
                marker_color=TAILWIND_COLORS["indigo-600"],
                customdata=monthly[["share_pct"]],
                hovertemplate=(
                    "Month: %{x}<br>Count: %{y}<br>Share: %{customdata[0]:.1f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["label"],
            y=rolling_counts,
            mode="lines",
            name="3-month avg",
            line=dict(color=TAILWIND_COLORS["indigo-300"], width=3),
            hovertemplate="Month: %{x}<br>3-month avg: %{y:.1f}<extra></extra>",
        )
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
    fig.update_layout(
        bargap=0.1,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    weekday_counts = add_share_columns(weekday_counts, "count")
    text_values = [
        count_share_text(row["count"], row["share_pct"]) for _, row in weekday_counts.iterrows()
    ]
    fig = px.bar(
        weekday_counts,
        x="weekday",
        y="count",
        text=text_values,
        color="weekday",
        color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]] * len(WEEKDAY_ORDER),
        custom_data=weekday_counts[["share_pct"]],
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
        hovertemplate="Weekday: %{x}<br>Count: %{y}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
    )
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


def _should_build_field(target_fields: set[str] | None, field_name: str) -> bool:
    """Check if a specific field should be built based on target fields."""
    return target_fields is None or field_name in target_fields


def _build_individual_field_charts(
    df: pd.DataFrame, theme: str, target_fields: set[str] | None
) -> dict[str, str]:
    """Build charts for individual encounter fields."""
    charts: dict[str, str] = {}
    for field in [
        "pcp_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
    ]:
        if target_fields is not None and field not in target_fields:
            continue
        chart = _build_chart_for_field(df, field, theme)
        if chart:
            charts[field] = chart
    return charts


def build_encounters_field_charts(
    theme: str,
    fields: Collection[str] | None = None,
) -> dict[str, str]:
    target_fields = {field for field in fields} if fields is not None else None
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

    if _should_build_field(target_fields, "encounters_counts_monthly"):
        monthly_chart = _build_monthly_chart(df, theme)
        if monthly_chart:
            charts["encounters_counts_monthly"] = monthly_chart

    if _should_build_field(target_fields, "encounters_counts_weekday"):
        weekday_chart = _build_weekday_chart(df, theme)
        if weekday_chart:
            charts["encounters_counts_weekday"] = weekday_chart

    # Build individual field charts (cat1, cat2, cat3, pcp_agency)
    field_charts = _build_individual_field_charts(df, theme, target_fields)
    charts.update(field_charts)

    if target_fields is None:
        return charts

    return {field: charts[field] for field in target_fields if field in charts}
