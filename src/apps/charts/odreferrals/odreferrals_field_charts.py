from collections.abc import Collection

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns, count_share_text
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import ODReferrals
from ..overdose.od_hist_monthly import (
    build_chart_od_hist_monthly,
    build_chart_top3_drugs_monthly,
)

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
    fig = px.pie(
        vc_df,
        names=label_col,
        values=value_col,
        hole=0.55,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=[vc_df["share_pct"].round(1)],
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
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


def _build_horizontal_bar(vc_df: pd.DataFrame, field: str, theme: str) -> str:
    vc_df["label_short"] = vc_df[field].astype(str).apply(lambda v: _shorten_label(v, 32))
    vc_df["full_label"] = vc_df[field].astype(str)
    vc_df = add_share_columns(vc_df, "count")
    text_values = [count_share_text(row["count"], row["share_pct"]) for _, row in vc_df.iterrows()]

    # Use go.Bar instead of px.bar to avoid template system bug
    fig = go.Figure(
        data=[
            go.Bar(
                x=vc_df["count"],
                y=vc_df["label_short"],
                orientation="h",
                text=text_values,
                textposition="outside",
                cliponaxis=False,
                marker_color=TAILWIND_COLORS["indigo-600"],
                customdata=vc_df[["full_label", "share_pct"]],
                hovertemplate="%{customdata[0]}<br>Count: %{x}<br>Share: %{customdata[1]:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(autorange="reversed")
    y_title = field.replace("_", " ").title()
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

    if field in {"narcan_given", "referral_to_sud_agency"}:
        s = df[field].map({True: "Yes", False: "No"}).fillna("Unknown")
    else:
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

    if field == "referral_agency" and not few_categories:
        vc2 = vc_df.rename(columns={field: "label"})
        return _build_treemap_chart(vc2, "label", "count", theme)
    if field in {
        "referral_source",
        "suspected_drug",
        "cpm_disposition",
        "living_situation",
        "engagement_location",
        "narcan_given",
        "referral_to_sud_agency",
        "delay_in_referral",
        "cpm_notification",
        "cpr_administered",
        "police_ita",
        "disposition",
        "transport_to_location",
        "transported_by",
    }:
        vc2 = vc_df.rename(columns={field: "label"})
        return _build_donut_chart(vc2, "label", "count", theme)

    return _build_horizontal_bar(vc_df, field, theme)


def _build_weekday_chart(df: pd.DataFrame, theme: str) -> str | None:
    if "od_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["od_date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return None
    dates = dates.dt.tz_localize(None, ambiguous="NaT", nonexistent="shift_forward")
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

    # Use go.Bar instead of px.bar to avoid template system bug
    fig = go.Figure(
        data=[
            go.Bar(
                x=weekday_counts["weekday"],
                y=weekday_counts["count"],
                text=text_values,
                textposition="outside",
                cliponaxis=False,
                marker_color=TAILWIND_COLORS["indigo-600"],
                customdata=weekday_counts[["share_pct"]],
                hovertemplate="Weekday: %{x}<br>Count: %{y}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
                showlegend=False,
            )
        ]
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title="OD referrals",
        margin={"t": 30, "l": 10, "r": 10, "b": 40},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        showticklabels=True,
        title="OD referrals",
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


def build_odreferrals_field_charts(
    theme: str,
    fields: Collection[str] | None = None,
) -> dict[str, str]:
    target_fields = {field for field in fields} if fields is not None else None
    qs = ODReferrals.objects.all().values(
        "od_date",
        "referral_source",
        "suspected_drug",
        "cpm_disposition",
        "living_situation",
        "engagement_location",
        "narcan_given",
        "referral_to_sud_agency",
        "delay_in_referral",
        "cpm_notification",
        "cpr_administered",
        "police_ita",
        "disposition",
        "transport_to_location",
        "transported_by",
    )
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    charts: dict[str, str] = {}
    if df.empty:
        return charts

    wants_monthly = target_fields is None or "odreferrals_counts_monthly" in target_fields
    if wants_monthly:
        monthly_chart = build_chart_od_hist_monthly(theme)
        top3_drugs_chart = build_chart_top3_drugs_monthly(theme)
        if monthly_chart and top3_drugs_chart:
            # Combine both charts with a separator
            combined_chart = f"""
            {monthly_chart}
            <div class="border-t border-slate-200/60 dark:border-slate-700/50 my-6"></div>
            <div class="px-2">
                <h4 class="text-md font-semibold text-gray-700 dark:text-gray-300 mb-4">Top 3 Suspected Drugs by Month</h4>
                {top3_drugs_chart}
            </div>
            """
            charts["odreferrals_counts_monthly"] = combined_chart

    wants_weekday = target_fields is None or "odreferrals_counts_weekday" in target_fields
    if wants_weekday:
        weekday_chart = _build_weekday_chart(df, theme)
        if weekday_chart:
            charts["odreferrals_counts_weekday"] = weekday_chart

    for field in [
        "referral_source",
        "suspected_drug",
        "cpm_disposition",
        "living_situation",
        "engagement_location",
        "narcan_given",
        "referral_to_sud_agency",
        "delay_in_referral",
        "cpm_notification",
        "cpr_administered",
        "police_ita",
        "disposition",
        "transport_to_location",
        "transported_by",
    ]:
        if target_fields is not None and field not in target_fields:
            continue
        chart = _build_chart_for_field(df, field, theme)
        if chart:
            charts[field] = chart

    if target_fields is None:
        return charts

    return {field: charts[field] for field in target_fields if field in charts}
