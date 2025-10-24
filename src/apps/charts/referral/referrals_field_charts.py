from collections.abc import Collection

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns, count_share_text, rolling_average
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Referrals


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _shorten_label(text: str, max_len: int = 24) -> str:
    try:
        s = str(text)
    except Exception:
        s = ""
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


# Use professional vibrant color palette for all charts
COLOR_SEQUENCE = CHART_COLORS_VIBRANT


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
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})


def _build_chart_for_field(df: pd.DataFrame, field: str, theme: str) -> str:
    s = df[field]
    if not _is_numeric_series(s):
        s = s.fillna("").replace({None: ""}).astype(str).str.strip()
        s = s.replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})

    if _is_numeric_series(df[field]):
        if field == "age":
            ages = pd.to_numeric(df[field], errors="coerce")
            bins = [-1, 17, 24, 34, 44, 54, 64, 74, 84, float("inf")]
            labels = ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]
            groups = pd.cut(ages, bins=bins, labels=labels, include_lowest=True, right=True)
            vc = groups.value_counts().reindex(labels, fill_value=0).reset_index()
            vc.columns = ["age_group", "count"]
            unknown_count = ages.isna().sum()
            if unknown_count > 0:
                vc = pd.concat(
                    [vc, pd.DataFrame([["Unknown", unknown_count]], columns=["age_group", "count"])]
                )
            vc = add_share_columns(vc, "count")
            text_values = [
                count_share_text(row["count"], row["share_pct"]) for _, row in vc.iterrows()
            ]
            fig = px.bar(
                vc,
                x="age_group",
                y="count",
                text=text_values,
                color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]],
                custom_data=vc[["share_pct"]],
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Age group: %{x}<br>Count: %{y}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
            )
            fig = style_plotly_layout(
                fig,
                theme=theme,
                height=360,
                x_title="Age group",
                y_title="Count",
                margin={"t": 30, "l": 10, "r": 10, "b": 40},
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=False,
                showticklabels=True,
                title="Count",
                ticklabelposition="outside",
                automargin=True,
            )
            fig.update_layout(bargap=0.1)
        else:
            fig = px.histogram(
                df, x=field, nbins=20, color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]]
            )
            fig.update_traces(hovertemplate=f"{field}: %{{x}}<br>Count: %{{y}}<extra></extra>")
            fig = style_plotly_layout(
                fig,
                theme=theme,
                height=360,
                x_title=field,
                y_title="Count",
                margin={"t": 30, "l": 10, "r": 10, "b": 40},
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=False,
                showticklabels=True,
                title="Count",
                ticklabelposition="outside",
                automargin=True,
            )
            fig.update_layout(bargap=0.1)
    else:
        s_clean = s[~s.str.lower().isin({"not disclosed", "single"})]
        # For encounter type categories, drop explicit 'No data' buckets
        if field.startswith("encounter_type_cat"):
            s_clean = s_clean[
                ~s_clean.str.strip().str.lower().isin({"no data", "no-data", "no_data"})
            ]
        vc_full = s_clean.value_counts()
        # Decide if we should use a simple bar for very few categories (<= 3)
        few_cats = int(s_clean.nunique()) <= 3
        top_n = 8
        vc_top = vc_full.head(top_n)
        other_count = int(vc_full.iloc[top_n:].sum()) if vc_full.size > top_n else 0
        vc = vc_top.reset_index()
        vc.columns = [field, "count"]
        if other_count > 0:
            vc = pd.concat(
                [vc, pd.DataFrame([{field: "Other", "count": other_count}])], ignore_index=True
            )

        # If very few categories, prefer a clean horizontal bar regardless of field
        if not few_cats:
            if field in {"insurance", "referral_closed_reason", "sex"}:
                vc2 = vc.rename(columns={field: "label"})
                return _build_donut_chart(vc2, "label", "count", theme)
            if field in {"referral_agency", "zipcode"}:
                vc2 = vc.rename(columns={field: "label"})
                return _build_treemap_chart(vc2, "label", "count", theme)

        vc["label_short"] = vc[field].astype(str).apply(lambda v: _shorten_label(v, 28))
        vc["full_label"] = vc[field].astype(str)
        vc = add_share_columns(vc, "count")
        text_values = [count_share_text(row["count"], row["share_pct"]) for _, row in vc.iterrows()]
        fig = px.bar(
            vc,
            x="count",
            y="label_short",
            orientation="h",
            text=text_values,
            color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]],
            custom_data=vc[["full_label", "share_pct"]],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_traces(
            hovertemplate="%{customdata[0]}<br>Count: %{x}<br>Share: %{customdata[1]:.1f}%<extra></extra>"
        )
        fig.update_yaxes(autorange="reversed")
        y_title = (
            "Referral Agency"
            if field == "referral_agency"
            else ("ZIP Code" if field == "zipcode" else field.replace("_", " ").title())
        )
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

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


def build_referrals_field_charts(
    theme: str,
    fields: Collection[str] | None = None,
) -> dict[str, str]:
    target_fields = {field for field in fields} if fields is not None else None
    qs = Referrals.objects.all().values(
        "age",
        "sex",
        "date_received",
        "referral_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
        "referral_closed_reason",
        "zipcode",
        "insurance",
        "referral_1",
        "referral_2",
        "referral_3",
        "referral_4",
        "referral_5",
    )
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    charts: dict[str, str] = {}
    for field in df.columns:
        try:
            if field in {"date_received"}:
                continue  # handled as quarterly chart below
            if target_fields is not None and field not in target_fields:
                continue
            charts[field] = _build_chart_for_field(df, field, theme)
        except Exception:
            continue

    # Add quarterly referral counts chart
    try:
        wants_quarterly = target_fields is None or "referrals_counts_quarterly" in target_fields
        if wants_quarterly and not df.empty and "date_received" in df.columns:
            dates = pd.to_datetime(df["date_received"], errors="coerce")
            qdf = pd.DataFrame(
                {
                    "year": dates.dt.year,
                    "quarter": "Q" + dates.dt.quarter.astype("Int64").astype(str),
                }
            ).dropna()
            qdf = qdf.groupby(["year", "quarter"], dropna=True).size().reset_index(name="count")
            if not qdf.empty:
                qdf2 = qdf.sort_values(["year", "quarter"]).reset_index(drop=True)
                qdf2 = add_share_columns(qdf2, "count")
                total_referrals = max(int(qdf2["count"].sum()), 1)
                qdf2["rate_per_100"] = (qdf2["count"] / total_referrals) * 100.0
                qdf2["text_label"] = [
                    count_share_text(row["count"], row["share_pct"]) for _, row in qdf2.iterrows()
                ]
                x_years = qdf2["year"].astype(str).tolist()
                x_quarters = qdf2["quarter"].astype(str).tolist()
                scatter_x = [[x_years[i], x_quarters[i]] for i in range(len(x_years))]
                rolling_counts = rolling_average(qdf2["count"], window=3)
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[x_years, x_quarters],
                            y=qdf2["count"],
                            text=qdf2["text_label"],
                            textposition="outside",
                            marker_color=TAILWIND_COLORS["indigo-600"],
                            customdata=qdf2[["share_pct", "rate_per_100"]],
                            hovertemplate=(
                                "Quarter %{x[0]} %{x[1]}<br>Count: %{y}<br>Share: %{customdata[0]:.1f}%<br>"
                                "Rate per 100 referrals: %{customdata[1]:.2f}<extra></extra>"
                            ),
                        )
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=scatter_x,
                        y=rolling_counts,
                        mode="lines",
                        name="3-quarter avg",
                        line=dict(color=TAILWIND_COLORS["indigo-300"], width=3),
                        hovertemplate="Quarter %{x[0]} %{x[1]}<br>3-quarter avg: %{y:.1f}<extra></extra>",
                    )
                )
                fig = style_plotly_layout(
                    fig,
                    theme=theme,
                    height=360,
                    x_title=None,
                    y_title="Count",
                    margin={"t": 30, "l": 10, "r": 10, "b": 60},
                )
                fig.update_xaxes(type="multicategory")
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(
                    showgrid=False,
                    showticklabels=True,
                    title="Count",
                    ticklabelposition="outside",
                    automargin=True,
                )
                fig.update_layout(
                    bargap=0.1,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                charts["referrals_counts_quarterly"] = plot(
                    fig, output_type="div", config={"responsive": True, "displaylogo": False}
                )
    except Exception:
        pass

    if target_fields is None:
        return charts

    return {field: charts[field] for field in target_fields if field in charts}
