import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Patients
from ..od_utils import get_quarterly_patient_counts
from .boxplots import build_patients_age_by_race_boxplot, build_patients_age_by_sex_boxplot


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


# Axis adjustments are applied inline after style_plotly_layout for each chart


def _shorten_label(text: str, max_len: int = 24) -> str:
    try:
        s = str(text)
    except Exception:
        s = ""
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


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
    # Use label as path for a single-level treemap
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


def _build_chart_for_field(df: pd.DataFrame, field: str, theme: str) -> str:
    s = df[field]
    # Clean values for categoricals
    if not _is_numeric_series(s):
        s = s.fillna("").replace({None: ""}).astype(str).str.strip()
        s = s.replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})

    if _is_numeric_series(df[field]):
        # Special-case medically common age bands
        if field == "age":
            ages = pd.to_numeric(df[field], errors="coerce")
            # Define medical age groups: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+
            bins = [-1, 17, 24, 34, 44, 54, 64, 74, 84, float("inf")]
            labels = [
                "0–17",
                "18–24",
                "25–34",
                "35–44",
                "45–54",
                "55–64",
                "65–74",
                "75–84",
                "85+",
            ]
            groups = pd.cut(ages, bins=bins, labels=labels, include_lowest=True, right=True)
            vc = groups.value_counts().reindex(labels, fill_value=0).reset_index()
            vc.columns = ["age_group", "count"]
            unknown_count = ages.isna().sum()
            if unknown_count > 0:
                # Append Unknown to the end
                vc = pd.concat(
                    [vc, pd.DataFrame([["Unknown", unknown_count]], columns=["age_group", "count"])]
                )

            fig = px.bar(
                vc,
                x="age_group",
                y="count",
                text="count",
                color_discrete_sequence=[TAILWIND_COLORS["indigo-600"]],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
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
        elif field == "sud":
            # Donut for SUD breakdown
            s2 = df[field].map({True: "Yes", False: "No"}).fillna("Unknown")
            vc = s2.value_counts().reindex(["Yes", "No", "Unknown"], fill_value=0).reset_index()
            vc.columns = ["label", "count"]
            return _build_donut_chart(vc, "label", "count", theme)
        else:
            # Generic numeric histogram with reasonable bins
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
        # Categorical: use creative chart types by field
        s_clean = s[~s.str.lower().isin({"not disclosed", "single"})]
        vc_full = s_clean.value_counts()
        # Build top list + "Other" for legibility
        top_n = 8
        vc_top = vc_full.head(top_n)
        other_count = int(vc_full.iloc[top_n:].sum()) if vc_full.size > top_n else 0
        vc = vc_top.reset_index()
        vc.columns = [field, "count"]
        if other_count > 0:
            vc = pd.concat(
                [vc, pd.DataFrame([{field: "Other", "count": other_count}])], ignore_index=True
            )

        # Choose chart by field
        if field in {"insurance", "marital_status", "veteran_status"}:
            vc2 = vc.rename(columns={field: "label"})
            return _build_donut_chart(vc2, "label", "count", theme)
        if field in {"pcp_agency", "zip_code"}:
            vc2 = vc.rename(columns={field: "label"})
            return _build_treemap_chart(vc2, "label", "count", theme)

        # Fallback: horizontal bar (shortened labels)
        vc["label_short"] = vc[field].astype(str).apply(lambda v: _shorten_label(v, 28))
        vc["full_label"] = vc[field].astype(str)
        fig = px.bar(
            vc,
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
        y_title = (
            "Primary Care Agency" if field == "pcp_agency" else field.replace("_", " ").title()
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


def build_patients_field_charts(theme: str) -> dict[str, str]:
    """
    Build a chart (HTML div) for each relevant field in Patients.
    Returns a mapping of field name -> chart HTML.
    """
    qs = Patients.objects.all().values(
        "age",
        "insurance",
        "pcp_agency",
        "race",
        "sex",
        "sud",
        "zip_code",
        "created_date",
        "modified_date",
        "marital_status",
        "veteran_status",
    )
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    charts: dict[str, str] = {}
    # Define which fields should be treated as categorical even if numeric-like
    categorical_overrides = {"zip_code", "sud"}
    # We will not chart raw created/modified dates; we'll add a single quarterly count chart instead
    for field in df.columns:
        try:
            series = df[field]
            # Skip raw created_date/modified_date charts
            if field in {"created_date", "modified_date"}:
                continue

            # Non-date fields
            if field in {"race", "sex"}:
                # Handled by dedicated boxplots appended later
                continue
            if field in categorical_overrides:
                charts[field] = _render_categorical_override(series, field, theme)
            else:
                charts[field] = _build_chart_for_field(df, field, theme)
        except Exception:
            # Skip problematic fields rather than failing page render
            continue

    # Add quarterly patient counts chart at the end
    try:
        q = get_quarterly_patient_counts()
        qdf = q["df"]
        if not qdf.empty:
            # Multicategory x-axis with go.Bar
            qdf2 = qdf.copy().sort_values(["year", "quarter"]).reset_index(drop=True)
            x_years = qdf2["year"].astype(str).tolist()
            x_quarters = qdf2["quarter"].astype(str).tolist()
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[x_years, x_quarters],
                        y=qdf2["count"],
                        text=qdf2["count"],
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
                y_title="Count",
                margin={"t": 30, "l": 10, "r": 10, "b": 60},
            )
            # Ensure multicategory axis
            fig.update_xaxes(type="multicategory")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=False,
                showticklabels=True,
                title="Count",
                ticklabelposition="outside",
                automargin=True,
            )
            fig.update_layout(bargap=0.1)
            charts["patient_counts_quarterly"] = plot(
                fig, output_type="div", config={"responsive": True, "displaylogo": False}
            )
        # Insert boxplots for Race and Sex (age distributions)
        box_sex = build_patients_age_by_sex_boxplot(theme)
        if box_sex:
            charts["sex_age_boxplot"] = box_sex
        box_race = build_patients_age_by_race_boxplot(theme)
        if box_race:
            charts["race_age_boxplot"] = box_race
    except Exception:
        pass

    return charts


def _render_categorical_override(series: pd.Series, field: str, theme: str) -> str:
    s = series.fillna("").astype(str).str.strip()
    s = s.replace({"": "Unknown"})
    s = s[~s.str.lower().isin({"not disclosed", "single"})]
    vc_full = s.value_counts()
    # Aggregate beyond top N as 'Other'
    top_n = 8
    vc_top = vc_full.head(top_n)
    other_count = int(vc_full.iloc[top_n:].sum()) if vc_full.size > top_n else 0
    vc = vc_top.reset_index()
    vc.columns = [field, "count"]
    if other_count > 0:
        vc = pd.concat(
            [vc, pd.DataFrame([{field: "Other", "count": other_count}])], ignore_index=True
        )

    if field == "sud":
        vc2 = vc.rename(columns={field: "label"})
        return _build_donut_chart(vc2, "label", "count", theme)
    if field in {"zip_code"}:
        vc2 = vc.rename(columns={field: "label"})
        return _build_treemap_chart(vc2, "label", "count", theme)

    # Fallback bar
    vc["label_short"] = vc[field].astype(str).apply(lambda v: _shorten_label(v, 28))
    vc["full_label"] = vc[field].astype(str)
    fig = px.bar(
        vc,
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
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})
