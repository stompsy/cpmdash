from collections.abc import Collection

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Patients
from ..od_utils import get_quarterly_patient_counts
from .boxplots import build_patients_age_by_race_boxplot


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


# Axis adjustments are applied inline after style_plotly_layout for each chart


def _shorten_label(text: str, max_len: int = 24) -> str:
    try:
        s = str(text)
    except Exception:
        s = ""
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


# Standardized color palette matching Veteran Care Coordination chart
# Using the same colors for consistency across all patient charts
PATIENT_CHART_COLORS = [
    CHART_COLORS_VIBRANT[0],  # Violet - primary
    CHART_COLORS_VIBRANT[1],  # Cyan - secondary
    CHART_COLORS_VIBRANT[3],  # Emerald - success/positive
    CHART_COLORS_VIBRANT[4],  # Amber - warning/attention
    CHART_COLORS_VIBRANT[5],  # Blue - tertiary
    CHART_COLORS_VIBRANT[2],  # Rose - quaternary (for variety)
    CHART_COLORS_VIBRANT[7],  # Teal - additional
    CHART_COLORS_VIBRANT[6],  # Pink - additional
    CHART_COLORS_VIBRANT[9],  # Indigo - additional
    CHART_COLORS_VIBRANT[10],  # Lime - additional
    CHART_COLORS_VIBRANT[8],  # Orange - additional
    CHART_COLORS_VIBRANT[11],  # Fuchsia - additional
]
COLOR_SEQUENCE = PATIENT_CHART_COLORS


MISSING_LABEL = "Missing"
ZIP_MISSING_LABEL = "Missing ZIP"
_MISSING_TOKENS = {"unknown", "not disclosed", "no data", "na", "none", "nan"}


def _top_n_counts_with_other_and_missing(
    s: pd.Series,
    field: str,
    *,
    top_n: int = 8,
    include_other: bool = True,
    missing_label: str = MISSING_LABEL,
    include_missing: bool = True,
) -> pd.DataFrame:
    vc_full = s.value_counts()
    missing_count = int(vc_full.get(missing_label, 0))
    vc_ranked = vc_full.drop(labels=[missing_label], errors="ignore")
    vc_top = vc_ranked.head(top_n)
    other_count = int(vc_ranked.iloc[top_n:].sum()) if vc_ranked.size > top_n else 0

    vc = vc_top.reset_index()
    vc.columns = [field, "count"]
    if include_other and other_count > 0:
        vc = pd.concat(
            [vc, pd.DataFrame([{field: "Other", "count": other_count}])], ignore_index=True
        )
    if include_missing and missing_count > 0:
        vc = pd.concat(
            [vc, pd.DataFrame([{field: missing_label, "count": missing_count}])],
            ignore_index=True,
        )
    return vc


def _build_donut_chart(
    vc_df: pd.DataFrame,
    label_col: str,
    value_col: str,
    theme: str,
    legend_filter: str | None = None,
) -> str:
    vc_df = add_share_columns(vc_df, value_col)
    # Plotly's numeric formatting can surface "NaN%" in pie text/hover in some edge cases.
    # Pre-format as strings so the browser never has to do numeric formatting.
    vc_df["share_text"] = vc_df["share_pct"].fillna(0.0).apply(lambda v: f"{v:.1f}%")
    fig = px.pie(
        vc_df,
        names=label_col,
        values=value_col,
        hole=0.5,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(
        textposition="inside",
        textfont=dict(size=16, color="#1e293b", family="Arial, sans-serif"),
        text=vc_df["share_text"],
        texttemplate="%{text}",
        hovertemplate="%{label}<br>Share: %{text}<extra></extra>",
        insidetextorientation="radial",
        marker=dict(line=dict(color="white", width=1)),
    )

    # If legend_filter is specified, hide all legend items except the specified one
    if legend_filter:
        fig.for_each_trace(
            lambda trace: trace.update(showlegend=False) if trace.name != legend_filter else None
        )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=460,  # Keep donuts consistent and larger
        x_title=None,
        y_title=None,
        margin={"t": 40, "l": 20, "r": 20, "b": 80},  # Extra bottom margin for legend
    )
    # Legend positioned below chart on all screen sizes for consistency
    # This ensures proper display on both desktop and mobile
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal orientation wraps better
            yanchor="top",
            y=-0.15,  # Position below chart
            xanchor="center",
            x=0.5,  # Center horizontally
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
        ),
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


def _build_chart_for_field(
    df: pd.DataFrame,
    field: str,
    theme: str,
    *,
    include_missing: bool = True,
) -> str:
    s = df[field]
    # Clean values for categoricals
    if not _is_numeric_series(s):
        s = s.fillna("").replace({None: ""}).astype(str).str.strip()

    if _is_numeric_series(df[field]) or field == "age":
        # Special-case medically common age bands
        if field == "age":
            # Create stacked bar chart by sex
            ages = pd.to_numeric(df["age"], errors="coerce")
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
            x_labels = labels + (["Missing age"] if include_missing else [])

            # Create a temporary dataframe with age groups and sex
            temp_df = df[["age", "sex"]].copy()
            temp_df["age_group"] = pd.cut(
                ages, bins=bins, labels=labels, include_lowest=True, right=True
            )

            if include_missing:
                temp_df["age_group"] = (
                    temp_df["age_group"].cat.add_categories(["Missing age"]).fillna("Missing age")
                )

            sex_lower = temp_df["sex"].fillna("").astype(str).str.strip().str.lower()
            male_mask = sex_lower.isin(["male", "m"])
            female_mask = sex_lower.isin(["female", "f"])
            missing_sex_mask = ~(male_mask | female_mask)

            male_df = temp_df[male_mask]
            female_df = temp_df[female_mask]
            missing_sex_df = temp_df[missing_sex_mask] if include_missing else temp_df.iloc[0:0]

            male_counts = male_df["age_group"].value_counts().reindex(x_labels, fill_value=0)
            female_counts = female_df["age_group"].value_counts().reindex(x_labels, fill_value=0)
            missing_sex_counts = (
                missing_sex_df["age_group"].value_counts().reindex(x_labels, fill_value=0)
                if include_missing
                else pd.Series([0] * len(x_labels), index=x_labels)
            )

            male_total = int(male_counts.sum())
            female_total = int(female_counts.sum())
            missing_sex_total = int(missing_sex_counts.sum())
            male_pct = [
                ((count / male_total) * 100.0) if male_total else 0.0
                for count in male_counts.values
            ]
            female_pct = [
                ((count / female_total) * 100.0) if female_total else 0.0
                for count in female_counts.values
            ]
            missing_sex_pct = [
                ((count / missing_sex_total) * 100.0) if missing_sex_total else 0.0
                for count in missing_sex_counts.values
            ]

            # Create stacked bar chart
            fig = go.Figure()

            # Add male bars with standardized color
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=male_counts.values,
                    name="Male",
                    marker_color=PATIENT_CHART_COLORS[0],  # Violet - non-gendered
                    customdata=male_pct,
                    hovertemplate="Age: %{x}<br>Male: %{customdata:.1f}%<extra></extra>",
                )
            )

            # Add female bars with standardized color
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=female_counts.values,
                    name="Female",
                    marker_color=PATIENT_CHART_COLORS[3],  # Amber - non-gendered
                    customdata=female_pct,
                    hovertemplate="Age: %{x}<br>Female: %{customdata:.1f}%<extra></extra>",
                )
            )

            if include_missing and missing_sex_total > 0:
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=missing_sex_counts.values,
                        name="Missing",
                        marker_color=TAILWIND_COLORS["slate-500"],
                        customdata=missing_sex_pct,
                        hovertemplate="Age: %{x}<br>Missing: %{customdata:.1f}%<extra></extra>",
                    )
                )

            fig = style_plotly_layout(
                fig,
                theme=theme,
                height=380,
                x_title="Age Group",
                y_title="Patient Count",
                margin={
                    "t": 40,
                    "l": 80,
                    "r": 20,
                    "b": 50,
                },  # Increased left margin for y-axis label gap
                show_legend=True,
            )

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(128,128,128,0.15)",
                ticklabelstandoff=10,  # Add gap between y-axis labels and bars
            )
            fig.update_layout(
                barmode="stack",
                bargap=0.15,  # Add gap between bars
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.15,
                    xanchor="left",
                    x=0,
                ),
            )
        elif field in {"sud", "behavioral_health"}:
            # Donut for boolean health flags
            s2 = df[field].map({True: "Yes", False: "No"}).fillna(MISSING_LABEL)
            vc = s2.value_counts().reindex(["Yes", "No", MISSING_LABEL], fill_value=0).reset_index()
            vc.columns = ["label", "count"]
            return _build_donut_chart(vc, "label", "count", theme)
        else:
            # Generic numeric histogram with reasonable bins
            fig = px.histogram(
                df,
                x=field,
                nbins=20,
                color_discrete_sequence=[PATIENT_CHART_COLORS[0]],  # Violet
            )
            fig.update_traces(hovertemplate=f"{field}: %{{x}}<br>Count: %{{y}}<extra></extra>")
            fig = style_plotly_layout(
                fig,
                theme=theme,
                height=360,
                x_title=field,
                y_title="Count",
                margin={"t": 30, "l": 60, "r": 10, "b": 40},  # Increased left margin
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=False,
                showticklabels=True,
                title="Count",
                ticklabelposition="outside",
                ticklabelstandoff=10,  # Add gap between y-axis labels and bars
                automargin=True,
            )
            fig.update_layout(bargap=0.15)  # Add gap between bars
    else:
        # Categorical: use creative chart types by field
        s_clean = s.astype(str).str.strip()
        s_lower = s_clean.str.lower()
        missing_mask = s_clean.eq("") | s_lower.isin(_MISSING_TOKENS)
        if field == "pcp_agency":
            s_clean = s_clean[~missing_mask]
        else:
            s_clean = (
                s_clean.mask(missing_mask, MISSING_LABEL)
                if include_missing
                else s_clean[~missing_mask]
            )

        vc = _top_n_counts_with_other_and_missing(
            s_clean,
            field,
            include_other=(field != "pcp_agency"),
            include_missing=include_missing,
        )

        # Choose chart by field
        if field in {"insurance", "marital_status", "veteran_status"}:
            vc2 = vc.rename(columns={field: "label"})
            return _build_donut_chart(vc2, "label", "count", theme)

        # Fallback: vertical bar (shortened labels)
        if field == "pcp_agency":
            # Remap labels as requested
            label_map = {
                "NOHN - Medical": "NOHN",
                "OMC - Primary": "OMC",
                "Jamestown - Medical": "Jamestown",
                "PBH - Medical": "PBH",
            }
            vc["label_short"] = (
                vc[field].replace(label_map).astype(str).apply(lambda v: _shorten_label(v, 16))
            )
            # Use different colors for each bar
            colors = [PATIENT_CHART_COLORS[i % len(PATIENT_CHART_COLORS)] for i in range(len(vc))]
        else:
            vc["label_short"] = vc[field].astype(str).apply(lambda v: _shorten_label(v, 28))
            colors = [PATIENT_CHART_COLORS[0]] * len(vc)

        vc["full_label"] = vc[field].astype(str)
        vc = add_share_columns(vc, "count")
        # Show only percentages with leading spaces for gap
        text_values = [f"  {row['share_pct']:.1f}%" for _, row in vc.iterrows()]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=vc["label_short"],
                y=vc["share_pct"],
                text=text_values,
                textposition="outside",
                marker=dict(color=colors),
                customdata=vc[["full_label", "share_pct"]].values,
                hovertemplate="%{customdata[0]}<br>%{customdata[1]:.1f}%<extra></extra>",
                cliponaxis=False,
                textfont=dict(size=14, color="#1e293b", family="Arial, sans-serif"),
            )
        )

        x_title = (
            "Primary Care Agency" if field == "pcp_agency" else field.replace("_", " ").title()
        )
        fig = style_plotly_layout(
            fig,
            theme=theme,
            height=400,  # Increased height to prevent cutoff
            x_title=x_title,
            y_title="Percentage",
            margin={"t": 40, "l": 70, "r": 20, "b": 90},  # More space for labels and text
        )
        fig.update_xaxes(
            showgrid=False,
            tickangle=0,  # Horizontal labels
        )
        fig.update_yaxes(
            showgrid=True,  # Enable horizontal gridlines
            gridcolor="rgba(128,128,128,0.15)",
            showticklabels=True,
            title="Percentage",
            automargin=True,
        )
        fig.update_layout(bargap=0.15, showlegend=False)

    config = {
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
    }
    return plot(fig, output_type="div", config=config)


def _patient_chart_html(
    df: pd.DataFrame,
    field: str,
    theme: str,
    categorical_overrides: set[str],
    *,
    zip_include_missing: bool,
    include_missing: bool,
) -> str:
    try:
        if field in categorical_overrides:
            return _render_categorical_override(
                df[field],
                field,
                theme,
                zip_include_missing=zip_include_missing,
                include_missing=include_missing,
            )
        return _build_chart_for_field(df, field, theme, include_missing=include_missing)
    except Exception:
        return ""


def build_patients_field_charts(
    theme: str,
    fields: Collection[str] | None = None,
    *,
    zip_include_missing: bool = False,
    include_missing: bool = True,
) -> dict[str, str]:
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
        "behavioral_health",
        "zip_code",
        "created_date",
        "modified_date",
        "marital_status",
        "veteran_status",
    )
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    requested = set(fields) if fields is not None else None
    charts: dict[str, str] = {}

    def wants(name: str) -> bool:
        return requested is None or name in requested

    categorical_overrides = {"zip_code", "sud", "behavioral_health"}
    skip_fields = {"created_date", "modified_date", "race", "sex"}

    for field in (col for col in df.columns if col not in skip_fields):
        if not wants(field):
            continue
        chart_html = _patient_chart_html(
            df,
            field,
            theme,
            categorical_overrides,
            zip_include_missing=zip_include_missing,
            include_missing=include_missing,
        )
        if chart_html:
            charts[field] = chart_html

    def add_chart(name: str, builder) -> None:
        if not wants(name) or name in charts:
            return
        try:
            chart = builder(theme)
        except Exception:
            return
        if chart:
            charts[name] = chart

    add_chart("patient_counts_quarterly", _build_quarterly_patient_bar)
    # Removed sex_age_boxplot - redundant with stacked age chart
    add_chart(
        "race_age_boxplot",
        lambda theme_arg: build_patients_age_by_race_boxplot(
            theme_arg, include_missing=include_missing
        ),
    )

    # Add enhanced production charts
    from .production_age_charts import build_enhanced_age_referral_sankey

    add_chart("age_referral_sankey", build_enhanced_age_referral_sankey)

    if requested is not None:
        charts = {k: v for k, v in charts.items() if k in requested}

    return charts


def _build_quarterly_patient_bar(theme: str) -> str | None:  # noqa: C901
    q = get_quarterly_patient_counts()
    qdf = q.get("df")
    if not isinstance(qdf, pd.DataFrame) or qdf.empty:
        return None

    qdf2 = qdf.copy().sort_values(["year", "quarter"]).reset_index(drop=True)
    qdf2 = add_share_columns(qdf2, "count")

    # Simple text labels showing just the count
    qdf2["text_label"] = qdf2["count"].apply(lambda x: f"{int(x)}")

    x_years = qdf2["year"].astype(str).tolist()
    x_quarters = qdf2["quarter"].astype(str).tolist()

    # Calculate dynamic average from the entire dataset
    normalized_baseline = qdf2["count"].mean()

    # Color bars based on whether they exceed the average
    # Emerald if above average, Cyan if at or below average (matching Veteran Care Coordination palette)
    bar_colors = []
    for _, row in qdf2.iterrows():
        if row["count"] > normalized_baseline:
            bar_colors.append(PATIENT_CHART_COLORS[2])  # Emerald - above average
        else:
            bar_colors.append(PATIENT_CHART_COLORS[1])  # Cyan - at or below average

    fig = go.Figure(
        data=[
            go.Bar(
                x=[x_years, x_quarters],  # Years on top, quarters below
                y=qdf2["count"],
                text=qdf2["text_label"],
                textposition="outside",
                marker_color=bar_colors,
                customdata=qdf2[["share_pct"]],
                hovertemplate=(
                    "Quarter %{x[0]} %{x[1]}<br>"
                    "Patients: %{y}<br>"
                    "Share of Total: %{customdata[0]:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    # Add annotation for Q1 2025 - 2 new CPMs hired
    q1_2025_idx = None
    for i, (year, quarter) in enumerate(zip(x_years, x_quarters, strict=False)):
        if year == "2025" and quarter == "1":
            q1_2025_idx = i
            break

    if q1_2025_idx is not None and q1_2025_idx < len(qdf2):
        q1_count = qdf2.iloc[q1_2025_idx]["count"]
        fig.add_annotation(
            x=[[x_years[q1_2025_idx]], [x_quarters[q1_2025_idx]]],
            y=q1_count,
            text="<b>2 New CPMs Hired</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=PATIENT_CHART_COLORS[2],  # Emerald
            ax=0,
            ay=-70,
            bgcolor="rgba(16, 185, 129, 0.8)",  # Emerald with transparency
            bordercolor=PATIENT_CHART_COLORS[2],  # Emerald
            borderwidth=2,
            borderpad=4,
            font=dict(size=11, color="white"),
        )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=620,  # Increased height for taller bars (~1.5 inches more at 72 DPI = ~108 pixels)
        x_title=None,
        y_title="Patient Count",
        margin={"t": 40, "l": 90, "r": 20, "b": 20},  # Minimal top margin to reduce gap with header
    )
    fig.update_xaxes(
        type="multicategory",
        showgrid=False,
        ticklabelstandoff=2,  # Reduced space between tick labels and axis baseline
    )
    # Set y-axis range with fixed top at 300 for consistent annotation positioning
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        showticklabels=True,
        title="Patient Count",
        ticklabelstandoff=10,  # Space between tick labels and axis
        ticklabelposition="outside",
        automargin=True,
        range=[0, 300],  # Fixed range with top at 300 for annotation space
    )

    # Calculate baseline average from 2021, 2023, and 2024 only
    # These are the "normal" years excluding 2022 (Behavioral Health spike) and 2025 (+2 CPMs)
    baseline_years = ["2021", "2023", "2024"]
    baseline_data = qdf2[qdf2["year"].astype(str).isin(baseline_years)]
    normalized_quarterly_avg = round(baseline_data["count"].mean(), 1)

    # Collect all annotations (including year labels above bars)
    all_annotations = []

    # Add the existing "2 New CPMs Hired" annotation if it exists
    if fig.layout.annotations:
        all_annotations.extend(list(fig.layout.annotations))

    # Add year-based context annotations ABOVE bars (vertical text)
    unique_years = sorted(set(x_years))

    year_labels = {
        "2021": "COVID-19",
        "2022": "Behavioral<br>Health",
        "2023": "Normalization",
        "2024": "Normalization",
        "2025": "+2 Community<br>Paramedics",
    }

    # Position annotations at y=270 (with chart top at y=300)
    annotation_y = 270

    for year in unique_years:
        if year in year_labels:
            # Find all quarter indices for this year
            year_indices = [i for i, y in enumerate(x_years) if y == year]
            if year_indices:
                # Calculate center position (middle of the year's quarters)
                # For multicategory axis, use numeric position at the center
                center_idx = (year_indices[0] + year_indices[-1]) / 2.0

                # Add vertical text annotation centered over the year
                all_annotations.append(
                    dict(
                        x=center_idx,  # Numeric position at center of year
                        xref="x",
                        y=annotation_y,  # Positioned at y=275 in data coordinates
                        yref="y",  # Use data coordinates for precise positioning
                        text=f"<b>{year_labels[year]}</b>",  # Semibold text for readability
                        textangle=270,  # Vertical text reading top-to-bottom
                        showarrow=False,
                        font=dict(size=12, color="#1e293b", family="Arial, sans-serif"),
                        xanchor="center",
                        yanchor="middle",  # Anchor to middle for centered alignment
                    )
                )

    # Create background rectangles for each year with distinct colors
    shapes = []

    # Define year colors from the patient chart palette
    year_colors = {
        "2021": PATIENT_CHART_COLORS[0],  # Violet
        "2022": PATIENT_CHART_COLORS[1],  # Cyan
        "2023": PATIENT_CHART_COLORS[3],  # Emerald
        "2024": PATIENT_CHART_COLORS[4],  # Amber
        "2025": PATIENT_CHART_COLORS[5],  # Blue
    }

    for year in unique_years:
        # Find all quarter indices for this year
        year_indices = [i for i, y in enumerate(x_years) if y == year]
        if year_indices:
            # Calculate the span of this year's quarters (left and right edges)
            x0 = year_indices[0] - 0.5  # Left edge of first quarter
            x1 = year_indices[-1] + 0.5  # Right edge of last quarter

            # Add semi-transparent background rectangle for this year
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,  # Full height - annotations positioned inside with clearance
                    fillcolor=year_colors.get(year, PATIENT_CHART_COLORS[0]),
                    opacity=0.08,  # Very subtle background
                    layer="below",
                    line_width=0,
                )
            )

    # Add the baseline average line
    shapes.append(
        dict(
            type="line",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=normalized_quarterly_avg,
            y1=normalized_quarterly_avg,
            line=dict(
                color="rgba(255, 255, 255, 0.5)",  # White with 50% opacity
                width=2,
                dash="dash",
            ),
        )
    )

    fig.update_layout(
        bargap=0.15,
        showlegend=False,
        annotations=all_annotations,
        shapes=shapes,
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


def _render_categorical_override(
    series: pd.Series,
    field: str,
    theme: str,
    *,
    zip_include_missing: bool = False,
    include_missing: bool = True,
) -> str:
    # Special handling for boolean health flags
    if field in {"sud", "behavioral_health"}:
        s2 = series.map({True: "Yes", False: "No"}).fillna(MISSING_LABEL)
        vc = s2.value_counts().reindex(["Yes", "No", MISSING_LABEL], fill_value=0).reset_index()
        vc.columns = ["label", "count"]
        return _build_donut_chart(vc, "label", "count", theme)

    s = series.fillna("").astype(str).str.strip()

    # Special handling for ZIP code chart: allow toggling missing ZIP values on/off.
    if field == "zip_code":
        missing_label = ZIP_MISSING_LABEL
        s_lower = s.str.lower()
        missing_mask = s.eq("") | s_lower.isin(_MISSING_TOKENS)
        s = s.mask(missing_mask, missing_label)

        # Some datasets appear to contain a stray "single" value; keep the existing behavior.
        s = s[~s.str.lower().isin({"single"})]
        if not zip_include_missing:
            s = s[s != missing_label]
        vc = _top_n_counts_with_other_and_missing(
            s,
            field,
            missing_label=missing_label,
            include_missing=zip_include_missing,
        )
    else:
        s_lower = s.str.lower()
        missing_mask = s.eq("") | s_lower.isin(_MISSING_TOKENS)
        s = s.mask(missing_mask, MISSING_LABEL) if include_missing else s[~missing_mask]
        vc = _top_n_counts_with_other_and_missing(s, field, include_missing=include_missing)

    # Special handling for ZIP codes with geographic grouping
    if field == "zip_code":
        return _render_zip_code_chart(vc, theme)

    # Fallback bar
    vc["label_short"] = vc[field].astype(str).apply(lambda v: _shorten_label(v, 28))
    vc["full_label"] = vc[field].astype(str)
    vc = add_share_columns(vc, "count")
    text_values = [
        f"  {row['share_pct']:.1f}%" for _, row in vc.iterrows()
    ]  # Add leading spaces for gap
    fig = px.bar(
        vc,
        x="share_pct",
        y="label_short",
        orientation="h",
        text=text_values,
        color_discrete_sequence=[PATIENT_CHART_COLORS[0]],  # Violet
        custom_data=vc[["full_label", "share_pct"]],
    )
    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=12))
    fig.update_traces(hovertemplate="%{customdata[0]}<br>%{customdata[1]:.1f}%<extra></extra>")

    # Add y-axis labels inside the bars (white text)
    for idx, row in vc.iterrows():
        fig.add_annotation(
            x=1,  # Position near left edge of bar
            y=idx,
            text=row["label_short"],
            showarrow=False,
            xanchor="left",
            xshift=5,  # Small shift from edge
            font=dict(color="white", size=11, family="Arial, sans-serif"),
        )

    fig.update_yaxes(autorange="reversed")
    y_title = field.replace("_", " ").title()
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title="Percentage",
        y_title=y_title,
        margin={"t": 30, "l": 100, "r": 80, "b": 60},  # Adjusted margins for in-bar labels
    )
    fig.update_xaxes(
        showgrid=False,
        title_standoff=25,  # Larger gap (~1/8 inch at typical viewing)
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,  # Hide default y-axis labels (using annotations instead)
        title=y_title,
        title_standoff=25,  # Larger gap (~1/8 inch at typical viewing)
        ticklabelposition="outside left",
        ticklabelstandoff=30,
        automargin=True,
    )
    fig.update_layout(bargap=0.05, showlegend=False)  # Smaller gap between bars (~1-2mm)
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


def _render_zip_code_chart(vc: pd.DataFrame, theme: str) -> str:
    """Render ZIP code chart with region names instead of ZIP codes."""
    # Define display names for ZIP codes and merge categories
    zip_display_names = {
        "98362": "PA East",
        "98363": "PA West",
        "98382": "Sequim",
        "98381": "Sekiu",
        "98326": "Clallam Bay",
        "98331": "Forks",
        "98343": "Joyce",
        "Homeless/Transient": "Transient",
        "Non-Clallam County ZIP Code": "Transient",
        "Other": "Transient",
        ZIP_MISSING_LABEL: "Missing ZIP",
    }

    # Define geographic groups for coloring
    zip_groups = {
        "Port Angeles": [
            "98362",
            "98363",
            "Homeless/Transient",
            "Non-Clallam County ZIP Code",
            "Other",
        ],
        "Missing": [ZIP_MISSING_LABEL],
        "Sequim": ["98382"],
        "Sekiu": ["98381"],
        "Clallam Bay": ["98326"],
        "Forks": ["98331"],
        "Joyce": ["98343"],
    }

    # Assign colors to groups
    group_colors = {
        "Port Angeles": PATIENT_CHART_COLORS[0],  # Violet
        "Sequim": PATIENT_CHART_COLORS[1],  # Cyan
        "Sekiu": PATIENT_CHART_COLORS[2],  # Emerald
        "Clallam Bay": PATIENT_CHART_COLORS[3],  # Amber
        "Forks": PATIENT_CHART_COLORS[4],  # Blue
        "Joyce": PATIENT_CHART_COLORS[6],  # Teal
        "Missing": TAILWIND_COLORS["slate-500"],
    }

    # Map ZIP codes to display names
    vc["display_name"] = vc["zip_code"].map(zip_display_names)
    # If no mapping exists, use the original value
    vc["display_name"] = vc["display_name"].fillna(vc["zip_code"])

    # Merge Transient categories by summing their counts
    transient_mask = vc["display_name"] == "Transient"
    if transient_mask.sum() > 1:
        # Sum all Transient entries
        transient_count = vc.loc[transient_mask, "count"].sum()
        # Remove all Transient rows
        vc = vc[~transient_mask].copy()
        # Add single Transient row
        transient_row = pd.DataFrame(
            [{"zip_code": "Transient", "display_name": "Transient", "count": transient_count}]
        )
        vc = pd.concat([vc, transient_row], ignore_index=True)

    # Create reverse lookup for groups
    zip_to_group = {}
    for group_name, zips in zip_groups.items():
        for zip_code in zips:
            zip_to_group[zip_code] = group_name

    # Add group information to dataframe
    vc["group"] = vc["zip_code"].apply(lambda z: zip_to_group.get(z, "Port Angeles"))
    vc = add_share_columns(vc, "count")

    # Sort globally by count so the bars run largest → smallest.
    vc = vc.sort_values(["count", "display_name"], ascending=[False, True])

    # Assign colors based on group
    vc["color"] = vc["group"].map(group_colors)

    text_values = [
        f"  {row['share_pct']:.1f}%" for _, row in vc.iterrows()
    ]  # Add leading spaces for gap

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=vc["display_name"],
            y=vc["share_pct"],
            orientation="v",
            text=text_values,
            textposition="outside",
            marker=dict(color=vc["color"].tolist()),
            customdata=vc[["display_name", "share_pct", "group"]].values,
            hovertemplate="%{customdata[0]} (%{customdata[2]})<br>%{customdata[1]:.1f}%<extra></extra>",
            cliponaxis=False,
            textfont=dict(size=14, color="#1e293b", family="Arial, sans-serif"),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=400,  # Increased height to prevent cutoff
        x_title="Region",
        y_title="Percentage",
        margin={"t": 40, "l": 70, "r": 20, "b": 90},  # More space for labels and text
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,  # Horizontal labels
        categoryorder="array",
        categoryarray=vc["display_name"].tolist(),
    )
    fig.update_yaxes(
        showgrid=True,  # Enable horizontal gridlines
        gridcolor="rgba(128,128,128,0.15)",
        showticklabels=True,
        title="Percentage",
        automargin=True,
    )
    fig.update_layout(
        bargap=0.15,
        showlegend=False,
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
