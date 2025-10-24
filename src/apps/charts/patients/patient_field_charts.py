from collections.abc import Collection

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns
from utils.plotly import style_plotly_layout

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


def _build_donut_chart(vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str) -> str:
    vc_df = add_share_columns(vc_df, value_col)
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
        textinfo="percent",
        textfont=dict(size=14, color="#000000", family="Arial, sans-serif"),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
        insidetextorientation="radial",
        marker=dict(line=dict(color="white", width=2)),
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title=None,
        margin={"t": 40, "l": 20, "r": 20, "b": 40},  # Balanced margins for centering
    )
    # Show legend on the right side for better label readability
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
    )
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

            # Create a temporary dataframe with age groups and sex
            temp_df = df[["age", "sex"]].copy()
            temp_df["age_group"] = pd.cut(
                ages, bins=bins, labels=labels, include_lowest=True, right=True
            )

            # Separate by sex
            male_df = temp_df[temp_df["sex"].str.lower().isin(["male", "m"])]
            female_df = temp_df[temp_df["sex"].str.lower().isin(["female", "f"])]

            male_counts = male_df["age_group"].value_counts().reindex(labels, fill_value=0)
            female_counts = female_df["age_group"].value_counts().reindex(labels, fill_value=0)

            # Create stacked bar chart
            fig = go.Figure()

            # Add male bars with standardized color
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=male_counts.values,
                    name="Male",
                    marker_color=PATIENT_CHART_COLORS[0],  # Violet - non-gendered
                    text=[f"{count}" if count > 0 else "" for count in male_counts.values],
                    textposition="inside",
                    hovertemplate="Age: %{x}<br>Male: %{y}<extra></extra>",
                )
            )

            # Add female bars with standardized color
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=female_counts.values,
                    name="Female",
                    marker_color=PATIENT_CHART_COLORS[3],  # Amber - non-gendered
                    text=[f"{count}" if count > 0 else "" for count in female_counts.values],
                    textposition="inside",
                    hovertemplate="Age: %{x}<br>Female: %{y}<extra></extra>",
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
                gridcolor="rgba(128,128,128,0.1)",
                ticklabelstandoff=10,  # Add gap between y-axis labels and bars
            )
            fig.update_layout(
                barmode="stack",
                bargap=0.15,  # Add gap between bars
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
            )
        elif field in {"sud", "behavioral_health"}:
            # Donut for boolean health flags
            s2 = df[field].map({True: "Yes", False: "No"}).fillna("Unknown")
            vc = s2.value_counts().reindex(["Yes", "No", "Unknown"], fill_value=0).reset_index()
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
                textfont=dict(size=12),
            )
        )

        x_title = (
            "Primary Care Agency" if field == "pcp_agency" else field.replace("_", " ").title()
        )
        fig = style_plotly_layout(
            fig,
            theme=theme,
            height=360,
            x_title=x_title,
            y_title="Percentage",
            margin={"t": 30, "l": 60, "r": 20, "b": 80},  # Full width with horizontal labels
        )
        fig.update_xaxes(
            showgrid=False,
            tickangle=0,  # Horizontal labels
            title_standoff=12,  # ~1/8 inch gap
        )
        fig.update_yaxes(
            showgrid=False,
            showticklabels=True,
            title="Percentage",
            title_standoff=12,  # ~1/8 inch gap
            automargin=True,
        )
        fig.update_layout(bargap=0.15, showlegend=False)

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


def _patient_chart_html(
    df: pd.DataFrame, field: str, theme: str, categorical_overrides: set[str]
) -> str:
    try:
        if field in categorical_overrides:
            return _render_categorical_override(df[field], field, theme)
        return _build_chart_for_field(df, field, theme)
    except Exception:
        return ""


def build_patients_field_charts(
    theme: str, fields: Collection[str] | None = None
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
        chart_html = _patient_chart_html(df, field, theme, categorical_overrides)
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
    add_chart("race_age_boxplot", build_patients_age_by_race_boxplot)

    # Add enhanced production charts
    from .enhanced_demographic_charts import build_veteran_service_bridge
    from .production_age_charts import build_enhanced_age_referral_sankey

    add_chart("age_referral_sankey", build_enhanced_age_referral_sankey)
    add_chart("veteran_service_bridge", build_veteran_service_bridge)

    if requested is not None:
        charts = {k: v for k, v in charts.items() if k in requested}

    return charts


def _build_quarterly_patient_bar(theme: str) -> str | None:
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
                x=[x_years, x_quarters],
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
        height=420,
        x_title=None,
        y_title="Patient Count",
        margin={"t": 60, "l": 60, "r": 20, "b": 100},  # Increased bottom margin for year labels
    )
    fig.update_xaxes(type="multicategory", showgrid=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.1)",
        showticklabels=True,
        title="Patient Count",
        ticklabelposition="outside",
        automargin=True,
    )

    # Add year-based context annotations below the x-axis
    # Collect unique years and their positions
    year_annotations = []
    unique_years = sorted(set(x_years))

    year_labels = {
        "2021": "COVID-19",
        "2022": "Behavioral Health",
        "2023": "Normalization",
        "2024": "Normalization",
        "2025": "+2 Community Paramedics",
    }

    for year in unique_years:
        if year in year_labels:
            # Find all indices for this year
            year_indices = [i for i, y in enumerate(x_years) if y == year]
            # Calculate center position (average of first and last index)
            center_position = (year_indices[0] + year_indices[-1]) / 2.0

            year_annotations.append(
                dict(
                    x=center_position,
                    y=-0.22,  # Position below the x-axis
                    xref="x",
                    yref="paper",
                    text=f"<i>{year_labels[year]}</i>",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xanchor="center",
                )
            )

    # Calculate dynamic average from the dataset
    normalized_quarterly_avg = round(qdf2["count"].mean(), 1)

    fig.update_layout(
        bargap=0.15,
        showlegend=False,
        annotations=[
            dict(
                text=f"<i>Bar color: Emerald = Above Average (>{normalized_quarterly_avg}) | Cyan = Average or Below (≤{normalized_quarterly_avg})</i>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
            ),
            # Label for the normalized baseline
            dict(
                text=f"<i>Quarterly Average (~{normalized_quarterly_avg})</i>",
                xref="paper",
                yref="y",
                x=1.02,
                y=normalized_quarterly_avg,
                showarrow=False,
                font=dict(size=9, color="white"),
                xanchor="left",
                yanchor="middle",
            ),
        ]
        + year_annotations
        + (fig.layout.annotations if fig.layout.annotations else []),
        shapes=[
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=normalized_quarterly_avg,
                y1=normalized_quarterly_avg,
                line=dict(
                    color="white",
                    width=2,
                    dash="dash",
                ),
            )
        ],
    )
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})


def _render_categorical_override(series: pd.Series, field: str, theme: str) -> str:
    # Special handling for boolean health flags
    if field in {"sud", "behavioral_health"}:
        s2 = series.map({True: "Yes", False: "No"}).fillna("Unknown")
        vc = s2.value_counts().reindex(["Yes", "No", "Unknown"], fill_value=0).reset_index()
        vc.columns = ["label", "count"]
        return _build_donut_chart(vc, "label", "count", theme)

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

    if field in {"sud", "behavioral_health"}:
        vc2 = vc.rename(columns={field: "label"})
        return _build_donut_chart(vc2, "label", "count", theme)

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
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})


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
        "Homeless/Transient": "Transient",
        "Non-Clallam County ZIP Code": "Transient",
        "Other": "Transient",
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
        "Sequim": ["98382"],
        "Sekiu": ["98381"],
        "Clallam Bay": ["98326"],
        "Forks": ["98331"],
    }

    # Assign colors to groups
    group_colors = {
        "Port Angeles": PATIENT_CHART_COLORS[0],  # Violet
        "Sequim": PATIENT_CHART_COLORS[1],  # Cyan
        "Sekiu": PATIENT_CHART_COLORS[2],  # Emerald
        "Clallam Bay": PATIENT_CHART_COLORS[3],  # Amber
        "Forks": PATIENT_CHART_COLORS[4],  # Blue
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

    # Sort by group, then by count within group
    group_order = ["Port Angeles", "Sequim", "Sekiu", "Clallam Bay", "Forks"]
    vc["group_order"] = vc["group"].apply(lambda g: group_order.index(g) if g in group_order else 0)
    vc = vc.sort_values(["group_order", "count"], ascending=[True, False])

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
            textfont=dict(size=12),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,  # Fixed height for consistency
        x_title="Region",
        y_title="Percentage",
        margin={"t": 30, "l": 60, "r": 20, "b": 80},  # Full width with horizontal labels
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,  # Horizontal labels
        title_standoff=12,  # ~1/8 inch gap
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=True,
        title="Percentage",
        title_standoff=12,  # ~1/8 inch gap
        automargin=True,
    )
    fig.update_layout(
        bargap=0.15,
        showlegend=False,
    )
    return plot(fig, output_type="div", config={"responsive": True, "displaylogo": False})
