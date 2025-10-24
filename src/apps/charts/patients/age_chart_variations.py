"""
Enhanced age chart variations demonstrating different visualization approaches
for presenting patient age demographics in compelling, insightful ways.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_normalization import add_share_columns, count_share_text
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Patients


def _prepare_age_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare age data with standardized medical age groups."""
    ages = pd.to_numeric(df["age"], errors="coerce")
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
    return vc


def _get_vulnerability_colors() -> dict[str, str]:
    """Return vulnerability-based color mapping for age groups."""
    return {
        "0–17": TAILWIND_COLORS["red-500"],  # Children - high vulnerability
        "18–24": TAILWIND_COLORS["slate-400"],  # Young adults
        "25–34": TAILWIND_COLORS["slate-400"],  # Adults
        "35–44": TAILWIND_COLORS["slate-400"],  # Adults
        "45–54": TAILWIND_COLORS["slate-400"],  # Middle age
        "55–64": TAILWIND_COLORS["amber-500"],  # Pre-senior - moderate
        "65–74": TAILWIND_COLORS["orange-500"],  # Young senior - higher
        "75–84": TAILWIND_COLORS["red-600"],  # Senior - high
        "85+": TAILWIND_COLORS["red-700"],  # Oldest - highest
        "Unknown": TAILWIND_COLORS["gray-300"],
    }


# ==============================================================================
# OPTION A: ANNOTATED BAR CHART with Benchmarks & Insights
# ==============================================================================


def build_age_chart_option_a_annotated(theme: str) -> str:
    """
    Option A: Enhanced bar chart with benchmark lines, target zones, and
    contextual annotations highlighting key insights.

    Features:
    - Vulnerability-based color coding
    - Benchmark line showing national average
    - Highlighted zones for high-risk populations
    - Callout annotations with insights
    - Comparison indicators
    """
    qs = Patients.objects.all().values("age", "sex")
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    if df.empty:
        return "<p>No data available</p>"

    vc = _prepare_age_data(df)
    color_map = _get_vulnerability_colors()

    # Create the base bar chart
    fig = go.Figure()

    for _, row in vc.iterrows():
        age_group = row["age_group"]
        count = row["count"]
        share_pct = row["share_pct"]

        fig.add_trace(
            go.Bar(
                x=[age_group],
                y=[count],
                marker_color=color_map.get(age_group, TAILWIND_COLORS["slate-400"]),
                text=count_share_text(count, share_pct),
                textposition="outside",
                hovertemplate=f"Age group: {age_group}<br>Count: {count}<br>Share: {share_pct:.1f}%<extra></extra>",
                showlegend=False,
            )
        )

    # Add benchmark line (hypothetical national average for demonstration)
    avg_count = vc["count"].mean()
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(vc) - 0.5,
        y0=avg_count,
        y1=avg_count,
        line=dict(color=TAILWIND_COLORS["blue-400"], width=2, dash="dash"),
        layer="above",
    )

    # Add benchmark annotation
    fig.add_annotation(
        x=len(vc) - 1,
        y=avg_count,
        text=f"Program Average: {int(avg_count)}",
        showarrow=False,
        yshift=10,
        font=dict(size=11, color=TAILWIND_COLORS["blue-400"]),
        bgcolor="rgba(59, 130, 246, 0.1)",
        bordercolor=TAILWIND_COLORS["blue-400"],
        borderwidth=1,
        borderpad=4,
    )

    # Add high-risk zone highlight (seniors 65+)
    senior_indices = [
        i for i, age in enumerate(vc["age_group"]) if age in ["65–74", "75–84", "85+"]
    ]
    if senior_indices:
        fig.add_shape(
            type="rect",
            x0=senior_indices[0] - 0.4,
            x1=senior_indices[-1] + 0.4,
            y0=0,
            y1=vc["count"].max() * 1.15,
            fillcolor="rgba(239, 68, 68, 0.08)",
            line=dict(width=0),
            layer="below",
        )

        # Add zone label
        fig.add_annotation(
            x=senior_indices[1] if len(senior_indices) > 1 else senior_indices[0],
            y=vc["count"].max() * 1.12,
            text="High-Risk Zone (65+)",
            showarrow=False,
            font=dict(size=10, color=TAILWIND_COLORS["red-600"]),
        )

    # Add insight callout for highest vulnerability group
    max_row = vc[vc["age_group"] != "Unknown"].nlargest(1, "count").iloc[0]
    fig.add_annotation(
        x=vc[vc["age_group"] == max_row["age_group"]].index[0],
        y=max_row["count"],
        text=f"📊 Largest cohort<br>{max_row['share_pct']:.1f}% of patients",
        showarrow=True,
        arrowhead=2,
        arrowcolor=TAILWIND_COLORS["indigo-500"],
        ax=50,
        ay=-50,
        font=dict(size=11),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=TAILWIND_COLORS["indigo-500"],
        borderwidth=2,
        borderpad=6,
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=420,
        x_title="Age Group",
        y_title="Patient Count",
        margin={"t": 50, "l": 60, "r": 20, "b": 50},
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    fig.update_layout(bargap=0.15)

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


# ==============================================================================
# OPTION B: POPULATION PYRAMID (Gender Comparison)
# ==============================================================================


def build_age_chart_option_b_pyramid(theme: str) -> str:
    """
    Option B: Population pyramid showing age distribution split by sex.

    Features:
    - Dual-axis horizontal bars (male/female)
    - Traditional demographic visualization
    - Immediate gender disparity insights
    - Mirrored design for easy comparison
    """
    qs = Patients.objects.all().values("age", "sex")
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    if df.empty:
        return "<p>No data available</p>"

    ages = pd.to_numeric(df["age"], errors="coerce")
    bins = [-1, 17, 24, 34, 44, 54, 64, 74, 84, float("inf")]
    labels = ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]
    df["age_group"] = pd.cut(ages, bins=bins, labels=labels, include_lowest=True, right=True)

    # Separate by sex
    male_df = df[df["sex"].str.lower().isin(["male", "m"])].copy()
    female_df = df[df["sex"].str.lower().isin(["female", "f"])].copy()

    male_counts = male_df["age_group"].value_counts().reindex(labels, fill_value=0)
    female_counts = female_df["age_group"].value_counts().reindex(labels, fill_value=0)

    fig = go.Figure()

    # Male bars (left side, negative values)
    male_values = [-int(x) for x in male_counts.values]
    fig.add_trace(
        go.Bar(
            y=labels,
            x=male_values,
            name="Male",
            orientation="h",
            marker_color=TAILWIND_COLORS["blue-500"],
            text=male_counts.values,
            textposition="inside",
            hovertemplate="Male %{y}<br>Count: %{text}<extra></extra>",
        )
    )

    # Female bars (right side, positive values)
    fig.add_trace(
        go.Bar(
            y=labels,
            x=female_counts.values,
            name="Female",
            orientation="h",
            marker_color=TAILWIND_COLORS["pink-500"],
            text=female_counts.values,
            textposition="inside",
            hovertemplate="Female %{y}<br>Count: %{text}<extra></extra>",
        )
    )

    # Calculate max for symmetric axis
    max_val = max(male_counts.max(), female_counts.max())

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=450,
        x_title="Patient Count",
        y_title="Age Group",
        margin={"t": 40, "l": 80, "r": 80, "b": 50},
        show_legend=True,
    )

    fig.update_xaxes(
        range=[-max_val * 1.2, max_val * 1.2],
        tickvals=[-max_val, -max_val / 2, 0, max_val / 2, max_val],
        ticktext=[
            str(int(max_val)),
            str(int(max_val / 2)),
            "0",
            str(int(max_val / 2)),
            str(int(max_val)),
        ],
        showgrid=True,
        gridcolor="rgba(128,128,128,0.1)",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        barmode="overlay",
        bargap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    # Add center line
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-0.5,
        y1=len(labels) - 0.5,
        line=dict(color="rgba(128,128,128,0.3)", width=2),
    )

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


# ==============================================================================
# OPTION C: SMALL MULTIPLES (Temporal Comparison)
# ==============================================================================


def build_age_chart_option_c_small_multiples(theme: str) -> str:
    """
    Option C: Small multiples showing age distribution across time periods.

    Features:
    - Side-by-side comparison of different time periods
    - Trend arrows showing increases/decreases
    - Compact, scannable layout
    - Immediate pattern recognition
    """
    qs = Patients.objects.all().values("age", "created_date")
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    if df.empty:
        return "<p>No data available</p>"

    df["created_date"] = pd.to_datetime(df["created_date"])

    # Create time periods (last 3 years by year)
    current_year = pd.Timestamp.now().year
    periods = [
        (f"{current_year - 2}", (current_year - 2, current_year - 1)),
        (f"{current_year - 1}", (current_year - 1, current_year)),
        (f"{current_year}", (current_year, current_year + 1)),
    ]

    ages = pd.to_numeric(df["age"], errors="coerce")
    bins = [-1, 17, 24, 34, 44, 54, 64, 74, 84, float("inf")]
    labels = ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]
    df["age_group"] = pd.cut(ages, bins=bins, labels=labels, include_lowest=True, right=True)

    # Create subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[p[0] for p in periods],
        horizontal_spacing=0.08,
    )

    color_map = _get_vulnerability_colors()
    period_data = []

    for col_idx, (period_name, (start_year, end_year)) in enumerate(periods, start=1):
        period_df = df[
            (df["created_date"].dt.year >= start_year) & (df["created_date"].dt.year < end_year)
        ]

        vc = period_df["age_group"].value_counts().reindex(labels, fill_value=0).reset_index()
        vc.columns = ["age_group", "count"]
        period_data.append(vc)

        for _, row in vc.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["age_group"]],
                    y=[row["count"]],
                    marker_color=color_map.get(row["age_group"], TAILWIND_COLORS["slate-400"]),
                    showlegend=False,
                    hovertemplate=f"{period_name}<br>%{{x}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

    # Add trend arrows between periods
    if len(period_data) >= 2:
        for i in range(len(period_data) - 1):
            prev_total = period_data[i]["count"].sum()
            curr_total = period_data[i + 1]["count"].sum()
            change_pct = ((curr_total - prev_total) / prev_total * 100) if prev_total > 0 else 0

            arrow_symbol = "↗" if change_pct > 5 else "↘" if change_pct < -5 else "→"
            arrow_color = (
                TAILWIND_COLORS["green-500"] if change_pct > 0 else TAILWIND_COLORS["red-500"]
            )

            fig.add_annotation(
                text=f"{arrow_symbol} {change_pct:+.1f}%",
                xref=f"x{i + 2}",
                yref=f"y{i + 2}",
                x=-0.5,
                y=max(period_data[i + 1]["count"]) * 0.9,
                showarrow=False,
                font=dict(size=14, color=arrow_color),
                bgcolor="rgba(255,255,255,0.8)",
                borderpad=4,
            )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        margin={"t": 60, "l": 50, "r": 20, "b": 80},
    )

    # Update all x-axes
    for i in range(1, 4):
        fig.update_xaxes(
            tickangle=-45,
            showgrid=False,
            row=1,
            col=i,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.1)",
            row=1,
            col=i,
        )

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


# ==============================================================================
# OPTION D: SANKEY FLOW (Age → Service Pathway)
# ==============================================================================


def build_age_chart_option_d_sankey(theme: str) -> str:
    """
    Option D: Sankey diagram showing flow from age groups through service types.

    Features:
    - Visual flow representation
    - Shows service utilization patterns by age
    - Thickness indicates volume
    - Interactive hover details
    """
    qs = Patients.objects.all().values("age", "sud", "behavioral_health")
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    if df.empty:
        return "<p>No data available</p>"

    ages = pd.to_numeric(df["age"], errors="coerce")
    bins = [-1, 24, 44, 64, float("inf")]
    age_labels = ["Youth (0-24)", "Adults (25-44)", "Middle Age (45-64)", "Seniors (65+)"]
    df["age_category"] = pd.cut(ages, bins=bins, labels=age_labels, include_lowest=True, right=True)

    # Create service categories
    def categorize_services(row):
        services = []
        if row["sud"]:
            services.append("SUD Services")
        if row["behavioral_health"]:
            services.append("Behavioral Health")
        if not services:
            services.append("General Care")
        return services[0]  # Take first for simplicity

    df["service_type"] = df.apply(categorize_services, axis=1)

    # Build flow data
    flow_counts = df.groupby(["age_category", "service_type"]).size().reset_index(name="count")

    # Create node labels
    age_nodes = age_labels
    service_nodes = ["SUD Services", "Behavioral Health", "General Care"]
    all_nodes = age_nodes + service_nodes

    # Create source/target/value lists
    sources = []
    targets = []
    values = []
    colors = []

    color_palette = [
        TAILWIND_COLORS["blue-400"],
        TAILWIND_COLORS["indigo-400"],
        TAILWIND_COLORS["purple-400"],
        TAILWIND_COLORS["pink-400"],
    ]

    for _, row in flow_counts.iterrows():
        if pd.notna(row["age_category"]):
            source_idx = all_nodes.index(row["age_category"])
            target_idx = all_nodes.index(row["service_type"])

            sources.append(source_idx)
            targets.append(target_idx)
            values.append(row["count"])
            colors.append(color_palette[source_idx % len(color_palette)])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=[
                        TAILWIND_COLORS["blue-500"],
                        TAILWIND_COLORS["indigo-500"],
                        TAILWIND_COLORS["purple-500"],
                        TAILWIND_COLORS["pink-500"],
                        TAILWIND_COLORS["red-400"],
                        TAILWIND_COLORS["orange-400"],
                        TAILWIND_COLORS["green-400"],
                    ],
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=[c.replace(")", ", 0.4)").replace("rgb", "rgba") for c in colors],
                    hovertemplate="%{source.label} → %{target.label}<br>Patients: %{value}<extra></extra>",
                ),
            )
        ]
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=450,
        margin={"t": 30, "l": 10, "r": 10, "b": 10},
    )

    fig.update_layout(
        title=dict(
            text="Patient Flow: Age Groups → Service Pathways",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
    )

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


# ==============================================================================
# OPTION E: COMBINATION CHART (Bars + Cumulative Line)
# ==============================================================================


def build_age_chart_option_e_combination(theme: str) -> str:
    """
    Option E: Combination chart with bars showing counts and line showing
    cumulative percentage distribution.

    Features:
    - Dual-axis visualization (count + cumulative %)
    - Shows both absolute numbers and distribution
    - Helps identify concentration points
    - Professional multi-layer presentation
    """
    qs = Patients.objects.all().values("age")
    data = list(qs)
    df = pd.DataFrame.from_records(data) if data else pd.DataFrame()

    if df.empty:
        return "<p>No data available</p>"

    vc = _prepare_age_data(df)
    vc = vc[vc["age_group"] != "Unknown"].copy()  # Exclude unknown for cumulative

    # Calculate cumulative percentage
    vc["cumulative_pct"] = vc["share_pct"].cumsum()

    color_map = _get_vulnerability_colors()

    fig = go.Figure()

    # Add bars for counts
    for _, row in vc.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["age_group"]],
                y=[row["count"]],
                name=row["age_group"],
                marker_color=color_map.get(row["age_group"], TAILWIND_COLORS["slate-400"]),
                text=count_share_text(row["count"], row["share_pct"]),
                textposition="outside",
                showlegend=False,
                hovertemplate=f"Age: {row['age_group']}<br>Count: {row['count']}<br>Share: {row['share_pct']:.1f}%<extra></extra>",
                yaxis="y",
            )
        )

    # Add cumulative percentage line
    fig.add_trace(
        go.Scatter(
            x=vc["age_group"],
            y=vc["cumulative_pct"],
            mode="lines+markers",
            name="Cumulative %",
            line=dict(
                color=TAILWIND_COLORS["emerald-500"],
                width=3,
            ),
            marker=dict(
                size=8,
                color=TAILWIND_COLORS["emerald-500"],
                line=dict(color="white", width=2),
            ),
            yaxis="y2",
            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
        )
    )

    # Add 50th and 75th percentile reference lines
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color=TAILWIND_COLORS["amber-400"],
        annotation_text="50th Percentile",
        annotation_position="right",
        yref="y2",
    )
    fig.add_hline(
        y=75,
        line_dash="dash",
        line_color=TAILWIND_COLORS["orange-400"],
        annotation_text="75th Percentile",
        annotation_position="right",
        yref="y2",
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=420,
        x_title="Age Group",
        margin={"t": 50, "l": 60, "r": 80, "b": 50},
        show_legend=True,
    )

    # Configure dual y-axes
    fig.update_layout(
        yaxis=dict(
            title="Patient Count",
            showgrid=False,
            side="left",
        ),
        yaxis2=dict(
            title="Cumulative Percentage",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.1)",
            overlaying="y",
            side="right",
            range=[0, 105],
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        bargap=0.15,
    )

    fig.update_xaxes(showgrid=False)

    config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }
    return plot(fig, output_type="div", config=config)


# ==============================================================================
# Build all variations
# ==============================================================================


def build_all_age_chart_variations(theme: str) -> dict[str, str]:
    """
    Build all 5 enhanced age chart variations.

    Returns:
        Dictionary mapping option names to chart HTML.
    """
    return {
        "option_a_annotated": build_age_chart_option_a_annotated(theme),
        "option_b_pyramid": build_age_chart_option_b_pyramid(theme),
        "option_c_small_multiples": build_age_chart_option_c_small_multiples(theme),
        "option_d_sankey": build_age_chart_option_d_sankey(theme),
        "option_e_combination": build_age_chart_option_e_combination(theme),
    }
