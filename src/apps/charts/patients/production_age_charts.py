"""
Production-ready age visualization charts for the patients dashboard.
- Stacked bar chart by sex (combines age distribution and gender breakdown)
- Enhanced Sankey diagram with specific referral pathways
"""

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import ODReferrals, Patients, Referrals

# Standardized color palette matching Veteran Care Coordination chart
PATIENT_CHART_COLORS = [
    CHART_COLORS_VIBRANT[0],  # Violet - primary
    CHART_COLORS_VIBRANT[1],  # Cyan - secondary
    CHART_COLORS_VIBRANT[3],  # Emerald - success/positive
    CHART_COLORS_VIBRANT[4],  # Amber - warning/attention
    CHART_COLORS_VIBRANT[5],  # Blue - tertiary
    CHART_COLORS_VIBRANT[2],  # Rose - quaternary
    CHART_COLORS_VIBRANT[7],  # Teal - additional
    CHART_COLORS_VIBRANT[6],  # Pink - additional
    CHART_COLORS_VIBRANT[9],  # Indigo - additional
    CHART_COLORS_VIBRANT[10],  # Lime - additional
    CHART_COLORS_VIBRANT[8],  # Orange - additional
    CHART_COLORS_VIBRANT[11],  # Fuchsia - additional
]


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
# STACKED BAR CHART: Age Distribution by Sex
# ==============================================================================


def build_simplified_age_bar_chart(theme: str) -> str:
    """
    Stacked bar chart showing age distribution split by sex with vulnerability-based colors.

    Features:
    - Stacked bars showing male/female breakdown
    - Vulnerability-based color coding for age groups
    - Count + percentage labels
    - Gender comparison at a glance
    - Interactive hover details
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

    male_total = int(male_counts.sum())
    female_total = int(female_counts.sum())
    male_pct = [
        ((count / male_total) * 100.0) if male_total else 0.0 for count in male_counts.values
    ]
    female_pct = [
        ((count / female_total) * 100.0) if female_total else 0.0 for count in female_counts.values
    ]

    fig = go.Figure()

    # Add male bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=male_counts.values,
            name="Male",
            marker_color=TAILWIND_COLORS["blue-500"],
            customdata=male_pct,
            hovertemplate="Age: %{x}<br>Male: %{customdata:.1f}%<extra></extra>",
        )
    )

    # Add female bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=female_counts.values,
            name="Female",
            marker_color=TAILWIND_COLORS["pink-500"],
            customdata=female_pct,
            hovertemplate="Age: %{x}<br>Female: %{customdata:.1f}%<extra></extra>",
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=400,
        x_title="Age Group",
        y_title="Patient Count",
        margin={"t": 40, "l": 60, "r": 20, "b": 50},
        show_legend=True,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    fig.update_layout(
        barmode="stack",
        bargap=0.15,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
        ),
    )

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


# ==============================================================================
# POPULATION PYRAMID: Age Distribution by Gender
# ==============================================================================


def build_age_gender_pyramid(theme: str) -> str:
    """
    Population pyramid showing age distribution split by sex.

    Features:
    - Dual-axis horizontal bars (male/female)
    - Traditional demographic visualization
    - Immediate gender disparity insights
    - Mirrored design for easy comparison
    - Vulnerability-based age groupings
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


# ==============================================================================
# ENHANCED SANKEY: Age Groups → Specific Referral Types
# ==============================================================================


def build_enhanced_age_referral_sankey(theme: str) -> str:  # noqa: C901
    """
    Enhanced Sankey diagram showing flow from age groups to specific referral types.

    Features:
    - Age categories → Specific high-impact referral types
    - Flow thickness indicates patient volume
    - Color-coded by age group vulnerability
    - Combines vaccination types and overdose data from multiple sources
    - Shows portion of all referral types (focused on key services)
    - Reveals consistent needs/gaps across all age groups (systemic patterns)

    Cross-references Patients, Referrals, and ODReferrals to show which age
    groups receive which types of services. The consistent distribution across
    age groups highlights systemic care needs rather than age-specific issues.
    """
    # Get patients with age data
    patients_qs = Patients.objects.all().values("id", "age")
    patients_data = list(patients_qs)
    df_patients = pd.DataFrame.from_records(patients_data) if patients_data else pd.DataFrame()

    if df_patients.empty:
        return "<p>No patient data available</p>"

    # Get referrals
    referrals_qs = Referrals.objects.all().values(
        "patient_ID", "referral_1", "referral_2", "referral_3", "referral_4", "referral_5"
    )
    referrals_data = list(referrals_qs)
    df_referrals = pd.DataFrame.from_records(referrals_data) if referrals_data else pd.DataFrame()

    if df_referrals.empty:
        return "<p>No referral data available</p>"

    # Get OD referrals to combine with regular referrals for overdose count
    od_referrals_qs = ODReferrals.objects.all().values("patient_id")
    od_referrals_data = list(od_referrals_qs)
    df_od = pd.DataFrame.from_records(od_referrals_data) if od_referrals_data else pd.DataFrame()

    # Prepare age categories
    ages = pd.to_numeric(df_patients["age"], errors="coerce")
    bins = [-1, 24, 44, 64, float("inf")]
    age_labels = ["Youth (0-24)", "Adults (25-44)", "Middle Age (45-64)", "Seniors (65+)"]
    df_patients["age_category"] = pd.cut(
        ages, bins=bins, labels=age_labels, include_lowest=True, right=True
    )

    # Melt referrals into long format (all referral columns into one)
    referral_cols = ["referral_1", "referral_2", "referral_3", "referral_4", "referral_5"]
    df_referrals_long = df_referrals.melt(
        id_vars=["patient_ID"], value_vars=referral_cols, value_name="referral_type"
    )

    # Clean referral types
    df_referrals_long["referral_type"] = (
        df_referrals_long["referral_type"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    df_referrals_long = df_referrals_long.dropna(subset=["referral_type"])
    df_referrals_long = df_referrals_long[df_referrals_long["referral_type"] != ""]

    # Normalize referral types
    def normalize_referral(ref_type: str) -> str | None:
        ref_lower = ref_type.lower().strip()

        # Rename 911/Walk-in to 911 calls
        if "911" in ref_lower or "walk-in" in ref_lower or "walk in" in ref_lower:
            return "911 calls"

        # Combine all Vax types
        if "vax -" in ref_lower or "vax-" in ref_lower or ref_lower.startswith("vax "):
            return "Vaccinations"

        # Specific referral types to include (case-insensitive matching)
        if "lab - blood draw" in ref_lower or "lab-blood draw" in ref_lower:
            return "Lab - Blood Draw"

        if "eval - assessment" in ref_lower or "eval-assessment" in ref_lower:
            return "Eval - Assessment"

        if "case management" in ref_lower:
            return "Case Management"

        if "eval - psych" in ref_lower or "dementia" in ref_lower or "crisis" in ref_lower:
            return "Eval - Psych/Dementia/Crisis"

        if "overdose" in ref_lower or ref_lower == "od":
            return "Overdose"

        if "med - rx reconciliation" in ref_lower or "med-rx reconciliation" in ref_lower:
            return "Med - Rx Reconciliation"

        if "med - antipsychotic im" in ref_lower or "antipsychotic" in ref_lower:
            return "Med - Antipsychotic IM"

        return None  # Exclude other types

    df_referrals_long["normalized_type"] = df_referrals_long["referral_type"].apply(
        normalize_referral
    )

    # Filter to only specified types
    df_referrals_long = df_referrals_long.dropna(subset=["normalized_type"])

    # Join with patients to get age categories
    df_merged = df_referrals_long.merge(
        df_patients[["id", "age_category"]], left_on="patient_ID", right_on="id", how="left"
    )

    # Add overdose referrals from ODReferrals table
    if not df_od.empty:
        # Join OD referrals with patients to get age categories
        df_od_merged = df_od.merge(
            df_patients[["id", "age_category"]], left_on="patient_id", right_on="id", how="left"
        )
        # Add normalized type for OD referrals
        df_od_merged["normalized_type"] = "Overdose"
        # Select only the columns we need to match df_merged structure
        df_od_merged = df_od_merged[["patient_id", "normalized_type", "age_category"]].rename(
            columns={"patient_id": "patient_ID"}
        )
        # Append to merged data
        df_merged = pd.concat(
            [df_merged[["patient_ID", "normalized_type", "age_category"]], df_od_merged],
            ignore_index=True,
        )

    # Count flows from age category to referral type
    flow_counts = (
        df_merged.groupby(["age_category", "normalized_type"], observed=False)
        .size()
        .reset_index(name="count")
    )

    # Remove flows with missing age category
    flow_counts = flow_counts.dropna(subset=["age_category"])

    if flow_counts.empty:
        return "<p>No referral pathway data available</p>"

    # Create node labels
    age_nodes = [label for label in age_labels if label in flow_counts["age_category"].values]
    referral_nodes = sorted(flow_counts["normalized_type"].unique().tolist())
    all_nodes = age_nodes + referral_nodes

    # Create source/target/value lists
    sources = []
    targets = []
    values = []
    link_colors = []

    # Color palette for age groups - using standardized patient chart colors
    age_color_map = {
        "Youth (0-24)": PATIENT_CHART_COLORS[0],  # Violet - youth energy
        "Adults (25-44)": PATIENT_CHART_COLORS[1],  # Cyan - prime adult
        "Middle Age (45-64)": PATIENT_CHART_COLORS[3],  # Amber - mature
        "Seniors (65+)": PATIENT_CHART_COLORS[5],  # Rose - senior care
    }

    # Calculate total for percentages
    total_flows = flow_counts["count"].sum()

    for _, row in flow_counts.iterrows():
        age_cat = row["age_category"]
        ref_type = row["normalized_type"]

        if age_cat in all_nodes and ref_type in all_nodes:
            source_idx = all_nodes.index(age_cat)
            target_idx = all_nodes.index(ref_type)

            sources.append(source_idx)
            targets.append(target_idx)
            values.append(row["count"])

            # Use age group color for links
            base_color = age_color_map.get(age_cat, TAILWIND_COLORS["slate-400"])
            # Make semi-transparent
            link_color = base_color.replace(")", ", 0.4)").replace("rgb", "rgba")
            link_colors.append(link_color)

    # Calculate percentages for display
    value_percentages = [(v / total_flows * 100) for v in values]
    link_custom_data = [[pct] for pct in value_percentages]

    # Calculate node values and percentages (sum of flows through each node)
    node_values = {}
    for node in all_nodes:
        node_idx = all_nodes.index(node)
        # Sum all flows that go through this node (as source or target)
        total = sum(values[i] for i, src in enumerate(sources) if src == node_idx)
        total += sum(values[i] for i, tgt in enumerate(targets) if tgt == node_idx)
        node_values[node] = total / 2  # Divide by 2 since we counted each flow twice

    # Create node customdata with percentages
    node_customdata = []
    for node in all_nodes:
        node_value = node_values.get(node, 0)
        percentage = (node_value / total_flows * 100) if total_flows > 0 else 0
        node_customdata.append([node_value, percentage])

    # Node colors
    node_colors = []
    for node in all_nodes:
        if node in age_color_map:
            node_colors.append(age_color_map[node])
        else:
            # Referral types get vibrant colors from the standardized palette
            idx = all_nodes.index(node) % len(PATIENT_CHART_COLORS)
            node_colors.append(PATIENT_CHART_COLORS[idx])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=25,
                    line=dict(color="rgba(0,0,0,0.3)", width=1),
                    label=all_nodes,
                    color=node_colors,
                    customdata=node_customdata,
                    hovertemplate="%{label}<br>%{customdata[1]:.1f}%<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=link_custom_data,
                    hovertemplate="%{source.label} → %{target.label}<br>%{customdata[0]:.1f}%<extra></extra>",
                ),
            )
        ]
    )

    fig = style_plotly_layout(
        fig, theme=theme, height=500, margin={"t": 10, "l": 10, "r": 10, "b": 10}
    )

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


# ==============================================================================
# SUNBURST CHART: Age Groups → Referral Services
# ==============================================================================


def build_age_referral_sunburst(theme: str) -> str:  # noqa: C901
    """
    Radial bar chart showing age groups and referral services in a stunning circular layout.

    Features:
    - Radial bars extending from center
    - Bar length represents percentage of total
    - Grouped by age category
    - Color-coded with vibrant gradients
    - Interactive hover with detailed stats
    """
    # Get patients with age data
    patients_qs = Patients.objects.all().values("id", "age")
    patients_data = list(patients_qs)
    df_patients = pd.DataFrame.from_records(patients_data) if patients_data else pd.DataFrame()

    if df_patients.empty:
        return "<p>No patient data available</p>"

    # Get referrals
    referrals_qs = Referrals.objects.all().values(
        "patient_ID", "referral_1", "referral_2", "referral_3", "referral_4", "referral_5"
    )
    referrals_data = list(referrals_qs)
    df_referrals = pd.DataFrame.from_records(referrals_data) if referrals_data else pd.DataFrame()

    if df_referrals.empty:
        return "<p>No referral data available</p>"

    # Get OD referrals
    od_referrals_qs = ODReferrals.objects.all().values("patient_id")
    od_referrals_data = list(od_referrals_qs)
    df_od = pd.DataFrame.from_records(od_referrals_data) if od_referrals_data else pd.DataFrame()

    # Prepare age categories
    ages = pd.to_numeric(df_patients["age"], errors="coerce")
    bins = [-1, 24, 44, 64, float("inf")]
    age_labels = ["Youth (0-24)", "Adults (25-44)", "Middle Age (45-64)", "Seniors (65+)"]
    df_patients["age_category"] = pd.cut(
        ages, bins=bins, labels=age_labels, include_lowest=True, right=True
    )

    # Melt referrals into long format
    referral_cols = ["referral_1", "referral_2", "referral_3", "referral_4", "referral_5"]
    df_referrals_long = df_referrals.melt(
        id_vars=["patient_ID"], value_vars=referral_cols, value_name="referral_type"
    )

    # Clean and normalize referral types (same logic as Sankey)
    df_referrals_long["referral_type"] = (
        df_referrals_long["referral_type"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    df_referrals_long = df_referrals_long.dropna(subset=["referral_type"])
    df_referrals_long = df_referrals_long[df_referrals_long["referral_type"] != ""]

    def normalize_referral(ref_type: str) -> str | None:
        ref_lower = ref_type.lower().strip()
        if "911" in ref_lower or "walk-in" in ref_lower or "walk in" in ref_lower:
            return "911 calls"
        if "vax -" in ref_lower or "vax-" in ref_lower or ref_lower.startswith("vax "):
            return "Vaccinations"
        if "lab - blood draw" in ref_lower or "lab-blood draw" in ref_lower:
            return "Lab - Blood Draw"
        if "eval - assessment" in ref_lower or "eval-assessment" in ref_lower:
            return "Eval - Assessment"
        if "case management" in ref_lower:
            return "Case Management"
        if "eval - psych" in ref_lower or "dementia" in ref_lower or "crisis" in ref_lower:
            return "Eval - Psych/Dementia/Crisis"
        if "overdose" in ref_lower or ref_lower == "od":
            return "Overdose"
        if "med - rx reconciliation" in ref_lower or "med-rx reconciliation" in ref_lower:
            return "Med - Rx Reconciliation"
        if "med - antipsychotic im" in ref_lower or "antipsychotic" in ref_lower:
            return "Med - Antipsychotic IM"
        return None

    df_referrals_long["normalized_type"] = df_referrals_long["referral_type"].apply(
        normalize_referral
    )
    df_referrals_long = df_referrals_long.dropna(subset=["normalized_type"])

    # Join with patients to get age categories
    df_merged = df_referrals_long.merge(
        df_patients[["id", "age_category"]], left_on="patient_ID", right_on="id", how="left"
    )

    # Add overdose referrals from ODReferrals table
    if not df_od.empty:
        df_od_merged = df_od.merge(
            df_patients[["id", "age_category"]], left_on="patient_id", right_on="id", how="left"
        )
        df_od_merged["normalized_type"] = "Overdose"
        df_od_merged = df_od_merged[["patient_id", "normalized_type", "age_category"]].rename(
            columns={"patient_id": "patient_ID"}
        )
        df_merged = pd.concat(
            [df_merged[["patient_ID", "normalized_type", "age_category"]], df_od_merged],
            ignore_index=True,
        )

    # Count flows
    flow_counts = (
        df_merged.groupby(["age_category", "normalized_type"], observed=False)
        .size()
        .reset_index(name="count")
    )
    flow_counts = flow_counts.dropna(subset=["age_category"])

    if flow_counts.empty:
        return "<p>No referral pathway data available</p>"

    # Calculate total and percentages
    total_count = flow_counts["count"].sum()
    flow_counts["percentage"] = (flow_counts["count"] / total_count * 100).round(1)

    # Sort by age category and percentage for better visual grouping
    flow_counts = flow_counts.sort_values(["age_category", "percentage"], ascending=[True, False])

    # Age group colors with enhanced vibrancy
    age_color_map = {
        "Youth (0-24)": PATIENT_CHART_COLORS[0],  # Violet
        "Adults (25-44)": PATIENT_CHART_COLORS[1],  # Cyan
        "Middle Age (45-64)": PATIENT_CHART_COLORS[3],  # Amber
        "Seniors (65+)": PATIENT_CHART_COLORS[5],  # Rose
    }

    # Create radial bar chart with one trace per age group
    fig = go.Figure()

    # Track theta positions for spacing
    theta_position = 0
    bar_width = 8  # Width of each bar in degrees
    gap_between_services = 2  # Gap between services
    gap_between_age_groups = 8  # Larger gap between age groups

    # Store age group positions for labels
    age_group_positions = {}

    for age_idx, age_cat in enumerate(age_labels):
        if age_cat not in flow_counts["age_category"].values:
            continue

        age_data = flow_counts[flow_counts["age_category"] == age_cat]

        if age_idx > 0:
            theta_position += gap_between_age_groups

        # Record start position for this age group
        age_group_start = theta_position

        # Get color for this age group
        base_color = age_color_map.get(age_cat, TAILWIND_COLORS["slate-400"])

        for _, row in age_data.iterrows():
            service = row["normalized_type"]
            percentage = row["percentage"]
            count = row["count"]

            # Add radial bar
            fig.add_trace(
                go.Barpolar(
                    r=[percentage],
                    theta=[theta_position],
                    width=[bar_width],
                    marker=dict(
                        color=base_color,
                        line=dict(color="white", width=1),
                    ),
                    name=f"{service}",
                    text=[f"{service}<br>{percentage}%"],
                    hovertemplate=(
                        f"<b>{service}</b><br>"
                        f"{age_cat}<br>"
                        f"{count:,} patients ({percentage}%)<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

            theta_position += bar_width + gap_between_services

        # Calculate middle position for age group label
        age_group_middle = (age_group_start + theta_position - gap_between_services) / 2
        age_group_positions[age_cat] = age_group_middle

    # Apply base styling
    fig = style_plotly_layout(
        fig, theme=theme, height=600, margin={"t": 80, "l": 40, "r": 40, "b": 40}
    )

    # Configure polar layout for radial bars
    fig.update_layout(
        title=dict(
            text="Radial View: Age Groups → Referral Services",
            x=0.5,
            xanchor="center",
            font=dict(size=18, family="Roboto, sans-serif"),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, flow_counts["percentage"].max() * 1.1],
                showticklabels=True,
                ticksuffix="%",
                tickfont=dict(size=10),
                gridcolor="rgba(128, 128, 128, 0.2)",
                gridwidth=1,
            ),
            angularaxis=dict(
                visible=False,  # Hide angular axis for cleaner look
                direction="clockwise",
                rotation=90,
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )

    # Add age group labels as annotations
    annotations = []
    label_radius = flow_counts["percentage"].max() * 1.15

    for age_cat, theta in age_group_positions.items():
        # Convert theta to radians for positioning
        theta_rad = math.radians(90 - theta)  # Adjust for plotly's coordinate system

        x = label_radius * math.cos(theta_rad)
        y = label_radius * math.sin(theta_rad)

        # Simplify age group labels
        label_short = age_cat.split("(")[0].strip()

        annotations.append(
            dict(
                x=x,
                y=y,
                xref="x",
                yref="y",
                text=f"<b>{label_short}</b>",
                showarrow=False,
                font=dict(
                    size=13,
                    color=age_color_map.get(age_cat, TAILWIND_COLORS["slate-400"]),
                    family="Roboto, sans-serif",
                ),
                xanchor="center",
                yanchor="middle",
            )
        )

    fig.update_layout(annotations=annotations)

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


# ==============================================================================
# PARALLEL CATEGORIES: Age Groups → Referral Services
# ==============================================================================


def build_age_referral_parallel_categories(theme: str) -> str:  # noqa: C901
    """
    Parallel categories chart showing structured flow from age groups to referral types.

    Features:
    - Two distinct columns: Age Groups | Referral Services
    - Flow ribbons show connections
    - Color-coded by age group
    - Clean, organized visualization
    """
    # Get patients with age data
    patients_qs = Patients.objects.all().values("id", "age")
    patients_data = list(patients_qs)
    df_patients = pd.DataFrame.from_records(patients_data) if patients_data else pd.DataFrame()

    if df_patients.empty:
        return "<p>No patient data available</p>"

    # Get referrals
    referrals_qs = Referrals.objects.all().values(
        "patient_ID", "referral_1", "referral_2", "referral_3", "referral_4", "referral_5"
    )
    referrals_data = list(referrals_qs)
    df_referrals = pd.DataFrame.from_records(referrals_data) if referrals_data else pd.DataFrame()

    if df_referrals.empty:
        return "<p>No referral data available</p>"

    # Get OD referrals
    od_referrals_qs = ODReferrals.objects.all().values("patient_id")
    od_referrals_data = list(od_referrals_qs)
    df_od = pd.DataFrame.from_records(od_referrals_data) if od_referrals_data else pd.DataFrame()

    # Prepare age categories
    ages = pd.to_numeric(df_patients["age"], errors="coerce")
    bins = [-1, 24, 44, 64, float("inf")]
    age_labels = ["Youth (0-24)", "Adults (25-44)", "Middle Age (45-64)", "Seniors (65+)"]
    df_patients["age_category"] = pd.cut(
        ages, bins=bins, labels=age_labels, include_lowest=True, right=True
    )

    # Melt referrals into long format
    referral_cols = ["referral_1", "referral_2", "referral_3", "referral_4", "referral_5"]
    df_referrals_long = df_referrals.melt(
        id_vars=["patient_ID"], value_vars=referral_cols, value_name="referral_type"
    )

    # Clean and normalize referral types
    df_referrals_long["referral_type"] = (
        df_referrals_long["referral_type"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    df_referrals_long = df_referrals_long.dropna(subset=["referral_type"])
    df_referrals_long = df_referrals_long[df_referrals_long["referral_type"] != ""]

    def normalize_referral(ref_type: str) -> str | None:
        ref_lower = ref_type.lower().strip()
        if "911" in ref_lower or "walk-in" in ref_lower or "walk in" in ref_lower:
            return "911 calls"
        if "vax -" in ref_lower or "vax-" in ref_lower or ref_lower.startswith("vax "):
            return "Vaccinations"
        if "lab - blood draw" in ref_lower or "lab-blood draw" in ref_lower:
            return "Lab - Blood Draw"
        if "eval - assessment" in ref_lower or "eval-assessment" in ref_lower:
            return "Eval - Assessment"
        if "case management" in ref_lower:
            return "Case Management"
        if "eval - psych" in ref_lower or "dementia" in ref_lower or "crisis" in ref_lower:
            return "Eval - Psych/Dementia/Crisis"
        if "overdose" in ref_lower or ref_lower == "od":
            return "Overdose"
        if "med - rx reconciliation" in ref_lower or "med-rx reconciliation" in ref_lower:
            return "Med - Rx Reconciliation"
        if "med - antipsychotic im" in ref_lower or "antipsychotic" in ref_lower:
            return "Med - Antipsychotic IM"
        return None

    df_referrals_long["normalized_type"] = df_referrals_long["referral_type"].apply(
        normalize_referral
    )
    df_referrals_long = df_referrals_long.dropna(subset=["normalized_type"])

    # Join with patients to get age categories
    df_merged = df_referrals_long.merge(
        df_patients[["id", "age_category"]], left_on="patient_ID", right_on="id", how="left"
    )

    # Add overdose referrals from ODReferrals table
    if not df_od.empty:
        df_od_merged = df_od.merge(
            df_patients[["id", "age_category"]], left_on="patient_id", right_on="id", how="left"
        )
        df_od_merged["normalized_type"] = "Overdose"
        df_od_merged = df_od_merged[["patient_id", "normalized_type", "age_category"]].rename(
            columns={"patient_id": "patient_ID"}
        )
        df_merged = pd.concat(
            [df_merged[["patient_ID", "normalized_type", "age_category"]], df_od_merged],
            ignore_index=True,
        )

    # Remove rows with missing age category
    df_merged = df_merged.dropna(subset=["age_category"])

    if df_merged.empty:
        return "<p>No referral pathway data available</p>"

    # Convert age categories to strings for parallel categories
    df_merged["age_category"] = df_merged["age_category"].astype(str)

    # Create color column based on age group
    df_merged["color_num"] = df_merged["age_category"].map(
        {
            "Youth (0-24)": 0,
            "Adults (25-44)": 1,
            "Middle Age (45-64)": 2,
            "Seniors (65+)": 3,
        }
    )

    fig = go.Figure(
        data=[
            go.Parcats(
                dimensions=[
                    dict(
                        label="Age Group",
                        values=df_merged["age_category"],
                        categoryorder="array",
                        categoryarray=age_labels,
                    ),
                    dict(
                        label="Referral Service",
                        values=df_merged["normalized_type"],
                        categoryorder="category ascending",
                    ),
                ],
                line=dict(
                    color=df_merged["color_num"],
                    colorscale=[
                        [0, PATIENT_CHART_COLORS[0]],
                        [0.33, PATIENT_CHART_COLORS[1]],
                        [0.67, PATIENT_CHART_COLORS[3]],
                        [1, PATIENT_CHART_COLORS[5]],
                    ],
                    shape="hspline",
                ),
                hoveron="dimension",
                hoverinfo="count+probability",
            )
        ]
    )

    fig = style_plotly_layout(
        fig, theme=theme, height=500, margin={"t": 50, "l": 10, "r": 10, "b": 50}
    )

    fig.update_layout(
        title=dict(
            text="Parallel Categories: Age Groups → Referral Services",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
    )

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


# ==============================================================================
# Build both production charts
# ==============================================================================


def build_production_age_charts(theme: str) -> dict[str, str]:
    """
    Build production-ready age charts for patients dashboard.

    Returns:
        Dictionary with 'age_bar', 'age_referral_sankey', 'age_referral_sunburst',
        and 'age_referral_parallel' chart HTML.
    """
    return {
        "age_bar": build_simplified_age_bar_chart(theme),
        "age_referral_sankey": build_enhanced_age_referral_sankey(theme),
        "age_referral_sunburst": build_age_referral_sunburst(theme),
        "age_referral_parallel": build_age_referral_parallel_categories(theme),
    }
