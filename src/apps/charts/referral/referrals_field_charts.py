from collections.abc import Collection

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.chart_normalization import add_share_columns, count_share_text
from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Referrals

# Import patient chart colors for consistency with patients quarterly chart
from ..patients.patient_field_charts import PATIENT_CHART_COLORS


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
    vc_df["share_pct_rounded"] = vc_df["share_pct"].round(1)
    fig = px.pie(
        vc_df,
        names=label_col,
        values=value_col,
        hole=0.55,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=["share_pct_rounded"],
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{customdata[0]:.1f}%",
        hovertemplate="%{label}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=400,  # Increased to accommodate bottom legend
        x_title=None,
        y_title=None,
        margin={"t": 30, "l": 24, "r": 24, "b": 80},  # Extra bottom margin for legend
        show_legend=True,
    )
    # Legend positioned below chart on all screen sizes for consistency
    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal orientation wraps better
            yanchor="top",
            y=-0.15,  # Position below chart
            xanchor="center",
            x=0.5,  # Center horizontally
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
        )
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


def _build_donut_chart_top_legend(
    vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str
) -> str:
    """Build a donut chart with legend positioned at the top for better mobile display."""
    vc_df = add_share_columns(vc_df, value_col)

    # Truncate long labels for better outside label display
    def truncate_label(label: str, max_length: int = 25) -> str:
        """Truncate labels intelligently, preserving key information."""
        if len(label) <= max_length:
            return label

        # Try to break at hyphen and keep first part
        if " - " in label:
            first_part = label.split(" - ")[0]
            if len(first_part) <= max_length:
                return first_part + "..."

        # Try to keep first words up to max_length
        words = label.split()
        truncated = ""
        for word in words:
            if len(truncated) + len(word) + 4 <= max_length:  # +4 for " ..."
                truncated += word + " "
            else:
                break

        if truncated:
            return truncated.strip() + "..."

        # Last resort: hard truncate
        return label[: max_length - 3] + "..."

    # Apply truncation to labels and store full labels for hover
    vc_df = vc_df.copy()
    vc_df["full_label"] = (
        vc_df[label_col].astype(str).copy()
    )  # Keep original for hover BEFORE truncation
    vc_df[label_col] = vc_df[label_col].apply(truncate_label)  # Now apply truncation

    # Pre-compute share percentages to avoid Plotly NaN percent behavior
    vc_df = add_share_columns(vc_df, value_col)
    vc_df["share_pct_rounded"] = vc_df["share_pct"].fillna(0.0).round(1)

    # Use go.Pie for more control over hover behavior
    fig = go.Figure(
        data=[
            go.Pie(
                labels=vc_df[label_col],
                values=vc_df[value_col],
                hole=0.55,
                marker=dict(colors=COLOR_SEQUENCE[: len(vc_df)], line=dict(color="white", width=1)),
                textposition="outside",
                textinfo="none",  # texttemplate provides the label and share
                textfont=dict(size=11),
                customdata=vc_df[["full_label", "share_pct_rounded"]].values,
                texttemplate="%{label}<br>%{customdata[1]:.1f}%",
                hovertemplate="<b>%{customdata[0]}</b><br>Share: %{customdata[1]:.1f}%<extra></extra>",
            )
        ]
    )
    fig = go.Figure(fig)  # Wrap in Figure
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=500,  # Increased height to accommodate outside labels
        x_title=None,
        y_title=None,
        margin={"t": 20, "l": 20, "r": 20, "b": 20},  # Balanced margins for labels
        show_legend=False,  # No legend needed - labels are on the chart
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


def _render_zipcode_chart(vc: pd.DataFrame, theme: str) -> str:
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
    vc["display_name"] = vc["zipcode"].map(zip_display_names)
    # If no mapping exists, use the original value
    vc["display_name"] = vc["display_name"].fillna(vc["zipcode"])

    # Merge Transient categories by summing their counts
    transient_mask = vc["display_name"] == "Transient"
    if transient_mask.sum() > 1:
        # Sum all Transient entries
        transient_count = vc.loc[transient_mask, "count"].sum()
        # Remove all Transient rows
        vc = vc[~transient_mask].copy()
        # Add single Transient row
        transient_row = pd.DataFrame(
            [{"zipcode": "Transient", "display_name": "Transient", "count": transient_count}]
        )
        vc = pd.concat([vc, transient_row], ignore_index=True)

    # Create reverse lookup for groups
    zip_to_group = {}
    for group_name, zips in zip_groups.items():
        for zip_code in zips:
            zip_to_group[zip_code] = group_name

    # Add group information to dataframe
    vc["group"] = vc["zipcode"].apply(lambda z: zip_to_group.get(z, "Port Angeles"))
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


def _build_sex_stacked_bar_chart(df: pd.DataFrame, theme: str) -> str:
    """Build stacked bar chart for sex field - stacked by age groups."""
    # Define age bins and labels
    ages = pd.to_numeric(df["age"], errors="coerce")
    bins = [-1, 17, 24, 34, 44, 54, 64, 74, 84, float("inf")]
    labels = ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]

    # Create a temporary dataframe with age groups and sex
    temp_df = df[["age", "sex"]].copy()
    temp_df["age_group"] = pd.cut(ages, bins=bins, labels=labels, include_lowest=True, right=True)

    # Separate by sex
    male_df = temp_df[temp_df["sex"].str.lower().isin(["male", "m"])]
    female_df = temp_df[temp_df["sex"].str.lower().isin(["female", "f"])]

    # Count by age group for each sex
    male_counts = male_df["age_group"].value_counts().reindex(labels, fill_value=0)
    female_counts = female_df["age_group"].value_counts().reindex(labels, fill_value=0)

    # Create stacked bar chart
    fig = go.Figure()

    # Add male bars
    fig.add_trace(
        go.Bar(
            x=["Male"],
            y=[male_counts.sum()],
            name="Male",
            marker_color=COLOR_SEQUENCE[0],
            hovertemplate="Male<br>Count: %{y}<extra></extra>",
        )
    )

    # Add female bars
    fig.add_trace(
        go.Bar(
            x=["Female"],
            y=[female_counts.sum()],
            name="Female",
            marker_color=COLOR_SEQUENCE[3],
            hovertemplate="Female<br>Count: %{y}<extra></extra>",
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=500,  # Match referral_closed_reason chart height
        x_title="Sex",
        y_title="Referral Count",
        margin={"t": 40, "l": 80, "r": 20, "b": 50},
        show_legend=False,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        ticklabelstandoff=10,
    )
    fig.update_layout(
        bargap=0.15,
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


def _build_referral_agency_bar_chart(
    vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str
) -> str:
    """Build a horizontal bar chart for referral agencies with counts and percentages."""
    vc_df = add_share_columns(vc_df, value_col)
    vc_df["share_pct_rounded"] = vc_df["share_pct"].round(1)

    # Sort by count descending
    vc_df = vc_df.sort_values(value_col, ascending=True)

    # Define color groups for agencies
    fire_dept_group = {"PAFD - Line", "911 - CPM Primary", "PAFD - CPM"}
    patient_group = {"Patient"}
    healthcare_group = {"NOHN", "OPCC", "OMC", "3C"}

    # Assign colors based on agency grouping
    def get_agency_color(agency: str) -> str:
        if agency in fire_dept_group:
            return PATIENT_CHART_COLORS[0]  # Violet for fire dept
        elif agency in patient_group:
            return PATIENT_CHART_COLORS[2]  # Emerald for patient
        elif agency in healthcare_group:
            return PATIENT_CHART_COLORS[1]  # Cyan for healthcare partners
        else:
            return PATIENT_CHART_COLORS[3]  # Amber for other agencies

    # Create color list for each bar
    bar_colors = [get_agency_color(label) for label in vc_df[label_col]]

    fig = px.bar(
        vc_df,
        y=label_col,
        x=value_col,
        orientation="h",
        custom_data=["share_pct_rounded"],
    )

    fig.update_traces(
        texttemplate="<b>%{customdata[0]}%</b>",
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=14, family="Roboto, sans-serif", color="white"),
        hovertemplate="<b>%{y}</b><br>Share: %{customdata[0]}%<extra></extra>",
        marker=dict(color=bar_colors, line=dict(width=0)),
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=max(300, len(vc_df) * 50),  # Dynamic height based on number of agencies
        x_title="Referral Count",
        y_title=None,
        margin={"t": 40, "l": 0, "r": 0, "b": 50},  # Top margin for modebar clearance
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(
        showgrid=False,
        ticklabelstandoff=9,  # 1/8-inch gap (9px at 72dpi) between labels and chart
    )
    fig.update_layout(
        showlegend=False,
        bargap=0.075,  # Reduced from default ~0.15 to half
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


def _build_treemap_chart(vc_df: pd.DataFrame, label_col: str, value_col: str, theme: str) -> str:
    vc_df = add_share_columns(vc_df, value_col)
    vc_df["share_pct_rounded"] = vc_df["share_pct"].round(1)

    # Use explicit color value
    text_color = "#0f172a" if theme == "light" else "#f8fafc"

    fig = px.treemap(
        vc_df,
        path=[label_col],
        values=value_col,
        color=label_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=["share_pct_rounded"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]}%",  # Add spaces for padding
        textposition="top left",
        textfont=dict(size=16, family="Roboto, sans-serif", color=text_color),
        hovertemplate="<extra></extra>",
        marker=dict(line=dict(width=2)),
        root_color="rgba(0,0,0,0)",
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=360,
        x_title=None,
        y_title=None,
        margin={"t": 0, "l": 0, "r": 0, "b": 0},  # Add margin for spacing
    )
    fig.update_layout(showlegend=False)
    return plot(
        fig,
        output_type="div",
        config={
            "responsive": True,
            "displaylogo": False,
            "staticPlot": True,  # Disable all interactions including hover cursor
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


def _process_encounter_categories(
    df: pd.DataFrame, category_labels: dict[int, str], cat_data: list[dict]
) -> None:
    """Process encounter type categories from a dataframe and add to cat_data list."""
    for cat_num, col_name in enumerate(
        ["encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"], 1
    ):
        if col_name not in df.columns:
            continue

        # Clean and count values for this category
        series = df[col_name].fillna("").astype(str).str.strip()
        series = series.replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})

        # Filter out unwanted values
        excluded = {"not disclosed", "no data", "no-data", "no_data", "", "unknown"}
        series = series[~series.str.lower().isin(excluded)]

        if series.empty:
            continue

        # Get value counts and add to existing data
        vc = series.value_counts()
        for value, count in vc.items():
            # Check if this category-value combination already exists
            existing_idx = None
            for idx, item in enumerate(cat_data):
                if item["category"] == category_labels[cat_num] and item["value"] == str(value):
                    existing_idx = idx
                    break

            if existing_idx is not None:
                # Add to existing count
                cat_data[existing_idx]["count"] += int(count)
            else:
                # Add new entry
                cat_data.append(
                    {"category": category_labels[cat_num], "value": str(value), "count": int(count)}
                )


def _build_hierarchical_encounter_treemap(df: pd.DataFrame, theme: str) -> str | None:
    """Build a treemap showing three category sections: Cat1, Cat2, Cat3 with their values.

    This function combines data from both Referrals (passed as df) and Encounters tables
    to show the complete picture of encounter types across the program.
    """
    from ..encounters.encounters_field_charts import Encounters

    # Descriptive labels for each category
    category_labels = {
        1: "Engagement Method",  # How: in-person visit, phone call, meeting
        2: "Communication Type",  # What: conversation, email
        3: "Contact Type",  # Who: patient, healthcare team, social services
    }

    # Prepare data list
    cat_data: list[dict] = []

    # Process referrals data (from the passed df)
    _process_encounter_categories(df, category_labels, cat_data)

    # Now fetch and process encounters data
    try:
        encounters_qs = Encounters.objects.all().values(
            "encounter_type_cat1",
            "encounter_type_cat2",
            "encounter_type_cat3",
        )
        encounters_data = list(encounters_qs)
        if encounters_data:
            encounters_df = pd.DataFrame.from_records(encounters_data)
            _process_encounter_categories(encounters_df, category_labels, cat_data)
    except Exception:
        # If encounters table isn't available or there's an error, continue with just referrals data
        pass

    if not cat_data:
        return None

    treemap_df = pd.DataFrame(cat_data)

    # Calculate percentages within each category
    treemap_df["percentage"] = treemap_df.groupby("category")["count"].transform(
        lambda x: (x / x.sum() * 100)
    )

    # Use explicit color value for text
    text_color = "#0f172a" if theme == "light" else "#f8fafc"

    # Parent category background colors - muted slate like accordion buttons
    parent_color = "#f1f5f9" if theme == "light" else "#334155"  # slate-100 / slate-700

    # First create the figure to get the structure
    fig = px.treemap(
        treemap_df,
        path=["category", "value"],
        values="count",
        color="value",  # Color by actual values, not categories
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=["percentage"],  # Include percentage for hover
    )

    # Build color mapping: parents get slate, children get vibrant colors from discrete sequence
    # Get unique values and assign colors
    unique_values = treemap_df["value"].unique()
    value_color_map = {
        val: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, val in enumerate(unique_values)
    }

    # Parent category labels for identification
    parent_labels = {"Engagement Method", "Communication Type", "Contact Type"}

    # Extract labels and parents from the trace to build color list and hover text list
    trace_data = fig.data[0]
    color_list = []
    hover_list = []
    for label, parent in zip(trace_data.labels, trace_data.parents, strict=True):  # type: ignore[attr-defined]
        if label in parent_labels or parent == "":  # parent == "" means it's the root
            # This is a parent category or root - use muted slate and disable hover
            color_list.append(parent_color)
            hover_list.append("<extra></extra>")
        else:
            # This is a child value - use vibrant color from map and show hover with percentage
            color_list.append(value_color_map.get(label, COLOR_SEQUENCE[0]))
            # Find the percentage for this label
            pct = treemap_df[treemap_df["value"] == label]["percentage"].iloc[0]
            hover_list.append(f"<b>{label}</b><br>{pct:.1f}%<extra></extra>")

    # Update trace with custom colors and hover templates
    fig.update_traces(
        textposition="top left",
        textfont=dict(size=14, family="Roboto, sans-serif", color=text_color),
        hovertemplate=hover_list,
        marker=dict(
            line=dict(width=1),
            colors=color_list,
        ),
        root_color="rgba(0,0,0,0)",
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=500,  # Taller for better visibility
        x_title=None,
        y_title=None,
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
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


def _build_special_chart(df: pd.DataFrame, vc: pd.DataFrame, field: str, theme: str) -> str | None:
    """Build special chart types (stacked bar, donut, treemap) for specific fields."""
    if field == "sex":
        # Use original dataframe for stacked bar chart by age
        return _build_sex_stacked_bar_chart(df, theme)
    if field == "insurance":
        vc2 = vc.rename(columns={field: "label"})
        return _build_donut_chart(vc2, "label", "count", theme)
    if field == "referral_closed_reason":
        # Use donut chart with legend on top for better mobile display
        vc2 = vc.rename(columns={field: "label"})
        return _build_donut_chart_top_legend(vc2, "label", "count", theme)
    if field == "zipcode":
        return _render_zipcode_chart(vc, theme)
    if field == "referral_agency":
        vc2 = vc.rename(columns={field: "label"})
        return _build_referral_agency_bar_chart(vc2, "label", "count", theme)
    return None


def _apply_field_specific_transforms(s: pd.Series, field: str) -> pd.Series:
    """Apply field-specific data transformations and filtering."""
    # For referral_agency field, rename agencies for cleaner display
    if field == "referral_agency":
        s = s.replace(
            {
                "NOHN - Medical": "NOHN",
                "OPCC - REdisCOVERY": "OPCC",
                "OMC - Primary": "OMC",
            }
        )

    # For referral_closed_reason field, transform labels for better display
    if field == "referral_closed_reason":
        s = s.replace(
            {
                "Referred - Successful": "Referred<br>Success",
                "Monitored - CPM Not Needed": "CPM Not Needed",
                "RP States CPM Not Needed": "CPM Not Needed",
            }
        )

    # For insurance field, combine categories BEFORE filtering
    if field == "insurance":
        s = s.replace(
            {
                "Medicare, Private": "Medicare",
                "MedicarePrivate": "Medicare",
                "Medicare, Tricare": "Tricare",
                "MedicareTricare": "Tricare",
                "MedicareOther": "Medicare",
                "MedicareOtherPrivate": "Medicare",
                "Medicare, Other": "Medicare",
                "OtherMedicare": "Medicare",
                "PrivateMedicare": "Medicare",
                "OtherTricareMedicare": "Tricare",
                "TricareMedicare": "Tricare",
                "Medicaid, Other": "Medicaid",
                "MedicaidOther": "Medicaid",
                "MedicaidPrivate": "Medicaid",
                "MedicaidTricare": "Tricare",
                "PrivateMedicaid": "Medicaid",
                "Dual Medicaid/MedicarePrivate": "Dual Medicaid/Medicare",
                "Dual Medicaid/Medicare, Tricare": "Dual Medicaid/Medicare",
                "Dual Medicaid/MedicareTricare": "Dual Medicaid/Medicare",
                "Dual Medicaid/Medicare, Private": "Dual Medicaid/Medicare",
                "Dual Medicaid/Medicare, Other": "Dual Medicaid/Medicare",
                "Dual Medicaid/MedicareOther": "Dual Medicaid/Medicare",
                "MedicarePrivateMedicaid": "Dual Medicaid/Medicare",
                "MedicareMedicaid": "Dual Medicaid/Medicare",
                "MedicaidMedicare": "Dual Medicaid/Medicare",
                "Indian Health ServiceTricareMedicare": "Indian Health Service",
                "Indian Health Service, Medicaid": "Indian Health Service",
                "Medicare, Private, Other": "Medicare",
                "MedicarePrivateOther": "Medicare",
                "OtherMedicarePrivate": "Medicare",
                "Private, Other": "Private",
                "Private, Medicaid": "Medicaid",
                "Other, Medicare": "Medicare",
                "OtherMedicaid": "Medicaid",
                "Other": "Private",
                "Unknown": "Not disclosed",
                "Not Applicable": "Not disclosed",
            }
        )

    # Filter out unwanted values based on field type
    # For all fields including insurance, filter out "not disclosed" and "single"
    s_clean = s[~s.str.lower().isin({"not disclosed", "single"})]

    # For encounter type categories, drop explicit 'No data' buckets
    if field.startswith("encounter_type_cat"):
        s_clean = s_clean[~s_clean.str.strip().str.lower().isin({"no data", "no-data", "no_data"})]

    return s_clean


def _build_chart_for_field(df: pd.DataFrame, field: str, theme: str) -> str:
    s = df[field]

    if not _is_numeric_series(s):
        s = s.fillna("").replace({None: ""}).astype(str).str.strip()
        s = s.replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})

    if _is_numeric_series(df[field]) or field == "age":
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

            # Create color list - highlight 55-84 age groups
            highlight_groups = {"55–64", "65–74", "75–84"}
            colors = [
                TAILWIND_COLORS["amber-500"]
                if row["age_group"] in highlight_groups
                else TAILWIND_COLORS["indigo-600"]
                for _, row in vc.iterrows()
            ]

            # No text labels above bars
            fig = px.bar(
                vc,
                x="age_group",
                y="count",
                custom_data=vc[["count", "share_pct"]],
            )
            fig.update_traces(
                marker_color=colors,
                hovertemplate="Age group: %{x}<br>Count: %{customdata[0]}<br>Share: %{customdata[1]:.1f}%<extra></extra>",
            )
            fig = style_plotly_layout(
                fig,
                theme=theme,
                height=360,
                x_title="Age group",
                y_title="Count",
                margin={"t": 30, "l": 60, "r": 10, "b": 40},  # Consistent margins
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=False,
                showticklabels=True,
                title="Count",
                ticklabelposition="outside",
                automargin=True,
            )
            fig.update_layout(bargap=0.15)  # ~1/16 inch spacing between bars
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
                margin={"t": 30, "l": 60, "r": 10, "b": 40},  # Consistent margins
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(
                showgrid=False,
                showticklabels=True,
                title="Count",
                ticklabelposition="outside",
                automargin=True,
            )
            fig.update_layout(bargap=0.15)  # ~1/16 inch spacing between bars
    else:
        # Apply field-specific transformations and filtering
        s_clean = _apply_field_specific_transforms(s, field)

        vc_full = s_clean.value_counts()

        # If no data after filtering, return empty string
        # Exception: for insurance field, we might have all "Not disclosed" which is valid
        if vc_full.empty or vc_full.size == 0:
            return ""

        # Decide if we should use a simple bar for very few categories (<= 3)
        few_cats = int(s_clean.nunique()) <= 3
        top_n = 13 if field == "referral_agency" else 8
        vc_top = vc_full.head(top_n)
        other_count = int(vc_full.iloc[top_n:].sum()) if vc_full.size > top_n else 0
        vc = vc_top.reset_index()
        vc.columns = [field, "count"]
        # Add "Other" category for fields that have more than top_n entries
        # Exception: Don't add "Other" for referral_agency field
        if other_count > 0 and field != "referral_agency":
            vc = pd.concat(
                [vc, pd.DataFrame([{field: "Other", "count": other_count}])], ignore_index=True
            )

        # If very few categories, try special chart types for certain fields
        # Always check for sex and insurance fields which need special treatment
        if not few_cats or field in {"sex", "insurance"}:
            special_chart = _build_special_chart(df, vc, field, theme)
            if special_chart is not None:
                return special_chart

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
        fig.update_layout(bargap=0.15)  # ~1/16 inch spacing between bars

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


def _build_quarterly_referrals_chart(df: pd.DataFrame, theme: str) -> str | None:  # noqa: C901
    """Build quarterly referrals bar chart styled like the patients quarterly chart."""
    if df.empty or "date_received" not in df.columns:
        return None

    dates = pd.to_datetime(df["date_received"], errors="coerce")
    qdf = pd.DataFrame(
        {
            "year": dates.dt.year,
            "quarter": dates.dt.quarter,
        }
    ).dropna()
    qdf = qdf.groupby(["year", "quarter"], dropna=True).size().reset_index(name="count")

    if qdf.empty:
        return None

    # Move 2022 Q3 referral(s) to 2023 Q1
    q3_2022_count = qdf.loc[(qdf["year"] == 2022) & (qdf["quarter"] == 3), "count"].sum()
    if q3_2022_count > 0:
        # Remove 2022 Q3
        qdf = qdf[~((qdf["year"] == 2022) & (qdf["quarter"] == 3))]
        # Add to 2023 Q1
        q1_2023_idx = qdf[(qdf["year"] == 2023) & (qdf["quarter"] == 1)].index
        if len(q1_2023_idx) > 0:
            qdf.loc[q1_2023_idx[0], "count"] += q3_2022_count
        else:
            # Create 2023 Q1 if it doesn't exist
            qdf = pd.concat(
                [qdf, pd.DataFrame([{"year": 2023, "quarter": 1, "count": q3_2022_count}])],
                ignore_index=True,
            )

    # Add placeholder rows for 2022 Q1, Q2, and Q3 (did not track referrals)
    placeholder_rows = pd.DataFrame(
        [
            {"year": 2022, "quarter": 1, "count": 0},
            {"year": 2022, "quarter": 2, "count": 0},
            {"year": 2022, "quarter": 3, "count": 0},
        ]
    )
    qdf = pd.concat([qdf, placeholder_rows], ignore_index=True)
    qdf = qdf.drop_duplicates(subset=["year", "quarter"], keep="first")

    qdf2 = qdf.sort_values(["year", "quarter"]).reset_index(drop=True)
    qdf2 = add_share_columns(qdf2, "count")

    # Simple text labels showing just the count (but not for placeholder rows)
    qdf2["text_label"] = qdf2["count"].apply(lambda x: f"{int(x)}" if x > 0 else "")

    x_years = qdf2["year"].astype(str).tolist()
    x_quarters = [f"Q{q}" for q in qdf2["quarter"].tolist()]  # Format as Q1, Q2, Q3, Q4

    # Calculate baseline average from 2021, 2023, and 2024 (normal operational years)
    # Exclude 2022 (Behavioral Health expansion) and 2025 (staffing changes)
    baseline_years = [2021, 2023, 2024]
    baseline_data = qdf2[qdf2["year"].isin(baseline_years)]
    if not baseline_data.empty:
        normalized_quarterly_avg = round(baseline_data["count"].mean(), 1)
    else:
        normalized_quarterly_avg = round(qdf2["count"].mean(), 1)

    # Color bars based on whether they exceed the average
    # Emerald if above average, Cyan if at or below average (matching patient chart palette)
    # Use transparent for placeholder rows
    bar_colors = []
    for _, row in qdf2.iterrows():
        if row["year"] == 2022 and row["quarter"] in [1, 2, 3]:
            bar_colors.append("rgba(0, 0, 0, 0)")  # Transparent for "did not track" quarters
        elif row["count"] > normalized_quarterly_avg:
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
                    "Referrals: %{y}<br>"
                    "Share of Total: %{customdata[0]:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=620,  # Increased height to match patients chart
        x_title=None,
        y_title="Referral Count",
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
        title="Referral Count",
        ticklabelstandoff=10,  # Space between tick labels and axis
        ticklabelposition="outside",
        automargin=True,
        range=[0, 300],  # Fixed range with top at 300 for annotation space
    )

    # Initialize annotations list
    all_annotations = []

    # Add "Did not track" annotations for 2022 Q1, Q2, and Q3
    for i, (year, quarter) in enumerate(zip(x_years, x_quarters, strict=False)):
        if year == "2022" and quarter in ["Q1", "Q2", "Q3"]:
            all_annotations.append(
                dict(
                    x=i,
                    xref="x",
                    y=normalized_quarterly_avg * 0.5,  # Position at half the baseline
                    yref="y",
                    text="<b>Did not track</b>",
                    textangle=270,  # Vertical text
                    showarrow=False,
                    font=dict(size=11, color="#64748b", family="Arial, sans-serif"),
                    xanchor="center",
                    yanchor="middle",
                )
            )

    # Add year-based context annotations above bars
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
            year_indices = [i for i, y in enumerate(x_years) if y == year]
            if year_indices:
                center_idx = (year_indices[0] + year_indices[-1]) / 2.0
                all_annotations.append(
                    dict(
                        x=center_idx,
                        xref="x",
                        y=annotation_y,  # Positioned at y=275 in data coordinates
                        yref="y",  # Use data coordinates for precise positioning
                        text=f"<b>{year_labels[year]}</b>",
                        textangle=270,  # Vertical text reading top-to-bottom
                        showarrow=False,
                        font=dict(size=12, color="#1e293b", family="Arial, sans-serif"),
                        xanchor="center",
                        yanchor="middle",  # Anchor to middle for centered alignment
                    )
                )

    # Create background rectangles for each year
    shapes = []
    year_colors = {
        "2021": PATIENT_CHART_COLORS[0],  # Violet
        "2022": PATIENT_CHART_COLORS[1],  # Cyan
        "2023": PATIENT_CHART_COLORS[3],  # Emerald
        "2024": PATIENT_CHART_COLORS[4],  # Amber
        "2025": PATIENT_CHART_COLORS[5],  # Blue
    }

    for year in unique_years:
        year_indices = [i for i, y in enumerate(x_years) if y == year]
        if year_indices:
            x0 = year_indices[0] - 0.5
            x1 = year_indices[-1] + 0.5
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor=year_colors.get(year, PATIENT_CHART_COLORS[0]),
                    opacity=0.08,
                    layer="below",
                    line_width=0,
                )
            )

    # Add baseline average line
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
                color="rgba(255, 255, 255, 0.5)",
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


def _build_quarterly_encounters_chart(theme: str) -> str | None:  # noqa: C901
    """Build stacked quarterly chart showing referrals and encounters."""
    from ...core.models import ODReferrals
    from ..encounters.encounters_field_charts import Encounters

    try:
        # Get referrals data
        referrals_qs = Referrals.objects.all().values("date_received")
        referrals_data = list(referrals_qs)

        # Get encounters data
        encounters_qs = Encounters.objects.all().values("encounter_date")
        encounters_data = list(encounters_qs)

        # Get PORT referrals data from ODReferrals table
        od_referrals_qs = ODReferrals.objects.all().values("od_date")
        od_referrals_data = list(od_referrals_qs)

        if not referrals_data and not encounters_data and not od_referrals_data:
            return None

        # Process referrals data
        referrals_df = (
            pd.DataFrame.from_records(referrals_data) if referrals_data else pd.DataFrame()
        )

        # Process encounters data
        encounters_df = (
            pd.DataFrame.from_records(encounters_data) if encounters_data else pd.DataFrame()
        )

        # Process PORT referrals from ODReferrals table
        od_referrals_df = (
            pd.DataFrame.from_records(od_referrals_data) if od_referrals_data else pd.DataFrame()
        )

        # Process regular referrals
        if not referrals_df.empty and "date_received" in referrals_df.columns:
            dates = pd.Series(
                pd.to_datetime(referrals_df["date_received"], errors="coerce")
            ).dropna()
            referrals_qdf = pd.DataFrame({"year": dates.dt.year, "quarter": dates.dt.quarter})
            referrals_qdf = (
                referrals_qdf.groupby(["year", "quarter"], dropna=True)
                .size()
                .reset_index(name="referrals_count")
            )

            # Move 2022 Q3 referral(s) to 2023 Q1
            q3_2022_count = referrals_qdf.loc[
                (referrals_qdf["year"] == 2022) & (referrals_qdf["quarter"] == 3), "referrals_count"
            ].sum()
            if q3_2022_count > 0:
                # Remove 2022 Q3
                referrals_qdf = referrals_qdf[
                    ~((referrals_qdf["year"] == 2022) & (referrals_qdf["quarter"] == 3))
                ]
                # Add to 2023 Q1
                q1_2023_idx = referrals_qdf[
                    (referrals_qdf["year"] == 2023) & (referrals_qdf["quarter"] == 1)
                ].index
                if len(q1_2023_idx) > 0:
                    referrals_qdf.loc[q1_2023_idx[0], "referrals_count"] += q3_2022_count
                else:
                    # Create 2023 Q1 if it doesn't exist
                    referrals_qdf = pd.concat(
                        [
                            referrals_qdf,
                            pd.DataFrame(
                                [{"year": 2023, "quarter": 1, "referrals_count": q3_2022_count}]
                            ),
                        ],
                        ignore_index=True,
                    )
        else:
            referrals_qdf = pd.DataFrame(columns=["year", "quarter", "referrals_count"])

        # Process PORT referrals from ODReferrals table
        if not od_referrals_df.empty and "od_date" in od_referrals_df.columns:
            dates = pd.Series(pd.to_datetime(od_referrals_df["od_date"], errors="coerce")).dropna()
            port_qdf = pd.DataFrame({"year": dates.dt.year, "quarter": dates.dt.quarter})
            port_qdf = (
                port_qdf.groupby(["year", "quarter"], dropna=True)
                .size()
                .reset_index(name="port_referrals_count")
            )
        else:
            port_qdf = pd.DataFrame(columns=["year", "quarter", "port_referrals_count"])

        # Process regular encounters
        if not encounters_df.empty and "encounter_date" in encounters_df.columns:
            dates = pd.to_datetime(encounters_df["encounter_date"], errors="coerce")
            encounters_qdf = pd.DataFrame(
                {"year": dates.dt.year, "quarter": dates.dt.quarter}
            ).dropna()
            encounters_qdf = (
                encounters_qdf.groupby(["year", "quarter"], dropna=True)
                .size()
                .reset_index(name="encounters_count")
            )
        else:
            encounters_qdf = pd.DataFrame(columns=["year", "quarter", "encounters_count"])

        # Get all years from all three datasets
        all_years_set = set()
        if not referrals_qdf.empty:
            all_years_set.update(referrals_qdf["year"].unique())
        if not port_qdf.empty:
            all_years_set.update(port_qdf["year"].unique())
        if not encounters_qdf.empty:
            all_years_set.update(encounters_qdf["year"].unique())

        if not all_years_set:
            return None

        all_years = sorted(all_years_set)

        # Create complete set of year-quarter combinations
        all_quarters = []
        for year in all_years:
            for quarter in [1, 2, 3, 4]:
                all_quarters.append({"year": year, "quarter": quarter})

        complete_df = pd.DataFrame(all_quarters)

        # Merge all three datasets
        qdf = complete_df.copy()
        if not referrals_qdf.empty:
            qdf = qdf.merge(referrals_qdf, on=["year", "quarter"], how="left")
        else:
            qdf["referrals_count"] = 0

        if not port_qdf.empty:
            qdf = qdf.merge(port_qdf, on=["year", "quarter"], how="left")
        else:
            qdf["port_referrals_count"] = 0

        if not encounters_qdf.empty:
            qdf = qdf.merge(encounters_qdf, on=["year", "quarter"], how="left")
        else:
            qdf["encounters_count"] = 0

        qdf["referrals_count"] = qdf["referrals_count"].fillna(0).astype(int)
        qdf["port_referrals_count"] = qdf["port_referrals_count"].fillna(0).astype(int)
        qdf["encounters_count"] = qdf["encounters_count"].fillna(0).astype(int)
        qdf["total_count"] = (
            qdf["referrals_count"] + qdf["port_referrals_count"] + qdf["encounters_count"]
        )

        # Mark quarters with 0 counts as "did not track"
        qdf["is_missing"] = (
            (qdf["total_count"] == 0) & (qdf["year"] == 2022) & (qdf["quarter"].isin([1, 2, 3]))
        )

        qdf = qdf.sort_values(["year", "quarter"]).reset_index(drop=True)

        x_years = qdf["year"].astype(str).tolist()
        x_quarters = [f"Q{q}" for q in qdf["quarter"].tolist()]

        # Create stacked bar chart with three categories
        fig = go.Figure()

        # Add encounters bars (bottom of stack)
        fig.add_trace(
            go.Bar(
                name="Encounters",
                x=[x_years, x_quarters],
                y=qdf["encounters_count"],
                text=[
                    f"{int(val)}" if val > 0 and not is_missing else ""
                    for val, is_missing in zip(
                        qdf["encounters_count"], qdf["is_missing"], strict=False
                    )
                ],
                textposition="inside",
                textfont=dict(size=12, color="white", family="Roboto, sans-serif"),
                marker_color=PATIENT_CHART_COLORS[2],  # Emerald for encounters
                hovertemplate="Encounters: %{y}<extra></extra>",
            )
        )

        # Add referrals bars (middle of stack)
        fig.add_trace(
            go.Bar(
                name="Referrals",
                x=[x_years, x_quarters],
                y=qdf["referrals_count"],
                text=[
                    f"{int(val)}" if val > 0 and not is_missing else ""
                    for val, is_missing in zip(
                        qdf["referrals_count"], qdf["is_missing"], strict=False
                    )
                ],
                textposition="inside",
                textfont=dict(size=12, color="white", family="Roboto, sans-serif"),
                marker_color=PATIENT_CHART_COLORS[1],  # Cyan for referrals
                hovertemplate="Referrals: %{y}<extra></extra>",
            )
        )

        # Add PORT referrals bars (top of stack)
        fig.add_trace(
            go.Bar(
                name="PORT Referrals",
                x=[x_years, x_quarters],
                y=qdf["port_referrals_count"],
                text=[
                    f"{int(val)}" if val > 0 and not is_missing else ""
                    for val, is_missing in zip(
                        qdf["port_referrals_count"], qdf["is_missing"], strict=False
                    )
                ],
                textposition="inside",
                textfont=dict(size=12, color="white", family="Roboto, sans-serif"),
                marker_color=PATIENT_CHART_COLORS[0],  # Violet for PORT referrals
                hovertemplate="PORT Referrals: %{y}<extra></extra>",
            )
        )

        # Add invisible bar trace for total labels (uses base to position at top of stack)
        fig.add_trace(
            go.Bar(
                name="",
                x=[x_years, x_quarters],
                y=[0] * len(qdf),  # Zero height
                base=qdf["total_count"].tolist(),  # Position at top of stacked bars
                text=[
                    f"{int(val)}" if val > 0 and not is_missing else ""
                    for val, is_missing in zip(qdf["total_count"], qdf["is_missing"], strict=False)
                ],
                textposition="outside",
                textfont=dict(size=14),
                marker=dict(color="rgba(0,0,0,0)"),  # Transparent
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Style the chart
        fig = style_plotly_layout(
            fig,
            theme=theme,
            height=620,  # Increased height to match patients chart
            x_title=None,
            y_title="Count",
            margin={
                "t": 40,
                "l": 90,
                "r": 20,
                "b": 20,
            },  # Minimal top margin to reduce gap with header
        )

        fig.update_xaxes(
            type="multicategory",
            showgrid=False,
            ticklabelstandoff=2,  # Reduced space between tick labels and axis baseline
        )
        # Set y-axis range with fixed top at 1300 for consistent annotation positioning
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            showticklabels=True,
            title="Count",
            ticklabelstandoff=10,  # Space between tick labels and axis
            ticklabelposition="outside",
            automargin=True,
            range=[0, 1300],  # Fixed range with top at 1300 for annotation space
        )

        # Add "Did not track" annotations for missing quarters
        all_annotations = []

        for i, (_year, _quarter, is_missing) in enumerate(
            zip(x_years, x_quarters, qdf["is_missing"], strict=False)
        ):
            if is_missing:
                all_annotations.append(
                    dict(
                        x=i,
                        xref="x",
                        y=500,  # Fixed position for "Did not track" text
                        yref="y",
                        text="<b>Did not track</b>",
                        textangle=270,
                        showarrow=False,
                        font=dict(size=11, color="#64748b", family="Arial, sans-serif"),
                        xanchor="center",
                        yanchor="middle",
                    )
                )

        # Add year-based context annotations above bars
        unique_years = sorted(set(x_years))
        year_labels = {
            "2021": "COVID-19",
            "2022": "Behavioral<br>Health",
            "2023": "Normalization",
            "2024": "Normalization",
            "2025": "+2 Community<br>Paramedics",
        }

        # Position annotations at y=1170 (with chart top at y=1300)
        annotation_y = 1105

        for year in unique_years:
            if year in year_labels:
                year_indices = [i for i, y in enumerate(x_years) if y == year]
                if year_indices:
                    center_idx = (year_indices[0] + year_indices[-1]) / 2.0
                    all_annotations.append(
                        dict(
                            x=center_idx,
                            xref="x",
                            y=annotation_y,  # Positioned at y=1175 in data coordinates
                            yref="y",  # Use data coordinates for precise positioning
                            text=f"<b>{year_labels[year]}</b>",
                            textangle=270,  # Vertical text reading top-to-bottom
                            showarrow=False,
                            font=dict(size=12, color="#1e293b", family="Arial, sans-serif"),
                            xanchor="center",
                            yanchor="middle",  # Anchor to middle for centered alignment
                        )
                    )

        # Create background rectangles for each year
        shapes = []
        year_colors = {
            "2021": PATIENT_CHART_COLORS[0],  # Violet
            "2022": PATIENT_CHART_COLORS[1],  # Cyan
            "2023": PATIENT_CHART_COLORS[3],  # Emerald
            "2024": PATIENT_CHART_COLORS[4],  # Amber
            "2025": PATIENT_CHART_COLORS[5],  # Blue
        }

        for year in unique_years:
            year_indices = [i for i, y in enumerate(x_years) if y == year]
            if year_indices:
                x0 = year_indices[0] - 0.5
                x1 = year_indices[-1] + 0.5
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=x0,
                        x1=x1,
                        y0=0,
                        y1=1,
                        fillcolor=year_colors.get(year, PATIENT_CHART_COLORS[0]),
                        opacity=0.08,
                        layer="below",
                        line_width=0,
                    )
                )

        # Add baseline average line (based on encounters only from 2023 and 2024)
        baseline_years = [int(y) for y in unique_years if y in ["2023", "2024"]]
        if baseline_years:
            baseline_data = qdf[(qdf["year"].isin(baseline_years)) & (~qdf["is_missing"])]
            if not baseline_data.empty:
                # Calculate baseline using only encounters count (not total)
                baseline_avg = round(baseline_data["encounters_count"].mean(), 1)
                shapes.append(
                    dict(
                        type="line",
                        xref="paper",
                        yref="y",
                        x0=0,
                        x1=1,
                        y0=baseline_avg,
                        y1=baseline_avg,
                        line=dict(
                            color="rgba(255, 255, 255, 0.5)",
                            width=2,
                            dash="dash",
                        ),
                    )
                )

        fig.update_layout(
            barmode="stack",
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
    except Exception:
        return None


def _build_individual_field_charts(
    df: pd.DataFrame, theme: str, target_fields: set[str] | None
) -> dict[str, str]:
    """Build charts for individual referral fields."""
    charts: dict[str, str] = {}
    for field in df.columns:
        try:
            if field in {"date_received"}:
                continue  # handled as quarterly chart
            # Skip encounter_type_cat fields - they're handled by the hierarchical treemap
            if field in {"encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"}:
                continue
            if target_fields is not None and field not in target_fields:
                continue
            chart_html = _build_chart_for_field(df, field, theme)
            if chart_html:
                charts[field] = chart_html
        except Exception:
            continue
    return charts


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

    # Build individual field charts
    charts = _build_individual_field_charts(df, theme, target_fields)

    # Add hierarchical encounter type treemap
    wants_encounter_type = target_fields is None or "encounter_type" in target_fields
    has_columns = all(
        col in df.columns
        for col in ["encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"]
    )

    if wants_encounter_type and not df.empty and has_columns:
        encounter_type_chart = _build_hierarchical_encounter_treemap(df, theme)
        if encounter_type_chart:
            charts["encounter_type"] = encounter_type_chart

    # Add quarterly referral counts chart
    try:
        wants_quarterly = target_fields is None or "referrals_counts_quarterly" in target_fields
        if wants_quarterly:
            quarterly_chart = _build_quarterly_referrals_chart(df, theme)
            if quarterly_chart:
                charts["referrals_counts_quarterly"] = quarterly_chart
    except Exception:
        pass

    # Add quarterly encounters counts chart
    try:
        wants_encounters_quarterly = (
            target_fields is None or "encounters_counts_quarterly" in target_fields
        )
        if wants_encounters_quarterly:
            encounters_quarterly_chart = _build_quarterly_encounters_chart(theme)
            if encounters_quarterly_chart:
                charts["encounters_counts_quarterly"] = encounters_quarterly_chart
    except Exception:
        pass

    if target_fields is None:
        return charts

    return {field: charts[field] for field in target_fields if field in charts}
