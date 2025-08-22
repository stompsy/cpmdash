from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_all_cases_scatter(theme):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "patient_id",
            "od_date",
            "disposition",
            "patient_age",
            "patient_sex",
            "narcan_doses_prior_to_ems",
            "narcan_prior_to_ems_dosage",
            "jail_start_1",
            "jail_end_1",
            "jail_start_2",
            "jail_end_2",
        )
    )

    # If the dataframe is empty, return a message
    if df.empty:
        return "No overdose data available to display."

    # Classify outcome
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    # Remove placeholder fake date
    df = df[df["od_date"] != pd.Timestamp("2000-01-01")]
    # Ensure od_date is timezone-naive
    if df["od_date"].dt.tz is not None:
        df["od_date"] = df["od_date"].dt.tz_localize(None)

    fatal_conditions = ["CPR attempted", "DOA"]
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if str(x).strip().lower() in fatal_conditions else "Non-Fatal"
    )

    # Process jail times
    jail_df = df[["patient_id", "jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]].copy()

    # Convert all jail date columns to timezone-naive
    for col in ["jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]:
        jail_df[col] = pd.to_datetime(jail_df[col], errors="coerce")
        if jail_df[col].dt.tz is not None:
            jail_df[col] = jail_df[col].dt.tz_localize(None)

    jail_df.dropna(subset=["jail_start_1", "jail_start_2"], how="all", inplace=True)
    jail_df.drop_duplicates(inplace=True)

    jail_periods = []
    today = datetime.now()

    for _, row in jail_df.iterrows():
        patient_id = row["patient_id"]
        if pd.notna(row["jail_start_1"]):
            start = row["jail_start_1"]
            end = row["jail_end_1"] if pd.notna(row["jail_end_1"]) else today
            jail_periods.append({"patient_id": patient_id, "start": start, "end": end})

        if pd.notna(row["jail_start_2"]):
            start = row["jail_start_2"]
            end = row["jail_end_2"] if pd.notna(row["jail_end_2"]) else today
            jail_periods.append({"patient_id": patient_id, "start": start, "end": end})

    jail_periods_df = pd.DataFrame(jail_periods)

    # Map patient age and short sex
    age_map = df.groupby("patient_id", observed=False)["patient_age"].min().to_dict()
    sex_map = (
        df.groupby("patient_id", observed=False)["patient_sex"]
        .last()
        .map({"Male": "M", "Female": "F"})
        .to_dict()
    )
    df["merged_label"] = df["patient_id"].map(
        lambda pid: f"{age_map.get(pid, 'N/A')} {sex_map.get(pid, 'N/A')}"
    )

    # Count overdoses per patient
    od_count_map = df.groupby("patient_id", observed=False).size().to_dict()
    df["od_count"] = df["patient_id"].map(od_count_map)
    df["is_repeat"] = df["od_count"] > 1
    df["sort_key"] = df["patient_id"].map(age_map.get)

    # Sort: single ODs first (by age), then repeat ODs (by age)
    label_sort_df = df.drop_duplicates("merged_label").copy()
    label_sort_df = label_sort_df.sort_values(["is_repeat", "sort_key"])
    label_order = label_sort_df["merged_label"]

    df["merged_label"] = pd.Categorical(df["merged_label"], categories=label_order, ordered=True)

    # Sort and calculate time difference
    df.sort_values(by=["merged_label", "od_date"], inplace=True)
    df["days_since_last_od"] = df.groupby("merged_label", observed=False)["od_date"].diff().dt.days
    df["days_since_last_od"] = df["days_since_last_od"].fillna("First OD")

    # Assign darker color palette
    color_palette = px.colors.qualitative.Pastel
    unique_labels = df["merged_label"].cat.categories
    color_map = {
        label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)
    }

    # Build multiple chart visualizations
    charts_html = ""

    # Get chart Y-range (merged_label categories)
    y_categories = list(df["merged_label"].cat.categories)

    # 1. ORIGINAL SCATTER PLOT
    charts_html += "<h2 style='margin: 30px 0 15px 0; color: #333; font-size: 24px;'>1. Original Scatter Plot Timeline</h2>"
    fig1 = create_scatter_timeline(df, jail_periods_df, y_categories, color_map, age_map, sex_map)
    charts_html += plot(fig1, output_type="div", config=fig1._config)

    # 2. GANTT CHART
    charts_html += "<h2 style='margin: 30px 0 15px 0; color: #333; font-size: 24px;'>2. Gantt Chart Timeline</h2>"
    fig2 = create_gantt_chart(df, jail_periods_df, y_categories, color_map)
    charts_html += plot(fig2, output_type="div", config=fig2._config)

    # 3. SWIMLANE CHART WITH SUBPLOTS
    charts_html += "<h2 style='margin: 30px 0 15px 0; color: #333; font-size: 24px;'>3. Swimlane Chart (Single vs Repeat Cases)</h2>"
    fig3 = create_swimlane_chart(df, jail_periods_df, color_map, age_map, sex_map)
    charts_html += plot(fig3, output_type="div", config=fig3._config)

    # 4. HEATMAP CALENDAR VIEW
    charts_html += "<h2 style='margin: 30px 0 15px 0; color: #333; font-size: 24px;'>4. Calendar Heatmap View</h2>"
    fig4 = create_calendar_heatmap(df, y_categories)
    charts_html += plot(fig4, output_type="div", config=fig4._config)

    return charts_html


def create_scatter_timeline(df, jail_periods_df, y_categories, color_map, age_map, sex_map):
    """Original scatter plot timeline"""
    fig = go.Figure()

    # Add jail time traces first to be in the background
    if not jail_periods_df.empty:
        jail_periods_df["merged_label"] = jail_periods_df["patient_id"].map(
            lambda pid: f"{age_map.get(pid, 'N/A')} {sex_map.get(pid, 'N/A')}"
        )
        for _, row in jail_periods_df.iterrows():
            if row["merged_label"] in y_categories:
                fig.add_trace(
                    go.Scatter(
                        x=[row["start"], row["end"]],
                        y=[row["merged_label"], row["merged_label"]],
                        mode="lines",
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=6),
                        hoverinfo="text",
                        text=f"Incarcerated<br>Start: {row['start'].strftime('%Y-%m-%d')}<br>End: {row['end'].strftime('%Y-%m-%d')}",
                        showlegend=False,
                    )
                )

    # Find range of years
    years = sorted(df["od_date"].dt.year.dropna().unique())

    # Add transparent vertical bands for each year
    year_colors = {
        2021: "rgba(100, 149, 237, 0.07)",
        2022: "rgba(34, 139, 34, 0.07)",
        2023: "rgba(255, 165, 0, 0.07)",
        2024: "rgba(220, 20, 60, 0.03)",
    }

    for year in years:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=f"{year}-01-01",
            x1=f"{year}-12-31",
            y0=0,
            y1=1,
            fillcolor=year_colors.get(year, "rgba(0,0,0,0.05)"),
            line=dict(width=0),
            layer="below",
        )

    # Add scatter plot for ODs
    for label in df["merged_label"].cat.categories:
        df_label = df[df["merged_label"] == label]
        fig.add_trace(
            go.Scatter(
                x=df_label["od_date"],
                y=df_label["merged_label"],
                mode="markers+lines",
                marker=dict(
                    color=color_map[label],
                    size=10,
                    symbol=df_label["overdose_outcome"].map({"Fatal": "x", "Non-Fatal": "circle"}),
                ),
                line=dict(color=color_map[label], width=1),
                name=label,
                hoverlabel=dict(namelength=-1),
                hovertemplate="<b>Patient: %{y}</b><br>"
                + "OD Date: %{x|%Y-%m-%d}<br>"
                + "Outcome: %{customdata[0]}<br>"
                + "Days Since Last OD: %{customdata[1]}<br>"
                + "<extra></extra>",
                customdata=df_label[["overdose_outcome", "days_since_last_od"]],
            )
        )

    # Style the layout
    fig = style_plotly_layout(
        fig,
        scroll_zoom=False,
        x_title="Date",
        y_title="Patient (Age Sex)",
        margin=dict(t=0, l=50, r=20, b=20),
        height=600,
    )

    return fig


def create_gantt_chart(df, jail_periods_df, y_categories, color_map):
    """Gantt chart showing patient timelines"""
    fig = go.Figure()

    # Get date range for each patient
    patient_ranges = (
        df.groupby("merged_label", observed=False)["od_date"].agg(["min", "max"]).reset_index()
    )

    # Create base timeline bars for each patient
    for _idx, row in patient_ranges.iterrows():
        label = row["merged_label"]
        start_date = row["min"]
        end_date = row["max"]

        # If single OD, extend end date slightly for visibility
        if start_date == end_date:
            end_date = start_date + timedelta(days=30)

        fig.add_trace(
            go.Bar(
                x=[end_date - start_date],
                y=[label],
                base=[start_date],
                orientation="h",
                marker=dict(color="rgba(200, 200, 200, 0.3)", line=dict(width=0)),
                name="Timeline",
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}<extra></extra>",
            )
        )

    # Add jail periods as red bars
    if not jail_periods_df.empty:
        for _, row in jail_periods_df.iterrows():
            if row["merged_label"] in y_categories:
                duration = row["end"] - row["start"]
                fig.add_trace(
                    go.Bar(
                        x=[duration],
                        y=[row["merged_label"]],
                        base=[row["start"]],
                        orientation="h",
                        marker=dict(color="rgba(255, 0, 0, 0.7)", line=dict(width=0)),
                        name="Incarcerated",
                        showlegend=False,
                        hovertemplate=f"<b>Incarcerated</b><br>Start: {row['start'].strftime('%Y-%m-%d')}<br>End: {row['end'].strftime('%Y-%m-%d')}<extra></extra>",
                    )
                )

    # Add overdose events as markers
    for label in df["merged_label"].cat.categories:
        df_label = df[df["merged_label"] == label]
        for _, od_row in df_label.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[od_row["od_date"]],
                    y=[od_row["merged_label"]],
                    mode="markers",
                    marker=dict(
                        color="red" if od_row["overdose_outcome"] == "Fatal" else "blue",
                        size=12,
                        symbol="x" if od_row["overdose_outcome"] == "Fatal" else "circle",
                        line=dict(width=2, color="white"),
                    ),
                    name=od_row["overdose_outcome"],
                    showlegend=False,
                    hovertemplate=f"<b>Overdose</b><br>Date: {od_row['od_date'].strftime('%Y-%m-%d')}<br>Outcome: {od_row['overdose_outcome']}<extra></extra>",
                )
            )

    fig = style_plotly_layout(
        fig,
        scroll_zoom=False,
        x_title="Date",
        y_title="Patient (Age Sex)",
        margin=dict(t=0, l=50, r=20, b=20),
        height=600,
    )

    return fig


def create_swimlane_chart(df, jail_periods_df, color_map, age_map, sex_map):
    """Swimlane chart with separate subplots for single vs repeat cases"""

    # Separate single vs repeat cases
    single_cases = df[~df["is_repeat"]]["merged_label"].cat.categories
    repeat_cases = df[df["is_repeat"]]["merged_label"].cat.categories

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Single Overdose Cases", "Repeat Overdose Cases"),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.7],
    )

    # Single cases subplot
    for label in single_cases:
        df_label = df[df["merged_label"] == label]
        fig.add_trace(
            go.Scatter(
                x=df_label["od_date"],
                y=df_label["merged_label"],
                mode="markers",
                marker=dict(
                    color="blue",
                    size=10,
                    symbol=df_label["overdose_outcome"].map({"Fatal": "x", "Non-Fatal": "circle"}),
                ),
                name=label,
                showlegend=False,
                hovertemplate="<b>Patient: %{y}</b><br>OD Date: %{x|%Y-%m-%d}<br>Outcome: %{customdata[0]}<extra></extra>",
                customdata=df_label[["overdose_outcome"]],
            ),
            row=1,
            col=1,
        )

    # Repeat cases subplot
    for label in repeat_cases:
        df_label = df[df["merged_label"] == label]
        fig.add_trace(
            go.Scatter(
                x=df_label["od_date"],
                y=df_label["merged_label"],
                mode="markers+lines",
                marker=dict(
                    color=color_map.get(label, "red"),
                    size=10,
                    symbol=df_label["overdose_outcome"].map({"Fatal": "x", "Non-Fatal": "circle"}),
                ),
                line=dict(color=color_map.get(label, "red"), width=1),
                name=label,
                showlegend=False,
                hovertemplate="<b>Patient: %{y}</b><br>OD Date: %{x|%Y-%m-%d}<br>Outcome: %{customdata[0]}<extra></extra>",
                customdata=df_label[["overdose_outcome"]],
            ),
            row=2,
            col=1,
        )

    # Add jail periods to repeat cases
    if not jail_periods_df.empty:
        jail_periods_df["merged_label"] = jail_periods_df["patient_id"].map(
            lambda pid: f"{age_map.get(pid, 'N/A')} {sex_map.get(pid, 'N/A')}"
        )
        for _, row in jail_periods_df.iterrows():
            if row["merged_label"] in repeat_cases:
                fig.add_trace(
                    go.Scatter(
                        x=[row["start"], row["end"]],
                        y=[row["merged_label"], row["merged_label"]],
                        mode="lines",
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=6),
                        showlegend=False,
                        hovertemplate=f"Incarcerated<br>Start: {row['start'].strftime('%Y-%m-%d')}<br>End: {row['end'].strftime('%Y-%m-%d')}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

    fig = style_plotly_layout(
        fig, scroll_zoom=False, show_legend=False, margin=dict(t=60, l=50, r=20, b=20), height=800
    )

    return fig


def create_calendar_heatmap(df, y_categories):
    """Calendar heatmap showing overdose frequency"""

    # Create a pivot table for heatmap
    df_copy = df.copy()
    df_copy["year_month"] = df_copy["od_date"].dt.to_period("M").astype(str)

    # Count overdoses per patient per month
    heatmap_data = (
        df_copy.groupby(["merged_label", "year_month"], observed=True)
        .size()
        .reset_index(name="od_count")
    )

    # Pivot for heatmap
    heatmap_pivot = heatmap_data.pivot(
        index="merged_label", columns="year_month", values="od_count"
    ).fillna(0)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale="Reds",
            hovertemplate="<b>Patient: %{y}</b><br>Month: %{x}<br>Overdoses: %{z}<extra></extra>",
            colorbar=dict(title="Overdoses per Month"),
        )
    )

    fig = style_plotly_layout(
        fig,
        scroll_zoom=False,
        x_title="Month",
        y_title="Patient (Age Sex)",
        margin=dict(t=0, l=50, r=20, b=20),
        height=600,
    )

    return fig
