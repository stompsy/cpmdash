from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_repeats_scatter(theme):  # noqa: C901
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
    df = df[df["od_date"] != pd.Timestamp("2000-01-01")]
    fatal_conditions = ["CPR attempted", "DOA"]
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if str(x).strip().lower() in fatal_conditions else "Non-Fatal"
    )

    # Convert jail date columns to datetime and filter out '2000-01-01' in main df
    for col in ["jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df.loc[df[col] == pd.Timestamp("2000-01-01"), col] = pd.NaT

    # Filter for repeat patients
    repeat_ids = df["patient_id"].value_counts()
    repeat_ids = repeat_ids[repeat_ids > 1].index
    df = df[df["patient_id"].isin(repeat_ids)].copy()

    # Process jail times
    jail_df = df[df["patient_id"].isin(repeat_ids)][
        ["patient_id", "jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]
    ].copy()
    jail_df.dropna(subset=["jail_start_1", "jail_start_2"], how="all", inplace=True)
    jail_df.drop_duplicates(inplace=True)

    # Ensure jail_df columns are datetime and filter out '2000-01-01' as pd.NaT
    for col in ["jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]:
        jail_df[col] = pd.to_datetime(jail_df[col], errors="coerce")
        jail_df.loc[jail_df[col] == pd.Timestamp("2000-01-01"), col] = pd.NaT

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
    age_map = df.groupby("patient_id", observed=False)["patient_age"].min()
    sex_map = (
        df.groupby("patient_id", observed=False)["patient_sex"]
        .last()
        .map({"Male": "M", "Female": "F"})
    )
    df["merged_label"] = df["patient_id"].map(lambda pid: f"{age_map[pid]} {sex_map[pid]}")
    df["sort_key"] = df["patient_id"].map(age_map)

    # Sort merged_label by age only
    label_order = df.drop_duplicates("merged_label").sort_values("sort_key")["merged_label"]
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

    # Build the line chart
    fig = go.Figure()

    # Add jail time traces first to be in the background
    if not jail_periods_df.empty:
        jail_periods_df["merged_label"] = jail_periods_df["patient_id"].map(
            lambda pid: f"{age_map.get(pid, 'N/A')} {sex_map.get(pid, 'N/A')}"
        )
        for _, row in jail_periods_df.iterrows():
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
    )

    return plot(fig, output_type="div", config=fig._config)
