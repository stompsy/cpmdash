from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_repeats_scatter(theme):
    odreferrals_qs = ODReferrals.objects.values(
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
    df = pd.DataFrame.from_records(list(odreferrals_qs))

    # If the dataframe is empty, return a message
    if df.empty:
        return "No overdose data available to display."

    # Classify outcome
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")

    # Filter out sentinel date (2000-01-01) and null dates
    df = df[(df["od_date"] != pd.Timestamp("2000-01-01")) & (df["od_date"].notna())].copy()

    fatal_conditions = ["cpr attempted", "doa", "fatal", "death", "deceased", "died"]
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if str(x).strip().lower() in fatal_conditions else "Non-Fatal"
    )

    # Convert jail date columns to datetime and filter out '2000-01-01' in main df
    for col in ["jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df.loc[df[col] == pd.Timestamp("2000-01-01"), col] = pd.NaT

    # Filter for repeat patients (count ALL overdoses per patient_id, not just after filtering)
    repeat_ids = df["patient_id"].value_counts()
    repeat_ids = repeat_ids[repeat_ids > 1].index
    df = df[df["patient_id"].isin(repeat_ids)].copy()

    # JAIL PROCESSING DISABLED — insufficient data to plot reliably
    # jail_df = df[df["patient_id"].isin(repeat_ids)][
    #     ["patient_id", "jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]
    # ].copy()
    # jail_df.dropna(subset=["jail_start_1", "jail_start_2"], how="all", inplace=True)
    # jail_df.drop_duplicates(inplace=True)
    #
    # # Ensure jail_df columns are datetime and filter out sentinel dates
    # for col in ["jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]:
    #     jail_df[col] = pd.to_datetime(jail_df[col], errors="coerce")
    #     jail_df.loc[jail_df[col] == pd.Timestamp("2000-01-01"), col] = pd.NaT
    #
    # jail_periods = []
    # today = datetime.now()
    #
    # for _, row in jail_df.iterrows():
    #     patient_id = row["patient_id"]
    #     if pd.notna(row["jail_start_1"]):
    #         start = row["jail_start_1"]
    #         end = row["jail_end_1"] if pd.notna(row["jail_end_1"]) else today
    #         jail_periods.append({"patient_id": patient_id, "start": start, "end": end})
    #     if pd.notna(row["jail_start_2"]):
    #         start = row["jail_start_2"]
    #         end = row["jail_end_2"] if pd.notna(row["jail_end_2"]) else today
    #         jail_periods.append({"patient_id": patient_id, "start": start, "end": end})
    #
    # jail_periods_df = pd.DataFrame(jail_periods)

    # Map patient age and short sex, including patient_id to ensure uniqueness
    age_map = df.groupby("patient_id", observed=False)["patient_age"].min()
    sex_map = (
        df.groupby("patient_id", observed=False)["patient_sex"]
        .last()
        .map({"Male": "M", "Female": "F"})
    )
    # Include patient_id in the label to prevent different patients from being merged
    df["merged_label"] = df["patient_id"].map(
        lambda pid: f"ID {pid}: {age_map[pid]} {sex_map[pid]}"
    )
    df["sort_key"] = df["patient_id"].map(age_map)

    # Sort merged_label by age, then by patient_id for consistent ordering
    label_order = df.drop_duplicates("merged_label").sort_values(["sort_key", "patient_id"])[
        "merged_label"
    ]
    df["merged_label"] = pd.Categorical(df["merged_label"], categories=label_order, ordered=True)

    # Sort and calculate time difference (now properly grouped by unique patient_id)
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

    # Jail time traces disabled
    # if not jail_periods_df.empty:
    #     jail_periods_df["merged_label"] = jail_periods_df["patient_id"].map(
    #         lambda pid: f"{age_map.get(pid, 'N/A')} {sex_map.get(pid, 'N/A')}"
    #     )
    #     for _, row in jail_periods_df.iterrows():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=[row["start"], row["end"]],
    #                 y=[row["merged_label"], row["merged_label"]],
    #                 mode="lines",
    #                 line=dict(color="rgba(255, 0, 0, 0.5)", width=6),
    #                 hoverinfo="text",
    #                 text=f"Incarcerated<br>Start: {row['start'].strftime('%Y-%m-%d')}<br>End: {row['end'].strftime('%Y-%m-%d')}",
    #                 showlegend=False,
    #             )
    #         )

    # Add scatter plot for ODs
    for label in df["merged_label"].cat.categories:
        df_label = df[df["merged_label"] == label].sort_values("od_date")

        # Create marker colors: red for fatal, patient color for non-fatal
        marker_colors = [
            "red" if outcome == "Fatal" else color_map[label]
            for outcome in df_label["overdose_outcome"]
        ]

        # Create marker line colors: dark red for fatal, match marker for non-fatal
        marker_line_colors = [
            "darkred" if outcome == "Fatal" else color_map[label]
            for outcome in df_label["overdose_outcome"]
        ]

        # Create hover text that shows FATAL in red for fatal overdoses
        hover_text = []
        for _, row in df_label.iterrows():
            # Format days as int if it's a number, otherwise keep as-is (e.g., "First OD")
            days_display = (
                int(row["days_since_last_od"])
                if isinstance(row["days_since_last_od"], int | float)
                else row["days_since_last_od"]
            )

            if row["overdose_outcome"] == "Fatal":
                hover_text.append(
                    f"<b>Patient: {row['merged_label']}</b><br>"
                    f"OD Date: {row['od_date'].strftime('%Y-%m-%d')}<br>"
                    f"<b style='color:red'>Outcome: FATAL</b><br>"
                    f"Days Since Last OD: {days_display}"
                )
            else:
                hover_text.append(
                    f"<b>Patient: {row['merged_label']}</b><br>"
                    f"OD Date: {row['od_date'].strftime('%Y-%m-%d')}<br>"
                    f"Outcome: {row['overdose_outcome']}<br>"
                    f"Days Since Last OD: {days_display}"
                )

        # Add all overdoses for this patient with connecting lines
        fig.add_trace(
            go.Scatter(
                x=df_label["od_date"],
                y=df_label["merged_label"],
                mode="markers+lines",
                marker=dict(
                    color=marker_colors,
                    size=10,
                    symbol="circle",
                    line=dict(color=marker_line_colors, width=1.5),
                ),
                line=dict(color=color_map[label], width=1),
                name=label,
                hoverlabel=dict(namelength=-1),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                showlegend=True,
            )
        )

    # Style the layout
    fig = style_plotly_layout(
        fig,
        scroll_zoom=False,
        x_title="Date",
        y_title="Patient (Age Sex)",
        margin=dict(t=40, l=50, r=20, b=20),  # Added top margin for modebar clearance
    )

    # Customize grid lines and axis spacing (AFTER style_plotly_layout to override defaults)
    fig.update_xaxes(
        showgrid=False,  # Hide vertical grid lines
        dtick="M1",  # Show tick every month
        ticklabelmode="period",  # Center labels on the period (month)
        range=[pd.Timestamp("2024-03-01"), datetime.now()],  # March 2024 to current date
    )
    fig.update_yaxes(
        showgrid=True,  # Show horizontal grid lines
        gridcolor="rgba(128,128,128,0.25)",  # Darker gray grid lines matching Patients page
        ticklabelstandoff=10,  # ~1/8 inch gap between y-axis labels and chart
    )

    return plot(fig, output_type="div", config=fig._config)
