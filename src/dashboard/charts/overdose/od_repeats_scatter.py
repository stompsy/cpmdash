import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


def build_chart_repeats_scatter(theme):

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
        )
    )

    # Classify outcome
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    fatal_conditions = ["CPR attempted", "DOA"]
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if str(x).strip().lower() in fatal_conditions else "Non-Fatal"
    )

    # Filter for repeat patients
    repeat_ids = df["patient_id"].value_counts()
    repeat_ids = repeat_ids[repeat_ids > 1].index
    df = df[df["patient_id"].isin(repeat_ids)].copy()

    # Map patient age and short sex
    age_map = df.groupby("patient_id", observed=False)["patient_age"].min()
    sex_map = (
        df.groupby("patient_id", observed=False)["patient_sex"]
        .last()
        .map({"Male": "M", "Female": "F"})
    )
    df["merged_label"] = df["patient_id"].map(
        lambda pid: f"{age_map[pid]} {sex_map[pid]}"
    )
    df["sort_key"] = df["patient_id"].map(age_map)

    # Sort merged_label by age only
    label_order = df.drop_duplicates("merged_label").sort_values("sort_key")[
        "merged_label"
    ]
    df["merged_label"] = pd.Categorical(
        df["merged_label"], categories=label_order, ordered=True
    )

    # Sort and calculate time difference
    df.sort_values(by=["merged_label", "od_date"], inplace=True)
    df["days_since_last_od"] = (
        df.groupby("merged_label", observed=False)["od_date"]
        .diff()
        .dt.days
    )
    df["days_since_last_od"] = df["days_since_last_od"].fillna("First OD")

    # Assign darker color palette
    color_palette = px.colors.qualitative.Pastel
    unique_labels = df["merged_label"].cat.categories
    color_map = {
        label: color_palette[i % len(color_palette)]
        for i, label in enumerate(unique_labels)
    }

    # Build the line chart
    fig = go.Figure()
    for label in unique_labels:
        patient_df = df[df["merged_label"] == label]
        fig.add_trace(
            go.Scatter(
                x=patient_df["od_date"],
                y=[label] * len(patient_df),
                mode="lines+markers",
                name=label,
                line=dict(color=color_map[label], width=2),
                marker=dict(
                    color=[
                        "red" if o == "Fatal" else color_map[label]
                        for o in patient_df["overdose_outcome"]
                    ],
                    symbol=[
                        "diamond" if o == "Fatal" else "circle"
                        for o in patient_df["overdose_outcome"]
                    ],
                    size=10,
                ),
                customdata=patient_df[
                    [
                        "merged_label",
                        "od_date",
                        "days_since_last_od",
                        "narcan_doses_prior_to_ems",
                        "narcan_prior_to_ems_dosage",
                    ]
                ],
                hovertemplate=(
                    "Patient: %{customdata[0]}<br>"
                    "OD Date: %{customdata[1]|%b %d, %Y}<br>"
                    "Days Since Last OD: %{customdata[2]}<br>"
                    "Narcan Doses: %{customdata[3]}<br>"
                    "Total Narcan: %{customdata[4]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        
    # Add a "Narcan Doses" trace with ▲ markers, sized by dose count
    for label in unique_labels:
        narcan_df = df[
            (df["merged_label"] == label)
            & (df["narcan_doses_prior_to_ems"].fillna(0) > 0)
        ]
        if not narcan_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=narcan_df["od_date"],
                    y=[label] * len(narcan_df),
                    mode="markers",
                    name="Narcan Doses",
                    marker=dict(
                        symbol="triangle-up",
                        size=narcan_df["narcan_doses_prior_to_ems"] * 4,  # scale as you like
                        color="royalblue",
                    ),
                    # <-- send the actual dose count into customdata
                    customdata=narcan_df[
                        ["narcan_doses_prior_to_ems", "narcan_prior_to_ems_dosage"]
                    ].values,
                    hovertemplate=(
                        "Patient: %{text}<br>"
                        "Date: %{x|%b %d, %Y}<br>"
                        "Narcan Doses: %{customdata[0]}<br>"
                        "Total Narcan: %{customdata[1]}mg<extra></extra>"
                    ),
                    text=[label] * len(narcan_df),
                    showlegend=(label == unique_labels[0]),
                )
            )
    
    # Update x-axis: show every month, abbreviated year
    fig.update_xaxes(
        tickformat="%b ’%y", dtick="M1", ticklabelmode="period"  # e.g., Jan ’24
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_repeats_scatter",
        scroll_zoom=False,
        margin=dict(t=0, l=75, r=20, b=65),
    )
    return plot(fig, output_type="div", config=fig._config)
