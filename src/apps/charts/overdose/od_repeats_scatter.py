from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def _alpha_suffix(index: int) -> str:
    """Convert 1-based index to A, B, ..., Z, AA, AB, ..."""
    if index <= 0:
        return "A"
    n = index
    chars: list[str] = []
    while n > 0:
        n -= 1
        chars.append(chr(ord("A") + (n % 26)))
        n //= 26
    return "".join(reversed(chars))


def _build_display_label_map(unique_labels: list[str]) -> dict[str, str]:
    """Map internal unique labels ("ID 123: 44 M") to user-facing axis labels.

    Shows only age/sex, with an alphabetic disambiguator when collisions occur.
    """
    base_by_label = {label: label.split(": ", 1)[-1] for label in unique_labels}

    base_counts: dict[str, int] = {}
    for base in base_by_label.values():
        base_counts[base] = base_counts.get(base, 0) + 1

    display_map: dict[str, str] = {}
    running_index: dict[str, int] = {}
    for label in unique_labels:
        base = base_by_label[label]
        if base_counts.get(base, 0) <= 1:
            display_map[label] = base
            continue
        idx = running_index.get(base, 0) + 1
        running_index[base] = idx
        display_map[label] = f"{base} ({_alpha_suffix(idx)})"
    return display_map


def _month_to_season(month: int) -> str:
    # Meteorological seasons.
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Fall"


def _load_repeat_overdose_df() -> pd.DataFrame:
    """Load and normalize overdose referral records for repeat-patient analysis."""
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
    if df.empty:
        return df

    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df[(df["od_date"] != pd.Timestamp("2000-01-01")) & (df["od_date"].notna())].copy()
    if df.empty:
        return df

    fatal_conditions = {"cpr attempted", "doa", "fatal", "death", "deceased", "died"}
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if str(x).strip().lower() in fatal_conditions else "Non-Fatal"
    )

    # Convert jail date columns to datetime and filter out sentinel dates.
    for col in ["jail_start_1", "jail_end_1", "jail_start_2", "jail_end_2"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df.loc[df[col] == pd.Timestamp("2000-01-01"), col] = pd.NaT

    # Repeat patients only.
    repeat_ids = df["patient_id"].value_counts()
    repeat_ids = repeat_ids[repeat_ids > 1].index
    df = df[df["patient_id"].isin(repeat_ids)].copy()
    if df.empty:
        return df

    age_map = df.groupby("patient_id", observed=False)["patient_age"].min()
    sex_map = (
        df.groupby("patient_id", observed=False)["patient_sex"]
        .last()
        .map({"Male": "M", "Female": "F"})
    )

    df["merged_label"] = df["patient_id"].map(
        lambda pid: f"ID {pid}: {age_map[pid]} {sex_map[pid]}"
    )
    df["sort_key"] = df["patient_id"].map(age_map)
    label_order = df.drop_duplicates("merged_label").sort_values(["sort_key", "patient_id"])[
        "merged_label"
    ]
    df["merged_label"] = pd.Categorical(df["merged_label"], categories=label_order, ordered=True)

    # Sort and calculate time differences.
    df.sort_values(by=["merged_label", "od_date"], inplace=True)
    df["days_since_last_od"] = df.groupby("merged_label", observed=False)["od_date"].diff().dt.days
    df["days_since_last_od"] = df["days_since_last_od"].fillna("First OD")

    first_dates = df.groupby("merged_label", observed=False)["od_date"].transform("min")
    df["days_since_first_od"] = (df["od_date"] - first_dates).dt.days
    df["season"] = df["od_date"].dt.month.map(_month_to_season)
    return df


def build_repeat_overdose_quick_stats() -> list[dict[str, str]]:
    """Quick stats for the repeat overdose section."""
    df = _load_repeat_overdose_df()
    if df.empty:
        return []

    patients = int(df["merged_label"].nunique())
    events = int(df.shape[0])
    repeat_events = max(0, events - patients)

    dt = df[["merged_label", "od_date"]].sort_values(["merged_label", "od_date"]).copy()
    # NOTE: don't subtract `groupby().nth()` results directly.
    # `nth()` preserves the original row index, so subtraction aligns on row index (not patient),
    # producing all-NaN and empty stats.
    dt["event_rank"] = dt.groupby("merged_label", observed=False)["od_date"].rank(method="first")
    first_dates = dt[dt["event_rank"] == 1].set_index("merged_label")["od_date"]
    second_dates = dt[dt["event_rank"] == 2].set_index("merged_label")["od_date"]
    days_to_second = (second_dates - first_dates).dt.days.dropna()

    if days_to_second.empty:
        median_days_display = "—"
        repeat_window_display = "—"
    else:
        # Median can legitimately be 0 if repeats occur the same day.
        median_raw = days_to_second.median()
        try:
            median_days_display = str(int(round(float(median_raw))))
        except Exception:
            median_days_display = "—"

        try:
            pct_30 = round(float(days_to_second.le(30).mean() * 100.0), 1)
            pct_90 = round(float(days_to_second.le(90).mean() * 100.0), 1)
            repeat_window_display = f"{pct_30:.1f}% / {pct_90:.1f}%"
        except Exception:
            repeat_window_display = "—"

    df_season = df.copy()
    df_season["rank"] = df_season.groupby("merged_label", observed=False)["od_date"].rank(
        method="first"
    )
    repeats_only = df_season[df_season["rank"] >= 2]
    top_season = "—"
    if not repeats_only.empty:
        top_season = str(repeats_only["season"].value_counts().idxmax())

    return [
        {"label": "Repeat patients", "value": f"{patients:,}"},
        {"label": "Repeat events", "value": f"{repeat_events:,}"},
        {"label": "Median days to 2nd OD", "value": median_days_display},
        {"label": "Repeat within 30/90 days", "value": repeat_window_display},
        {"label": "Top repeat season", "value": top_season},
    ]


def build_chart_repeats_aligned_timeline(theme: str) -> str:
    """Aligned timeline where each patient starts at day 0 (their first recorded OD)."""
    df = _load_repeat_overdose_df()
    if df.empty:
        return "No overdose data available to display."

    unique_labels = list(df["merged_label"].cat.categories)
    display_label_map = _build_display_label_map(unique_labels)

    color_palette = px.colors.qualitative.Pastel
    color_map = {
        label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)
    }

    fig = go.Figure()
    for label in unique_labels:
        df_label = df[df["merged_label"] == label].sort_values("days_since_first_od")

        marker_colors = [
            "red" if outcome == "Fatal" else color_map[label]
            for outcome in df_label["overdose_outcome"]
        ]
        marker_line_colors = [
            "darkred" if outcome == "Fatal" else color_map[label]
            for outcome in df_label["overdose_outcome"]
        ]

        hover_text: list[str] = []
        for _, row in df_label.iterrows():
            days_display = (
                int(row["days_since_last_od"])
                if isinstance(row["days_since_last_od"], int | float)
                else row["days_since_last_od"]
            )
            display_patient = display_label_map.get(row["merged_label"], str(row["merged_label"]))
            day_zero = (
                int(row["days_since_first_od"]) if pd.notna(row["days_since_first_od"]) else 0
            )
            if row["overdose_outcome"] == "Fatal":
                hover_text.append(
                    f"<b>Patient: {display_patient}</b><br>"
                    f"Day: {day_zero}<br>"
                    f"OD Date: {row['od_date'].strftime('%Y-%m-%d')}<br>"
                    f"<b style='color:red'>Outcome: FATAL</b><br>"
                    f"Days Since Last OD: {days_display}"
                )
            else:
                hover_text.append(
                    f"<b>Patient: {display_patient}</b><br>"
                    f"Day: {day_zero}<br>"
                    f"OD Date: {row['od_date'].strftime('%Y-%m-%d')}<br>"
                    f"Outcome: {row['overdose_outcome']}<br>"
                    f"Days Since Last OD: {days_display}"
                )

        fig.add_trace(
            go.Scatter(
                x=df_label["days_since_first_od"],
                y=df_label["merged_label"],
                mode="markers+lines",
                marker=dict(
                    color=marker_colors,
                    size=9,
                    symbol="circle",
                    line=dict(color=marker_line_colors, width=1.2),
                ),
                line=dict(color=color_map[label], width=0.8),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                showlegend=False,
            )
        )

    patient_count = len(unique_labels)
    height = min(1200, max(520, 220 + patient_count * 18))

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title="Days since first recorded overdose (aligned)",
        y_title="Patient (Age Sex)",
        height=height,
        margin=dict(t=30, l=50, r=20, b=40),
        hovermode_unified=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", rangemode="tozero")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.25)",
        ticklabelstandoff=10,
        categoryorder="array",
        categoryarray=list(unique_labels),
        tickmode="array",
        tickvals=list(unique_labels),
        ticktext=[display_label_map.get(label, str(label)) for label in unique_labels],
        automargin=True,
        tickfont=dict(size=11),
    )
    return plot(fig, output_type="div", config=fig._config)


def build_chart_repeat_interval_hist(theme: str) -> str:
    """Histogram of time between overdoses for the repeat cohort (excludes first OD)."""
    df = _load_repeat_overdose_df()
    if df.empty:
        return "No overdose data available to display."

    intervals = df[df["days_since_last_od"] != "First OD"]["days_since_last_od"]
    intervals_num = pd.to_numeric(intervals, errors="coerce").dropna()
    if intervals_num.empty:
        return "<p>No repeat intervals available.</p>"

    fig = px.histogram(
        x=intervals_num,
        nbins=30,
        labels={"x": "Days between overdoses"},
        template=None,
    )
    fig.update_traces(hovertemplate="Days between: %{x}<br>Count: %{y}<extra></extra>")
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=320,
        scroll_zoom=False,
        x_title="Days between overdoses",
        y_title="Count",
        margin=dict(t=20, l=50, r=10, b=40),
        hovermode_unified=False,
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    return plot(fig, output_type="div", config=fig._config)


def build_chart_repeat_seasonality(theme: str) -> str:
    """Seasonal distribution of repeat overdoses (events after the first)."""
    df = _load_repeat_overdose_df()
    if df.empty:
        return "No overdose data available to display."

    tmp = df.copy()
    tmp["rank"] = tmp.groupby("merged_label", observed=False)["od_date"].rank(method="first")
    tmp = tmp[tmp["rank"] >= 2]
    if tmp.empty:
        return "<p>No repeat overdoses available.</p>"

    order = ["Winter", "Spring", "Summer", "Fall"]
    season_counts = (
        tmp.groupby(["season", "overdose_outcome"], observed=False).size().reset_index(name="count")
    )
    season_counts["season"] = pd.Categorical(
        season_counts["season"], categories=order, ordered=True
    )
    season_counts = season_counts.sort_values("season")

    fig = px.bar(
        season_counts,
        x="season",
        y="count",
        color="overdose_outcome",
        barmode="stack",
        category_orders={"season": order},
        color_discrete_map={"Fatal": "#ef4444", "Non-Fatal": "#6366f1"},
        template=None,
    )
    fig.update_traces(hovertemplate="Season: %{x}<br>Count: %{y}<extra></extra>")
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=300,
        scroll_zoom=False,
        x_title=None,
        y_title="Repeat overdoses",
        margin=dict(t=20, l=50, r=10, b=40),
        hovermode_unified=False,
    )
    return plot(fig, output_type="div", config=fig._config)


def build_chart_repeats_scatter(theme: str) -> str:
    df = _load_repeat_overdose_df()

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

    unique_labels = list(df["merged_label"].cat.categories)
    display_label_map = _build_display_label_map(unique_labels)

    # Assign darker color palette and preserve explicit category order for y-axis
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

            display_patient = display_label_map.get(row["merged_label"], str(row["merged_label"]))

            if row["overdose_outcome"] == "Fatal":
                hover_text.append(
                    f"<b>Patient: {display_patient}</b><br>"
                    f"OD Date: {row['od_date'].strftime('%Y-%m-%d')}<br>"
                    f"<b style='color:red'>Outcome: FATAL</b><br>"
                    f"Days Since Last OD: {days_display}"
                )
            else:
                hover_text.append(
                    f"<b>Patient: {display_patient}</b><br>"
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
                    size=9,
                    symbol="circle",
                    line=dict(color=marker_line_colors, width=1.2),
                ),
                line=dict(color=color_map[label], width=0.8),
                name=label,
                hoverlabel=dict(namelength=-1),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                showlegend=False,
            )
        )

    # Style the layout
    patient_count = len(unique_labels)
    # If you want every y label, you have to give the chart enough vertical real estate.
    # Clamp so the page doesn't become a 3-mile scroll if patient_count is huge.
    # Slightly tighter rows while keeping labels readable.
    height = min(1400, max(620, 240 + patient_count * 20))

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title="Date",
        y_title="Patient (Age Sex)",
        height=height,
        margin=dict(t=40, l=50, r=20, b=20),  # Added top margin for modebar clearance
        hovermode_unified=False,
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
        categoryorder="array",  # Respect our explicit ordering
        categoryarray=list(unique_labels),  # Ensure every patient row is rendered
        tickmode="array",
        tickvals=list(unique_labels),
        ticktext=[display_label_map.get(label, str(label)) for label in unique_labels],
        automargin=True,
        tickfont=dict(size=11),
    )

    return plot(fig, output_type="div", config=fig._config)
