import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


def build_chart_od_density_heatmap(theme):

    # Dataframe
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "od_date",
        )
    )

    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date"])
    df["od_date__hour"] = df["od_date"].dt.hour

    hours = list(range(24))
    df["od_date__hour"] = pd.Categorical(
        df["od_date__hour"], categories=hours, ordered=True
    )

    # Get day names and map to 3-letter abbreviations
    df["od_date__day_of_week_full"] = df["od_date"].dt.day_name()

    # Map full day names to 3-letter abbreviations
    day_mapping = {
        'Monday': 'Mon',
        'Tuesday': 'Tue', 
        'Wednesday': 'Wed',
        'Thursday': 'Thu',
        'Friday': 'Fri',
        'Saturday': 'Sat',
        'Sunday': 'Sun'
    }
    df["od_date__day_of_week_full"] = df["od_date__day_of_week_full"].map(day_mapping)

    # Order days with Monday at bottom and Sunday at top (reversed from typical order)
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # Monday to Sunday (bottom to top in chart)
    df["od_date__day_of_week_full"] = pd.Categorical(
        df["od_date__day_of_week_full"], categories=days_order, ordered=True
    )

    # Pivot table
    pivot = df.pivot_table(
        index="od_date__day_of_week_full",
        columns="od_date__hour",
        aggfunc="size",
        fill_value=0,
        observed=False,    # keep the current (in-place) behavior
    )

    # Custom hover text - flatten the array for customdata
    hover_text_flat = []
    for row in pivot.index:
        row_data = []
        for col in pivot.columns:
            row_data.append(f"Hour: {col}<br>Day: {row}<br>Count: {pivot.loc[row, col]}")
        hover_text_flat.append(row_data)

    light_palette = px.colors.sequential.Viridis   # matter

    # Region masks - updated to new time boundaries
    early_morning_mask = df["od_date"].dt.hour < 9  # 00:00-08:59

    working_hours_mask = (
        df["od_date"].dt.hour.between(9, 16)  # 09:00-16:59
        & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
    )

    weekend_daytime_mask = (
        df["od_date"].dt.hour.between(9, 16)  # 09:00-16:59 (same as working hours)
        & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
    )

    early_evening_mask = (
        df["od_date"].dt.hour.between(17, 18)  # 17:00-18:59
        & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
    )

    weekend_early_evening_mask = (
        df["od_date"].dt.hour.between(17, 18)  # 17:00-18:59
        & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
    )

    late_evening_mask = df["od_date"].dt.hour >= 19  # 19:00-23:59

    # Totals
    total_count = len(df)
    early_morning_count = early_morning_mask.sum()
    working_hours_count = working_hours_mask.sum()
    weekend_daytime_count = weekend_daytime_mask.sum()
    early_evening_count = early_evening_mask.sum()
    weekend_early_evening_count = weekend_early_evening_mask.sum()
    late_evening_count = late_evening_mask.sum()

    # Percentages
    percent = lambda x: round((x / total_count) * 100, 1) if total_count else 0
    early_morning_pct = percent(early_morning_count)
    working_hours_pct = percent(working_hours_count)
    weekend_daytime_pct = percent(weekend_daytime_count)
    early_evening_pct = percent(early_evening_count)
    weekend_early_evening_pct = percent(weekend_early_evening_count)
    late_evening_pct = percent(late_evening_count)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            customdata=hover_text_flat,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=light_palette,
            showscale=True,
        )
    )
    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        tickformat="02d", # pad with zero
        title="",  # Remove x-axis title
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=8,  # Set specific standoff distance
    )
    fig.update_yaxes(
        title="",  # Remove y-axis title
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklen=0,  # Remove tick marks
        ticks="",   # Hide tick marks
        ticklabelstandoff=10  # Set specific standoff distance
    )

    # Explicitly set hovermode for heatmap
    fig.update_layout(hovermode="closest")

    # Working Hours (08:00-16:00, Mon-Fri) - White solid border on top
    fig.add_shape(
        type="rect",
        x0=7.5,   # Start at 08:00
        x1=15.5,  # End at 16:00 (exclusive)
        y0=-0.5,
        y1=4.5,
        fillcolor="rgba(0, 0, 0, 0)",
        line=dict(color="white", width=3, dash="solid"),
        layer="above",
    )

    # Early Evening (16:00-18:00, Mon-Fri) - White dashed border
    fig.add_shape(
        type="rect",
        x0=15.5,  # Start at 16:00
        x1=17.5,  # End at 18:00 (exclusive)
        y0=-0.5,
        y1=4.5,
        fillcolor="rgba(0, 0, 0, 0)",                           # Transparent fill
        line=dict(color="white", width=3, dash="dash"),
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title=None,
        y_title=None,
        margin=dict(t=45, l=10, r=10, b=45),
        hovermode_unified=False,
    )

    chart_config = fig._config.copy()
    chart_config.update({
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",                              # Show modebar only on hover
        "staticPlot": False,                                    # Ensure interactivity
    })

    return plot(fig, output_type="div", config=chart_config), {
        "early_morning": {"count": early_morning_count, "pct": early_morning_pct},
        "working_hours": {"count": working_hours_count, "pct": working_hours_pct},
        "weekend_daytime": {"count": weekend_daytime_count, "pct": weekend_daytime_pct},
        "early_evening": {"count": early_evening_count, "pct": early_evening_pct},
        "weekend_early_evening": {"count": weekend_early_evening_count, "pct": weekend_early_evening_pct},
        "late_evening": {"count": late_evening_count, "pct": late_evening_pct},
    }
