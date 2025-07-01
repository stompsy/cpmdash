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
    
    df["od_date__day_of_week_full"] = df["od_date"].dt.day_name()
    df["od_date__day_of_week"] = df["od_date__day_of_week_full"].str[:3]

    # Order days
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["od_date__day_of_week"] = pd.Categorical(
        df["od_date__day_of_week"], categories=days_order, ordered=False
    )

    # Pivot table
    pivot = df.pivot_table(
        index="od_date__day_of_week",
        columns="od_date__hour",
        aggfunc="size",
        fill_value=0,
        observed=False,    # keep the current (in-place) behavior
    )

    # Lookup full names for hover
    day_lookup = (
        df.drop_duplicates("od_date__day_of_week")[
            ["od_date__day_of_week", "od_date__day_of_week_full"]
        ]
        .set_index("od_date__day_of_week")["od_date__day_of_week_full"]
        .to_dict()
    )

    # Custom hover text
    hover_text = [
        [
            f"Hour: {col}:00<br>Day: {day_lookup[row]}<br>Count: {pivot.loc[row, col]}"
            for col in pivot.columns
        ]
        for row in pivot.index
    ]
    
    light_palette = px.colors.sequential.Viridis   # matter
    
    # Region colors matching the time region cards
    region_colors = {
        "early_morning": "#3B82F6",     # bg-blue-500
        "working_hours": "#10B981",     # bg-green-500  
        "early_evening": "#F97316",     # bg-orange-500
        "late_evening": "#8B5CF6",      # bg-purple-500
        "weekend_daytime": "#EAB308",   # bg-yellow-500
        "weekend_evening": "#EC4899",   # bg-pink-500
    }

    # Region masks
    early_morning_mask = df["od_date"].dt.hour < 8

    working_hours_mask = (
        df["od_date"].dt.hour.between(8, 15)
        & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
    )

    weekend_daytime_mask = (
        df["od_date"].dt.hour.between(8, 15)
        & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
    )
    
    early_evening_mask = (
        df["od_date"].dt.hour.between(16, 18)  # 16:00 to 18:59
        & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
    )
    
    weekend_early_evening_mask = (
        df["od_date"].dt.hour.between(16, 18)  # 16:00 to 18:59
        & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
    )
    
    late_evening_mask = df["od_date"].dt.hour.between(19, 23)

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
            text=hover_text,
            hoverinfo="text",
            colorscale=light_palette,
            showscale=False,
        )
    )
    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        tickformat="02d", # pad with zero
        title="",  # Remove x-axis title
    )
    fig.update_yaxes(
        title="",  # Remove y-axis title
    )
    
    # Add color overlay masks for each time region
    # Early Morning (00:00-07:59, All days) - Blue
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=7.5,
        y0=-0.5,
        y1=6.5,
        fillcolor="rgba(59, 130, 246, 0.15)",  # Semi-transparent blue
        line=dict(width=0),
        layer="below"
    )
    
    # Working Hours (08:00-15:59, Mon-Fri) - Green
    fig.add_shape(
        type="rect",
        x0=7.5,
        x1=15.5,
        y0=-0.5,
        y1=4.5,
        fillcolor="rgba(16, 185, 129, 0.15)",  # Semi-transparent green
        line=dict(width=0),
        layer="below"
    )
    
    # Weekend Daytime (08:00-15:59, Sat-Sun) - Yellow
    fig.add_shape(
        type="rect",
        x0=7.5,
        x1=15.5,
        y0=4.5,
        y1=6.5,
        fillcolor="rgba(234, 179, 8, 0.15)",  # Semi-transparent yellow
        line=dict(width=0),
        layer="below"
    )
    
    # Early Evening (16:00-18:59, Mon-Fri) - Orange
    fig.add_shape(
        type="rect",
        x0=15.5,
        x1=18.5,
        y0=-0.5,
        y1=4.5,
        fillcolor="rgba(249, 115, 22, 0.15)",  # Semi-transparent orange
        line=dict(width=0),
        layer="below"
    )
    
    # Weekend Evening (16:00-18:59, Sat-Sun) - Pink
    fig.add_shape(
        type="rect",
        x0=15.5,
        x1=18.5,
        y0=4.5,
        y1=6.5,
        fillcolor="rgba(236, 72, 153, 0.15)",  # Semi-transparent pink
        line=dict(width=0),
        layer="below"
    )
    
    # Late Evening (19:00-23:59, All days) - Purple
    fig.add_shape(
        type="rect",
        x0=18.5,
        x1=23.5,
        y0=-0.5,
        y1=6.5,
        fillcolor="rgba(139, 92, 246, 0.15)",  # Semi-transparent purple
        line=dict(width=0),
        layer="below"
    )
    
    # Add working hours rectangle border
    fig.add_shape(
        type="rect",
        x0=7.55,  # Smaller gap from 7.6
        x1=15.45, # Smaller gap from 15.4
        y0=-0.45,  # Smaller inset from -0.4
        y1=4.45,   # Smaller inset from 4.4
        line=dict(color="white", width=2, dash="solid"),
        fillcolor="rgba(0,0,0,0)",  # Transparent fill
    )
    fig.add_annotation(
        x=7.6,  # Just inside left edge
        y=4.4,  # Just inside top edge
        text="<b>Working Hours</b>",
        font=dict(
            size=14,
            color="white",
            family="Arial, sans-serif",
        ),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    fig.add_annotation(
        x=7.6,
        y=4.1,
        text=f"{working_hours_count} overdoses • {working_hours_pct}%",
        font=dict(size=12, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )

    
    # Add early morning hours rectangle border
    fig.add_shape(
        type="rect",
        x0=-0.45,  # Smaller inset from -0.4
        x1=7.45,   # Smaller gap before working hours (was 7.4)
        y0=-0.45,  # Smaller inset from -0.4
        y1=6.45,   # Smaller inset from 6.4
        line=dict(color=region_colors["early_morning"], width=2, dash="solid"),
        fillcolor="rgba(0,0,0,0)",  # Transparent fill
    )
    fig.add_annotation(
        x=-0.4,
        y=6.4,
        text="<b>Early Morning</b>",
        font=dict(
            size=14,
            color="white",
            family="Arial, sans-serif",
        ),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    fig.add_annotation(
        x=-0.4,
        y=6.1,
        text=f"{early_morning_count} overdoses • {early_morning_pct}%",
        font=dict(size=12, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )

    
    # Add Saturday and Sunday from 08:00 to 16:00 rectangle
    fig.add_shape(
        type="rect",
        x0=7.55,   # Smaller gap from 7.6
        x1=15.45,  # Smaller gap from 15.4
        y0=4.55,   # Smaller gap after working hours (was 4.6)
        y1=6.45,   # Smaller inset from 6.4
        line=dict(color=region_colors["weekend_daytime"], width=2, dash="solid"),
        fillcolor="rgba(0,0,0,0)",  # Transparent
    )
    fig.add_annotation(
        x=7.6,
        y=6.4,
        text="<b>Weekend Daytime</b>",
        font=dict(
            size=14,
            color="white",
            family="Arial, sans-serif",
        ),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    fig.add_annotation(
        x=7.6,
        y=6.1,
        text=f"{weekend_daytime_count} overdoses • {weekend_daytime_pct}%",
        font=dict(size=12, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    
    # Add early evening hours rectangle
    # Covers Mon–Fri from 16:00 to 18:59
    fig.add_shape(
        type="rect",
        x0=15.55,  # Smaller gap after working hours (was 15.6)
        x1=18.45,  # Smaller gap from 18.4
        y0=-0.45,  # Smaller inset from -0.4
        y1=4.45,   # Smaller inset from 4.4
        line=dict(color=region_colors["early_evening"], width=2, dash="solid"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=15.6,
        y=4.4,
        text="<b>Early Evening</b>",
        font=dict(size=14, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    fig.add_annotation(
        x=15.6,
        y=4.1,
        text=f"{early_evening_count} overdoses • {early_evening_pct}%",
        font=dict(size=12, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    
    # Add weekend early evening hours rectangle
    # Covers Sat–Sun from 16:00 to 18:59
    fig.add_shape(
        type="rect",
        x0=15.55,  # Smaller gap after weekend daytime (was 15.6)
        x1=18.45,  # Smaller gap from 18.4
        y0=4.55,   # Smaller gap after weekend daytime (was 4.6)
        y1=6.45,   # Smaller inset from 6.4
        line=dict(color=region_colors["weekend_evening"], width=2, dash="solid"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=15.6,
        y=6.4,
        text="<b>Weekend Evening</b>",
        font=dict(size=14, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    fig.add_annotation(
        x=15.6,
        y=6.1,
        text=f"{weekend_early_evening_count} overdoses • {weekend_early_evening_pct}%",
        font=dict(size=12, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )

    # Add late evening hours rectangle
    # Covers all days from 19:00 to 23:59
    fig.add_shape(
        type="rect",
        x0=18.55,  # Smaller gap after early evening (was 18.6)
        x1=23.45,  # Smaller gap from 23.4
        y0=-0.45,  # Smaller inset from -0.4
        y1=6.45,   # Smaller inset from 6.4
        line=dict(color=region_colors["late_evening"], width=2, dash="solid"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=18.6,
        y=6.4,
        text="<b>Late Evening</b>",
        font=dict(size=14, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    fig.add_annotation(
        x=18.6,
        y=6.1,
        text=f"{late_evening_count} overdoses • {late_evening_pct}%",
        font=dict(size=12, color="white"),
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_heatmap_density",
        show_legend=False,
        scroll_zoom=False,
        x_title="",  # Remove x-axis title
        y_title="",  # Remove y-axis title
        margin=dict(t=0, l=0, r=16, b=0),
    )
    return plot(fig, output_type="div", config=fig._config), {
        "early_morning": {"count": early_morning_count, "pct": early_morning_pct},
        "working_hours": {"count": working_hours_count, "pct": working_hours_pct},
        "weekend_daytime": {"count": weekend_daytime_count, "pct": weekend_daytime_pct},
        "early_evening": {"count": early_evening_count, "pct": early_evening_pct},
        "weekend_early_evening": {"count": weekend_early_evening_count, "pct": weekend_early_evening_pct},
        "late_evening": {"count": late_evening_count, "pct": late_evening_pct},
    }
