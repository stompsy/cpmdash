"""
Detailed hourly breakdown analytics for overdose patterns
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from utils.plotly import get_color_palette, get_theme_colors, style_plotly_layout

from ...core.models import ODReferrals


def build_chart_od_hourly_breakdown(theme):
    """
    Build a comprehensive hour-by-hour breakdown chart showing:
    - Total overdoses by hour across all days
    - Weekday vs weekend patterns
    - Coverage gaps visualization
    """

    # Get all overdose data with timestamps
    overdoses = ODReferrals.objects.exclude(od_date__isnull=True)

    # Convert to pandas for easier time analysis
    data = []
    for od in overdoses:
        data.append(
            {
                "datetime": od.od_date,
                "hour": od.od_date.hour,
                "weekday": od.od_date.weekday(),  # 0=Monday, 6=Sunday
                "is_weekend": od.od_date.weekday() >= 5,
            }
        )

    df = pd.DataFrame(data)

    if df.empty:
        # Return empty chart if no data with proper styling
        fig = go.Figure()
        fig.add_annotation(
            text="No overdose data available for analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig = style_plotly_layout(
            fig,
            theme=theme,
            height=700,
            show_legend=True,
            scroll_zoom=False,
            margin=dict(t=45, l=10, r=10, b=45),
            x_title="Hour of Day",
            y_title="Number of Overdoses",
        )
        chart_config = fig._config.copy()
        chart_config.update(
            {
                "responsive": True,
                "displaylogo": False,
                "displayModeBar": "hover",  # Show modebar only on hover
                "staticPlot": False,  # Ensure interactivity
            }
        )

        return plot(fig, output_type="div", config=chart_config)

    # Get color palette
    colors = get_color_palette(theme)

    # Create hourly breakdown data
    hourly_all = df["hour"].value_counts().sort_index()
    hourly_weekday = df[df["is_weekend"] == False]["hour"].value_counts().sort_index()
    hourly_weekend = df[df["is_weekend"] == True]["hour"].value_counts().sort_index()

    # Ensure all hours are represented (0-23)
    all_hours = range(24)
    hourly_all = hourly_all.reindex(all_hours, fill_value=0)
    hourly_weekday = hourly_weekday.reindex(all_hours, fill_value=0)
    hourly_weekend = hourly_weekend.reindex(all_hours, fill_value=0)

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("All Days - Hourly Overdose Distribution", "Weekday vs Weekend Comparison"),
        vertical_spacing=0.2,
        row_heights=[0.5, 0.5],  # Equal height for both subplots
    )
    # Top chart: All days with coverage overlay
    fig.add_trace(
        go.Bar(
            x=list(all_hours),
            y=hourly_all.values,
            name="Total Overdoses",
            marker_color=colors["primary"],
            hovertemplate="<b>Hour %{x}:00</b><br>Overdoses: %{y}<extra></extra>",
            legendgroup="top",
            legendgrouptitle_text="All Days - Hourly Distribution",
        ),
        row=1,
        col=1,
    )

    # Add coverage overlay (current working hours 8-16)
    coverage_hours = list(range(8, 16))
    coverage_y = [hourly_all[h] for h in coverage_hours]

    fig.add_trace(
        go.Bar(
            x=coverage_hours,
            y=coverage_y,
            name="Current Coverage",
            marker_color=colors["success"],
            opacity=0.8,
            hovertemplate="<b>Hour %{x}:00</b><br>Covered Overdoses: %{y}<extra></extra>",
            legendgroup="top",
        ),
        row=1,
        col=1,
    )

    # Bottom chart: Weekday vs Weekend
    fig.add_trace(
        go.Bar(
            x=list(all_hours),
            y=hourly_weekday.values,
            name="Weekdays (Mon-Fri)",
            marker_color=colors["info"],
            opacity=0.8,
            hovertemplate="<b>Hour %{x}:00</b><br>Weekday Overdoses: %{y}<extra></extra>",
            legendgroup="bottom",
            legendgrouptitle_text="Weekday vs Weekend Comparison",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(all_hours),
            y=hourly_weekend.values,
            name="Weekends (Sat-Sun)",
            marker_color=colors["warning"],
            opacity=0.8,
            hovertemplate="<b>Hour %{x}:00</b><br>Weekend Overdoses: %{y}<extra></extra>",
            legendgroup="bottom",
        ),
        row=2,
        col=1,
    )

    # Configure legend to be horizontal and above charts
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, groupclick="toggleitem"
        )
    )

    # Update x-axes
    for row in [1, 2]:
        fig.update_xaxes(
            title_text=None,  # Remove x-axis titles
            tickmode="linear",
            tick0=0,
            dtick=1,  # Show all hours (0-23) instead of every 2 hours
            range=[-0.5, 23.5],
            tickfont=dict(size=14, family="Roboto"),
            ticklabelstandoff=8,
            row=row,
            col=1,
        )

    # Update y-axes
    fig.update_yaxes(
        title_text=None, tickfont=dict(size=14, family="Roboto"), ticklabelstandoff=10, row=1, col=1
    )
    fig.update_yaxes(
        title_text=None, tickfont=dict(size=14, family="Roboto"), ticklabelstandoff=10, row=2, col=1
    )

    # Update layout using style_plotly_layout
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=700,
        show_legend=True,
        scroll_zoom=False,
        margin=dict(t=45, l=45, r=40, b=45),
    )

    theme_colors = get_theme_colors(theme)

    # Ensure vertical grid lines are removed (override any theme defaults)
    fig.update_xaxes(
        showgrid=False,
        showline=True,  # Show the x-axis line
        linewidth=1,  # Set line width
        linecolor="lightgray",  # Set line color
    )

    # Ensure both charts have consistent grid colors
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme_colors["grid_color"],  # Force same grid color
        showline=False,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme_colors["grid_color"],  # Force same grid color
        showline=False,
        zeroline=False,
        row=2,
        col=1,
    )

    chart_config = fig._config.copy()
    chart_config.update(
        {
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",  # Show modebar only on hover
            "staticPlot": False,  # Ensure interactivity
        }
    )

    return plot(fig, output_type="div", config=chart_config)


def build_chart_day_of_week_totals(theme):
    """
    Bar chart: Daily Overdose Totals by day of week
    """
    overdoses = ODReferrals.objects.exclude(od_date__isnull=True)
    data = []
    for od in overdoses:
        data.append(
            {
                "datetime": od.od_date,
                "weekday": od.od_date.weekday(),
                "weekday_name": od.od_date.strftime("%A"),
            }
        )
    df = pd.DataFrame(data)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No overdose data available for daily totals",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig = style_plotly_layout(
            fig,
            theme=theme,
            height=300,
            show_legend=False,
            scroll_zoom=False,
            x_title="Day of Week",
            y_title="Number of Overdoses",
        )

        chart_config = fig._config.copy()
        chart_config.update(
            {
                "responsive": True,
                "displaylogo": False,
                "displayModeBar": "hover",  # Show modebar only on hover
                "staticPlot": False,  # Ensure interactivity
            }
        )

        return plot(fig, output_type="div", config=chart_config)

    colors = get_color_palette(theme)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_counts = df["weekday_name"].value_counts().reindex(day_names, fill_value=0)
    bar_colors = [
        colors["danger"] if day in ["Saturday", "Sunday"] else colors["primary"]
        for day in day_names
    ]

    fig = go.Figure(
        go.Bar(
            x=day_names,
            y=daily_counts.values,
            name="Daily Totals",
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Total Overdoses: %{y}<extra></extra>",
        )
    )

    fig.update_xaxes(
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=8,  # Set specific standoff distance
        showline=True,  # Show the x-axis line
        linewidth=1,  # Set line width
        linecolor="lightgray",  # Set line color
    )
    fig.update_yaxes(
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=10,  # Set specific standoff distance
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=350,
        show_legend=False,
        scroll_zoom=False,
        x_title=None,
        y_title=None,
        margin=dict(t=45, l=45, r=40, b=45),
    )

    # Remove vertical grid lines
    fig.update_xaxes(showgrid=False)

    chart_config = fig._config.copy()
    chart_config.update(
        {
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",  # Show modebar only on hover
            "staticPlot": False,  # Ensure interactivity
        }
    )

    return plot(fig, output_type="div", config=chart_config)
