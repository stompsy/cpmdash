import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_od_density_heatmap(theme):
    """
    Create an interactive Plotly heatmap that looks like matplotlib but maintains hover functionality
    """
    # Use the same data processing as the other versions
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
    df["od_date__hour"] = pd.Categorical(df["od_date__hour"], categories=hours, ordered=True)

    # Get day names and map to 3-letter abbreviations
    df["od_date__day_of_week_full"] = df["od_date"].dt.day_name()

    # Map full day names to 3-letter abbreviations
    day_mapping = {
        "Monday": "Mon",
        "Tuesday": "Tue",
        "Wednesday": "Wed",
        "Thursday": "Thu",
        "Friday": "Fri",
        "Saturday": "Sat",
        "Sunday": "Sun",
    }
    df["od_date__day_of_week_full"] = df["od_date__day_of_week_full"].map(day_mapping)

    # Order days with Monday at bottom and Sunday at top (reversed from typical order)
    days_order = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]  # Monday to Sunday (bottom to top in chart)
    df["od_date__day_of_week_full"] = pd.Categorical(
        df["od_date__day_of_week_full"], categories=days_order, ordered=True
    )

    # Pivot table - same as other versions
    pivot = df.pivot_table(
        index="od_date__day_of_week_full",
        columns="od_date__hour",
        aggfunc="size",
        fill_value=0,
        observed=False,
    )

    # Create annotations matrix for displaying values in cells
    annotations = []
    for _i, row in enumerate(pivot.index):
        for _j, col in enumerate(pivot.columns):
            value = pivot.loc[row, col]
            annotations.append(
                dict(
                    x=col,
                    y=row,
                    text=str(value),
                    xref="x",
                    yref="y",
                    showarrow=False,
                    font=dict(color="white", size=12, family="Roboto"),
                    xanchor="center",
                    yanchor="middle",
                )
            )

    # Custom hover text - same as original
    hover_text_flat = []
    for row in pivot.index:
        row_data = []
        for col in pivot.columns:
            count = pivot.loc[row, col]
            # Enhanced hover with additional context
            hover_info = (
                f"<b>Time:</b> {col:02d}:00<br><b>Day:</b> {row}<br><b>Overdoses:</b> {count}"
            )
            if count > 0:  # type: ignore
                percentage = round((count / pivot.values.sum()) * 100, 1)
                hover_info += f"<br><b>% of Total:</b> {percentage}%"
            row_data.append(hover_info)
        hover_text_flat.append(row_data)

    # Create the Plotly figure with matplotlib-like styling
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            customdata=hover_text_flat,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="Viridis",
            showscale=False,  # Remove legend/colorbar
            hoverongaps=False,
        )
    )

    # Add text annotations to show values in each cell (matplotlib-style)
    # Also add custom hour labels since we disabled default tick labels
    for hour in range(24):
        annotations.append(
            dict(
                x=hour,
                y=-1.0,  # Position below the heatmap
                text=f"{hour:02d}",
                xref="x",
                yref="y",
                showarrow=False,
                font=dict(
                    color="gray" if theme == "light" else "lightgray", size=14, family="Roboto"
                ),
                xanchor="center",
                yanchor="top",
            )
        )

    fig.update_layout(annotations=annotations)

    # Customize axes to match matplotlib style - no titles
    fig.update_xaxes(
        showticklabels=False,  # Completely hide default tick labels
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.3)",
        # Aggressively disable ALL default ticks and tick marks
        ticks="",
        ticklen=0,
        tickwidth=0,
        linewidth=0,  # Remove axis line
        mirror=False,  # Don't mirror to opposite side
        showline=False,  # Remove axis line completely
        zeroline=False,  # Remove zero line
        # Override automatic tick placement with explicit range
        range=[-0.5, 23.5],  # Set explicit range to match data
        tickmode="array",
        tickvals=[],  # Empty array = no ticks
        ticktext=[],  # Empty array = no labels
        # Additional properties to completely disable all automatic ticks
        autorange=False,  # Disable auto range to prevent automatic ticks
        dtick=None,  # Disable automatic tick spacing
        tick0=None,  # Disable tick starting point
        nticks=0,  # Set number of ticks to 0
        fixedrange=True,  # Prevent zooming which might trigger tick regeneration
    )

    fig.update_yaxes(
        tickfont=dict(size=14, family="Roboto"),  # Match original heatmap font size
        ticklen=0,
        ticks="",
        ticklabelstandoff=10,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.3)",
    )

    # Working Hours (08:00-16:00, Mon-Fri) - Bright border for visibility
    fig.add_shape(
        type="rect",
        x0=7.5,  # Start at 08:00
        x1=15.5,  # End at 16:00 (exclusive)
        y0=-0.5,
        y1=4.5,
        fillcolor="rgba(0, 0, 0, 0)",
        line=dict(color="yellow", width=4, dash="solid"),
        layer="above",
    )

    # Add fork-like tick marks for hour boundaries
    tick_color = "gray"
    tick_width = 1
    tick_height = 0.2
    bottom_y = -0.5  # Bottom of heatmap

    for hour in range(24):
        # Left L-shaped tick (start of hour)
        fig.add_shape(
            type="line",
            x0=hour - 0.5,
            y0=bottom_y,
            x1=hour - 0.5,
            y1=bottom_y - tick_height,
            line=dict(color=tick_color, width=tick_width),
            layer="above",
        )
        fig.add_shape(
            type="line",
            x0=hour - 0.5,
            y0=bottom_y - tick_height,
            x1=hour,
            y1=bottom_y - tick_height,
            line=dict(color=tick_color, width=tick_width),
            layer="above",
        )

        # Right backwards L-shaped tick (end of hour)
        fig.add_shape(
            type="line",
            x0=hour + 0.5,
            y0=bottom_y,
            x1=hour + 0.5,
            y1=bottom_y - tick_height,
            line=dict(color=tick_color, width=tick_width),
            layer="above",
        )
        fig.add_shape(
            type="line",
            x0=hour + 0.5,
            y0=bottom_y - tick_height,
            x1=hour,
            y1=bottom_y - tick_height,
            line=dict(color=tick_color, width=tick_width),
            layer="above",
        )

        # Central vertical tick from bottom of L's to hour label
        fig.add_shape(
            type="line",
            x0=hour,
            y0=bottom_y - tick_height,
            x1=hour,
            y1=bottom_y - tick_height - 0.2,
            line=dict(color=tick_color, width=tick_width),
            layer="above",
        )

    # Apply theme styling
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title=None,  # Remove x-axis title
        y_title=None,  # Remove y-axis title
        margin=dict(
            t=45, l=50, r=40, b=15
        ),  # Match daily totals style with space for custom elements
        hovermode_unified=False,
    )

    # Override some styles to make it look more like matplotlib - no title
    fig.update_layout(
        title=None,  # Remove title
        hovermode="closest",
        plot_bgcolor="rgba(255,255,255,0.2)"
        if theme == "light"
        else "rgba(31,41,55,0.2)",  # Match container bg
        paper_bgcolor="rgba(255,255,255,0.2)"
        if theme == "light"
        else "rgba(31,41,55,0.2)",  # Match container bg
    )

    chart_config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "staticPlot": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "overdose_heatmap_interactive",
            "height": 600,
            "width": 1200,
            "scale": 2,
        },
    }

    return plot(fig, output_type="div", config=chart_config)
