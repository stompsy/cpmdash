import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout


def build_chart_od_time_region_bars(theme):
    # Overdose data by time region
    regions = [
        "Early Morning (00:00–08:00)",
        "Working Hours (08:00–16:00, M–F)",
        "Early Evening (16:00–19:00, M–F)",
        "Late Evening (19:00–24:00)",
        "Weekend Daytime (08:00–16:00)",
        "Weekend Early Evening (16:00–19:00)",
    ]
    counts = [24, 33, 22, 23, 11, 9]

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=regions,
                y=counts,
                text=[f"{val} overdoses" for val in counts],
                textposition="auto",
            )
        ]
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        show_legend=False,
        scroll_zoom=False,
        x_title="Time Region",
        y_title="Number of Overdoses",
        margin=dict(t=0, l=60, r=20, b=65),
    )
    return plot(fig, output_type="div", config=fig._config)
