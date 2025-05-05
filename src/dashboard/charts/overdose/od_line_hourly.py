import pandas as pd
import plotly.express as px

from ...utils.plotly import style_plotly_layout

from ...models import *


def build_chart_od_line_hourly(df, theme="light"):

    # Dataframe
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "od_date",
        )
    )

    # Convert 'od_date' to datetime
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df["od_datetime__hour"] = df["od_date"].dt.hour

    # Create line chart
    # Aggregate hourly counts for line chart
    hourly_counts = df.groupby("od_datetime__hour").size().reset_index(name="counts")

    fig = px.line(
        hourly_counts,
        x="od_datetime__hour",
        y="counts",
        title=None,
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
    )
    fig.add_shape(
        type="rect",
        x0=8,  # Start of working hours
        x1=16,
        y0=0,
        y1=6.5,
        line=dict(color="red", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=12,
        y=6,
        text="Working Hours",
        showarrow=True,
        arrowhead=4,
        ax=0,
        ay=-40,
        font=dict(color="red"),
    )
    fig.update_traces(hovertemplate="Hour: %{x}:00<br>Overdoses: %{y}<extra></extra>")
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_line_hourly",
        scroll_zoom=False,
        x_title="Hour of Day",
        y_title="Overdose Count",
        margin=dict(t=0, l=75, r=20, b=65),
    )

    return fig
