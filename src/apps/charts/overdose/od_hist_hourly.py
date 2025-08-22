import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


def build_chart_od_hist_hourly(theme="light"):
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

    # Create histogram
    fig = px.histogram(
        df,
        x="od_datetime__hour",
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            tick0=0.5,
            dtick=1,
            tickformat=".0f",
            tickvals=np.arange(0, 24, 1),
        ),
        bargap=0.2,
    )
    fig.add_shape(
        type="rect",
        x0=8,
        x1=16,  # Working hours
        y0=0,
        y1=6.5,  # Covers all weekdays (Monday to Sunday)
        line=dict(color="red", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)",  # Transparent fill
    )
    fig.add_annotation(
        x=13,
        y=3,
        text="Working Hours",
        showarrow=True,
        arrowhead=4,
        ax=0,
        ay=-40,
        font=dict(color="red"),
    )
    fig.update_traces(
        xbins=dict(start=0, end=24, size=1),
        customdata=np.arange(1, 25),
        hovertemplate="Hour: %{x}:00 - %{customdata}:00<br>Overdoses: %{y}<extra></extra>",
        texttemplate="%{y}",
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=12, color="white"),
        hoverinfo="skip",
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title="Hour of Day",
        y_title="Overdose Count",
        margin=dict(t=0, l=75, r=20, b=65),
    )

    return plot(fig, output_type="div", config=fig._config)
