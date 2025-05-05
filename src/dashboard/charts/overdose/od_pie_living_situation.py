import pandas as pd
import plotly.express as px

from ...utils.plotly import style_plotly_layout

from ...models import *


def build_chart_od_living_situation(df, theme="light"):
    # Dataframe
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "living_situation",
        )
    )

    df = df.groupby("living_situation").size().reset_index(name="Count")
    df = df.sort_values(by="Count", ascending=False)

    fig = px.pie(
        df,
        names="living_situation",
        values="Count",
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4,
    )
    fig.update_traces(
        hovertemplate="Situation: %{label}<br>Count: %{value} (%{percent})<extra></extra>",
        texttemplate="%{percent}",
        textposition="inside",
        textfont=dict(size=14, color="white", family="Arial", weight="bold"),
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_living_situation",
        scroll_zoom=False,
        height=300,
        margin={"r": 40, "t": 0, "l": 40, "b": 40},
    )

    return fig
