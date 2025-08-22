import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Referrals


def build_chart_od_agency_treemap(theme):
    referrals = Referrals.objects.all()
    df = pd.DataFrame.from_records(
        referrals.values(
            "referral_agency",
        )
    )

    families = [
        "red",
        "orange",
        "amber",
        "yellow",
        "lime",
        "green",
        "emerald",
        "teal",
        "cyan",
        "sky",
        "blue",
        "indigo",
        "violet",
        "purple",
        "fuchsia",
        "pink",
        "rose",
        "slate",
        "gray",
        "zinc",
        "neutral",
        "stone",
    ]
    tokens = [f"{hue}-{shade}" for hue in families for shade in (600, 500, 400, 300)]
    color_sequence = [TAILWIND_COLORS[token] for token in tokens]

    top100 = df["referral_agency"].dropna().value_counts().nlargest(100).reset_index()

    fig = px.treemap(
        top100,
        path=["referral_agency"],  # <-- uses that column
        values="count",  # <-- and this column
        color="referral_agency",
        hover_data={"count": True},
        color_discrete_sequence=color_sequence,
    )

    fig.update_traces(
        marker_line_width=2,
        marker_line_color="white",
        textinfo="label+value",
    )

    fig.update_layout(
        font=dict(family="font-roboto", size=28, color="white"),
        title=dict(font=dict(family="font-droid", size=40, color="white")),
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        margin=dict(t=0, l=0, r=0, b=0),
    )

    return plot(fig, output_type="div", config=fig._config)
