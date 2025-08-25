import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import Referrals


def build_chart_od_agency_treemap(theme: str) -> str:
    # Get referral data - split to avoid mypy internal error
    qs = Referrals.objects.all()  # type: ignore[misc]
    data = list(qs.values("referral_agency"))  # type: ignore[misc]
    df = pd.DataFrame.from_records(data)  # type: ignore[misc]

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
    # Split the complex list comprehension to avoid mypy internal error
    shades = (600, 500, 400, 300)
    tokens: list[str] = []
    for hue in families:
        for shade in shades:
            tokens.append(f"{hue}-{shade}")
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
