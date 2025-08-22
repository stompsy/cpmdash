import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_referral_delay(theme="light"):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("delay_in_referral"))
    df["delay_in_referral"] = df["delay_in_referral"].astype(str)
    # Create a two-level grouping: 'CPM respond' vs. everything else
    df["delay_group"] = df["delay_in_referral"].apply(
        lambda x: "CPM respond" if x == "CPM respond" else "Delays in Referral"
    )
    # Turn it into an ordered categorical so plotly keeps order
    df["delay_group"] = pd.Categorical(
        df["delay_group"],
        categories=["CPM respond", "Delays in Referral"],
        ordered=True,
    )

    fig = px.histogram(
        df,
        x="delay_group",
        color="delay_group",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        labels={"delay_group": "Delays in Referral"},
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
    )

    return plot(fig, output_type="div", config=fig._config)
