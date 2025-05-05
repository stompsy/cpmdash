import pandas as pd
import plotly.express as px
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout

from ...models import ODReferrals


def build_chart_referral_delay(theme="light"):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("delay_in_referral"))

    fig = px.histogram(
        df,
        x="delay_in_referral",
        nbins=30,
        title=None,
        labels={"delay_in_referral": "Delay in Referral (Days)"},
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_notification",
    )

    return plot(fig, output_type="div", config=fig._config)
