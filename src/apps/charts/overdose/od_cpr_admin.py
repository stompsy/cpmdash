import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_cpr_admin(theme="light"):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("cpr_administered"))

    fig = px.histogram(
        df,
        x="cpr_administered",
        title=None,
        labels={"cpr_administered": "CPR Administered"},
    )

    # counts inside bars, bold, with a larger font
    # fig.update_traces(
    #     texttemplate="<b>%{y}</b>",
    #     textposition="inside",
    #     textfont=dict(size=12)  # adjust size as desired
    # )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
    )

    return plot(fig, output_type="div", config=fig._config)
