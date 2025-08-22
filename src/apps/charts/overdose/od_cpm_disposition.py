import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_cpm_disposition(theme):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("cpm_disposition"))

    fig = px.histogram(
        df,
        x="cpm_disposition",
        nbins=30,
        title=None,
        labels={"cpm_disposition": "CPM Disposition"},
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
    )

    return plot(fig, output_type="div", config=fig._config)
