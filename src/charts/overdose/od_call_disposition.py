import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from dashboard.models import ODReferrals


def build_chart_call_disposition(theme):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("disposition"))

    fig = px.histogram(
        df,
        x="disposition",
        nbins=30,
        title=None,
        labels={"disposition": "Call Disposition"},
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_call_disposition",
        scroll_zoom=False,
    )

    return plot(fig, output_type="div", config=fig._config)
