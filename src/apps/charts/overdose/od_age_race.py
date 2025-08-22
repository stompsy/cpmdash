import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


# Build the chart
def build_chart_od_age_race(theme):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "patient_age",
            "patient_race",
        )
    )

    # Count and sort by number of data points per race
    race_order = df["patient_race"].value_counts().sort_values(ascending=False).index.tolist()

    fig = px.box(
        df,
        x="patient_race",
        y="patient_age",
        color="patient_race",
        points="all",
        category_orders={"patient_race": race_order},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
    )
    return plot(fig, output_type="div", config=fig._config)
