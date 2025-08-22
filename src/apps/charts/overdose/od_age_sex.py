import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


# Build the chart
def build_chart_od_age_sex(theme):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("patient_age", "patient_sex"))

    # Count and sort by number of data points per sex
    sex_order = df["patient_sex"].value_counts().sort_values(ascending=False).index.tolist()
    fig = px.box(
        df,
        x="patient_sex",
        y="patient_age",
        color="patient_sex",
        points="all",
        category_orders={"patient_sex": sex_order},
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
    )
    return plot(fig, output_type="div", config=fig._config)
