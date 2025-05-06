import pandas as pd
import plotly.express as px
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals

def build_chart_narcan_given(theme):
    # 1) load into a DataFrame
    qs = ODReferrals.objects.values("narcan_given")
    df = pd.DataFrame.from_records(qs)

    # 2) map to Yes/No
    df["narcan_given"] = df["narcan_given"].map({True: "Yes", False: "No"})

    # 3) get counts as a DataFrame
    counts = (
        df["narcan_given"]
        .value_counts()
        .sort_index()             # will sort alphabetically: No, Yes
        .rename_axis("narcan_given")
        .reset_index(name="count")
    )

    # 4) build the bar chart
    fig = px.bar(
        counts,
        x="narcan_given",
        y="count",
        color="narcan_given",
        text="count",
        category_orders={"narcan_given": ["Yes", "No"]},  # force Yes first if you like
        labels={"narcan_given": "Narcan Given?", "count": "Referral Count"},
    )

    fig.update_traces(textposition="outside")  # optional, puts your counts above the bars

    # 5) style & export
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_narcan_given",
        scroll_zoom=False,
        margin=dict(t=0, l=75, r=20, b=65),
    )
    return plot(fig, output_type="div", config=fig._config)
