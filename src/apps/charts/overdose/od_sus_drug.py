import pandas as pd
import plotly.express as px
from django.db.models import Count
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_sus_drug(theme):
    qs = (
        ODReferrals.objects.values("suspected_drug")
        .annotate(count=Count("ID"))
        .order_by("count")
        .reverse()  # smallest → largest
    )

    df = pd.DataFrame(qs).fillna({"suspected_drug": "Unknown"})

    df["drug_group"] = df["suspected_drug"].apply(
        lambda d: "Fentanyl-related" if "fentanyl" in d.lower() else "Other"
    )

    fent_first = (
        df[df["drug_group"] == "Fentanyl-related"]["suspected_drug"].to_list()
        + df[df["drug_group"] == "Other"]["suspected_drug"].to_list()
    )

    custom_map = {
        "Fentanyl-related": "#EF553B",
        "Other": "#636EFA",
    }

    fig = px.bar(
        df,
        x="count",
        y="suspected_drug",
        orientation="h",
        color="drug_group",
        color_discrete_map=custom_map,
        category_orders={"suspected_drug": fent_first},
        labels={"count": "Number of Overdoses", "suspected_drug": "Suspected Drug"},
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=fent_first[::-1],  # or fent_first to flip
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
    )

    return plot(fig, output_type="div", config=fig._config)
