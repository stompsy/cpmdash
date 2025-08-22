import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_od_stack_livingsituation(theme):
    """
    Build a stacked bar chart comparing fatal and non-fatal overdoses by insurance type.
    """
    # Dataframe
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "disposition",
            "living_situation",
        )
    )

    # If the dataframe is empty, return a message
    if df.empty:
        return "No overdose data available to display."

    # Convert to datetime and classify fatal vs non-fatal
    fatal_conditions = ["CPR attempted", "DOA"]
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if x in fatal_conditions else "Non-Fatal"
    )

    # Group by insurance type and outcome
    grouped = (
        df.groupby(["living_situation", "overdose_outcome"])
        .size()
        .reset_index(name="count")
        .dropna(subset=["living_situation"])
        .sort_values("count", ascending=False)
    )

    fig = px.bar(
        grouped,
        x="living_situation",
        y="count",
        color="overdose_outcome",
        barmode="stack",
        labels={
            "living_situation": "Living Situation",
            "count": "Overdose Count",
            "overdose_outcome": "Outcome",
        },
        color_discrete_map={"Fatal": "#EF553B", "Non-Fatal": "#636EFA"},
        text="count",
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        y_title="Overdose Count",
        margin=dict(t=0, l=75, r=20, b=65),
    )
    return plot(fig, output_type="div", config=fig._config)
