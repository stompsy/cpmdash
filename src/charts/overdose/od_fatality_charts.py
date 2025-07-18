import pandas as pd
import plotly.express as px

from utils.plotly import style_plotly_layout

from dashboard.models import ODReferrals


def _prepare_overdose_data():
    df = pd.DataFrame.from_records(
        ODReferrals.objects.all().values(
            "disposition",
            "suspected_drug",
            "cpr_administered",
            "police_ita",
            "narcan_given",
        )
    )

    fatal_conditions = ["CPR attempted", "DOA"]
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if x in fatal_conditions else "Non-Fatal"
    )

    return df


def _build_fatality_chart(df, column, theme="light"):
    grouped = (
        df.groupby([column, "overdose_outcome"])
        .size()
        .reset_index(name="count")
        .dropna()
    )

    total_counts = grouped.groupby(column)["count"].sum().sort_values(ascending=False)

    grouped[column] = pd.Categorical(
        grouped[column], categories=total_counts.index.tolist(), ordered=True
    )

    grouped = grouped.sort_values(by=column)

    fig = px.bar(
        grouped,
        x=column,
        y="count",
        color="overdose_outcome",
        barmode="stack",
        labels={
            "suspected_drug": "Drug",
            "cpr_administered": "CPR",
            "police_ita": "ITA",
            "narcan_given": "Narcan",
            "disposition": "Disposition",
            "count": "OD Count",
            "overdose_outcome": "Outcome",
        },
        color_discrete_map={"Fatal": "red", "Non-Fatal": "#636EFA"},
        text="count",
    )

    fig.update_traces(textposition="inside", insidetextanchor="middle")

    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_fatality",
        scroll_zoom=False,
        margin=dict(t=0, l=75, r=20, b=65),
    )

    return fig


def get_fatality_charts(theme="light"):
    df = _prepare_overdose_data()
    return {
        "suspected_drug": _build_fatality_chart(
            df, "suspected_drug", theme  # "Suspected Drug vs Fatalities"
        ),
        "cpr_administered": _build_fatality_chart(
            df, "cpr_administered", theme  # "CPR Administered vs Fatalities"
        ),
        "police_ita": _build_fatality_chart(
            df, "police_ita", theme  # "Police ITA vs Fatalities"
        ),
        "disposition": _build_fatality_chart(
            df, "disposition", theme  # "Disposition vs Fatalities"
        ),
        "narcan_given": _build_fatality_chart(
            df, "narcan_given", theme  # "Narcan Given vs Fatalities"
        ),
    }
