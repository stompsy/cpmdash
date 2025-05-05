import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


# Dataframe
odreferrals = ODReferrals.objects.all()
df = pd.DataFrame.from_records(
    odreferrals.values(
        "disposition",
        "od_date",
    )
)

# Convert 'od_date' to datetime and extract features
df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
df["od_date__date"] = df["od_date"].dt.date
df["od_date__hour"] = df["od_date"].dt.hour
df["od_date__month"] = df["od_date"].dt.month
df["od_date__day_of_week"] = df["od_date"].dt.day_name()

# Classify overdoses as Fatal or Non-Fatal
fatal_conditions = ["CPR attempted", "DOA"]
df["overdose_outcome"] = df["disposition"].apply(
    lambda x: "Fatal" if x in fatal_conditions else "Non-Fatal"
)

# Add markers for daily overdose counts
daily_counts = (
    df.groupby(["od_date", "overdose_outcome"]).size().reset_index(name="count")
)

colors = {"Fatal": "red", "Non-Fatal": "#636EFA"}

# Create histogram colored by fatal/non-fatal outcomes
def build_chart_od_hist_monthly(theme):
    fig = px.histogram(
        df,
        x="od_date",
        color="overdose_outcome",
        barmode="stack",
        histfunc="count",
        color_discrete_map={"Fatal": "red", "Non-Fatal": "#636EFA"},
    )
    fig.update_traces(
        xbins_size="M1",
        hovertemplate=("Month: %{x|%m/%Y}<br>" "Overdose Count: %{y}<extra></extra>"),
        texttemplate="%{y}",
        textposition="inside",
        insidetextanchor="middle",
    )
    fig.update_xaxes(
        showgrid=True,
        ticklabelmode="period",
        dtick="M1",
        tickformat="%b\n%Y",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(label="All", step="all"),
                ]
            )
        ),
    )
    for outcome in ["Fatal", "Non-Fatal"]:
        df_outcome = daily_counts[daily_counts["overdose_outcome"] == outcome]
        fig.add_trace(
            go.Scatter(
                mode="markers",
                marker=dict(
                    size=7,
                    color=colors[outcome],
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                x=df_outcome["od_date"],
                y=df_outcome["count"],
                name=f"Daily Count ({outcome})",
                hovertemplate=(
                    "Outcome: <b>%{customdata[0]}</b><br>"
                    "Date: %{x|%m/%d/%Y}<br>"
                    "Overdose Count: %{y}<extra></extra>"
                ),
                customdata=df_outcome[["overdose_outcome"]].values,
                showlegend=False,
            )
        )
    fig.update_layout(
        bargap=0.1,
        xaxis_range=["2024-03-01", "2024-12-31"],
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_hist_monthly",
        scroll_zoom=False,
        x_title="Month",
        y_title="Overdose Count",
        margin=dict(t=0, l=60, r=20, b=55),
    )
    return plot(fig, output_type="div", config=fig._config)
