import pandas as pd
import plotly.express as px
from plotly.offline import plot

from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals


def build_chart_od_work_hours(theme):
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "od_date",
        )
    )

    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df["hour"] = df["od_date"].dt.hour

    # Define working hours (08:00 to 15:59)
    df["during_work_hours"] = df["hour"].between(8, 15)

    # Count overdoses during and outside working hours
    work_hours_count = (
        df["during_work_hours"]
        .value_counts()
        .rename(index={True: "During Work Hours", False: "Outside Work Hours"})
    )

    # Work vs. non-work hours comparison
    work_hour_df = work_hours_count.reset_index()
    work_hour_df.columns = ["Time Category", "Overdose Count"]

    fig = px.bar(
        work_hour_df,
        x="Time Category",
        y="Overdose Count",
        color="Time Category",
        text="Overdose Count",
        color_discrete_map={
            "During Work Hours": "#1f77b4",
            "Outside Work Hours": "#ff7f0e",
        },
    )

    # Apply theme styling
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title=None,
        y_title="Overdose Count",  # Remove y-axis title
        margin=dict(
            t=0, l=75, r=20, b=65
        ),  # Match daily totals style with space for custom elements
        hovermode_unified=False,
    )

    # Override some styles to make it look more like matplotlib - no title
    fig.update_layout(
        title=None,  # Remove title
        hovermode="closest",
        plot_bgcolor="rgba(255,255,255,0.2)"
        if theme == "light"
        else "rgba(31,41,55,0.2)",  # Match container bg
        paper_bgcolor="rgba(255,255,255,0.2)"
        if theme == "light"
        else "rgba(31,41,55,0.2)",  # Match container bg
    )

    chart_config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "staticPlot": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "overdose_histogram_monthly",
            "height": 600,
            "width": 1200,
            "scale": 2,
        },
    }

    return plot(fig, output_type="div", config=chart_config)
