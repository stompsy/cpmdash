from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from ...core.models import ODReferrals


def build_chart_od_hist_monthly(theme):
    # Get data including suspected drug information
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        list(odreferrals.values("disposition", "od_date", "suspected_drug"))
    )
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date"])

    # Exclude Feb 2024 (no data available until March)
    df = df[~((df["od_date"].dt.year == 2024) & (df["od_date"].dt.month == 2))]

    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if x in ["CPR attempted", "DOA"] else "Non-Fatal"
    )

    # Create monthly aggregations
    df["month"] = df["od_date"].dt.tz_localize(None).dt.to_period("M")

    # Get drug type data by month
    drug_counts = df.groupby(["month", "suspected_drug"]).size().reset_index(name="count")
    drug_counts["month_date"] = drug_counts["month"].dt.to_timestamp()

    daily_counts = df.groupby(["od_date", "overdose_outcome"]).size().reset_index(name="count")

    # stacked monthly histogram
    fig = px.histogram(
        df,
        x="od_date",
        color="overdose_outcome",
        barmode="stack",
        histfunc="count",
        color_discrete_map={
            "Fatal": TAILWIND_COLORS["red-500"],
            "Non-Fatal": TAILWIND_COLORS["indigo-600"],
        },
        template=None,  # Bypass Plotly template system to avoid marker pattern bug
    )

    # Show individual segment counts INSIDE the bars
    fig.update_traces(
        xbins_size="M1",
        hovertemplate="Month: %{x|%m/%Y}<br>Overdose Count: %{y}<extra></extra>",
        texttemplate="%{y}",  # Show the count for this segment
        textposition="inside",  # Place text inside the bar segment
        textangle=0,  # Keep text horizontal
        insidetextanchor="end",  # Center the text within the segment
        marker=dict(line=dict(width=0)),  # Remove bar borders
    )

    # add daily markers
    colors = {"Fatal": TAILWIND_COLORS["red-500"], "Non-Fatal": TAILWIND_COLORS["indigo-900"]}
    offsets = {"Fatal": 0.1, "Non-Fatal": -0.5}
    for outcome in ["Fatal", "Non-Fatal"]:
        df_o = daily_counts[daily_counts["overdose_outcome"] == outcome]
        y_positions = df_o["count"] + offsets[outcome]
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_o["od_date"],
                y=y_positions,
                marker=dict(
                    size=8,
                    color=colors[outcome],
                    line=dict(width=1, color=TAILWIND_COLORS["slate-800"]),
                ),
                hovertemplate=(
                    "Outcome: <b>%{customdata[0]}</b><br>Date: %{x|%m/%d/%Y}<extra></extra>"
                ),
                customdata=df_o[["overdose_outcome"]].values,
                showlegend=False,  # Show in legend
            )
        )

    fig.update_layout(
        bargap=0.1,
    )

    # Apply theme styling
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=600,
        scroll_zoom=False,
        x_title="Date range selector",
        y_title=None,  # Remove y-axis title
        margin=dict(t=30, l=0, r=5, b=40),
        hovermode_unified=False,
    )

    # Calculate date range for trailing 12 months (one year ago to today)
    now = datetime.now()
    one_year_ago = (now - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")

    # style x-axis & rangeselector (AFTER style_plotly_layout to override grid settings)
    fig.update_xaxes(
        showgrid=False,  # Remove vertical grid lines
        ticklabelmode="period",
        dtick="M1",
        tickformat="%b\n%Y",
        rangemode="tozero",  # Start range at 0
        rangeslider_visible=True,
        range=[one_year_ago, today],  # Default view: trailing 12 months
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ],
            bgcolor=TAILWIND_COLORS["slate-800"],  # dark background
            activecolor=TAILWIND_COLORS["slate-600"],  # slightly lighter when active
            font=dict(color=TAILWIND_COLORS["slate-100"]),  # light text
            y=1.15,  # Position buttons higher above chart (default is ~1.0)
            yanchor="bottom",  # Anchor to bottom of buttons, so y value pushes them up
        ),
    )

    # style y-axis (AFTER style_plotly_layout to override grid settings)
    fig.update_yaxes(
        showgrid=True,  # Remove horizontal grid lines
        showticklabels=True,
        ticklabelstandoff=10,  # Space between tick labels and axis (~1/8 inch)
        ticklabelposition="outside",
        automargin=True,
        rangemode="tozero",  # Start range at 0
        zeroline=True,
        zerolinecolor="rgba(128,128,128,0.25)",
    )

    # Override some styles to make it look more like matplotlib - no title
    fig.update_layout(
        title=None,  # Remove title
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        modebar=dict(
            orientation="h",  # Horizontal orientation
            # Note: Plotly doesn't support x/y positioning for modebar - it's hardcoded to top-right
            # The modebar will float in its default position when hovering over the chart
        ),
    )

    chart_config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "staticPlot": False,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
            "zoomInGeo",
            "zoomOutGeo",
            "resetGeo",
            "hoverClosestGeo",
            "sendDataToCloud",
            "hoverClosestGl2d",
            "hoverClosestPie",
            "toggleHover",
            "resetViews",
            "resetViewMapbox",
        ],
        "modeBarButtonsToAdd": ["toImage", "resetScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "overdose_histogram_monthly",
            "height": 600,
            "width": 1200,
            "scale": 2,
        },
    }

    return plot(fig, output_type="div", config=chart_config)


def build_chart_top5_drugs_monthly(theme):
    """
    Build a stacked area chart showing top 3 drugs by month
    """
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(list(odreferrals.values("od_date", "suspected_drug")))
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date"])

    if df.empty:
        return "<p>No data available</p>"

    # Create monthly aggregations
    # Cast to Series[datetime] to satisfy mypy, which gets confused otherwise
    od_date_series: pd.Series[datetime] = df["od_date"]
    # Mypy gets confused by the chained .dt accessors. Chaining them directly
    # seems to resolve the false positive.
    df["month"] = od_date_series.dt.tz_localize(None).dt.to_period("M")

    # Find top 5 drugs overall
    top_drugs = df["suspected_drug"].value_counts().head(5).index.tolist()

    # Reorder so "Fentanyl, Stimulant (Unknown)" appears on top of the stack
    # In stacked charts, last trace drawn is on top, so move it to the end
    fentanyl_stimulant = "Fentanyl, Stimulant (Unknown)"
    if fentanyl_stimulant in top_drugs:
        top_drugs.remove(fentanyl_stimulant)
        top_drugs.append(fentanyl_stimulant)  # Add to end so it's drawn last (on top)

    # Get drug counts by month for top 5 drugs
    drug_counts = (
        df[df["suspected_drug"].isin(top_drugs)]
        .groupby(["month", "suspected_drug"])
        .size()
        .reset_index(name="count")
    )
    drug_counts["month_date"] = drug_counts["month"].dt.to_timestamp()

    # Smooth gradient palette - flows naturally from indigo through purple to pink
    # Perfect for stacked areas where adjacent colors blend smoothly
    gradient_palette = [
        "#6366f1",  # Indigo
        "#8b5cf6",  # Violet
        "#a855f7",  # Purple
        "#c026d3",  # Fuchsia
        "#db2777",  # Pink
    ]

    # Create stacked area chart with smooth curves
    fig = go.Figure()

    for i, drug in enumerate(top_drugs):
        drug_data = drug_counts[drug_counts["suspected_drug"] == drug].sort_values("month_date")
        # Use gradient palette colors, cycling if we have more than 5 drugs
        color = gradient_palette[i % len(gradient_palette)]

        fig.add_trace(
            go.Scatter(
                x=drug_data["month_date"],
                y=drug_data["count"],
                name=drug,
                mode="lines",
                stackgroup="one",
                fillcolor=color,
                line=dict(
                    width=1,  # No border creates smooth color transitions
                    shape="spline",  # Smooth curves instead of sharp angles
                    smoothing=1.0,  # Maximum smoothness (range 0-1.3)
                ),
                hovertemplate="<b>%{fullData.name}</b><br>Month: %{x|%b %Y}<br>Count: %{y}<extra></extra>",
            )
        )

    # Apply theme styling
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=300,
        scroll_zoom=False,
        x_title=None,
        y_title="Overdose Count",
        margin=dict(t=20, l=50, r=0, b=40),  # Match margins with OD referrals by month chart
        hovermode_unified=False,
    )

    # Style axes
    fig.update_xaxes(
        showgrid=False,
        tickformat="%b\n%Y",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.25)",
        showticklabels=True,
        ticklabelstandoff=10,
        ticklabelposition="outside",
        automargin=True,
        rangemode="tozero",
    )

    # Transparent background
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    chart_config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",
        "staticPlot": False,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
            "zoomInGeo",
            "zoomOutGeo",
            "resetGeo",
            "hoverClosestGeo",
            "sendDataToCloud",
            "hoverClosestGl2d",
            "hoverClosestPie",
            "toggleHover",
            "resetViews",
            "resetViewMapbox",
        ],
        "modeBarButtonsToAdd": ["toImage", "resetScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "top3_drugs_monthly",
            "height": 300,
            "width": 1200,
            "scale": 2,
        },
    }

    return plot(fig, output_type="div", config=chart_config)
