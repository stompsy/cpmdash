import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from utils.plotly import style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS
from dashboard.models import ODReferrals


def build_chart_od_hist_monthly(theme):
    
    # daily counts
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("disposition", "od_date"))
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if x in ["CPR attempted", "DOA"] else "Non-Fatal"
    )
    daily_counts = (
        df.groupby(["od_date", "overdose_outcome"])
            .size().reset_index(name="count")
    )

    # stacked monthly histogram
    fig = px.histogram(
        df,
        x="od_date",
        color="overdose_outcome",
        barmode="stack",
        histfunc="count",
        color_discrete_map={
            "Fatal": TAILWIND_COLORS["red-500"],
            "Non-Fatal": TAILWIND_COLORS["indigo-600"]
        },
    )

    # ensure labels are always horizontal
    fig.update_traces(
        xbins_size="M1",
        hovertemplate="Month: %{x|%m/%Y}<br>Overdose Count: %{y}<extra></extra>",
        texttemplate="%{y}",
        textposition="inside",
        insidetextanchor="middle",
        textangle=0,   # prevent automatic rotation
    )

    # add daily markers
    colors = {
        "Fatal": TAILWIND_COLORS["red-500"],
        "Non-Fatal": TAILWIND_COLORS["indigo-600"]
    }
    offsets = {
        "Fatal": 0.5,
        "Non-Fatal": -0.5
    }
    for outcome in ["Fatal", "Non-Fatal"]:
        df_o = daily_counts[daily_counts["overdose_outcome"] == outcome]
        y_positions = df_o["count"] + offsets[outcome]
        fig.add_trace(go.Scatter(
            mode="markers",
            x=df_o["od_date"],
            y=y_positions,
            name=f"Daily Count ({outcome})",
            marker=dict(
                size=8,
                color=colors[outcome],
                line=dict(width=1, color=TAILWIND_COLORS["slate-800"])
            ),
            hovertemplate=(
                "Outcome: <b>%{customdata[0]}</b><br>"
                "Date: %{x|%m/%d/%Y}<br>"
                "Count: %{y}<extra></extra>"
            ),
            customdata=df_o[["overdose_outcome"]].values,
            showlegend=False,
        ))

    # style x-axis & rangeselector
    fig.update_xaxes(
        showgrid=True,
        ticklabelmode="period",
        dtick="M1",
        tickformat="%b\n%Y",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ],
            bgcolor=TAILWIND_COLORS["slate-800"],              # dark background
            activecolor=TAILWIND_COLORS["slate-600"],          # slightly lighter when active
            font=dict(color=TAILWIND_COLORS["slate-100"]),     # light text
        ),
    )

    fig.update_layout(
        bargap=0.1,
        # xaxis_range=["2024-03-01", "2024-12-31"],
    )

    # Apply theme styling
    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=False,
        x_title="Date range selector",
        y_title=None,  # Remove y-axis title
        margin=dict(t=0, l=20, r=20, b=55),  # Match daily totals style with space for custom elements
        hovermode_unified=False,
    )

    # Override some styles to make it look more like matplotlib - no title
    fig.update_layout(
        title=None,  # Remove title
        hovermode="closest",
        plot_bgcolor="rgba(255,255,255,0.2)" if theme == "light" else "rgba(31,41,55,0.2)",  # Match container bg
        paper_bgcolor="rgba(255,255,255,0.2)" if theme == "light" else "rgba(31,41,55,0.2)",  # Match container bg
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
            "scale": 2
        }
    }

    return plot(fig, output_type="div", config=chart_config)