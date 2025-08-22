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
    df = pd.DataFrame.from_records(odreferrals.values("disposition", "od_date", "suspected_drug"))
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date"])

    df["overdose_outcome"] = df["disposition"].apply(
        lambda x: "Fatal" if x in ["CPR attempted", "DOA"] else "Non-Fatal"
    )

    # Create monthly aggregations
    df["month"] = df["od_date"].dt.to_period("M")
    monthly_totals = df.groupby("month").size()

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
    )

    # ensure labels are always horizontal
    fig.update_traces(
        xbins_size="M1",
        hovertemplate="Month: %{x|%m/%Y}<br>Overdose Count: %{y}<extra></extra>",
        texttemplate="%{y}",
        textposition="inside",
        insidetextanchor="middle",
        textangle=0,  # prevent automatic rotation
    )

    # Add drug type indicators as vertical lines (mini bar charts)
    # First, find the top 3 most common drugs across all data
    top_drugs = df["suspected_drug"].value_counts().head(3).index.tolist()

    # Use more distinct, vibrant colors for better visibility
    drug_colors = {
        "Fentanyl": TAILWIND_COLORS["purple-600"],
        "Heroin": TAILWIND_COLORS["orange-600"],
        "Cocaine": TAILWIND_COLORS["emerald-600"],
        "Methamphetamine": TAILWIND_COLORS["blue-600"],
        "Opiate/opioid (Unknown)": TAILWIND_COLORS["pink-600"],
        "Fentanyl, Stimulant (Unknown)": TAILWIND_COLORS["cyan-600"],
        "Other": TAILWIND_COLORS["amber-600"],
        "Unknown": TAILWIND_COLORS["violet-600"],
    }

    # Create vertical line charts above each month's bar
    drug_list = top_drugs[:3]  # Use only top 3 drugs
    bar_width_days = 15  # Approximate width of monthly bar in days
    line_width = bar_width_days / 3  # Each line gets 1/3 of the bar width

    for i, drug in enumerate(drug_list):
        drug_data = drug_counts[drug_counts["suspected_drug"] == drug]
        if not drug_data.empty:
            x_positions = []
            y_starts = []
            hover_data = []

            for _, row in drug_data.iterrows():
                month_total = monthly_totals.get(row["month"], 0)
                drug_count = row["count"]

                # Calculate proportional line height (max 20% of month total)
                max_line_height = max(month_total * 0.2, 3)  # At least 3 units tall
                line_height = (drug_count / month_total) * max_line_height if month_total > 0 else 1

                # Position line horizontally within the month (centered, with equal spacing)
                # Create 3 equal sections across the bar width, shifted right to center over month
                section_start = -bar_width_days / 2  # Start from left edge
                x_offset = (
                    section_start + (i * line_width) + (line_width / 2)
                )  # Center within section
                # Add offset to shift right and better center over the monthly bar
                x_pos = row["month_date"] + pd.Timedelta(days=x_offset + 15)  # Shift 15 days right

                y_start = month_total + 1  # Start just above the bar
                y_end = y_start + line_height

                x_positions.extend([x_pos, x_pos, None])  # None creates line break
                y_starts.extend([y_start, y_end, None])
                hover_data.append(drug_count)

            # Add vertical lines as a single trace
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_starts,
                    mode="lines",
                    name=f"{drug}",
                    line=dict(
                        color=drug_colors.get(
                            drug, TAILWIND_COLORS["amber-600"]
                        ),  # More distinct fallback
                        width=10,  # Thicker lines to better fill the sections
                    ),
                    hovertemplate=f"Drug: <b>{drug}</b><br>Month: %{{x|%m/%Y}}<br>Cases: %{{customdata}}<extra></extra>",
                    customdata=[
                        val for val in hover_data for _ in range(3)
                    ],  # Repeat for line segments
                    showlegend=True,
                    connectgaps=False,
                )
            )

    # add daily markers (existing code)
    colors = {"Fatal": TAILWIND_COLORS["red-500"], "Non-Fatal": TAILWIND_COLORS["indigo-900"]}
    offsets = {"Fatal": 0.5, "Non-Fatal": -0.5}
    for outcome in ["Fatal", "Non-Fatal"]:
        df_o = daily_counts[daily_counts["overdose_outcome"] == outcome]
        y_positions = df_o["count"] + offsets[outcome]
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_o["od_date"],
                y=y_positions,
                name=f"Daily Count ({outcome})",
                marker=dict(
                    size=8,
                    color=colors[outcome],
                    line=dict(width=1, color=TAILWIND_COLORS["slate-800"]),
                ),
                hovertemplate=(
                    "Outcome: <b>%{customdata[0]}</b><br>"
                    "Date: %{x|%m/%d/%Y}<br>"
                    "Count: %{y}<extra></extra>"
                ),
                customdata=df_o[["overdose_outcome"]].values,
                showlegend=True,  # Show in legend
            )
        )

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
            bgcolor=TAILWIND_COLORS["slate-800"],  # dark background
            activecolor=TAILWIND_COLORS["slate-600"],  # slightly lighter when active
            font=dict(color=TAILWIND_COLORS["slate-100"]),  # light text
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
        height=600,
        scroll_zoom=False,
        x_title="Date range selector",
        y_title=None,  # Remove y-axis title
        margin=dict(
            t=30, l=20, r=30, b=55
        ),  # Match daily totals style with space for custom elements
        hovermode_unified=False,
    )

    # Override some styles to make it look more like matplotlib - no title
    fig.update_layout(
        title=None,  # Remove title
        hovermode="closest",
        plot_bgcolor="rgba(255,255,255,0.2)"
        if theme == "light"
        else TAILWIND_COLORS["slate-800"],  # Match container bg
        paper_bgcolor="rgba(255,255,255,0.2)"
        if theme == "light"
        else TAILWIND_COLORS["slate-800"],  # Match container bg
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
