import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


def build_chart_od_density_heatmap(theme):
    
    # Dataframe
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        odreferrals.values(
            "od_date",
        )
    )

    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df["od_date__hour"] = df["od_date"].dt.hour
    df["od_date__day_of_week_full"] = df["od_date"].dt.day_name()
    df["od_date__day_of_week"] = df["od_date__day_of_week_full"].str[:3]

    # Order days
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["od_date__day_of_week"] = pd.Categorical(
        df["od_date__day_of_week"], categories=days_order, ordered=True
    )

    # Pivot table
    pivot = df.pivot_table(
        index="od_date__day_of_week",
        columns="od_date__hour",
        aggfunc="size",
        fill_value=0,
    )

    # Lookup full names for hover
    day_lookup = (
        df.drop_duplicates("od_date__day_of_week")[
            ["od_date__day_of_week", "od_date__day_of_week_full"]
        ]
        .set_index("od_date__day_of_week")["od_date__day_of_week_full"]
        .to_dict()
    )

    # Custom hover text
    hover_text = [
        [
            f"Hour: {col}:00<br>Day: {day_lookup[row]}<br>Count: {pivot.loc[row, col]}"
            for col in pivot.columns
        ]
        for row in pivot.index
    ]
    
    light_palette = px.colors.sequential.Viridis   # matter 
    
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            text=hover_text,
            hoverinfo="text",
            colorscale=light_palette,
        )
    )
    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        tickformat="02d", # pad with zero
        title="Hour of Day",
    )
    fig.update_traces(
        colorbar=dict(
            title=None,
            orientation="h",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            len=0.5,
            thickness=15,
            tickfont=dict(size=12, color="white"),
            outlinewidth=1, # border line around the colorbar
            outlinecolor="lightgray",
        )
    )
    fig.add_shape(
        type="rect",
        x0=7.5,
        x1=15.5, # Working hours
        y0=-0.5,
        y1=4.5, # Covers all weekdays (Monday to Sunday)
        line=dict(color="white", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)", # Transparent fill
    )
    fig.add_annotation(
        x=11.7,
        y=4.2,
        text="<b>Working Hours</b>",
        font=dict(
            size=16,
            color="white",
            family="Arial, sans-serif",
        ),
        xanchor="center",
        yanchor="bottom",
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_heatmap_density",
        scroll_zoom=False,
        x_title="Hour of Day",
        y_title="Day of Week",
        margin=dict(t=0, l=75, r=20, b=65),
    )
    return plot(fig, output_type="div", config=fig._config)
