import pandas as pd
import plotly.express as px
from plotly.offline import plot
import json
import os
from django.conf import settings

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


# Dataframe
odreferrals = ODReferrals.objects.all()
df = pd.DataFrame.from_records(
    odreferrals.values(
        "disposition",
        "od_date",
        "long",
        "lat",
    )
)

# Classify overdoses as Fatal or Non-Fatal
fatal_conditions = ["CPR attempted", "DOA"]
df["overdose_outcome"] = df["disposition"].apply(
    lambda x: "Fatal" if x in fatal_conditions else "Non-Fatal"
)

df["count"] = 1  # each row = 1 overdose case

# Aggregate data to adjust bubble size for repeated locations
location_counts = (
    df.groupby(["lat", "long", "overdose_outcome"]).size().reset_index(name="count")
)


def build_chart_od_map(theme):
    fig = px.scatter_mapbox(
        location_counts,
        lat="lat",
        lon="long",
        size="count",
        size_max=25,  # max size of bubble
        color="overdose_outcome",
        color_continuous_scale="Reds",
        zoom=10,
        mapbox_style="open-street-map",
        title=None,
        hover_data={"count": True, "lat": False, "long": False},
    )

    # Center map on Port Angeles, WA
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=48.1181, lon=-123.4550),  # Port Angeles, WA
        ),
        legend_title_text=None,
    )

    fig.update_traces(
        hovertemplate=(
            "Location: %{lat:.2f}, %{lon:.2f}<br>" "Overdose Count: %{marker.size}<br>"
        )
    )

    # TODO: Check and see if City of Port Angeles boundary matches?
    # Load Port Angeles boundary GeoJSON
    boundary_path = os.path.join(settings.BASE_DIR, "staticfiles", "src", "data", "port_angeles_outer_boundary.geojson")
    with open(boundary_path) as f:
        pa_boundary = json.load(f)

    # Overlay city boundary clearly on the map
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": pa_boundary,
                    "type": "line",
                    "color": "green",
                    "line": {"width": 1},
                }
            ],
        },
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_od_map",
        scroll_zoom=True,
        margin={"r": 20, "t": 0, "l": 20, "b": 20},
    )
    return plot(fig, output_type="div", config=fig._config)
