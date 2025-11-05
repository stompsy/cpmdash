import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
from django.conf import settings
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT
from utils.plotly import style_plotly_layout

from ...core.models import ODReferrals

# Paths to geographic data
DATA_DIR = Path(settings.BASE_DIR) / "src/static/data"
CITY_BOUNDARY_PATH = (
    Path(settings.BASE_DIR) / "staticfiles/data/port_angeles_outer_boundary.geojson"
)
FIRE_DISTRICTS_PATH = DATA_DIR / "fire_districts.geojson"
AIANNH_PATH = DATA_DIR / "tl_2025_us_aiannh.shp"

# Colors from CHART_COLORS_VIBRANT for consistent district shading
CITY_COLOR = CHART_COLORS_VIBRANT[0]  # #8b5cf6 - Violet
FD2_COLOR = CHART_COLORS_VIBRANT[2]  # #f43f5e - Rose
FD4_COLOR = CHART_COLORS_VIBRANT[4]  # #f59e0b - Amber
ELWHA_COLOR = CHART_COLORS_VIBRANT[1]  # #06b6d4 - Cyan

# Opacity for filled areas
FILL_OPACITY = 0.4
HOVER_OPACITY = 0.7


def _load_boundary_layers():
    """
    Load all geographic boundaries and convert to GeoJSON for Plotly mapbox layers.

    Each jurisdiction gets TWO layers:
    1. A filled polygon with low opacity (25%) for shading
    2. A line outline for clear borders

    Colors from CHART_COLORS_VIBRANT palette for visual distinction.
    All layers render below OD marker traces so markers remain visible on top.
    """
    layers = []

    try:
        # Load city boundary
        city_geojson = json.loads(CITY_BOUNDARY_PATH.read_text(encoding="utf-8"))

        # Add fill layer (shading)
        layers.append(
            {
                "source": city_geojson,
                "type": "fill",
                "below": "traces",
                "color": CITY_COLOR,
                "opacity": FILL_OPACITY,
                "visible": True,
            }
        )
        # Add line layer (border)
        layers.append(
            {
                "source": city_geojson,
                "type": "line",
                "below": "traces",
                "color": CITY_COLOR,
                "line": {"width": 2.5},
                "visible": True,
            }
        )

        # Load fire districts and reproject to WGS84 (EPSG:4326)
        # Fire districts are in EPSG:2926 (WA State Plane) and need conversion for Plotly
        fire_gdf = gpd.read_file(FIRE_DISTRICTS_PATH)
        fire_gdf = fire_gdf.to_crs(epsg=4326)  # Reproject to lat/lon

        # Fire District 2 (rose)
        fd2 = fire_gdf[fire_gdf["LABEL"] == "Fire District 2"]
        if not fd2.empty:
            fd2_geojson = json.loads(fd2.to_json())
            # Fill layer
            layers.append(
                {
                    "source": fd2_geojson,
                    "type": "fill",
                    "below": "traces",
                    "color": FD2_COLOR,
                    "opacity": FILL_OPACITY,
                    "visible": True,
                }
            )
            # Line layer
            layers.append(
                {
                    "source": fd2_geojson,
                    "type": "line",
                    "below": "traces",
                    "color": FD2_COLOR,
                    "line": {"width": 2.0},
                    "visible": True,
                }
            )

        # Fire District 4 (amber)
        fd4 = fire_gdf[fire_gdf["LABEL"] == "Fire District 4"]
        if not fd4.empty:
            fd4_geojson = json.loads(fd4.to_json())
            # Fill layer
            layers.append(
                {
                    "source": fd4_geojson,
                    "type": "fill",
                    "below": "traces",
                    "color": FD4_COLOR,
                    "opacity": FILL_OPACITY,
                    "visible": True,
                }
            )
            # Line layer
            layers.append(
                {
                    "source": fd4_geojson,
                    "type": "line",
                    "below": "traces",
                    "color": FD4_COLOR,
                    "line": {"width": 2.0},
                    "visible": True,
                }
            )

        # Load Lower Elwha Klallam Reservation (cyan)
        # Reproject to WGS84 for consistency with other layers
        aiannh = gpd.read_file(AIANNH_PATH)
        elwha = aiannh[aiannh["NAME"].str.contains("Lower Elwha", case=False, na=False)]
        if not elwha.empty:
            # Dissolve multiple records if needed and convert to GeoJSON
            elwha_dissolved = elwha.dissolve()
            elwha_dissolved = elwha_dissolved.to_crs(epsg=4326)  # Reproject to lat/lon
            elwha_geojson = json.loads(elwha_dissolved.to_json())
            # Fill layer
            layers.append(
                {
                    "source": elwha_geojson,
                    "type": "fill",
                    "below": "traces",
                    "color": ELWHA_COLOR,
                    "opacity": FILL_OPACITY,
                    "visible": True,
                }
            )
            # Line layer
            layers.append(
                {
                    "source": elwha_geojson,
                    "type": "line",
                    "below": "traces",
                    "color": ELWHA_COLOR,
                    "line": {"width": 2.0},
                    "visible": True,
                }
            )

    except Exception as e:
        # If boundary data fails to load, log it but don't crash the chart
        print(f"Warning: Could not load boundary layers: {e}")

    return layers


def _get_basemap_config():
    """
    Get basemap configuration for Plotly mapbox.
    Uses Esri NatGeoWorldMap as the base tile layer.
    """
    # Esri NatGeoWorldMap - beautiful terrain and geography styling
    # No API key required for Esri's public tile services
    return {
        "use_custom_tiles": True,
        "tile_url": "https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri — National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC",
    }


def build_chart_od_map(theme):
    # Dataframe
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        list(
            odreferrals.values(
                "disposition",
                "od_date",
                "long",
                "lat",
            )
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

    # Get basemap configuration
    basemap_config = _get_basemap_config()

    # Create initial figure with white background (tiles will be added as layers)
    fig = px.scatter_mapbox(
        location_counts,
        lat="lat",
        lon="long",
        size="count",
        size_max=25,  # max size of bubble
        color="overdose_outcome",
        color_continuous_scale="Reds",
        zoom=11.5,  # type: ignore
        mapbox_style="white-bg",  # Blank canvas for custom tiles
        title=None,
        hover_data={"count": True, "lat": False, "long": False},
    )

    # Load boundary layers
    boundary_layers = _load_boundary_layers()

    # Build layer stack: Esri tiles (bottom) -> boundaries -> OD markers (top)
    # The key is proper z-ordering:
    # 1. Raster tiles render below everything
    # 2. Boundaries render above tiles but below scatter traces
    # 3. OD markers (traces) render on top

    all_layers = []

    # Add Esri NatGeoWorldMap as the base tile layer (deepest layer)
    if basemap_config["use_custom_tiles"]:
        all_layers.append(
            {
                "below": "traces",  # Below everything
                "sourcetype": "raster",
                "sourceattribution": basemap_config["attribution"],
                "source": [basemap_config["tile_url"]],
            }
        )

    # Add boundary layers - they'll render above the raster but below traces
    # We need to ensure they're visible by NOT using "below" at all, or by using "below: traces"
    # but since the raster already has that, we need a different approach
    all_layers.extend(boundary_layers)

    # Configure mapbox with all layers
    mapbox_dict = {
        "style": "white-bg",  # Blank canvas so our custom tiles show
        "center": {"lat": 48.1125, "lon": -123.4550},
        "zoom": 11.5,
        "layers": all_layers,
    }

    fig.update_layout(mapbox=mapbox_dict, legend_title_text=None)

    fig.update_traces(
        hovertemplate=("Location: %{lat:.2f}, %{lon:.2f}<br>Overdose Count: %{marker.size}<br>")
    )

    # Add custom legend for jurisdictions in upper right
    # Create legend as an annotation box with colored squares for each jurisdiction
    legend_text = (
        "<b>Jurisdictions</b><br>"
        f'<span style="color:{CITY_COLOR}">■</span> City of Port Angeles<br>'
        f'<span style="color:{FD2_COLOR}">■</span> Fire District 2<br>'
        f'<span style="color:{FD4_COLOR}">■</span> Fire District 4<br>'
        f'<span style="color:{ELWHA_COLOR}">■</span> Lower Elwha Reservation'
    )

    fig.add_annotation(
        text=legend_text,
        xref="paper",
        yref="paper",
        x=0.98,  # Upper right
        y=0.98,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)" if theme == "light" else "rgba(30, 41, 59, 0.9)",
        bordercolor="rgba(0, 0, 0, 0.2)" if theme == "light" else "rgba(255, 255, 255, 0.2)",
        borderwidth=1,
        borderpad=8,
        font=dict(size=11, color="black" if theme == "light" else "white"),
        align="left",
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        scroll_zoom=True,
        margin={"r": 20, "t": 0, "l": 20, "b": 20},
    )
    return plot(fig, output_type="div", config=fig._config)
