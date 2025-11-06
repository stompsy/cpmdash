"""
Leaflet-based interactive overdose hotspot map with choropleth jurisdictions.
Based on https://leafletjs.com/examples/choropleth/
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from django.conf import settings
from django.template.loader import render_to_string

from utils.chart_colors import CHART_COLORS_VIBRANT

from ...core.models import ODReferrals

# Paths to geographic data
DATA_DIR = Path(settings.BASE_DIR) / "src/static/data"
CITY_BOUNDARY_PATH = (
    Path(settings.BASE_DIR) / "staticfiles/data/port_angeles_outer_boundary.geojson"
)
FIRE_DISTRICTS_PATH = DATA_DIR / "fire_districts.geojson"
AIANNH_PATH = DATA_DIR / "tl_2025_us_aiannh.shp"

# Colors from CHART_COLORS_VIBRANT
CITY_COLOR = CHART_COLORS_VIBRANT[0]  # #8b5cf6 - Violet
FD2_COLOR = CHART_COLORS_VIBRANT[2]  # #f43f5e - Rose
FD4_COLOR = CHART_COLORS_VIBRANT[4]  # #f59e0b - Amber
ELWHA_COLOR = CHART_COLORS_VIBRANT[1]  # #06b6d4 - Cyan


def _prepare_jurisdictions_geojson():
    """
    Load all jurisdictions and prepare a single GeoJSON with properties for each feature.
    Returns GeoJSON string with name, color, and OD count for each jurisdiction.
    """
    features = []

    try:
        # Load city boundary
        city_geojson = json.loads(CITY_BOUNDARY_PATH.read_text(encoding="utf-8"))
        for feature in city_geojson.get("features", []):
            feature["properties"] = {
                "name": "City of Port Angeles",
                "color": CITY_COLOR,
                "type": "city",
            }
            features.append(feature)

        # Load fire districts and reproject to WGS84
        fire_gdf = gpd.read_file(FIRE_DISTRICTS_PATH)
        fire_gdf = fire_gdf.to_crs(epsg=4326)

        # Fire District 4 (add first so it renders on bottom)
        fd4 = fire_gdf[fire_gdf["LABEL"] == "Fire District 4"]
        if not fd4.empty:
            fd4_json = json.loads(fd4.to_json())
            for feature in fd4_json.get("features", []):
                feature["properties"] = {
                    "name": "Fire District 4",
                    "color": FD4_COLOR,
                    "type": "fire_district",
                }
                features.append(feature)

        # Fire District 2
        fd2 = fire_gdf[fire_gdf["LABEL"] == "Fire District 2"]
        if not fd2.empty:
            fd2_json = json.loads(fd2.to_json())
            for feature in fd2_json.get("features", []):
                feature["properties"] = {
                    "name": "Fire District 2",
                    "color": FD2_COLOR,
                    "type": "fire_district",
                }
                features.append(feature)

        # Load Elwha reservation (add AFTER FD2 so it renders on top)
        aiannh = gpd.read_file(AIANNH_PATH)
        elwha = aiannh[aiannh["NAME"].str.contains("Lower Elwha", case=False, na=False)]
        if not elwha.empty:
            elwha_dissolved = elwha.dissolve()
            elwha_dissolved = elwha_dissolved.to_crs(epsg=4326)
            elwha_json = json.loads(elwha_dissolved.to_json())
            for feature in elwha_json.get("features", []):
                feature["properties"] = {
                    "name": "Lower Elwha Klallam Reservation",
                    "color": ELWHA_COLOR,
                    "type": "reservation",
                }
                features.append(feature)

    except Exception as e:
        print(f"Warning: Could not load jurisdiction boundaries: {e}")

    return json.dumps({"type": "FeatureCollection", "features": features})


def _get_od_markers_data():
    """
    Get overdose location data as JSON array for Leaflet markers.
    """
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
        list(odreferrals.values("disposition", "od_date", "long", "lat"))
    )

    # Return empty array if no data
    if df.empty:
        return json.dumps([])

    # Classify overdoses
    fatal_conditions = ["CPR attempted", "DOA"]
    df["fatal"] = df["disposition"].apply(lambda x: x in fatal_conditions)

    # Aggregate by location
    location_counts = df.groupby(["lat", "long"]).agg({"fatal": ["sum", "count"]}).reset_index()
    location_counts.columns = ["lat", "lon", "fatal_count", "total_count"]

    # Convert to list of dicts
    markers = location_counts.to_dict("records")
    return json.dumps(markers)


def build_chart_od_map(theme):
    """
    Build an interactive Leaflet.js map with choropleth jurisdictions and OD markers.
    Based on https://leafletjs.com/examples/choropleth/
    """
    # Prepare data
    jurisdictions_geojson = _prepare_jurisdictions_geojson()
    od_markers_json = _get_od_markers_data()

    # Determine theme colors
    if theme == "dark":
        info_bg = "rgba(30, 41, 59, 0.95)"
        info_color = "white"
        legend_bg = "rgba(30, 41, 59, 0.9)"
        legend_color = "white"
    else:
        info_bg = "rgba(255, 255, 255, 0.95)"
        info_color = "black"
        legend_bg = "rgba(255, 255, 255, 0.9)"
        legend_color = "black"

    # Generate unique map ID to avoid conflicts and bust cache
    import hashlib
    import time

    timestamp = str(int(time.time()))
    map_id = f"od_map_{hashlib.md5((jurisdictions_geojson + timestamp).encode(), usedforsecurity=False).hexdigest()[:8]}"

    # Render the HTML/JS from a template to avoid constructing a large f-string in Python.
    # Pass JSON strings through directly (they will be marked safe in the template).
    context = {
        "map_id": map_id,
        "timestamp": timestamp,
        "jurisdictions_geojson": jurisdictions_geojson,
        "od_markers_json": od_markers_json,
        "info_bg": info_bg,
        "info_color": info_color,
        "legend_bg": legend_bg,
        "legend_color": legend_color,
        "CITY_COLOR": CITY_COLOR,
        "FD2_COLOR": FD2_COLOR,
        "FD4_COLOR": FD4_COLOR,
        "ELWHA_COLOR": ELWHA_COLOR,
    }

    html = render_to_string("charts/overdose/od_map.html", context)

    return html
