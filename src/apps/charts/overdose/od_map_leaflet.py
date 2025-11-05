"""
Leaflet-based interactive overdose hotspot map with choropleth jurisdictions.
Based on https://leafletjs.com/examples/choropleth/
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from django.conf import settings

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

        # Fire District 4
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

        # Load Elwha reservation
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

    # Generate unique map ID to avoid conflicts
    import hashlib

    map_id = f"od_map_{hashlib.md5(jurisdictions_geojson.encode(), usedforsecurity=False).hexdigest()[:8]}"  # nosec B324

    html = f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <div id="{map_id}" style="width: 100%; height: 600px;"></div>

    <script>
    (function() {{
        // Initialize map
        var map = L.map('{map_id}').setView([48.1125, -123.4550], 12);

        // Add Esri NatGeo basemap
        L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: '© Esri, National Geographic',
            maxZoom: 18
        }}).addTo(map);

        // Jurisdictions GeoJSON data
        var jurisdictionsData = {jurisdictions_geojson};

        // OD markers data
        var odMarkersData = {od_markers_json};

        // Style function for jurisdictions
        function style(feature) {{
            return {{
                fillColor: feature.properties.color,
                weight: 2,
                opacity: 1,
                color: feature.properties.color,
                dashArray: '',
                fillOpacity: 0.15
            }};
        }}

        // Highlight feature on hover
        function highlightFeature(e) {{
            var layer = e.target;

            layer.setStyle({{
                weight: 3,
                color: layer.feature.properties.color,
                dashArray: '',
                fillOpacity: 0.2
            }});

            layer.bringToFront();
            info.update(layer.feature.properties);
        }}

        // Reset highlight
        function resetHighlight(e) {{
            geojson.resetStyle(e.target);
            info.update();
        }}

        // Zoom to feature on click
        function zoomToFeature(e) {{
            map.fitBounds(e.target.getBounds());
        }}

        // Attach event listeners
        function onEachFeature(feature, layer) {{
            layer.on({{
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: zoomToFeature
            }});
        }}

        // Add jurisdictions layer
        var geojson = L.geoJson(jurisdictionsData, {{
            style: style,
            onEachFeature: onEachFeature
        }}).addTo(map);

        // Custom info control
        var info = L.control();

        info.onAdd = function (map) {{
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        }};

        info.update = function (props) {{
            this._div.innerHTML = '<h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">Service Area</h4>' +
                (props ? '<b>' + props.name + '</b>' : 'Hover over a jurisdiction');
        }};

        info.addTo(map);

        // Add OD markers
        odMarkersData.forEach(function(marker) {{
            var isFatal = marker.fatal_count > 0;
            var color = isFatal ? '#dc2626' : '#f97316';
            var radius = 4 + (marker.total_count * 2);

            L.circleMarker([marker.lat, marker.lon], {{
                radius: radius,
                fillColor: color,
                color: '#fff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.7
            }}).bindPopup(
                '<b>Overdose Location</b><br>' +
                'Total: ' + marker.total_count + '<br>' +
                'Fatal: ' + marker.fatal_count + '<br>' +
                'Non-Fatal: ' + (marker.total_count - marker.fatal_count)
            ).addTo(map);
        }});

        // Legend control
        var legend = L.control({{position: 'topright'}});

        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'info legend');
            var jurisdictions = [
                {{name: 'City of Port Angeles', color: '{CITY_COLOR}'}},
                {{name: 'Fire District 2', color: '{FD2_COLOR}'}},
                {{name: 'Fire District 4', color: '{FD4_COLOR}'}},
                {{name: 'Lower Elwha Reservation', color: '{ELWHA_COLOR}'}}
            ];

            div.innerHTML = '<h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">Jurisdictions</h4>';

            jurisdictions.forEach(function(item) {{
                div.innerHTML +=
                    '<i style="background:' + item.color + '; width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; border: 1px solid ' + item.color + ';"></i> ' +
                    '<span style="font-size: 12px;">' + item.name + '</span><br style="clear: both;">';
            }});

            return div;
        }};

        legend.addTo(map);

        // Add custom CSS for info and legend
        var style = document.createElement('style');
        style.innerHTML = `
            .info {{
                padding: 10px 14px;
                font: 12px/1.5 Arial, sans-serif;
                background: {info_bg};
                color: {info_color};
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                border: 1px solid rgba(0,0,0,0.2);
            }}
            .legend {{
                padding: 10px 14px;
                font: 12px/1.5 Arial, sans-serif;
                background: {legend_bg};
                color: {legend_color};
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                border: 1px solid rgba(0,0,0,0.2);
                line-height: 24px;
            }}
            .legend i {{
                border-radius: 3px;
            }}
        `;
        document.head.appendChild(style);
    }})();
    </script>
    """

    return html
