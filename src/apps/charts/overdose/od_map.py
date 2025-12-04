"""
Leaflet-based interactive overdose hotspot map with choropleth jurisdictions.
Based on https://leafletjs.com/examples/choropleth/
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from django.conf import settings
from django.template import Context, Template

from utils.chart_colors import CHART_COLORS_VIBRANT

from ...core.models import ODReferrals

# Paths to geographic data
DATA_DIR = Path(settings.BASE_DIR) / "src/static/data"
CITY_BOUNDARY_PATH = (
    Path(settings.BASE_DIR) / "staticfiles/data/port_angeles_outer_boundary.geojson"
)
FIRE_DISTRICTS_PATH = DATA_DIR / "fire_districts.geojson"
ELWHA_PATH = DATA_DIR / "lower_elwha_reservation.geojson"

# Colors from CHART_COLORS_VIBRANT
CITY_COLOR = CHART_COLORS_VIBRANT[0]  # #8b5cf6 - Violet
FD2_COLOR = CHART_COLORS_VIBRANT[2]  # #f43f5e - Rose
FD4_COLOR = CHART_COLORS_VIBRANT[4]  # #f59e0b - Amber
ELWHA_COLOR = CHART_COLORS_VIBRANT[1]  # #06b6d4 - Cyan

OD_MAP_TEMPLATE = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<div id="{{ map_id }}" class="w-full h-[600px]"></div>

<script>
(function() {
    function initMap() {
        if (typeof L === 'undefined') {
            setTimeout(initMap, 50);
            return;
        }

    // Cache buster: {{ timestamp }}
    // Initialize map with authentication-aware zoom behavior
    var isRestricted = {% if zoom_mode == "restricted" %}true{% else %}false{% endif %};

    var mapOptions = {};
    if (isRestricted) {
        mapOptions = {
            // United States-ish bounds: roughly continental US
            maxBounds: L.latLngBounds(
                L.latLng(24.396308, -124.848974), // Southwest (Hawaii/AK excluded on purpose)
                L.latLng(49.384358, -66.885444)   // Northeast
            ),
            minZoom: 10,  // Zoomed out to national view
            maxZoom: 12, // Restricted to city/neighborhood level to protect privacy
        };
    }

    var map = L.map('{{ map_id }}', mapOptions).setView([48.1125, -123.4550], isRestricted ? 8 : 12);

    // Add Esri NatGeo basemap
    L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri, National Geographic',
        maxZoom: isRestricted ? 13 : 18
    }).addTo(map);

    // Jurisdictions GeoJSON data
    var jurisdictionsData = {{ jurisdictions_geojson|safe }};

    // OD markers data
    var odMarkersData = {{ od_markers_json|safe }};

    // Style function for jurisdictions
    function style(feature) {
        return {
            fillColor: feature.properties.color,
            weight: 2,
            opacity: 1,
            color: feature.properties.color,
            dashArray: '',
            fillOpacity: 0.15  // Increased for better visibility
        };
    }

    // Highlight feature on hover
    function highlightFeature(e) {
        var layer = e.target;

        layer.setStyle({
            weight: 3,
            color: layer.feature.properties.color,
            dashArray: '',
            fillOpacity: 0.3
        });

        // Don't call bringToFront() to keep OD markers on top
        // layer.bringToFront();
        info.update({
            type: 'jurisdiction',
            name: layer.feature.properties.name
        });
    }

    // Reset highlight
    function resetHighlight(e) {
        geojson.resetStyle(e.target);
        info.update();
    }

    // Zoom to feature on click
    function zoomToFeature(e) {
        map.fitBounds(e.target.getBounds());
    }

    // Attach event listeners
    function onEachFeature(feature, layer) {
        layer.on({
            mouseover: highlightFeature,
            mouseout: resetHighlight,
            click: zoomToFeature
        });
    }

    // Add jurisdictions layer
    var geojson = L.geoJson(jurisdictionsData, {
        style: style,
        onEachFeature: onEachFeature
    }).addTo(map);

    // Legend control (at top right, added first so it appears on top)
    var legend = L.control({position: 'topright'});

    legend.onAdd = function (map) {
        var div = L.DomUtil.create('div', 'info legend w-60');
        var jurisdictions = [
            {name: 'City of Port Angeles', color: '{{ CITY_COLOR }}'},
            {name: 'Fire District 2', color: '{{ FD2_COLOR }}'},
            {name: 'Fire District 4', color: '{{ FD4_COLOR }}'},
            {name: 'Lower Elwha Reservation', color: '{{ ELWHA_COLOR }}'}
        ];

        div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Jurisdictions</h4>';

        jurisdictions.forEach(function(item) {
            div.innerHTML +=
                '<i class="w-4.5 h-4.5 float-left mr-2 opacity-70 border" style="background:' + item.color + '; border-color:' + item.color + ';"></i> ' +
                '<span class="text-xs">' + item.name + '</span><br class="clear-both">';
        });

        return div;
    };

    legend.addTo(map);

    // Custom info control (below legend at top right)
    var info = L.control({position: 'topright'});

    info.onAdd = function (map) {
        this._div = L.DomUtil.create('div', 'info w-60 mt-2.5');
        this.update();
        return this._div;
    };

    info.update = function (data) {
        if (!data) {
            // Default state
            this._div.innerHTML = ' <h4 class="m-0 mb-2 text-sm font-semibold">Map Info</h4>' +
                '<span class="text-slate-400 text-xs">Hover over a jurisdiction or overdose marker</span>';
        } else if (data.type === 'jurisdiction') {
            // Jurisdiction info
            this._div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Service Area</h4>' +
                '<b class="text-13">' + data.name + '</b>';
        } else if (data.type === 'overdose') {
            // Overdose marker info
            this._div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Overdose Location</h4>' +
                '<b class="text-13">Total Cases: ' + data.total_count + '</b><br>' +
                '<b class="text-13">Fatal: ' + data.fatal_count + ' | Non-Fatal: ' + (data.total_count - data.fatal_count) + '</b>';
        }
    };

    info.addTo(map);

    // Create a custom pane for OD markers to ensure they're always on top
    map.createPane('markerPane');
    map.getPane('markerPane').style.zIndex = 650; // Higher than overlays (400) and tooltips (600)

    // Add OD markers to the custom pane
    odMarkersData.forEach(function(marker) {
        var isFatal = marker.fatal_count > 0;
        var color = isFatal ? '#dc2626' : '#f97316';
        var radius = 4 + (marker.total_count * 2);

        var circleMarker = L.circleMarker([marker.lat, marker.lon], {
            radius: radius,
            fillColor: color,
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.7,
            pane: 'markerPane', // Use custom pane for proper z-index
            interactive: true // Ensure markers are clickable
        });

        // Add hover events to update info control
        circleMarker.on('mouseover', function(e) {
            info.update({
                type: 'overdose',
                total_count: marker.total_count,
                fatal_count: marker.fatal_count
            });
            // Highlight marker on hover
            this.setStyle({
                weight: 2,
                fillOpacity: 0.9
            });
        });

        circleMarker.on('mouseout', function(e) {
            info.update();
            // Reset marker style
            this.setStyle({
                weight: 1,
                fillOpacity: 0.7
            });
        });

        circleMarker.addTo(map);
    });

    // Add custom CSS for info and legend
    var style = document.createElement('style');
    style.innerHTML = `
        .info {
            padding: 10px 14px;
            font: 12px/1.5 Arial, sans-serif;
            background: {{ info_bg }};
            color: {{ info_color }};
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
            border: 1px solid rgba(0,0,0,0.2);
        }
        .legend {
            padding: 10px 14px;
            font: 12px/1.5 Arial, sans-serif;
            background: {{ legend_bg }};
            color: {{ legend_color }};
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
            border: 1px solid rgba(0,0,0,0.2);
            line-height: 24px;
        }
        .legend i {
            border-radius: 3px;
        }
    `;
    document.head.appendChild(style);
    }
    initMap();
})();
</script>
"""


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
        elwha_geojson = json.loads(ELWHA_PATH.read_text(encoding="utf-8"))
        for feature in elwha_geojson.get("features", []):
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


def build_chart_od_map(theme, zoom_mode="full"):
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

    # Render the HTML/JS from the template string
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
        "zoom_mode": zoom_mode,
    }

    html = Template(OD_MAP_TEMPLATE).render(Context(context))

    return html
