"""
Leaflet-based interactive patient address map for the Referrals page.

Plots every referral, encounter, and OD referral record at the owning patient's
home address (lat/long from the Patients table).  Markers are aggregated by
*location* — ``(lat, lon, source)`` — so that multiple patients sharing the
same coordinates (e.g. a shelter) collapse into a single circle whose area is
proportional to the total record count (sqrt scaling).

A **density contour overlay** computed via scipy KDE + contourpy delineates
high-activity zones with five nested levels (Low → Extreme).  Each contour
polygon has a gradient fill and a glowing border, producing a topographic-map
aesthetic that makes hot-spot boundaries immediately visible.  The contour
layer is toggleable via an Alpine.js button alongside the three source layers.

Architecture mirrors the existing OD hotspot map
(``src/apps/charts/overdose/od_map.py``), reusing the same jurisdiction
GeoJSON layers, Esri basemap, info/legend controls, and authentication-aware
zoom restrictions.
"""

import hashlib
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from contourpy import contour_generator
from django.conf import settings
from django.template import Context, Template
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

from utils.chart_colors import CHART_COLORS_VIBRANT

from ...core.models import Encounters, ODReferrals, Patients, Referrals

# ---------------------------------------------------------------------------
# Geographic data paths – identical to od_map.py
# ---------------------------------------------------------------------------
DATA_DIR = Path(settings.BASE_DIR) / "src/static/data"
CITY_BOUNDARY_PATH = (
    Path(settings.BASE_DIR) / "staticfiles/data/port_angeles_outer_boundary.geojson"
)
FIRE_DISTRICTS_PATH = DATA_DIR / "fire_districts.geojson"
ELWHA_PATH = DATA_DIR / "lower_elwha_reservation.geojson"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
CITY_COLOR = CHART_COLORS_VIBRANT[0]  # #8b5cf6 – Violet
FD2_COLOR = CHART_COLORS_VIBRANT[2]  # #f43f5e – Rose
FD4_COLOR = CHART_COLORS_VIBRANT[4]  # #f59e0b – Amber
ELWHA_COLOR = CHART_COLORS_VIBRANT[1]  # #06b6d4 – Cyan

# Per-source marker colours
REFERRAL_COLOR = "#3b82f6"  # Blue-500
ENCOUNTER_COLOR = "#10b981"  # Emerald-500
OD_REFERRAL_COLOR = "#f43f5e"  # Rose-500

# Density contour colours (Low → Extreme) — same gradient as the old heatmap
# but now with crisp borders and defined shapes
CONTOUR_COLORS = ["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]
CONTOUR_LABELS = ["Low", "Moderate", "High", "Very High", "Extreme"]
CONTOUR_FILL_OPACITIES = [0.08, 0.12, 0.18, 0.25, 0.35]
CONTOUR_BORDER_WEIGHTS = [1.5, 2.0, 2.5, 3.0, 3.5]

# ---------------------------------------------------------------------------
# Leaflet + Alpine.js template
# ---------------------------------------------------------------------------
PATIENT_MAP_TEMPLATE = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

{# --- Alpine.js toggle controls --- #}
<div x-data="{
        showReferrals: true,
        showEncounters: true,
        showOdReferrals: true,
        showHeatmap: true,
        mapReady: false,
        totalVisible: {{ total_records }},
        counts: { referral: {{ referral_total }}, encounter: {{ encounter_total }}, od_referral: {{ od_referral_total }} },
     }"
     x-init="$watch('showReferrals', v => {
         var el = document.getElementById('{{ map_id }}');
         if (el && el._layerGroups) { v ? el._layerGroups.referral.addTo(el._map) : el._map.removeLayer(el._layerGroups.referral); }
         totalVisible = (showReferrals ? counts.referral : 0) + (showEncounters ? counts.encounter : 0) + (showOdReferrals ? counts.od_referral : 0);
     }); $watch('showEncounters', v => {
         var el = document.getElementById('{{ map_id }}');
         if (el && el._layerGroups) { v ? el._layerGroups.encounter.addTo(el._map) : el._map.removeLayer(el._layerGroups.encounter); }
         totalVisible = (showReferrals ? counts.referral : 0) + (showEncounters ? counts.encounter : 0) + (showOdReferrals ? counts.od_referral : 0);
     }); $watch('showOdReferrals', v => {
         var el = document.getElementById('{{ map_id }}');
         if (el && el._layerGroups) { v ? el._layerGroups.od_referral.addTo(el._map) : el._map.removeLayer(el._layerGroups.od_referral); }
         totalVisible = (showReferrals ? counts.referral : 0) + (showEncounters ? counts.encounter : 0) + (showOdReferrals ? counts.od_referral : 0);
     }); $watch('showHeatmap', v => {
         var el = document.getElementById('{{ map_id }}');
         if (el && el._contourLayer) { v ? el._contourLayer.addTo(el._map) : el._map.removeLayer(el._contourLayer); }
     });"
     class="space-y-3">

    {# Toggle bar #}
    <div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <p class="text-xs font-medium text-slate-500 dark:text-slate-400">
            Showing <span class="font-semibold text-slate-700 dark:text-slate-200"
                          x-text="totalVisible.toLocaleString()"></span> records
        </p>
        <div class="inline-flex overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm
                    dark:border-slate-700/60 dark:bg-slate-900/40">
            <button type="button"
                    @click="showReferrals = !showReferrals"
                    :class="showReferrals
                        ? 'bg-blue-500 text-white'
                        : 'text-slate-600 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800/60'"
                    class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold transition-colors">
                <span class="inline-block h-2.5 w-2.5 rounded-full"
                      :class="showReferrals ? 'bg-white' : 'bg-blue-500'"
                      style="flex-shrink:0"></span>
                Referrals
                <span class="opacity-70" x-text="'(' + counts.referral.toLocaleString() + ')'"></span>
            </button>
            <button type="button"
                    @click="showEncounters = !showEncounters"
                    :class="showEncounters
                        ? 'bg-emerald-500 text-white'
                        : 'text-slate-600 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800/60'"
                    class="flex items-center gap-1.5 border-x border-slate-200 px-3 py-1.5 text-xs font-semibold transition-colors
                           dark:border-slate-700/60">
                <span class="inline-block h-2.5 w-2.5 rounded-full"
                      :class="showEncounters ? 'bg-white' : 'bg-emerald-500'"
                      style="flex-shrink:0"></span>
                Encounters
                <span class="opacity-70" x-text="'(' + counts.encounter.toLocaleString() + ')'"></span>
            </button>
            <button type="button"
                    @click="showOdReferrals = !showOdReferrals"
                    :class="showOdReferrals
                        ? 'bg-rose-500 text-white'
                        : 'text-slate-600 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800/60'"
                    class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold transition-colors">
                <span class="inline-block h-2.5 w-2.5 rounded-full"
                      :class="showOdReferrals ? 'bg-white' : 'bg-rose-500'"
                      style="flex-shrink:0"></span>
                OD Referrals
                <span class="opacity-70" x-text="'(' + counts.od_referral.toLocaleString() + ')'"></span>
            </button>
            <button type="button"
                    @click="showHeatmap = !showHeatmap"
                    :class="showHeatmap
                        ? 'bg-amber-500 text-white'
                        : 'text-slate-600 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800/60'"
                    class="flex items-center gap-1.5 border-l border-slate-200 px-3 py-1.5 text-xs font-semibold transition-colors
                           dark:border-slate-700/60">
                <span class="inline-block h-2.5 w-2.5 rounded-full"
                      :class="showHeatmap ? 'bg-white' : 'bg-amber-500'"
                      style="flex-shrink:0"></span>
                Density Zones
            </button>
        </div>
    </div>

    {# Map container #}
    <div id="{{ map_id }}" class="w-full h-[600px] rounded-lg"></div>
</div>

<script>
(function() {
    function initMap() {
        if (typeof L === 'undefined') {
            setTimeout(initMap, 50);
            return;
        }

    // Cache buster: {{ timestamp }}
    var isRestricted = {% if zoom_mode == "restricted" %}true{% else %}false{% endif %};

    var mapOptions = {};
    if (isRestricted) {
        mapOptions = {
            maxBounds: L.latLngBounds(
                L.latLng(24.396308, -124.848974),
                L.latLng(49.384358, -66.885444)
            ),
            minZoom: 10,
            maxZoom: 12,
        };
    }

    var mapEl = document.getElementById('{{ map_id }}');
    var map = L.map('{{ map_id }}', mapOptions).setView([48.1125, -123.4550], isRestricted ? 8 : 12);
    mapEl._map = map;  // Expose for Alpine.js layer toggling

    L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri, National Geographic',
        maxZoom: isRestricted ? 13 : 18
    }).addTo(map);

    // --- Jurisdictions layer ---
    var jurisdictionsData = {{ jurisdictions_geojson|safe }};

    function style(feature) {
        return {
            fillColor: feature.properties.color,
            weight: 2,
            opacity: 1,
            color: feature.properties.color,
            dashArray: '',
            fillOpacity: 0.15
        };
    }

    function highlightFeature(e) {
        var layer = e.target;
        layer.setStyle({ weight: 3, color: layer.feature.properties.color, dashArray: '', fillOpacity: 0.3 });
        info.update({ type: 'jurisdiction', name: layer.feature.properties.name });
    }

    function resetHighlight(e) {
        geojson.resetStyle(e.target);
        info.update();
    }

    function zoomToFeature(e) {
        map.fitBounds(e.target.getBounds());
    }

    function onEachFeature(feature, layer) {
        layer.on({ mouseover: highlightFeature, mouseout: resetHighlight, click: zoomToFeature });
    }

    var geojson = L.geoJson(jurisdictionsData, { style: style, onEachFeature: onEachFeature }).addTo(map);

    // --- Legend ---
    var legend = L.control({position: 'topright'});
    legend.onAdd = function () {
        var div = L.DomUtil.create('div', 'info legend w-60');
        var items = [
            {name: 'City of Port Angeles', color: '{{ CITY_COLOR }}'},
            {name: 'Fire District 2', color: '{{ FD2_COLOR }}'},
            {name: 'Fire District 4', color: '{{ FD4_COLOR }}'},
            {name: 'Lower Elwha Reservation', color: '{{ ELWHA_COLOR }}'}
        ];
        div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Jurisdictions</h4>';
        items.forEach(function(item) {
            div.innerHTML +=
                '<i class="w-4.5 h-4.5 float-left mr-2 opacity-70 border" style="background:' + item.color + '; border-color:' + item.color + ';"></i> ' +
                '<span class="text-xs">' + item.name + '</span><br class="clear-both">';
        });

        // Marker legend
        div.innerHTML += '<h4 class="m-0 mt-3 mb-2 text-sm font-semibold">Record Type</h4>';
        var markers = [
            {name: 'Referrals', color: '{{ REFERRAL_COLOR }}'},
            {name: 'Encounters', color: '{{ ENCOUNTER_COLOR }}'},
            {name: 'OD Referrals', color: '{{ OD_REFERRAL_COLOR }}'}
        ];
        markers.forEach(function(m) {
            div.innerHTML +=
                '<i class="w-4.5 h-4.5 float-left mr-2 opacity-90 border" style="background:' + m.color + '; border-color:#fff; border-radius:50%;"></i> ' +
                '<span class="text-xs">' + m.name + '</span><br class="clear-both">';
        });

        // Dot sizing note
        div.innerHTML += '<p class="m-0 mt-2 text-[10px] leading-tight text-slate-400">Dot area &prop; record count</p>';

        // Density contour zones
        div.innerHTML += '<h4 class="m-0 mt-3 mb-2 text-sm font-semibold">Density Zones</h4>';
        var zones = {{ contour_legend_json|safe }};
        zones.forEach(function(z) {
            div.innerHTML +=
                '<i class="w-4.5 h-4.5 float-left mr-2" style="background:' + z.color + '; opacity:0.6; border: 2px solid ' + z.color + '; border-radius: 2px;"></i> ' +
                '<span class="text-xs">' + z.label + '</span><br class="clear-both">';
        });

        return div;
    };
    legend.addTo(map);

    // --- Info control ---
    var info = L.control({position: 'topright'});
    info.onAdd = function () {
        this._div = L.DomUtil.create('div', 'info w-60 mt-2.5');
        this.update();
        return this._div;
    };
    info.update = function (data) {
        if (!data) {
            this._div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Map Info</h4>' +
                '<span class="text-slate-400 text-xs">Hover over a jurisdiction or marker</span>';
        } else if (data.type === 'jurisdiction') {
            this._div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Service Area</h4>' +
                '<b class="text-13">' + data.name + '</b>';
        } else if (data.type === 'marker') {
            var label = data.source === 'referral' ? 'Referrals'
                      : data.source === 'encounter' ? 'Encounters'
                      : 'OD Referrals';
            this._div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Patient Location</h4>' +
                '<b class="text-13">' + label + ': ' + data.count + ' records</b><br>' +
                '<span class="text-xs text-slate-400">' + data.patients + ' patient' + (data.patients !== 1 ? 's' : '') + ' at this address</span>';
        } else if (data.type === 'contour') {
            this._div.innerHTML = '<h4 class="m-0 mb-2 text-sm font-semibold">Density Zone</h4>' +
                '<b class="text-13" style="color:' + data.color + '">' + data.label + ' Activity</b>';
        }
    };
    info.addTo(map);

    // --- Marker data & layer groups ---
    var markersData = {{ patient_markers_json|safe }};
    map.createPane('markerPane');
    map.getPane('markerPane').style.zIndex = 650;

    // Radius scaler: sqrt so circle AREA is proportional to record count
    function markerRadius(count) {
        return Math.max(3, Math.min(30, 2 + Math.sqrt(count) * 1.5));
    }

    var colorMap = {
        referral: '{{ REFERRAL_COLOR }}',
        encounter: '{{ ENCOUNTER_COLOR }}',
        od_referral: '{{ OD_REFERRAL_COLOR }}'
    };

    var layerGroups = {
        referral: L.layerGroup(),
        encounter: L.layerGroup(),
        od_referral: L.layerGroup()
    };

    markersData.forEach(function(m) {
        var cm = L.circleMarker([m.lat, m.lon], {
            radius: markerRadius(m.count),
            fillColor: colorMap[m.source],
            color: '#fff',
            weight: 1,
            opacity: 0.9,
            fillOpacity: 0.55,
            pane: 'markerPane',
            interactive: true
        });

        cm.on('mouseover', function() {
            info.update({ type: 'marker', source: m.source, count: m.count, patients: m.patients });
            this.setStyle({ weight: 2, fillOpacity: 0.85 });
        });
        cm.on('mouseout', function() {
            info.update();
            this.setStyle({ weight: 1, fillOpacity: 0.55 });
        });

        layerGroups[m.source].addLayer(cm);
    });

    // Add all layer groups to map (all visible by default)
    layerGroups.referral.addTo(map);
    layerGroups.encounter.addTo(map);
    layerGroups.od_referral.addTo(map);

    // Expose layer groups for Alpine.js toggling
    mapEl._layerGroups = layerGroups;

    // --- Density contour overlay (KDE-derived polygons) ---
    var contourData = {{ contour_geojson|safe }};

    // Create a dedicated pane so contours render BELOW markers but ABOVE the basemap
    map.createPane('contourPane');
    map.getPane('contourPane').style.zIndex = 450;

    var contourLayer = L.geoJson(contourData, {
        pane: 'contourPane',
        style: function(feature) {
            var p = feature.properties;
            return {
                fillColor: p.color,
                fillOpacity: p.fill_opacity,
                color: p.color,
                weight: p.border_weight,
                opacity: 0.85,
                dashArray: '',
                className: 'contour-zone contour-level-' + p.level
            };
        },
        onEachFeature: function(feature, layer) {
            layer.on({
                mouseover: function(e) {
                    var p = feature.properties;
                    e.target.setStyle({
                        fillOpacity: p.fill_opacity + 0.12,
                        weight: p.border_weight + 1.5,
                        opacity: 1.0
                    });
                    info.update({ type: 'contour', label: p.label, color: p.color });
                },
                mouseout: function(e) {
                    contourLayer.resetStyle(e.target);
                    info.update();
                }
            });
        }
    });
    contourLayer.addTo(map);
    mapEl._contourLayer = contourLayer;

    // --- Theme-aware CSS + contour glow ---
    var cssEl = document.createElement('style');
    cssEl.innerHTML = `
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
        .legend i { border-radius: 3px; }
        /* Glowing border effect on contour zones */
        .contour-zone {
            filter: drop-shadow(0 0 3px currentColor);
            transition: fill-opacity 0.15s, stroke-width 0.15s;
        }
        .contour-level-3, .contour-level-4 {
            filter: drop-shadow(0 0 6px currentColor);
        }
    `;
    document.head.appendChild(cssEl);
    }
    initMap();
})();
</script>
"""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _prepare_jurisdictions_geojson():
    """
    Load jurisdiction boundaries and return a GeoJSON FeatureCollection string.
    Identical logic to od_map._prepare_jurisdictions_geojson().
    """
    features = []

    try:
        city_geojson = json.loads(CITY_BOUNDARY_PATH.read_text(encoding="utf-8"))
        for feature in city_geojson.get("features", []):
            feature["properties"] = {
                "name": "City of Port Angeles",
                "color": CITY_COLOR,
                "type": "city",
            }
            features.append(feature)

        fire_gdf = gpd.read_file(FIRE_DISTRICTS_PATH)
        fire_gdf = fire_gdf.to_crs(epsg=4326)

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


def _get_patient_markers_data():
    """
    Build location-aggregated marker data and per-location record totals.

    For every referral, encounter, and OD referral record, look up the owning
    patient's lat/long, then aggregate by ``(lat, lon, source)``.  Each group
    becomes one circle marker whose radius scales with the total record count.

    Also returns a ``location_totals`` dict mapping ``(lat, lon) → total_count``
    across all sources, used by ``_compute_density_contours()`` for KDE input.

    Returns
    -------
    tuple[str, dict[tuple[float, float], int], int, int, int]
        ``(markers_json, location_totals,
        referral_total, encounter_total, od_referral_total)``
    """
    # ---- Fetch patient coordinates ----
    patients = {
        row["id"]: (row["latitude"], row["longitude"])
        for row in Patients.objects.exclude(latitude__isnull=True)
        .exclude(longitude__isnull=True)
        .values("id", "latitude", "longitude")
    }

    # ---- Referrals: count per patient ----
    ref_qs = list(Referrals.objects.exclude(patient_ID__isnull=True).values("patient_ID"))
    ref_df = pd.DataFrame.from_records(ref_qs) if ref_qs else pd.DataFrame(columns=["patient_ID"])

    # ---- Encounters: count per patient ----
    enc_qs = list(Encounters.objects.exclude(patient_ID__isnull=True).values("patient_ID"))
    enc_df = pd.DataFrame.from_records(enc_qs) if enc_qs else pd.DataFrame(columns=["patient_ID"])

    # ---- OD Referrals: count per patient ----
    od_qs = list(ODReferrals.objects.exclude(patient_id__isnull=True).values("patient_id"))
    od_df = pd.DataFrame.from_records(od_qs) if od_qs else pd.DataFrame(columns=["patient_id"])
    if not od_df.empty:
        od_df = od_df.rename(columns={"patient_id": "patient_ID"})

    # ---- Aggregate per (lat, lon, source) ----
    markers = []
    referral_total = 0
    encounter_total = 0
    od_referral_total = 0

    # Also accumulate per-location totals for density contour computation
    location_totals: dict[tuple[float, float], int] = {}

    for source_label, df in [
        ("referral", ref_df),
        ("encounter", enc_df),
        ("od_referral", od_df),
    ]:
        if df.empty:
            continue

        # Records per patient for this source
        counts = df["patient_ID"].value_counts()

        # Group by (lat, lon) → {count: total_records, patients: unique_patient_count}
        loc_agg: dict[tuple[float, float], dict[str, int]] = {}
        for pid, count in counts.items():
            coords = patients.get(pid)
            if coords is None:
                continue
            lat, lon = float(coords[0]), float(coords[1])
            key = (lat, lon)
            if key not in loc_agg:
                loc_agg[key] = {"count": 0, "patients": 0}
            loc_agg[key]["count"] += int(count)
            loc_agg[key]["patients"] += 1

        source_total = 0
        for (lat, lon), agg in loc_agg.items():
            markers.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "count": agg["count"],
                    "patients": agg["patients"],
                    "source": source_label,
                }
            )
            source_total += agg["count"]

            # Accumulate for density contours (combined across all sources)
            location_totals[(lat, lon)] = location_totals.get((lat, lon), 0) + agg["count"]

        if source_label == "referral":
            referral_total = source_total
        elif source_label == "encounter":
            encounter_total = source_total
        else:
            od_referral_total = source_total

    return (
        json.dumps(markers),
        location_totals,
        referral_total,
        encounter_total,
        od_referral_total,
    )


def _extract_contour_polygons(points_arrays, offsets_arrays):
    """
    Convert contourpy's OuterOffset output into a list of shapely Polygons.

    Parameters
    ----------
    points_arrays : list[np.ndarray]
        Per-group Nx2 coordinate arrays from ``contour_generator.filled()``.
    offsets_arrays : list[np.ndarray]
        Per-group ring-boundary index arrays.

    Returns
    -------
    list[Polygon]
        Valid, non-empty shapely polygons extracted from the contour fill.
    """
    polys = []
    for group_idx in range(len(points_arrays)):
        pts = points_arrays[group_idx]
        ofs = offsets_arrays[group_idx]

        rings = []
        for r in range(len(ofs) - 1):
            ring_pts = pts[int(ofs[r]) : int(ofs[r + 1])]
            if len(ring_pts) >= 4:
                rings.append(ring_pts.tolist())

        if not rings:
            continue

        try:
            poly = Polygon(rings[0], rings[1:] if len(rings) > 1 else [])
            if not poly.is_valid:
                poly = make_valid(poly)
            if not poly.is_empty:
                polys.append(poly)
        except Exception:
            continue
    return polys


def _compute_density_contours(location_totals):
    """
    Compute KDE-based density contour polygons from location-aggregated data.

    Uses scipy's ``gaussian_kde`` (bandwidth 0.08 ≈ 8–9 km) to estimate a 2-D
    density surface from the weighted point data, then hands the resulting grid
    to ``contourpy`` for marching-squares contour extraction at five threshold
    levels.  Raw polygons are merged and simplified via shapely, then packaged
    as a GeoJSON FeatureCollection with per-level colour/opacity/weight
    properties baked in for direct Leaflet consumption.

    Weights are sqrt-compressed so that extreme outliers (e.g. a shelter with
    2,000+ records) spread influence without obliterating the gradient for
    every other neighbourhood.

    Parameters
    ----------
    location_totals : dict[tuple[float, float], int]
        Mapping of ``(lat, lon) → total_records_across_all_sources``.

    Returns
    -------
    str
        JSON-serialised GeoJSON FeatureCollection of contour polygons.
    """
    if not location_totals:
        return json.dumps({"type": "FeatureCollection", "features": []})

    lats = np.array([k[0] for k in location_totals])
    lons = np.array([k[1] for k in location_totals])
    weights = np.sqrt(np.array(list(location_totals.values()), dtype=float))

    # KDE — bandwidth 0.08 ≈ 8-9 km, good for county-scale visualisation
    positions = np.vstack([lons, lats])
    kde = gaussian_kde(positions, weights=weights, bw_method=0.08)

    # Evaluate on a 300×300 grid with slight padding
    pad = 0.03
    xi = np.linspace(float(lons.min()) - pad, float(lons.max()) + pad, 300)
    yi = np.linspace(float(lats.min()) - pad, float(lats.max()) + pad, 300)
    x_grid, y_grid = np.meshgrid(xi, yi)
    z_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

    max_density = float(z_grid.max())
    if max_density == 0:
        return json.dumps({"type": "FeatureCollection", "features": []})

    # Five cumulative threshold fractions of peak density
    level_fracs = [0.02, 0.08, 0.20, 0.40, 0.65]
    levels = [max_density * f for f in level_fracs]

    gen = contour_generator(x_grid, y_grid, z_grid, name="serial", fill_type="OuterOffset")

    features = []
    for level_idx, level in enumerate(levels):
        upper = max_density * 1.5  # Cumulative: everything above this threshold
        filled_result = gen.filled(level, upper)
        # OuterOffset fill_type returns a 2-tuple: (points, offsets).
        # points: list of Nx2 coordinate arrays per polygon group.
        # offsets: list of ring-boundary index arrays per group.
        points_arrays = filled_result[0]
        offsets_arrays = filled_result[1]

        polys_for_level = _extract_contour_polygons(points_arrays, offsets_arrays)
        if not polys_for_level:
            continue

        merged = unary_union(polys_for_level)
        # Simplify for smaller GeoJSON — 0.001° ≈ 100m tolerance
        merged = merged.simplify(0.001, preserve_topology=True)

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "level": level_idx,
                    "label": CONTOUR_LABELS[level_idx],
                    "color": CONTOUR_COLORS[level_idx],
                    "fill_opacity": CONTOUR_FILL_OPACITIES[level_idx],
                    "border_weight": CONTOUR_BORDER_WEIGHTS[level_idx],
                },
                "geometry": mapping(merged),
            }
        )

    return json.dumps({"type": "FeatureCollection", "features": features})


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_chart_patient_map(theme, zoom_mode="full"):
    """
    Build an interactive Leaflet map showing patient addresses for every
    referral, encounter, and OD referral — with density contour overlay and
    Alpine.js toggle controls.

    Parameters
    ----------
    theme : str
        ``"dark"`` or ``"light"``.
    zoom_mode : str
        ``"full"`` for authenticated users (unrestricted zoom),
        ``"restricted"`` for anonymous users (limited zoom to protect privacy).

    Returns
    -------
    str
        Raw HTML/JS string ready to be injected into a template.
    """
    jurisdictions_geojson = _prepare_jurisdictions_geojson()
    (
        patient_markers_json,
        location_totals,
        referral_total,
        encounter_total,
        od_referral_total,
    ) = _get_patient_markers_data()
    total_records = referral_total + encounter_total + od_referral_total

    # Compute density contour polygons from combined location data
    contour_geojson = _compute_density_contours(location_totals)

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

    timestamp = str(int(time.time()))
    map_id = f"patient_map_{hashlib.md5((str(total_records) + timestamp).encode(), usedforsecurity=False).hexdigest()[:8]}"

    # Build legend data for contour zones
    contour_legend = [
        {"label": CONTOUR_LABELS[i], "color": CONTOUR_COLORS[i]} for i in range(len(CONTOUR_LABELS))
    ]

    context = {
        "map_id": map_id,
        "timestamp": timestamp,
        "jurisdictions_geojson": jurisdictions_geojson,
        "patient_markers_json": patient_markers_json,
        "contour_geojson": contour_geojson,
        "contour_legend_json": json.dumps(contour_legend),
        "total_records": total_records,
        "referral_total": referral_total,
        "encounter_total": encounter_total,
        "od_referral_total": od_referral_total,
        "info_bg": info_bg,
        "info_color": info_color,
        "legend_bg": legend_bg,
        "legend_color": legend_color,
        "CITY_COLOR": CITY_COLOR,
        "FD2_COLOR": FD2_COLOR,
        "FD4_COLOR": FD4_COLOR,
        "ELWHA_COLOR": ELWHA_COLOR,
        "REFERRAL_COLOR": REFERRAL_COLOR,
        "ENCOUNTER_COLOR": ENCOUNTER_COLOR,
        "OD_REFERRAL_COLOR": OD_REFERRAL_COLOR,
        "zoom_mode": zoom_mode,
    }

    return Template(PATIENT_MAP_TEMPLATE).render(Context(context))
