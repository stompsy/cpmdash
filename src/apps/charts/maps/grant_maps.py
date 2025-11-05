from __future__ import annotations

from pathlib import Path
from typing import cast

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from django.conf import settings
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.base import BaseGeometry
from xyzservices import TileProvider

DATA_DIR = Path(settings.BASE_DIR) / "src/static/data"
CITY_BOUNDARY_PATH = (
    Path(settings.BASE_DIR) / "staticfiles/data/port_angeles_outer_boundary.geojson"
)
FIRE_DISTRICTS_PATH = DATA_DIR / "fire_districts.geojson"
COUNTY_PATH = DATA_DIR / "cb_2023_us_county_20m.shp"
AIANNH_PATH = DATA_DIR / "tl_2025_us_aiannh.shp"
DEFAULT_OUTPUT_DIR = Path(settings.BASE_DIR) / "static/media/maps"

CRS_PROJECTED = "EPSG:3857"

CITY_COLOR = "#2B6CB0"
FD2_COLOR = "#C53030"
FD4_COLOR = "#ED8936"
ELWHA_COLOR = "#805AD5"
CITY_EDGE = "#1A365D"
FD2_EDGE = "#7F1D1D"
FD4_EDGE = "#9C4221"
ELWHA_EDGE = "#553C9A"
COUNTY_EDGE = "#444444"
COMBINED_COLOR = "#319795"
COMBINED_EDGE = "#1d4044"


def _load_geodata() -> dict[str, gpd.GeoDataFrame]:
    county = gpd.read_file(COUNTY_PATH)
    county = county[(county["STATEFP"] == "53") & (county["COUNTYFP"] == "009")]
    if county.empty:
        raise ValueError("Clallam County geometry not found in county shapefile.")

    city = gpd.read_file(CITY_BOUNDARY_PATH)
    if city.empty:
        raise ValueError("Port Angeles city boundary GeoJSON is empty.")

    fire = gpd.read_file(FIRE_DISTRICTS_PATH)
    required_labels = {"Port Angeles", "Fire District 2", "Fire District 4"}
    if not required_labels.issubset(set(fire["LABEL"])):
        missing = required_labels - set(fire["LABEL"])
        raise ValueError(f"Missing fire district geometries: {', '.join(sorted(missing))}")

    aiannh = gpd.read_file(AIANNH_PATH)
    elwha = aiannh[aiannh["NAME"].str.contains("Lower Elwha", case=False, na=False)]
    if elwha.empty:
        raise ValueError("Lower Elwha Klallam Reservation not found in AIANNH shapefile.")

    return {
        "county": county.to_crs(CRS_PROJECTED),
        "city": city.to_crs(CRS_PROJECTED),
        "fire": fire.to_crs(CRS_PROJECTED),
        "elwha": elwha.to_crs(CRS_PROJECTED),
    }


def _dissolve_features(
    fire: gpd.GeoDataFrame,
    label: str,
) -> gpd.GeoDataFrame:
    subset = fire[fire["LABEL"] == label]
    if subset.empty:
        raise ValueError(f"Fire district geometry not found for label '{label}'.")
    dissolved = subset.dissolve()
    dissolved["LABEL"] = label
    return dissolved.reset_index(drop=True)


def _prepare_elwha(elwha: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geometry = elwha.unary_union
    return gpd.GeoDataFrame(
        {"NAME": ["Lower Elwha Klallam Reservation"]}, geometry=[geometry], crs=elwha.crs
    )


def _configure_axes(
    ax: Axes,
    geoms: list[gpd.GeoDataFrame],
    buffer_ratio: float = 0.12,
    min_buffer: float = 3500.0,
) -> None:
    combined = gpd.GeoSeries([gdf.unary_union for gdf in geoms], crs=geoms[0].crs)
    bounds = combined.unary_union.bounds
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    buffer_x = max(min_buffer, width * buffer_ratio)
    buffer_y = max(min_buffer, height * buffer_ratio)
    ax.set_xlim(minx - buffer_x, maxx + buffer_x)
    ax.set_ylim(miny - buffer_y, maxy + buffer_y)
    ax.set_axis_off()


def _build_jawg_provider() -> TileProvider:
    token = getattr(settings, "JAWG_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError(
            "w_ACCESS_TOKEN is not configured. Add it to your .env file to render Jawg basemaps."
        )
    return TileProvider(
        name="Jawg Streets",
        url=f"https://tile.jawg.io/jawg-sunny/{{z}}/{{x}}/{{y}}{{r}}.png?access-token={token}",
        attribution=(
            '<a href="https://jawg.io" title="Tiles Courtesy of Jawg Maps" target="_blank">'
            '&copy; <b>Jawg</b>Maps</a> &copy; <a href="https://www.openstreetmap.org/copyright">'
            "OpenStreetMap</a> contributors"
        ),
        min_zoom=0,
        max_zoom=22,
    )


def _add_basemap(ax: Axes) -> None:
    ctx.add_basemap(
        ax,
        crs=CRS_PROJECTED,
        source="Esri.NatGeoWorldMap",
        alpha=0.93,
        attribution=True,
    )


def _get_centroid(geom: gpd.GeoDataFrame) -> tuple[float, float]:
    geometry = cast(BaseGeometry, geom.geometry.iloc[0])
    centroid = geometry.centroid
    return centroid.x, centroid.y


def generate_port_angeles_maps(
    output_dir: str | Path | None = None, dpi: int = 300
) -> dict[str, Path]:
    data = _load_geodata()
    county = data["county"]
    fire = data["fire"]
    elwha = _prepare_elwha(data["elwha"])

    fd2 = _dissolve_features(fire, "Fire District 2")
    fd4 = _dissolve_features(fire, "Fire District 4")
    pafd = _dissolve_features(fire, "Port Angeles")
    focus_geoms = [fd2, fd4, elwha, pafd]
    buffer_ratio = 0.12
    min_buffer = 6000.0
    component_geoms = [
        cast(BaseGeometry, fd2.geometry.iloc[0]),
        cast(BaseGeometry, fd4.geometry.iloc[0]),
        cast(BaseGeometry, pafd.geometry.iloc[0]),
        cast(BaseGeometry, elwha.geometry.iloc[0]),
    ]
    combined_geom = gpd.GeoSeries(component_geoms, crs=fd2.crs).unary_union
    combined = gpd.GeoDataFrame(
        {"LABEL": ["Combined Coverage"]}, geometry=[combined_geom], crs=fd2.crs
    )

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    # Map 1: City boundary only
    fig, ax = plt.subplots(figsize=(7.5, 9.5), dpi=dpi)
    county.boundary.plot(ax=ax, color=COUNTY_EDGE, linewidth=1.0, alpha=0.9)
    pafd.plot(ax=ax, facecolor=CITY_COLOR, edgecolor=CITY_EDGE, linewidth=1.8, alpha=0.85)
    _configure_axes(ax, focus_geoms, buffer_ratio=buffer_ratio, min_buffer=min_buffer)
    _add_basemap(ax)
    ax.add_artist(
        ScaleBar(dx=1, units="m", dimension="si-length", location="lower right", box_alpha=0.3)
    )
    legend_handles = [
        Patch(facecolor=CITY_COLOR, edgecolor=CITY_EDGE, label="City of Port Angeles")
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    plt.tight_layout()
    city_output = out_dir / "port_angeles_city.png"
    fig.savefig(city_output, facecolor="white")
    plt.close(fig)
    saved_paths["city"] = city_output

    # Map 2: District overlays
    fig2, ax2 = plt.subplots(figsize=(7.5, 9.5), dpi=dpi)
    county.boundary.plot(ax=ax2, color=COUNTY_EDGE, linewidth=1.0, alpha=0.9)
    fd4.plot(ax=ax2, facecolor=FD4_COLOR, edgecolor=FD4_EDGE, linewidth=1.4, alpha=0.6)
    fd2.plot(ax=ax2, facecolor=FD2_COLOR, edgecolor=FD2_EDGE, linewidth=1.4, alpha=0.55)
    elwha.plot(ax=ax2, facecolor=ELWHA_COLOR, edgecolor=ELWHA_EDGE, linewidth=1.6, alpha=0.75)
    pafd.plot(ax=ax2, facecolor=CITY_COLOR, edgecolor=CITY_EDGE, linewidth=1.8, alpha=0.65)
    _configure_axes(ax2, focus_geoms, buffer_ratio=buffer_ratio, min_buffer=min_buffer)
    _add_basemap(ax2)
    ax2.add_artist(
        ScaleBar(dx=1, units="m", dimension="si-length", location="lower right", box_alpha=0.3)
    )
    county_x2, county_y2 = _get_centroid(county)
    ax2.text(
        county_x2,
        county_y2 - 26000,
        "Clallam County",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
        alpha=0.75,
        fontweight="semibold",
    )
    legend_handles = [
        Patch(facecolor=CITY_COLOR, edgecolor=CITY_EDGE, label="City of Port Angeles"),
        Patch(facecolor=FD2_COLOR, edgecolor=FD2_EDGE, label="Clallam County Fire District 2"),
        Patch(facecolor=FD4_COLOR, edgecolor=FD4_EDGE, label="Clallam County Fire District 4"),
        Patch(facecolor=ELWHA_COLOR, edgecolor=ELWHA_EDGE, label="Lower Elwha Klallam Reservation"),
    ]
    ax2.legend(
        handles=legend_handles,
        loc="lower left",
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    plt.tight_layout()
    districts_output = out_dir / "port_angeles_fire_districts.png"
    fig2.savefig(districts_output, facecolor="white")
    plt.close(fig2)
    saved_paths["districts"] = districts_output

    # Map 3: Combined coverage outline
    fig3, ax3 = plt.subplots(figsize=(7.5, 9.5), dpi=dpi)
    county.boundary.plot(ax=ax3, color=COUNTY_EDGE, linewidth=1.0, alpha=0.9)
    combined.plot(
        ax=ax3, facecolor=COMBINED_COLOR, edgecolor=COMBINED_EDGE, linewidth=2.2, alpha=0.45
    )
    _configure_axes(ax3, [combined], buffer_ratio=buffer_ratio, min_buffer=min_buffer)
    _add_basemap(ax3)
    ax3.add_artist(
        ScaleBar(dx=1, units="m", dimension="si-length", location="lower right", box_alpha=0.3)
    )
    ax3.legend(
        handles=[
            Patch(facecolor=COMBINED_COLOR, edgecolor=COMBINED_EDGE, label="Combined Service Area")
        ],
        loc="lower left",
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    plt.tight_layout()
    combined_output = out_dir / "port_angeles_combined_coverage.png"
    fig3.savefig(combined_output, facecolor="white")
    plt.close(fig3)
    saved_paths["combined"] = combined_output

    return saved_paths


if __name__ == "__main__":
    generate_port_angeles_maps()
