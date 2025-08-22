# dashboard/utils/plotly.py
from collections.abc import Callable
from typing import Any

from utils.tailwind_colors import TAILWIND_COLORS


def get_theme_colors(theme: str = "dark") -> dict[str, str]:
    """
    Centralized theme color configuration for maximum contrast and readability
    """
    if theme == "dark":
        return {
            # Maximum contrast colors for dark mode
            "axis_font_color": TAILWIND_COLORS["slate-50"],  # Near white for maximum contrast
            "font_color": TAILWIND_COLORS["slate-50"],  # Near white for maximum contrast
            "plot_bg": TAILWIND_COLORS["transparent"],
            "paper_bg": TAILWIND_COLORS["transparent"],
            "grid_color": TAILWIND_COLORS["gray-600"],  # More visible grid
            "tick_color": TAILWIND_COLORS["slate-50"],  # Maximum contrast tick labels
            "hover_bg": "rgba(31, 41, 55, 0.95)",  # Dark gray with transparency
            "hover_border": "rgba(156, 163, 175, 0.8)",  # Light gray border
            "hover_font": TAILWIND_COLORS["gray-50"],  # Maximum contrast for hover text
        }
    else:  # light theme
        return {
            # Maximum contrast colors for light mode
            "axis_font_color": TAILWIND_COLORS["slate-950"],  # Near black for maximum contrast
            "font_color": TAILWIND_COLORS["slate-950"],  # Near black for maximum contrast
            "plot_bg": TAILWIND_COLORS["transparent"],
            "paper_bg": TAILWIND_COLORS["transparent"],
            "grid_color": TAILWIND_COLORS["gray-400"],  # More visible grid
            "tick_color": TAILWIND_COLORS["slate-950"],  # Maximum contrast tick labels
            "hover_bg": "rgba(255, 255, 255, 0.95)",  # White with transparency
            "hover_border": "rgba(107, 114, 128, 0.8)",  # Gray border
            "hover_font": TAILWIND_COLORS["gray-950"],  # Maximum contrast for hover text
        }


def style_plotly_layout(
    fig: Any,
    theme: str = "dark",
    hovermode_unified: bool = False,
    axis_font_size: int = 13,
    font_family: str = "Roboto",
    scroll_zoom: bool = True,
    show_legend: bool = False,
    show_modebar: bool = True,
    height: int = 500,
    x_title: str | None = None,
    y_title: str | None = None,
    margin: dict[str, int] | None = None,
) -> Any:
    """
    Enhanced Plotly layout styling with robust theme support
    """

    # Get theme colors
    colors = get_theme_colors(theme)

    # default margin if not supplied
    margin = margin or dict(t=40, l=20, r=20, b=20)

    # if using unified hovers, ensure top margin is tall enough
    if hovermode_unified:
        margin["t"] = max(margin.get("t", 0), 80)

    layout_updates = dict(
        title=None,
        showlegend=show_legend,
        font=dict(family=font_family, size=axis_font_size, color=colors["font_color"]),
        xaxis=dict(
            title=x_title if x_title else None,
            title_font=dict(size=axis_font_size, color=colors["axis_font_color"]),
            showgrid=True,
            gridcolor=colors["grid_color"],
            zeroline=False,
            tickfont=dict(color=colors["tick_color"]),
        ),
        yaxis=dict(
            title=y_title if y_title else None,
            title_font=dict(size=axis_font_size, color=colors["axis_font_color"]),
            showgrid=True,
            gridcolor=colors["grid_color"],
            zeroline=False,
            tickfont=dict(color=colors["tick_color"]),
        ),
        margin=margin,
        height=height,
        autosize=True,
        plot_bgcolor=colors["plot_bg"],
        paper_bgcolor=colors["paper_bg"],
        modebar={
            "orientation": "h",
        },
        # Modern hover styling applied by default
        hoverlabel=dict(
            bgcolor=colors["hover_bg"],
            bordercolor=colors["hover_border"],
            font=dict(family="Roboto, sans-serif", size=14, color=colors["hover_font"]),
            namelength=-1,
            align="left",
        ),
    )

    # add unified hovermode + styling if requested
    if hovermode_unified:
        layout_updates["hovermode"] = "x unified"
        layout_updates["hoverlabel"] = {
            "bgcolor": colors["hover_bg"],
            "bordercolor": colors["hover_border"],
            "font": {
                "family": font_family,
                "size": axis_font_size,
                "color": colors["hover_font"],
            },
        }

    fig.update_layout(**layout_updates)

    config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": scroll_zoom,
        "displayModeBar": show_modebar,
        "modeBarButtonsToRemove": [
            "zoom",
            "pan",
            "lasso2d",
            "select2d",
            "autoScale",
            "zoomIn",
            "zoomOut",
        ],
    }

    # Plotly Figure objects allow attaching a _config attribute; ignore for static typing
    fig._config = config  # type: ignore[attr-defined]
    return fig


def create_theme_aware_chart(
    chart_function: Callable[..., Any], request: Any, *args: Any, **kwargs: Any
) -> Any:
    """
    Helper function to create charts with automatic theme detection

    Usage:
        fig_html = create_theme_aware_chart(build_chart_od_density_heatmap, request)
    """
    from .theme import get_theme_from_request

    theme = get_theme_from_request(request)

    # Call the chart building function with theme
    if "theme" not in kwargs:
        kwargs["theme"] = theme

    return chart_function(*args, **kwargs)


def get_plotly_theme(theme: str = "dark") -> dict[str, Any]:
    """
    Get plotly theme configuration for backward compatibility
    """
    colors = get_theme_colors(theme)
    return {
        "font": {"color": colors["font_color"]},
        "plot_bgcolor": colors["plot_bg"],
        "paper_bgcolor": colors["paper_bg"],
    }


def get_color_palette(theme: str = "dark") -> dict[str, str]:
    """
    Get color palette for charts
    """
    from .tailwind_colors import TAILWIND_COLORS

    if theme == "dark":
        return {
            "primary": TAILWIND_COLORS["blue-500"],
            "success": TAILWIND_COLORS["green-500"],
            "warning": TAILWIND_COLORS["yellow-500"],
            "danger": TAILWIND_COLORS["red-500"],
            "info": TAILWIND_COLORS["cyan-500"],
            "purple": TAILWIND_COLORS["purple-500"],
        }
    else:
        return {
            "primary": TAILWIND_COLORS["blue-600"],
            "success": TAILWIND_COLORS["green-600"],
            "warning": TAILWIND_COLORS["yellow-600"],
            "danger": TAILWIND_COLORS["red-600"],
            "info": TAILWIND_COLORS["cyan-600"],
            "purple": TAILWIND_COLORS["purple-600"],
        }
