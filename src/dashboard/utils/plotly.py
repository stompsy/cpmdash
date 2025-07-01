# dashboard/utils/plotly.py
from dashboard.utils.tailwind_colors import TAILWIND_COLORS


def style_plotly_layout(
    fig,
    theme="dark",
    hovermode_unified=False,
    axis_font_size=13,
    font_family="Roboto",
    export_filename="pafd_cpm_chart",
    enable_image_export=True,
    scroll_zoom=True,
    show_legend=False,
    show_modebar=True,
    height=500,
    x_title=None,
    y_title=None,
    margin=None,
):

    if theme == "dark":
        axis_font_color =   TAILWIND_COLORS["slate-200"]
        font_color =        TAILWIND_COLORS["slate-200"]
        plot_bg =           TAILWIND_COLORS["transparent"]
        paper_bg =          TAILWIND_COLORS["transparent"]
        grid_color =        TAILWIND_COLORS["gray-800"]
    else:
        axis_font_color =   TAILWIND_COLORS["slate-800"]
        font_color =        TAILWIND_COLORS["slate-800"]
        plot_bg =           TAILWIND_COLORS["transparent"]  # Make both themes transparent for better blending
        paper_bg =          TAILWIND_COLORS["transparent"]  # Make both themes transparent for better blending
        grid_color =        TAILWIND_COLORS["slate-200"]   # Lighter grid for light mode

    # default margin if not supplied
    margin = margin or dict(t=40, l=20, r=20, b=20)

    # if using unified hovers, ensure top margin is tall enough
    if hovermode_unified:
        margin["t"] = max(margin.get("t", 0), 80)

    layout_updates = dict(
        title=None,
        showlegend=show_legend,
        font=dict(family=font_family, size=axis_font_size, color=font_color),
        xaxis=dict(
            title=x_title if x_title else None,
            title_font=dict(size=axis_font_size, color=axis_font_color),
            showgrid=True,
            gridcolor=grid_color,
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title if y_title else None,
            title_font=dict(size=axis_font_size, color=axis_font_color),
            showgrid=True,
            gridcolor=grid_color,
            zeroline=False,
        ),
        margin=margin,
        height=height,
        autosize=True,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        modebar={"orientation": "h",},
    )
    
    # add unified hovermode + styling if requested
    if hovermode_unified:
        layout_updates["hovermode"] = "x unified"
        layout_updates["hoverlabel"] = {
            "bgcolor": TAILWIND_COLORS["gray-800"],
            "bordercolor": TAILWIND_COLORS["gray-600"],
            "font": {
                "family": font_family,
                "size": axis_font_size,
                "color": TAILWIND_COLORS["gray-50"],
            }
        }

    fig.update_layout(**layout_updates)

    config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": scroll_zoom,
        "displayModeBar": show_modebar,     # ← respect the new flag
    }

    if enable_image_export:
        config.update({
            "toImageButtonOptions": {
                "format": "svg",
                "filename": export_filename,
                "height": 500,
                "width": 800,
                "scale": 1,
            },
            "modeBarButtonsToRemove": [
                "zoom", "pan", "lasso2d", "select2d",
                "autoScale", "zoomIn", "zoomOut",
            ],
        })
    
    fig._config = config
    return fig
