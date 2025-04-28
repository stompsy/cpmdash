# dashboard/utils/plotly.py


def style_plotly_layout(
    fig,
    theme="light",
    axis_font_size=12,
    font_family="Courier New",
    export_filename="pafd_cpm_chart",
    enable_image_export=True,
    scroll_zoom=True,
    show_legend=False,
    x_title=None,
    y_title=None,
    height=400,
    margin=dict(t=0, l=40, r=20, b=40),
):

    if theme == "dark":
        axis_font_color = "#d1d5db"
        font_color = "#e5e7eb"
        plot_bg = "#111827"
        paper_bg = "#1f2937"
        grid_color = "#374151"
    else:
        axis_font_color = "#374151"
        font_color = "#1f2937"
        plot_bg = "#ffffff"
        paper_bg = "#ffffff"
        grid_color = "#e5e7eb"

    fig.update_layout(
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
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        modebar={
            "orientation": "h",
        },
    )

    if enable_image_export:
        fig._config = {
            "displaylogo": False,
            "scrollZoom": scroll_zoom,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": export_filename,
                "height": 500,
                "width": 800,
                "scale": 1,
            },
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

    return fig
