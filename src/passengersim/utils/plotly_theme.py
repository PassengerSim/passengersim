from .colors import passengersim_colors

try:
    import plotly.io as pio
except ImportError:
    pxsim_plotly_template = None
else:
    _plot_background_color = "rgb(242, 242, 245)"
    _plot_grid_color = "white"
    pxsim_plotly_template = dict(
        layout={
            "annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1},
            "autotypenumbers": "strict",
            "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
            "colorscale": {
                "diverging": [
                    [0, "#8e0152"],
                    [0.1, "#c51b7d"],
                    [0.2, "#de77ae"],
                    [0.3, "#f1b6da"],
                    [0.4, "#fde0ef"],
                    [0.5, "#f7f7f7"],
                    [0.6, "#e6f5d0"],
                    [0.7, "#b8e186"],
                    [0.8, "#7fbc41"],
                    [0.9, "#4d9221"],
                    [1, "#276419"],
                ],
                "sequential": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "sequentialminus": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
            },
            "colorway": passengersim_colors,
            "dragmode": "pan",
            "font": {"color": "#2a3f5f", "family": "Roboto, Arial, sans-serif"},
            "geo": {
                "bgcolor": "white",
                "lakecolor": "white",
                "landcolor": "white",
                "showlakes": True,
                "showland": True,
                "subunitcolor": "#C8D4E3",
            },
            "hoverlabel": {"align": "left"},
            "hovermode": "closest",
            "mapbox": {"style": "light"},
            "paper_bgcolor": "white",
            "plot_bgcolor": _plot_background_color,
            "polar": {
                "angularaxis": {"gridcolor": _plot_grid_color, "linecolor": _plot_grid_color, "ticks": ""},
                "bgcolor": _plot_background_color,
                "radialaxis": {"gridcolor": _plot_grid_color, "linecolor": _plot_grid_color, "ticks": ""},
            },
            "scene": {
                "xaxis": {
                    "backgroundcolor": "white",
                    "gridcolor": _plot_grid_color,
                    "gridwidth": 2,
                    "linecolor": _plot_grid_color,
                    "showbackground": True,
                    "ticks": "",
                    "zerolinecolor": _plot_grid_color,
                },
                "yaxis": {
                    "backgroundcolor": "white",
                    "gridcolor": _plot_grid_color,
                    "gridwidth": 2,
                    "linecolor": _plot_grid_color,
                    "showbackground": True,
                    "ticks": "",
                    "zerolinecolor": _plot_grid_color,
                },
                "zaxis": {
                    "backgroundcolor": "white",
                    "gridcolor": _plot_grid_color,
                    "gridwidth": 2,
                    "linecolor": _plot_grid_color,
                    "showbackground": True,
                    "ticks": "",
                    "zerolinecolor": _plot_grid_color,
                },
            },
            "shapedefaults": {"line": {"color": "#2a3f5f"}},
            "ternary": {
                "aaxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""},
                "baxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""},
                "bgcolor": "white",
                "caxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""},
            },
            "title": {"x": 0.05},
            "xaxis": {
                "automargin": True,
                "gridcolor": _plot_grid_color,
                "linecolor": _plot_grid_color,
                "ticks": "",
                "title": {"standoff": 15},
                "zerolinecolor": _plot_grid_color,
                "zerolinewidth": 2,
                "title_font": dict(size=13, color="black", weight="bold"),
            },
            "yaxis": {
                "automargin": True,
                "gridcolor": _plot_grid_color,
                "linecolor": _plot_grid_color,
                "ticks": "",
                "title": {"standoff": 15},
                "zerolinecolor": _plot_grid_color,
                "zerolinewidth": 2,
                "title_font": dict(size=13, color="black", weight="bold"),
            },
        },
    )

    # add to templates and set as default
    pio.templates["passengersim"] = pxsim_plotly_template
    pio.templates.default = "passengersim"
