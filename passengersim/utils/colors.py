import altair as alt

atl_at_gt = [
    "#b3a16a",  # gold
    "#003258",  # navy
    "#a5a5a5",  # silver
    "#b4122b",  # red
    "#6f8db9",  # blue
    "#db9e0d",  # yellow
    "#5F249F",  # purple
    "#A4D233",  # green
    "#64CCC9",  # electric blue
    "#E04F39",  # orange
]

passengersim_colors = [
    "#6a3d9a",  # dark purple
    "#cab2d6",  # light purple
    "#33a02c",  # dark green
    "#b2df8a",  # light green
    "#ff7f00",  # dark orange
    "#fdbf6f",  # light orange
    "#1f78b4",  # dark blue
    "#a6cee3",  # light blue
    "#e31a1c",  # dark red
    "#fb9a99",  # light red
]


@alt.theme.register("passengersim", enable=True)
def passengersim_theme():
    font = "Roboto, Arial, sans-serif"
    return alt.theme.ThemeConfig(
        {
            "config": {
                "view": {
                    "continuousHeight": 300,
                    "continuousWidth": 400,
                },  # from the default theme
                "range": {"category": passengersim_colors, "Sequential": "plasma"},
                "title": {"font": font, "fontSize": 18, "anchor": "start", "offset": 8},
                "axis": {"labelFont": font, "titleFont": font},
                "header": {"labelFont": font, "titleFont": font},
                "legend": {"labelFont": font, "titleFont": font},
            }
        }
    )
