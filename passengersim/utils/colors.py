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


def atl_at_gt_theme():
    font = "Roboto, Arial, sans-serif"
    return {
        "config": {
            "view": {
                "continuousHeight": 300,
                "continuousWidth": 400,
            },  # from the default theme
            "range": {"category": atl_at_gt},
            "title": {"font": font},
            "axis": {"labelFont": font, "titleFont": font},
            "header": {"labelFont": font, "titleFont": font},
            "legend": {"labelFont": font, "titleFont": font},
        }
    }


alt.themes.register("atl_at_gt", atl_at_gt_theme)
alt.themes.enable("atl_at_gt")
