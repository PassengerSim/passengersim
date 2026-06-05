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

DarkPurple = "#6a3d9a"
LightPurple = "#cab2d6"
DarkGreen = "#33a02c"
LightGreen = "#b2df8a"
DarkOrange = "#ff7f00"
LightOrange = "#fdbf6f"
DarkBlue = "#1f78b4"
LightBlue = "#a6cee3"
DarkRed = "#e31a1c"
LightRed = "#fb9a99"
DarkYellow = "#b15928"
LightYellow = "#e3d086"

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
    "#b15928",  # dark yellow
    "#e3d086",  # light yellow
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
                "range": {"category": passengersim_colors, "heatmap": "plasma"},
                "title": {"font": font, "fontSize": 18, "anchor": "start", "offset": 8},
                "axis": {"labelFont": font, "titleFont": font},
                "header": {"labelFont": font, "titleFont": font},
                "legend": {"labelFont": font, "titleFont": font},
                "bar": {"color": passengersim_colors[0]},
                "mark": {"color": passengersim_colors[0]},
                "line": {"color": passengersim_colors[0]},
            }
        }
    )


def common_carrier_colors(carrier: str):
    """Get a common color for a given carrier, to be used across multiple charts."""
    commons = dict(
        AA="#0078D2",  # American
        AC="#F01428",  # Air Canada
        AF="#FF0000",  # Air France
        AS="#00b2d6",  # Alaska
        BA="#0035AD",  # British Airways
        DL="#9B1631",  # Delta
        EK="#D71A21",  # Emirates
        EY="#C4921B",  # Etihad
        FR="#F4CA35",  # Ryanair
        KL="#00A2DF",  # KLM
        LH="#0A1D3D",  # Lufthansa
        LX="#E60005",  # Swiss
        QR="#660033",  # Qatar
        TK="#C70A0C",  # Turkish
        UA="#0033A0",  # United
        U2="#FF5E00",  # Easyjet
        WN="#FFBF27",  # Southwest
        WS="#00AAA6",  # WestJet
        Bison="#724e3a",
        Eagle="#e3c622",
        AL1=DarkPurple,
        AL2=DarkGreen,
        AL3=DarkOrange,
        AL4=DarkBlue,
    )
    return commons.get(carrier, DarkPurple)
