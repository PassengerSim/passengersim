from __future__ import annotations

import json
import warnings
import zoneinfo
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd

from passengersim.summaries import GenericSimulationTables

# mapping dependencies
try:
    import geopandas as gpd
except ImportError:
    gpd = None
try:
    import pyproj
except ImportError:
    pyproj = None
try:
    import shapely
    from shapely.geometry import LineString, Point
except ImportError:
    shapely = None
    LineString = None
    Point = None

from passengersim.core import Generator
from passengersim.driver._constructors import make_core_choice_model, make_core_demand
from passengersim.utils.colors import common_carrier_colors
from passengersim.utils.mapping import PROJECTION1_CONUS, PROJECTION1_WORLD, conus_only, world_only

from .base import Config
from .dataframes import booking_curves_to_dataframe, frat5_curves_to_dataframe
from .legs import Leg
from .places import Place

__all__ = [
    "fig_booking_curves",
    "fig_frat5_curves",
    "fig_route_map",
    "fig_speed_by_distance",
    "fig_duration_by_distance",
    "plotly_route_map",
    "fig_demand_vs_capacity",
    "fig_max_wtp_distributions",
    "fig_hub_schedule",
    "fig_market_schedule",
    "fig_reference_price_by_distance",
    "fig_maximum_fare_ratio_by_distance",
]


def fig_booking_curves(cfg: Config, raw_df: bool = False) -> alt.Chart | pd.DataFrame:
    """Create a figure showing all booking curves defined in the config.

    Each curve is plotted as a line showing the cumulative proportion of bookings
    made at each number of days prior to departure. Curves are colored by passenger
    segment and labeled to indicate which demand(s) use them.

    Parameters
    ----------
    cfg : Config
        Configuration object containing booking curves and demands.
    raw_df : bool, optional
        If True, return the underlying pandas DataFrame instead of the chart.
        Default is False.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An Altair line chart of the booking curves, or the raw DataFrame if
        ``raw_df`` is True.
    """
    data = booking_curves_to_dataframe(cfg.booking_curves, add_zero_days=True)
    data["proportion"] = data.groupby(["curve_name"])["proportion"].shift(periods=1, fill_value=0)

    segments = {}
    for demand in cfg.demands:
        full_label = f"{demand.segment}/{demand.curve}"
        if demand.curve not in segments or segments[demand.curve] == full_label:
            segments[demand.curve] = full_label
        else:
            segments[demand.curve] = f"VARIOUS/{demand.curve}"
    data["segment"] = data["curve_name"].map(lambda x: segments.get(x, f"UNUSED/{x}"))
    if raw_df:
        return data
    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("days_prior:Q", scale=alt.Scale(reverse=True), title="Days Prior to Departure"),
            y=alt.Y("proportion:Q", title="Cumulative Proportion of Bookings"),
            color=alt.Color("segment:N", title="Passenger Segment"),
        )
    )


def fig_frat5_curves(cfg: Config) -> alt.Chart:
    """Create a figure showing all Frat5 curves defined in the config.

    Each Frat5 curve is plotted as a line showing how the fare ratio adjustment
    factor varies with days prior to departure.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Frat5 curves.

    Returns
    -------
    alt.Chart
        An Altair line chart of all Frat5 curves, colored by curve name.
    """
    data = frat5_curves_to_dataframe(cfg.frat5_curves)

    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("days_prior:Q", scale=alt.Scale(reverse=True), title="Days Prior to Departure"),
            y=alt.Y("frat5_value:Q", title="Frat5 Value"),
            color=alt.Color("curve_name:N", title="Frat5 Curve Name"),
        )
    )


def fig_route_map(cfg: Config, carrier: str | None = None) -> alt.Chart:
    """Create a figure showing the route map for the configured network.

    Routes are drawn as geodesic arcs on either a CONUS or world projection,
    determined automatically by the geographic extent of the network. Airport
    locations are shown as points, with major hubs (airports serving more than
    20% of all legs) rendered as stars.

    Parameters
    ----------
    cfg : Config
        Configuration object containing legs and places.
    carrier : str, optional
        If provided, only show legs operated by this carrier. If None (default),
        all carriers are shown.

    Returns
    -------
    alt.Chart
        An Altair layered chart combining the base map, route arcs, and
        airport point markers.
    """
    return _legs_conus_or_world(cfg.legs, cfg.places, title="Route Map", line_color="red", carrier=carrier)


def _legs_as_geodataframe(
    legs: list[Leg],
    places: dict[str, Place],
    *,
    carrier: str | None = None,
    fix_longitude_wrap: bool = True,
) -> gpd.GeoDataFrame:
    """Convert a list of legs to a GeoDataFrame of geodesic arc geometries.

    Each leg is represented as a ``LineString`` consisting of intermediate
    points along the geodesic great-circle path between origin and destination.
    The number of interpolated points scales with the leg distance so that
    long-haul routes appear smooth on a map.

    Parameters
    ----------
    legs : list of Leg
        Legs to convert. Each leg must have ``orig``, ``dest``, and optionally
        ``distance`` and ``carrier`` attributes.
    places : dict of str to Place
        Mapping from airport code to :class:`Place`, used to look up the
        latitude and longitude of each endpoint.
    carrier : str, optional
        If provided, only include legs operated by this carrier.
    fix_longitude_wrap : bool, default True
        If True, shift negative longitudes by +360° when the average absolute
        longitude of a leg's coordinates exceeds 90°, which prevents incorrect
        rendering of routes that cross the antimeridian.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame indexed by ``market`` (``"ORIG-DEST"`` strings) with a
        single ``geometry`` column containing the geodesic ``LineString`` for
        each leg.

    Raises
    ------
    ImportError
        If ``geopandas`` or ``shapely`` is not installed.
    """

    if gpd is None:
        raise ImportError("geopandas is not installed, it is required for this function.")
    if shapely is None:
        raise ImportError("shapely is not installed, it is required for this function.")

    # get rid of all places that are not connected by any leg (by carrier).
    relevant_places = set()
    for leg in legs:
        if carrier is not None and leg.carrier != carrier:
            continue
        relevant_places.add(leg.orig)
        relevant_places.add(leg.dest)

    points = {}
    for place_code, place_data in places.items():
        if place_code not in relevant_places:
            continue
        points[place_code] = Point(place_data.lon, place_data.lat)

    geod = pyproj.Geod(ellps="WGS84")

    leg_lines = {}
    n_legs = 0

    for leg in legs:
        if carrier is not None and leg.carrier != carrier:
            continue
        market = f"{leg.orig}-{leg.dest}"  # (leg.orig, leg.dest)
        n_legs += 1

        orig = points[leg.orig]
        dest = points[leg.dest]
        dist = leg.distance or 1000
        npts = max(30, int(dist / 100))  # at least 10 points, more for longer distances
        leg_coords = geod.npts(orig.x, orig.y, dest.x, dest.y, npts=npts)
        leg_coords.insert(0, (orig.x, orig.y))
        leg_coords.append((dest.x, dest.y))
        if fix_longitude_wrap:
            wrap_fix_needed = np.absolute(np.asarray(leg_coords)[:, 0]).mean() > 90
            if wrap_fix_needed:
                leg_coords = [(i + 360 if i < 0 else i, j) for (i, j) in leg_coords]
        leg_lines[market] = LineString(leg_coords)

    return gpd.GeoDataFrame(
        list(leg_lines.keys()),
        geometry=list(leg_lines.values()),
        crs="EPSG:4326",
        columns=["market"],
    ).set_index("market")


def _legs_conus_or_world(
    legs: list[Leg],
    places: dict[str, Place],
    *,
    carrier: str | None = None,
    line_color: str = "red",
    title: str = "Flight Routes",
) -> alt.Chart:
    """Build a route-map Altair chart, choosing CONUS or world projection automatically.

    If all legs are entirely within the contiguous United States, a CONUS
    Albers projection is used. If any leg has at least one non-CONUS endpoint,
    a world (Robinson) projection is used. Airport markers are sized by
    connectivity: airports serving more than 20% of all visible legs are
    rendered as stars; all others are rendered as circles.

    Parameters
    ----------
    legs : list of Leg
        Legs to display. Each leg must have ``orig``, ``dest``, ``carrier``,
        and optionally ``distance`` attributes.
    places : dict of str to Place
        Mapping from airport code to :class:`Place`.
    carrier : str, optional
        If provided, only display legs operated by this carrier.
    line_color : str, default ``"red"``
        CSS color string used for the route arc lines.
    title : str, default ``"Flight Routes"``
        Chart title. If ``carrier`` is also provided, the carrier code is
        prepended (e.g., ``"AA Flight Routes"``).

    Returns
    -------
    alt.Chart
        An Altair layered chart comprising the base map, route arcs, and
        airport point markers.

    Raises
    ------
    ImportError
        If ``geopandas``, ``pyproj``, or ``shapely`` is not installed.
    """

    if gpd is None:
        raise ImportError("geopandas is not installed, it is required for this function.")
    if pyproj is None:
        raise ImportError("pyproj is not installed, it is required for this function.")
    if shapely is None:
        raise ImportError("shapely is not installed, it is required for this function.")

    # get rid of all places that are not connected by any leg (by carrier).
    relevant_places = set()
    for leg in legs:
        if carrier is not None and leg.carrier != carrier:
            continue
        relevant_places.add(leg.orig)
        relevant_places.add(leg.dest)

    points = {}
    place_is_conus = {}
    for place_code, place_data in places.items():
        if place_code not in relevant_places:
            continue
        points[place_code] = Point(place_data.lon, place_data.lat)
        if place_data.country is None:
            from passengersim.utils.geocoding import is_conus

            place_is_conus[place_code] = is_conus(place_data.lat, place_data.lon)
        else:
            place_is_conus[place_code] = place_data.country == "US"

    geod = pyproj.Geod(ellps="WGS84")

    leg_lines_conus = {}
    leg_lines_intl = {}
    n_legs = 0
    n_legs_by_place = {}

    for leg in legs:
        if carrier is not None and leg.carrier != carrier:
            continue
        market = f"{leg.orig}-{leg.dest}"  # (leg.orig, leg.dest)
        n_legs += 1

        # count the number of legs touching each place, this will be used
        # to identify hubs or significant operational bases
        n_legs_by_place[leg.orig] = n_legs_by_place.get(leg.orig, 0) + 1
        n_legs_by_place[leg.dest] = n_legs_by_place.get(leg.dest, 0) + 1
        orig = points[leg.orig]
        dest = points[leg.dest]
        dist = leg.distance or 1000
        npts = max(30, int(dist / 100))  # at least 10 points, more for longer distances
        leg_coords = geod.npts(orig.x, orig.y, dest.x, dest.y, npts=npts)
        leg_coords.insert(0, (orig.x, orig.y))
        leg_coords.append((dest.x, dest.y))
        if place_is_conus[leg.orig] and place_is_conus[leg.dest]:
            leg_lines_conus[market] = LineString(leg_coords)
        else:
            leg_lines_intl[market] = LineString(leg_coords)

    # now we create two sets of map lines: one for CONUS legs, and one for
    # the rest of the world.
    gdf_conus = gpd.GeoDataFrame(
        list(leg_lines_conus.keys()),
        geometry=list(leg_lines_conus.values()),
        crs="EPSG:4326",
        columns=["market"],
    ).set_index("market")
    gdf_intl = gpd.GeoDataFrame(
        list(leg_lines_intl.keys()),
        geometry=list(leg_lines_intl.values()),
        crs="EPSG:4326",
        columns=["market"],
    ).set_index("market")

    # now we create data that will map the places as points or stars (for hubs)
    point_data_conus = []
    point_data_world = []
    for point_code, point_value in points.items():
        if point_code not in relevant_places:
            continue
        if point_code not in n_legs_by_place:
            continue
        shape = "circle"
        if n_legs_by_place[point_code] > n_legs * 0.2:
            shape = "star"
        if shape == "star" or place_is_conus[point_code]:
            point_data_conus.append(
                {
                    "place_code": point_code,
                    "latitude": point_value.y,
                    "longitude": point_value.x,
                    "place_shape": shape,
                }
            )
        if shape == "star" or not place_is_conus[point_code]:
            point_data_world.append(
                {
                    "place_code": point_code,
                    "latitude": point_value.y,
                    "longitude": point_value.x,
                    "place_shape": shape,
                }
            )

    # df = pd.DataFrame(
    #     {"market": list(leg_lines_conus.keys()), "constant": [1] * len(leg_lines_conus)},
    #     columns=["market", "constant"],
    # ).set_index("market")

    conus_geojson = json.loads(gdf_conus.to_json())
    any_intl = len(gdf_intl) > 0
    intl_geojson = json.loads(pd.concat([gdf_conus, gdf_intl], ignore_index=True).to_json())
    # if raw_geojson:
    #     return conus_geojson, intl_geojson

    STAR = "M0,.5L.6,.8L.5,.1L1,-.3L.3,-.4L0,-1L-.3,-.4L-1,-.3L-.5,.1L-.6,.8L0,.5Z"

    def _make_place_points(point_data_, proj_args):
        return (
            alt.Chart(alt.Data(values=point_data_))
            .mark_point(
                fill="black",
                stroke="white",
                strokeWidth=0.5,
                opacity=1.0,
            )
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                tooltip=[alt.Tooltip("place_code:N", title="Place")],
                shape=alt.Shape(
                    "place_shape:N",
                    scale=alt.Scale(
                        domain=["circle", "star"],
                        range=["circle", STAR],
                    ),
                    legend=None,
                ),
                size=alt.Size(
                    "place_shape:N",
                    scale=alt.Scale(
                        domain=["circle", "star"],
                        range=[40, 200],  # Assign specific sizes
                    ),
                    legend=None,
                ),
            )
            .project(**proj_args)
        )

    def _make_route_lines(geojson, proj_args):
        return (
            alt.Chart(alt.Data(values=geojson["features"]))
            .mark_geoshape(
                stroke=line_color,
                strokeWidth=1,
                fill="none",
            )
            .project(**proj_args)
        )

    if not any_intl:
        # there are no international legs, so just return the CONUS map
        place_points_conus = _make_place_points(point_data_conus, PROJECTION1_CONUS)
        route_lines = _make_route_lines(conus_geojson, PROJECTION1_CONUS)
        base_conus = conus_only()
        if title:
            if carrier is not None:
                title = f"{carrier} {title}"
            route_lines = route_lines.properties(title=title)
        return route_lines + base_conus + place_points_conus

    # there are international legs, so return an international map
    place_points_intl = _make_place_points(point_data_conus + point_data_world, PROJECTION1_WORLD)
    route_lines_intl = _make_route_lines(intl_geojson, PROJECTION1_WORLD)

    base_world = world_only()
    if title:
        if carrier is not None:
            title = f"{carrier} {title}"
        route_lines_intl = route_lines_intl.properties(title=title)
    return route_lines_intl + base_world + place_points_intl


def fig_speed_by_distance(cfg: Config, carrier: str | None = None) -> alt.Chart:
    """Create a figure showing gate-to-gate average speed versus leg distance.

    Each leg is represented by a wedge marker whose angle indicates the
    approximate bearing of the flight. This makes it easy to spot outliers
    in gate-to-gate block times and to compare operational performance across
    carriers.

    Parameters
    ----------
    cfg : Config
        Configuration object containing legs and places.
    carrier : str, optional
        If provided, only show legs operated by this carrier.

    Returns
    -------
    alt.Chart
        An interactive Altair scatter chart with distance on the x-axis and
        average speed (miles per hour) on the y-axis.
    """

    from passengersim.config.places import calculate_mean_bearing

    df = cfg.dataframes.legs.eval("speed = distance / (duration_minutes / 60)")

    df = df.join(cfg.dataframes.places[["lat", "lon", "name"]].set_index("name").add_prefix("orig_"), on="orig").join(
        cfg.dataframes.places[["lat", "lon", "name"]].set_index("name").add_prefix("dest_"), on="dest"
    )
    df["bearing"] = df.apply(
        lambda row: calculate_mean_bearing(row["orig_lat"], row["orig_lon"], row["dest_lat"], row["dest_lon"]), axis=1
    )

    if carrier is not None:
        df = df[df["carrier"] == carrier]

    # Determine if all carriers have well-defined colors
    try:
        carrier_colors = {}
        for c in df["carrier"].unique():
            carrier_colors[c] = common_carrier_colors(c, error_if_missing=True)
        color_scale = alt.Scale(domain=list(carrier_colors.keys()), range=list(carrier_colors.values()))
    except KeyError:
        color_scale = alt.Scale()

    return (
        alt.Chart(df)
        .mark_point(shape="wedge", filled=True, size=300)
        .encode(
            x=alt.X("distance", title="Leg Distance (mi)"),
            y=alt.Y("speed", title="Gate-to-Gate Average Speed (mph)"),
            color=alt.Color("carrier", scale=color_scale),
            angle=alt.Angle("bearing").scale(domain=[0, 360], range=[0, 360]),
            tooltip=[
                alt.Tooltip("leg_id", title="Leg ID"),
                alt.Tooltip("distance", title="Distance (mi)"),
                alt.Tooltip("duration_minutes", title="Duration (minutes)"),
                alt.Tooltip("speed", title="Speed (mph)"),
                alt.Tooltip("carrier", title="Carrier"),
                alt.Tooltip("fltno", title="Flight No"),
                "orig",
                "dest",
            ],
        )
        .interactive()
    )


def fig_duration_by_distance(cfg: Config, carrier: str | None = None) -> alt.Chart:
    """Create a figure showing gate-to-gate flight duration versus leg distance.

    Each leg is represented by a wedge marker whose angle indicates the
    approximate bearing of the flight. This complements
    :func:`fig_speed_by_distance` by showing block times directly, which is
    useful for validating schedule inputs.

    Parameters
    ----------
    cfg : Config
        Configuration object containing legs and places.
    carrier : str, optional
        If provided, only show legs operated by this carrier.

    Returns
    -------
    alt.Chart
        An interactive Altair scatter chart with distance on the x-axis and
        gate-to-gate duration (hours) on the y-axis.
    """

    from passengersim.config.places import calculate_mean_bearing

    df = cfg.dataframes.legs.eval("speed = distance / (duration_minutes / 60)").eval(
        "duration_hours = duration_minutes / 60"
    )

    df = df.join(cfg.dataframes.places[["lat", "lon", "name"]].set_index("name").add_prefix("orig_"), on="orig").join(
        cfg.dataframes.places[["lat", "lon", "name"]].set_index("name").add_prefix("dest_"), on="dest"
    )
    df["bearing"] = df.apply(
        lambda row: calculate_mean_bearing(row["orig_lat"], row["orig_lon"], row["dest_lat"], row["dest_lon"]), axis=1
    )

    if carrier is not None:
        df = df[df["carrier"] == carrier]

    # Determine if all carriers have well-defined colors
    try:
        carrier_colors = {}
        for c in df["carrier"].unique():
            carrier_colors[c] = common_carrier_colors(c, error_if_missing=True)
        color_scale = alt.Scale(domain=list(carrier_colors.keys()), range=list(carrier_colors.values()))
    except KeyError:
        color_scale = alt.Scale()

    return (
        alt.Chart(df)
        .mark_point(shape="wedge", filled=True, size=300)
        .encode(
            x=alt.X("distance", title="Leg Distance (mi)"),
            y=alt.Y("duration_hours", title="Gate-to-Gate Duration (hours)"),
            color=alt.Color("carrier", scale=color_scale),
            angle=alt.Angle("bearing").scale(domain=[0, 360], range=[0, 360]),
            tooltip=[
                alt.Tooltip("leg_id", title="Leg ID"),
                alt.Tooltip("distance", title="Distance (mi)"),
                alt.Tooltip("duration_minutes", title="Duration (minutes)"),
                alt.Tooltip("speed", title="Speed (mph)"),
                alt.Tooltip("carrier", title="Carrier"),
                alt.Tooltip("fltno", title="Flight No"),
                "orig",
                "dest",
            ],
        )
        .interactive()
    )


def plotly_route_map(
    cfg: Config,
    carrier: str | None = None,
    line_color: str | None = None,
    center_lon: float | None = None,
    center_lat: float = 10,
    projection_type: str = "robinson",
    height: int = 400,
    title: str | bool = True,
):
    """Plot a route map using Plotly.

    Each leg is drawn as a straight geodesic line on a world map projection.
    Airport locations are shown as small black dot markers with hover tooltips
    that include the airport name, label, and time zone.

    Parameters
    ----------
    cfg : Config
        Configuration object containing legs and places.
    carrier : str, optional
        If provided, only show legs operated by this carrier.
    line_color : str, optional
        CSS color string for the route lines. If None (default), the color is
        determined automatically from :func:`~passengersim.utils.colors.common_carrier_colors`
        using the ``carrier`` argument.
    center_lon : float, optional
        Longitude (degrees) used as the center of the map projection rotation.
        If None (default), the longitude of the busiest hub airport is used.
    center_lat : float, default 10
        Latitude (degrees) used as the center of the map projection rotation.
    projection_type : str, default ``"robinson"``
        Plotly map projection type (e.g., ``"robinson"``, ``"natural earth"``,
        ``"mercator"``).
    height : int, default 400
        Height of the figure in pixels.
    title : str or bool, default True
        Chart title. If True, a title is generated automatically from the
        carrier name (e.g., ``"AA Routes"`` or ``"All Routes"``). If False or
        an empty string, no title is shown.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the route map.
    """

    import plotly.graph_objects as go

    if line_color is None:
        line_color = common_carrier_colors(carrier)

    df_places = cfg.dataframes.places
    df_legs = cfg.dataframes.legs
    if carrier is not None:
        df_legs = df_legs[df_legs["carrier"] == carrier]
    df_legs = df_legs.join(
        cfg.dataframes.places[["lat", "lon", "name"]].set_index("name").add_prefix("orig_"), on="orig"
    ).join(cfg.dataframes.places[["lat", "lon", "name"]].set_index("name").add_prefix("dest_"), on="dest")
    df_legs = df_legs.reset_index()

    # If not given explicitly, center the longitude on the biggest hub for this carrier
    if center_lon is None:
        _arr = np.concat([df_legs["orig_lon"], df_legs["dest_lon"]])
        _vals, _counts = np.unique(_arr, return_counts=True)
        _most_common = _vals[_counts.argmax()]
        center_lon = float(_most_common)

    fig = go.Figure()
    for i in range(len(df_legs)):
        fig.add_trace(
            go.Scattergeo(
                # locationmode = 'USA-states',
                lon=[df_legs["orig_lon"][i], df_legs["dest_lon"][i]],
                lat=[df_legs["orig_lat"][i], df_legs["dest_lat"][i]],
                mode="lines",
                line=dict(width=1, color=line_color),
            )
        )

    hover_text = df_places.apply(
        lambda row: "<br>".join(
            filter(
                None,
                [
                    row["name"],
                    row.get("label") or None,
                    row.get("time_zone") or None,
                ],
            )
        ),
        axis=1,
    )

    fig.add_trace(
        go.Scattergeo(
            lon=df_places["lon"],
            lat=df_places["lat"],
            hoverinfo="text",
            text=hover_text,
            mode="markers",
            marker=dict(size=4, color="rgb(0, 0, 0)", line=dict(width=3, color="rgba(68, 68, 68, 0)")),
        )
    )

    fig.update_layout(
        showlegend=False,
        geo=dict(
            scope="world",
            projection_type=projection_type,
            projection_rotation=dict(lon=center_lon, lat=center_lat, roll=0),
        ),
        height=height,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    if title is True:
        if carrier is not None:
            title = f"{carrier} Routes"
        else:
            title = "All Routes"
    if title:
        fig.update_layout(
            title={
                "text": title,
                "y": 0.95,
                "x": 0.05,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 24, "color": line_color, "weight": "bold"},
            }
        )
    return fig


def fig_demand_vs_capacity(cfg: Config) -> alt.Chart:
    """Create a figure comparing base demand to seat capacity by airport.

    Two scatter charts are produced side-by-side:

    * **Departing** – total base demand originating at each airport vs. total
      seats departing from that airport.
    * **Arriving** – total base demand destined for each airport vs. total
      seats arriving at that airport.

    Points are colored by the airport's *path flow share* (the fraction of
    leg-touches at that airport that are connecting passengers), using a
    viridis color scale. A red diagonal 1:1 reference line is included in
    each panel.

    Parameters
    ----------
    cfg : Config
        Configuration object containing demands, legs, paths, and places.

    Returns
    -------
    alt.Chart
        An Altair horizontally concatenated chart with two interactive panels.
    """

    place_counters = {name: [0, 0, 0] for name in cfg.places}
    leg_map = {leg.leg_id: leg for leg in cfg.legs}

    for pth in cfg.paths:
        for leg_id in pth.legs:
            leg = leg_map[leg_id]
            if leg.orig == pth.orig:
                place_counters[leg.orig][0] += 1
            if leg.dest == pth.dest:
                place_counters[leg.dest][2] += 1
            else:
                place_counters[leg.dest][1] += 1
    flow_shares = pd.Series({k: v[1] / (sum(v) + 0.0001) for k, v in place_counters.items()}).rename("path_flow_share")

    _demands = cfg.dataframes.demands
    _demands["base_demand"] *= cfg.simulation_controls.demand_multiplier

    ## Arriving
    df = cfg.dataframes.places[["name"]].set_index("name")
    df = (
        df.join(_demands.groupby("dest").agg(demand_arriving=("base_demand", "sum")))
        .join(cfg.dataframes.legs.groupby("dest").agg(seats_arriving=("capacity", "sum")))
        .join(flow_shares)
        .fillna(0)
        .reset_index()
    )

    limit = max(df["demand_arriving"].max(), df["seats_arriving"].max())

    flow_color = alt.Color(
        "path_flow_share",
        title="Path Flow Share",
        scale=alt.Scale(
            scheme=alt.SchemeParams(name="viridis", extent=[0.1, 0.9]),
            nice=True,
        ),
    )

    points = (
        alt.Chart(df)
        .mark_point(filled=True)
        .encode(
            x=alt.X("demand_arriving", title="Demand Arriving"),
            y=alt.Y("seats_arriving", title="Seats Arriving"),
            color=flow_color,
            tooltip=[
                alt.Tooltip("name", title="Place Code"),
                alt.Tooltip("demand_arriving", title="Total Base Demand Arriving", format=".2f"),
                alt.Tooltip("seats_arriving", title="Total Seats Arriving"),
                alt.Tooltip("path_flow_share", title="Path Flow Share", format=".2%"),
            ],
        )
    )

    one_to_one = (
        alt.Chart(pd.DataFrame({"v": [0, limit]})).mark_line(color="red", strokeWidth=0.5).encode(x="v:Q", y="v:Q")
    )

    arrival_fig = (points + one_to_one).properties(
        title=alt.TitleParams(text="Arriving Each Destination", anchor="middle", fontWeight="bold", fontSize=12),
    )

    ## Departing
    df1 = cfg.dataframes.places[["name"]].set_index("name")
    df1 = (
        df1.join(_demands.groupby("orig").agg(demand_departing=("base_demand", "sum")))
        .join(cfg.dataframes.legs.groupby("orig").agg(seats_departing=("capacity", "sum")))
        .join(flow_shares)
        .fillna(0)
        .reset_index()
    )
    limit1 = max(df1["demand_departing"].max(), df1["seats_departing"].max())

    points1 = (
        alt.Chart(df1)
        .mark_point(filled=True)
        .encode(
            x=alt.X("demand_departing", title="Demand Departing"),
            y=alt.Y("seats_departing", title="Seats Departing"),
            color=flow_color,
            tooltip=[
                alt.Tooltip("name", title="Place Code"),
                alt.Tooltip("demand_departing", title="Total Base Demand Departing", format=".2f"),
                alt.Tooltip("seats_departing", title="Total Seats Departing"),
                alt.Tooltip("path_flow_share", title="Path Flow Share", format=".2%"),
            ],
        )
    )
    one_to_one1 = (
        alt.Chart(pd.DataFrame({"v": [0, limit1]})).mark_line(color="red", strokeWidth=0.5).encode(x="v:Q", y="v:Q")
    )

    depart_fig = (points1 + one_to_one1).properties(
        title=alt.TitleParams(text="Departing Each Origin", anchor="middle", fontWeight="bold", fontSize=12),
    )

    return (depart_fig.interactive(name="dep_param") | arrival_fig.interactive(name="arr_param")).properties(
        title="Demand vs Capacity by Place"
    )


def fig_max_wtp_distributions(
    cfg: Config,
    orig: str,
    dest: str,
    *,
    n_draws: int = 1_000_000,
    lower_bound: float = 0.0,
    upper_bound: float | None = None,
    random_seed: int = 42,
    raw_df: bool = False,
) -> alt.Chart | pd.DataFrame:
    """Create a figure showing cumulative distributions of maximum willingness to pay.

    For each choice model that has demand in the given O&D market, this function
    simulates a large number of draws and computes the maximum willingness to pay
    (WTP) for each draw. The resulting empirical cumulative distribution is plotted
    as "percentage of draws with WTP greater than threshold" vs. WTP threshold,
    providing an intuitive view of price sensitivity by passenger segment.

    Parameters
    ----------
    cfg : Config
        Configuration object containing choice models and demands.
    orig : str
        Origin airport code for the market of interest.
    dest : str
        Destination airport code for the market of interest.
    n_draws : int, default 1_000_000
        Number of draws to simulate per choice model. More draws produce a
        smoother and more accurate distribution estimate.
    lower_bound : float, default 0.0
        Lower bound of the WTP threshold range shown on the x-axis.
    upper_bound : float, optional
        Upper bound of the WTP threshold range shown on the x-axis. If None
        (default), it is set to five times the maximum reference fare found
        across all demands in the market.
    random_seed : int, default 42
        Random seed passed to the core random-number generator for
        reproducibility.
    raw_df : bool, default False
        If True, return the underlying pandas DataFrame instead of the chart.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An Altair line chart of WTP cumulative distributions, colored by choice
        model name. Returns a DataFrame if ``raw_df`` is True.

    Raises
    ------
    ValueError
        If no demand is found for any choice model in the specified market.
    """
    # collect demands for this market by passengers segment
    demands = {}
    max_reference_price = 0
    for d in cfg.demands:
        if d.orig == orig and d.dest == dest:
            demands[d.segment] = d
            if d.reference_price > max_reference_price:
                max_reference_price = d.reference_price

    prng = Generator(seed=random_seed)
    choice_models = {name: make_core_choice_model(cfg, prng) for name, cfg in cfg.choice_models.items()}
    if upper_bound is None:
        upper_bound = 5 * max_reference_price
    pct_greater = {}
    for cm_name, cm in choice_models.items():
        if cm_name not in demands:
            warnings.warn(f"No demand for {cm_name}", stacklevel=2)
        dmd_cfg = demands[cm_name]
        dmd = make_core_demand(dmd_cfg, markets={}, choice_models=choice_models, booking_curves={})
        z = cm.max_wtp(dmd.reference_price, n_draws=n_draws, raw=True)
        raw_sorted = np.sort(z["raw"])
        x_values = np.linspace(lower_bound, upper_bound, 200)
        pct_greater[cm_name] = pd.Series(
            100.0 * (1 - np.searchsorted(raw_sorted, x_values, side="left") / raw_sorted.size),
            index=x_values,
            name=cm_name,
        )
    if len(pct_greater) == 0:
        raise ValueError("no demand present for any segment with a choice model")
    df = (
        pd.concat(pct_greater, axis=1)
        .reset_index()
        .melt(id_vars="index", var_name="choice_model", value_name="pct_greater_than")
    )
    if raw_df:
        return df
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("index", title="WTP Threshold", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("pct_greater_than", title="% of Draws with WTP > Threshold"),
            color=alt.Color("choice_model", title="Choice Model"),
            tooltip=["index", "pct_greater_than"],
        )
        .properties(title=f"Distribution of Maximum Willingness to Pay ({orig}~{dest})")
    )
    return chart


def fig_hub_schedule(
    cfg: Config,
    hub: str,
    *,
    carrier: str | None = None,
    raw_df: bool = False,
    width: int = 600,
    min_height: int = 100,
    max_height: int = 600,
    spoke: str | None = None,
    coloring: str | None = None,
    connections: bool = True,
) -> alt.TopLevelMixin:
    """Visualize the schedule at a given hub.

    This does not initially show the actually allowed connections, just the whole schedule
    of arrivals and departures at the hub. This can be used to visually inspect
    potential connection opportunities and the overall structure of the schedule
    at that hub. Click on any leg to see the allowed connections highlighted.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    hub : str
        Airport code of the hub to visualize.
    carrier : str, optional
        If provided, only show legs operated by this carrier.
    raw_df : bool, optional
        If True, return the raw dataframe instead of a chart.
    width : int, default 600
        Width of the chart in pixels.
    min_height : int, default 100
        Minimum height of the chart in pixels.
    max_height : int, default 600
        Maximum height of the chart in pixels.
    spoke : str, optional
        If provided, highlight legs connecting to/from this spoke airport.
    coloring : str, optional
        Controls how leg lines are colored. Options are ``"spoke"`` (color by
        spoke airport) or ``"spoke.region"`` (color by spoke region tag, with
        legend-click interactivity). If None, legs are colored by carrier.
    connections : bool, default True
        If True, clicking on any leg line highlights all legs that connect to
        that leg (i.e., share a path through ``cfg.paths``). Legs that do not
        connect to the selected leg are dimmed to 0.1 opacity. Click the same
        leg again to deselect and restore full opacity for all legs.
    """

    # Compute the time zone offset from UTC at this hub
    hub_place = cfg.places.get(hub)
    if hub_place is None:
        raise ValueError(f"No place found for {hub}")
    hub_tz_name = hub_place.time_zone
    tz = zoneinfo.ZoneInfo(hub_tz_name)
    reference_time = datetime.fromtimestamp(cfg.simulation_controls.reference_epoch(), tz=tz)
    hub_offset_milliseconds = reference_time.utcoffset().total_seconds() * 1000

    base_df = cfg.dataframes.legs
    if carrier is not None:
        base_df = base_df[base_df["carrier"] == carrier]
    df1 = base_df.query(f"orig == '{hub}'")
    df1 = df1[
        [
            "orig",
            "dest",
            "dep_hour_local",
            "arr_hour_local",
            "duration_minutes",
            "leg_id",
            "fltno",
            "carrier",
            "dep_time",
            "arr_time",
            "capacity",
        ]
    ].eval("duration_hours = duration_minutes / 60")
    df1["spoke"] = df1["dest"]
    if spoke:
        df1["highlight"] = (df1["dest"] == spoke).astype(int)
    df1 = df1.eval("hub_hour = dep_hour_local").eval("far_hour = dep_hour_local + duration_hours")
    # altair renders epoch in milliseconds
    df1["hub_unixtime_ms"] = df1["dep_time"] * 1000 + hub_offset_milliseconds
    df1["far_unixtime_ms"] = df1["arr_time"] * 1000 + hub_offset_milliseconds

    df2 = base_df.query(f"dest == '{hub}'")
    df2 = df2[
        [
            "orig",
            "dest",
            "arr_hour_local",
            "dep_hour_local",
            "duration_minutes",
            "leg_id",
            "fltno",
            "carrier",
            "dep_time",
            "arr_time",
            "capacity",
        ]
    ].eval("duration_hours = duration_minutes / 60")
    df2["spoke"] = df2["orig"]
    if spoke:
        df2["highlight"] = (df2["orig"] == spoke).astype(int)
    df2 = df2.eval("hub_hour = arr_hour_local").eval("far_hour = arr_hour_local - duration_hours")
    # altair renders epoch in milliseconds
    df2["hub_unixtime_ms"] = df2["arr_time"] * 1000 + hub_offset_milliseconds
    df2["far_unixtime_ms"] = df2["dep_time"] * 1000 + hub_offset_milliseconds

    df = (
        pd.concat([df1, df2], axis=0)
        .sort_values(by=["orig", "dest", "hub_hour"])
        .sort_values(by=["hub_hour"])
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(columns={"index": "rownum"})
    )
    df["hub_time"] = pd.to_datetime(df["hub_hour"], unit="h").dt.strftime("%H:%M")
    df["dep_time_nominal"] = pd.to_datetime(df["dep_hour_local"], unit="h").dt.strftime("%H:%M")
    df["arr_time_nominal"] = pd.to_datetime(df["arr_hour_local"], unit="h").dt.strftime("%H:%M")

    if connections:
        # Build a mapping from each leg_id visible in the hub schedule to the set of
        # leg_ids it connects with (i.e., they share at least one path together).
        # Each leg is always connected to itself so clicking it stays highlighted.
        hub_leg_ids = set(df["leg_id"])
        connections_map: dict = {leg_id: {leg_id} for leg_id in hub_leg_ids}
        for path in cfg.paths:
            # Find the subset of this path's legs that appear in the hub schedule
            hub_path_legs = [lid for lid in path.legs if lid in hub_leg_ids]
            if len(hub_path_legs) > 1:
                # All hub-schedule legs in this path are mutually connected
                for lid in hub_path_legs:
                    connections_map[lid].update(hub_path_legs)
        # Encode the connected-leg set as a pipe-delimited string so we can use
        # Vega's indexof() to test membership in the interactive expression below.
        df["connected_leg_ids_str"] = df["leg_id"].map(
            lambda lid: "|" + "|".join(str(x) for x in sorted(connections_map.get(lid, {lid}))) + "|"
        )

    height = len(df) * 5
    if height < min_height:
        height = min_height
    if height > max_height:
        height = max_height

    if raw_df:
        return df

    title = f"Flight Schedule at {hub}"
    if carrier is not None:
        title = f"{carrier} {title}"

    timeaxis = alt.Axis(format="%a %H:%M", labelAngle=-25)

    # Determine if all carriers have well-defined colors
    try:
        carrier_colors = {}
        for c in df["carrier"].unique():
            carrier_colors[c] = common_carrier_colors(c, error_if_missing=True)
        color_scale = alt.Scale(domain=list(carrier_colors.keys()), range=list(carrier_colors.values()))
    except KeyError:
        color_scale = alt.Scale()

    common_encoding = dict(
        x=alt.X("hub_unixtime_ms:T", title=f"Time at {hub} (local hour)", axis=timeaxis, scale=alt.Scale(type="utc")),
        x2="far_unixtime_ms:T",
        y=alt.Y("rownum:N", axis=None),
        color=alt.Color("carrier:N", scale=color_scale),
    )
    if spoke:
        common_encoding["strokeWidth"] = alt.StrokeWidth(
            "highlight:N", scale=alt.Scale(domain=[0, 1], range=[1, 2]), legend=None
        )

    if coloring == "spoke":
        common_encoding["color"] = alt.Color("spoke:N", legend=None)
    elif coloring == "spoke.region":
        regions = {p: plc.tags["region"] for p, plc in cfg.places.items()}
        df["region"] = df["spoke"].map(regions)
        common_encoding["color"] = alt.Color("region:N")
    elif coloring is not None:
        raise ValueError("unrecognized coloring")

    # Build the selection and opacity encoding for connection highlighting.
    # When a leg is clicked, the Vega expression checks whether the clicked
    # leg's id appears in each datum's pipe-delimited connected_leg_ids_str.
    # If nothing is selected (!isValid), all legs remain at full opacity.
    sel_leg = None
    if connections:
        sel_leg = alt.selection_point(fields=["leg_id"], name="sel_leg")
        connected_expr = (
            "!isValid(sel_leg.leg_id) || "
            "indexof(datum.connected_leg_ids_str, '|' + toString(sel_leg.leg_id) + '|') >= 0"
        )
        common_encoding["opacity"] = alt.condition(
            connected_expr,  # plain string → serialized as {"test": expr} in Vega-Lite
            alt.value(1.0),
            alt.value(0.1),
        )

    chart = alt.Chart(df.query(f"orig == '{hub}'")).mark_rule(point={"shape": "diamond"}).encode(
        tooltip=[
            "leg_id:N",
            "carrier:N",
            "fltno:N",
            "orig:N",
            "dest:N",
            alt.Tooltip("dep_time_nominal:N", title=f"Departure Time ({hub})"),
            alt.Tooltip("arr_time_nominal:N", title="Arrival Time"),
            alt.Tooltip("duration_hours:Q", title="Duration (hours)", format=".2f"),
            "capacity:N",
        ],
        **common_encoding,
    ).properties(width=width, height=height) + alt.Chart(df.query(f"dest == '{hub}'")).mark_rule(
        point={"shape": "diamond"}
    ).encode(
        tooltip=[
            "leg_id:N",
            "carrier:N",
            "fltno:N",
            "orig:N",
            "dest:N",
            alt.Tooltip("dep_time_nominal:N", title="Departure Time"),
            alt.Tooltip("hub_time:N", title=f"Arrival Time ({hub})"),
            alt.Tooltip("duration_hours:Q", title="Duration (hours)", format=".2f"),
            "capacity:N",
        ],
        **common_encoding,
    ).properties(width=width, height=height, title=title)

    if coloring == "spoke.region":
        selection = alt.selection_point(fields=["region"], bind="legend")
        chart = chart.encode(
            strokeDash=alt.when(selection).then(alt.value([1, 0])).otherwise(alt.value([2, 3]))
        ).add_params(selection)

    if sel_leg is not None:
        chart = chart.add_params(sel_leg)

    return chart


def fig_market_schedule(
    cfg: Config | GenericSimulationTables,
    orig: str,
    dest: str,
    *,
    timezone: str | None = None,
    raw_df: bool = False,
    width: int = 600,
    min_height: int = 100,
    max_height: int = 600,
    color_connect_time: bool = False,
) -> alt.TopLevelMixin:
    """Visualize the schedules for a given O-D market.

    This shows the schedule for all paths serving a market. Each path is displayed
    as a rule that spans from the departure time at the origin to the arrival time
    at the destination. The X-axis represents local time at the origin.

    Parameters
    ----------
    cfg : Config | GenericSimulationTables
        Configuration object containing paths, legs, and places information.
        Optionally this can be a GenericSimulationTables object, from which the
        configuration is extracted, along with leg simulation results
    orig : str
        Origin airport code.
    dest : str
        Destination airport code.
    timezone : str, optional
        Time zone for horizontal axis labels. If not provided, the time zone at the
        market origin will be used.
    raw_df : bool, optional
        If True, return the raw dataframe instead of the chart. Default is False.
    width : int, default 600
        Width of the chart in pixels.
    min_height : int, default 100
        Minimum height of the chart in pixels.
    max_height : int, default 600
        Maximum height of the chart in pixels.
    color_connect_time : bool, default False
        Whether to color connection time marks by length.

    Returns
    -------
    alt.TopLevelMixin or pd.DataFrame
        An Altair chart object, or a pandas DataFrame if raw_df is True.
    """
    if isinstance(cfg, GenericSimulationTables):
        simtab = cfg
        cfg = simtab.config
    else:
        simtab = None

    if timezone is None:
        # Compute the time zone offset from UTC at the origin
        orig_place = cfg.places.get(orig)
        if orig_place is None:
            raise ValueError(f"No place found for {orig}")
        timezone = orig_place.time_zone
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except zoneinfo.ZoneInfoNotFoundError:
        tz_place = cfg.get_place(timezone, error_if_missing=False)
        if tz_place is None:
            raise
        timezone = tz_place.time_zone
        tz = zoneinfo.ZoneInfo(timezone)
    reference_time = datetime.fromtimestamp(cfg.simulation_controls.reference_epoch(), tz=tz)
    tz_offset_milliseconds = reference_time.utcoffset().total_seconds() * 1000

    # Get legs dataframe for quick lookup
    legs_df = cfg.dataframes.legs
    leg_map = {row["leg_id"]: row for _, row in legs_df.iterrows()}

    # Collect all paths for this market
    paths = [p for p in cfg.paths if p.orig == orig and p.dest == dest]

    if not paths:
        raise ValueError(f"No paths found for market {orig}~{dest}")

    leg_data = []
    connect_data = []
    for path_num, path in enumerate(paths):
        path_overall_start_time = leg_map[path.legs[0]]["dep_time"]
        path_overall_end_time = leg_map[path.legs[-1]]["arr_time"]
        path_overall_duration_minutes = (path_overall_end_time - path_overall_start_time) / 60
        path_overall_duration = f"{path_overall_duration_minutes // 60:.0f}:{path_overall_duration_minutes % 60:02.0f}"
        path_n_legs = len(path.legs)

        prior_leg_arr_time = None
        for leg_id in path.legs:
            leg = leg_map[leg_id]
            leg_ = {}
            leg_["leg_id"] = leg_id
            leg_["path_id"] = path.path_id
            leg_["path_quality_index"] = path.path_quality_index
            leg_["carrier"] = leg["carrier"]
            leg_["rownum"] = path_num
            leg_["orig"] = leg["orig"]
            leg_["dest"] = leg["dest"]
            leg_["dep_time"] = leg["dep_time"] * 1000 + tz_offset_milliseconds
            leg_["arr_time"] = leg["arr_time"] * 1000 + tz_offset_milliseconds
            leg_["dep_hour_local"] = leg["dep_hour_local"]
            leg_["arr_hour_local"] = leg["arr_hour_local"]
            leg_duration = (leg["arr_time"] - leg["dep_time"]) / 60  # in minutes
            leg_["leg_duration"] = f"{leg_duration // 60:.0f}:{leg_duration % 60:02.0f}"
            leg_["path_duration"] = path_overall_duration
            leg_["path_n_legs"] = path_n_legs
            if simtab is not None:
                leg_sim = simtab.legs.loc[leg_id]
                leg_["avg_load_factor"] = leg_sim["avg_load_factor"]
                leg_["avg_local"] = leg_sim["avg_local"]
            leg_data.append(leg_)

            if prior_leg_arr_time is not None:
                connect_ = {}
                connect_["path_id"] = path.path_id
                connect_["carrier"] = leg["carrier"]
                connect_["layover_start"] = prior_leg_arr_time * 1000 + tz_offset_milliseconds
                connect_["layover_end"] = leg["dep_time"] * 1000 + tz_offset_milliseconds
                connect_["connection_time_minutes"] = (leg["dep_time"] - prior_leg_arr_time) / 60
                connect_["path_duration"] = path_overall_duration
                connect_["rownum"] = path_num
                connect_["path_n_legs"] = path_n_legs
                connect_["location"] = leg["orig"]
                connect_data.append(connect_)

            prior_leg_arr_time = leg["arr_time"]

    df = pd.DataFrame(leg_data)
    df["dep_time_nominal"] = pd.to_datetime(df["dep_hour_local"], unit="h").dt.strftime("%H:%M")
    df["arr_time_nominal"] = pd.to_datetime(df["arr_hour_local"], unit="h").dt.strftime("%H:%M")
    df_connect = pd.DataFrame(connect_data)
    df_connect["layover_duration"] = pd.to_datetime(
        (df_connect["layover_end"] - df_connect["layover_start"]) / 1000, unit="s"
    ).dt.strftime("%H:%M")

    if raw_df:
        return df

    height = len(df) * 10
    if height < min_height:
        height = min_height
    if height > max_height:
        height = max_height

    timeaxis = alt.Axis(format="%a %H:%M", labelAngle=-25)

    title = f"Paths Schedules {orig}~{dest}"

    if simtab is None:
        # Determine if all carriers have well-defined colors
        try:
            carrier_colors = {}
            for c in df["carrier"].unique():
                carrier_colors[c] = common_carrier_colors(c, error_if_missing=True)
            color_scale = alt.Scale(domain=list(carrier_colors.keys()), range=list(carrier_colors.values()))
        except KeyError:
            color_scale = alt.Scale()
        colors = alt.Color("carrier:N", title="Carrier", scale=color_scale)
    else:
        colors = alt.Color("avg_load_factor:Q", title="Avg Load Factor", legend=alt.Legend(format=".2f"))

    leg_bars = (
        alt.Chart(df)
        .mark_rule(strokeWidth=7)
        .encode(
            x=alt.X(
                "dep_time:T",
                title=f"Time ({timezone})",
                axis=timeaxis,
                scale=alt.Scale(type="utc"),
            ),
            x2="arr_time:T",
            y=alt.Y("rownum:N", axis=None),
            color=colors,
            tooltip=[
                alt.Tooltip("path_id:N", title="Path ID"),
                alt.Tooltip("carrier:N", title="Carrier"),
                alt.Tooltip("path_n_legs:Q", title="Number of Legs"),
                alt.Tooltip("path_duration:N", title="Path Overall Duration"),
                alt.Tooltip("leg_id:N", title="Leg ID"),
                alt.Tooltip("leg_duration:N", title="Leg Duration"),
                alt.Tooltip("orig:N", title="Leg Orig"),
                alt.Tooltip("dep_time_nominal:N", title="Departure Time"),
                alt.Tooltip("dest:N", title="Leg Dest"),
                alt.Tooltip("arr_time_nominal:N", title="Arrival Time"),
            ]
            + [
                alt.Tooltip("avg_load_factor:Q", title="Avg Load Factor", format=".2f"),
                alt.Tooltip("avg_local:Q", title="Avg Local", format=".2f"),
            ]
            if simtab is not None
            else [],
        )
    )

    connect_bars = (
        alt.Chart(df_connect)
        .mark_rule(strokeWidth=3, strokeDash=[5, 1])
        .encode(
            x=alt.X(
                "layover_start:T",
                title=f"Time ({timezone})",
                axis=timeaxis,
                scale=alt.Scale(type="utc"),
            ),
            x2="layover_end:T",
            y=alt.Y("rownum:N", axis=None),
            tooltip=[
                alt.Tooltip("path_id:N", title="Path ID"),
                alt.Tooltip("carrier:N", title="Carrier"),
                alt.Tooltip("path_n_legs:Q", title="Number of Legs"),
                alt.Tooltip("path_duration:N", title="Path Overall Duration"),
                alt.Tooltip("location:N", title="Connection at"),
                alt.Tooltip("layover_duration:N", title="Layover Duration"),
            ],
        )
    )

    if simtab is not None:
        # do not color connect bars with data, it's too busy
        connect_bars = connect_bars.mark_rule(color="#cccccc")
    elif color_connect_time:
        connect_bars = connect_bars.encode(
            color=alt.Color("connection_time_minutes:Q", title="Connection Time", legend=alt.Legend(gradientLength=65)),
        )
    else:
        connect_bars = connect_bars.encode(
            color=colors,
        )

    return (
        (leg_bars + connect_bars).resolve_scale(color="independent").properties(width=width, height=height, title=title)
    )


def fig_reference_price_by_distance(cfg: Config) -> alt.Chart:
    """Create a figure showing reference price versus distance for all demands.

    Each point represents one demand record. Points are colored by passenger
    segment, making it easy to verify that reference prices scale
    appropriately with distance and to compare pricing assumptions across
    segments.

    Parameters
    ----------
    cfg : Config
        Configuration object containing demand definitions (with ``distance``
        and ``reference_price`` columns in the demands dataframe).

    Returns
    -------
    alt.Chart
        An Altair scatter chart with distance (miles) on the x-axis and
        reference price (dollars) on the y-axis.
    """
    return (
        alt.Chart(cfg.dataframes.demands)
        .mark_point(filled=True)
        .encode(
            x=alt.X("distance:Q", axis=alt.Axis(format=",.0f"), title="Distance (miles)"),
            y=alt.Y("reference_price:Q", axis=alt.Axis(format="$,.0f"), title="Reference Price"),
            color="segment:N",
            tooltip=["orig", "dest", "distance", "reference_price", "segment"],
        )
    )


def fig_maximum_fare_ratio_by_distance(cfg: Config) -> alt.Chart:
    """Create a figure showing the maximum fare ratio versus distance by market and carrier.

    The fare ratio for a market is defined as the highest published fare
    divided by the lowest published fare. Plotting this against distance
    helps verify that the fare ladder is consistent and that high-fare
    buckets are priced at a sensible premium relative to the lowest bucket.

    Parameters
    ----------
    cfg : Config
        Configuration object containing fares and demands (used to look up
        market distances).

    Returns
    -------
    alt.Chart
        An Altair scatter chart with distance (miles) on the x-axis and the
        maximum-to-minimum fare ratio on the y-axis, colored by carrier.
    """
    frat = (
        cfg.dataframes.fares.groupby(["carrier", "orig", "dest"])
        .agg(
            Max_Price=("price", "max"),
            Min_Price=("price", "min"),
        )
        .eval("FareRatio = Max_Price / Min_Price")
    )
    dist = cfg.dataframes.demands.groupby(["orig", "dest"]).agg(
        distance=("distance", "max"),
    )
    df = frat.join(dist).reset_index()

    # Determine if all carriers have well-defined colors
    try:
        carrier_colors = {}
        for c in df["carrier"].unique():
            carrier_colors[c] = common_carrier_colors(c, error_if_missing=True)
        color_scale = alt.Scale(domain=list(carrier_colors.keys()), range=list(carrier_colors.values()))
    except KeyError:
        color_scale = alt.Scale()

    return (
        alt.Chart(df)
        .mark_point(filled=True)
        .encode(
            x=alt.X("distance:Q", axis=alt.Axis(format=",.0f"), title="Distance (miles)"),
            y=alt.Y("FareRatio", title="Maximum Fare Ratio"),
            color=alt.Color("carrier:N", scale=color_scale),
            tooltip=["carrier", "orig", "dest", "distance", "FareRatio"],
        )
    )
