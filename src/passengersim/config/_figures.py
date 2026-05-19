from __future__ import annotations

import json
import warnings

import altair as alt
import numpy as np
import pandas as pd

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
    "plotly_route_map",
    "fig_demand_vs_capacity",
    "fig_max_wtp_distributions",
    "fig_hub_schedule",
]


def fig_booking_curves(cfg: Config, raw_df: bool = False) -> alt.Chart:
    """Create a figure showing all booking curves in the config."""
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
    """Create a figure showing all Frat5 curves in the config."""
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
    """Create a figure showing the route map."""
    return _legs_conus_or_world(cfg.legs, cfg.places, title="Route Map", line_color="red", carrier=carrier)


def _legs_as_geodataframe(
    legs: list[Leg],
    places: dict[str, Place],
    *,
    carrier: str | None = None,
    fix_longitude_wrap: bool = True,
) -> gpd.GeoDataFrame:

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
    """Figure showing leg speed by distance."""

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

    return (
        alt.Chart(df)
        .mark_point(shape="wedge", filled=True, size=300)
        .encode(
            x=alt.X("distance", title="Leg Distance (mi)"),
            y=alt.Y("speed", title="Gate-to-Gate Average Speed (mph)"),
            color=alt.Color("carrier"),
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
    center_lon: float = -90,
    center_lat: float = 10,
    projection_type: str = "robinson",
    height: int = 400,
    title: str | bool = True,
):
    """Plot route map using plotly."""

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


def fig_demand_vs_capacity(cfg: Config):
    """Figure showing demand vs. capacity by place."""

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
    """
    Figure showing cumulative distribution of maximum willingness to pay by choice model.

    Parameters
    ----------
    cfg : Config
        Use the choice models defined in this config.
    reference_price : float
        Willingness to pay is computed relative to this reference price. In general, WTP
        scales with a reference fare, so the generated figure needs to have a reference
        fare to define a concrete scale.
    n_draws : int
        Number of draws to generate. The WTP distribution is estimated by simulating a
        large number of draws from the choice model, and computing the maximum WTP for
        each draw. More draws will lead to a smoother and more accurate estimate of the
        distribution, but will take more time to compute (but this should run pretty
        fast even with a million draws).
    lower_bound, upper_bound : float, optional
        Lower and upper bounds for the WTP thresholds shown on the x-axis. If
        upper_bound is None, it will be set to 5 times the maximum reference fare.
    random_seed : int, optional
        Random seed for reproducibility of the draws.
    raw_df
        If True, return the raw dataframe underlying the chart, instead of the chart itself.

    Returns
    -------
    alt.Chart or pd.DataFrame
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
    min_height: int = 200,
    max_height: int = 600,
) -> alt.TopLevelMixin:
    """Visualize the schedule at a given hub.

    This does not show the actually allowed connections, just the whole schedule
    of arrivals and departures at the hub, which can be used to visually inspect
    potential connection opportunities and the overall structure of the schedule
    at that hub.
    """
    base_df = cfg.dataframes.legs
    if carrier is not None:
        base_df = base_df[base_df["carrier"] == carrier]
    df1 = base_df.query(f"orig == '{hub}'")
    df1 = df1[
        ["orig", "dest", "dep_hour_local", "arr_hour_local", "duration_minutes", "leg_id", "fltno", "carrier"]
    ].eval("duration_hours = duration_minutes / 60")
    df1 = df1.eval("hub_hour = dep_hour_local").eval("far_hour = dep_hour_local + duration_hours")

    df2 = base_df.query(f"dest == '{hub}'")
    df2 = df2[
        ["orig", "dest", "arr_hour_local", "dep_hour_local", "duration_minutes", "leg_id", "fltno", "carrier"]
    ].eval("duration_hours = duration_minutes / 60")
    df2 = df2.eval("hub_hour = arr_hour_local").eval("far_hour = arr_hour_local - duration_hours")

    df = (
        pd.concat([df1, df2], axis=0)
        .sort_values(by=["orig", "dest", "hub_hour"])
        .sort_values(by=["hub_hour"])
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(columns={"index": "rownum"})
    )
    df["hub_time"] = pd.to_datetime(df["hub_hour"], unit="h").dt.strftime("%H:%M")
    df["dep_time"] = pd.to_datetime(df["dep_hour_local"], unit="h").dt.strftime("%H:%M")
    df["arr_time"] = pd.to_datetime(df["arr_hour_local"], unit="h").dt.strftime("%H:%M")

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

    return alt.Chart(df.query(f"orig == '{hub}'")).mark_rule(point={"shape": "diamond"}).encode(
        x=alt.X("hub_hour", title=f"Time at {hub} (local hour)"),
        x2="far_hour",
        y=alt.Y("rownum:N", axis=None),
        color="carrier",
        tooltip=[
            "orig",
            "dest",
            alt.Tooltip("dep_time:N", title=f"Departure Time ({hub})"),
            alt.Tooltip("arr_time:N", title="Arrival Time"),
            "leg_id",
            "fltno",
            "carrier",
        ],
    ).properties(width=width, height=height) + alt.Chart(df.query(f"dest == '{hub}'")).mark_rule(
        point={"shape": "diamond"}
    ).encode(
        x=alt.X("hub_hour", title=f"Time at {hub} (local hour)"),
        x2="far_hour",
        y=alt.Y("rownum:N", axis=None),
        color="carrier",
        tooltip=[
            "orig",
            "dest",
            alt.Tooltip("dep_time:N", title="Departure Time"),
            alt.Tooltip("hub_time:N", title=f"Arrival Time ({hub})"),
            "leg_id",
            "fltno",
            "carrier",
        ],
    ).properties(width=width, height=height, title=title)
