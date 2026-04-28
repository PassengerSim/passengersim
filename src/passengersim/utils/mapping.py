import json
from typing import Literal

import altair as alt
import geopandas as gpd
import pandas as pd
import pyproj
import vega_datasets
from shapely.geometry import LineString, Point

from passengersim.config import Leg, Place


def north_america_basemap():
    projection_args = dict(
        type="conicEqualArea",
        center=[0.1, 41],
        parallels=[20, 60],
        rotate=[96, 0, 0],
        scale=450,
    )
    projection_args = dict(
        type="equirectangular",
        center=[-100, 41],
        # parallels=[20, 60],
        # rotate=[96, 0, 0],
        # scale=450,
    )
    source = alt.topo_feature(vega_datasets.data.world_110m.url, "countries")
    basemap = (
        alt.Chart(source, width=500, height=300).mark_geoshape(fill="none", stroke="gray").project(**projection_args)
    )
    return basemap, projection_args


def world_map():
    projection_args = dict(type="naturalEarth1")
    source = alt.topo_feature(vega_datasets.data.world_110m.url, "countries")
    basemap = (
        alt.Chart(source, width=500, height=300).mark_geoshape(fill="none", stroke="gray").project(**projection_args)
    )
    return basemap, projection_args


def route_map(
    basemap_name: Literal["north-america"],
    legs: list[Leg],
    places: dict[str, Place],
    *,
    carrier: str | None = None,
    color_on: Literal["n_legs", "n_seats", None] = None,
    line_color: str = "red",
    title: str = "Flight Routes",
    raw_geojson: bool = False,
):
    if basemap_name == "north-america":
        basemap, projection_args = north_america_basemap()
    else:
        raise ValueError(f"Unknown basemap: {basemap_name}")

    basemap2, projection2_args = world_map()

    points = {}
    conus = {}
    for place_code, place_data in places.items():
        points[place_code] = Point(place_data.lon, place_data.lat)
        if place_data.country is None:
            from .geocoding import is_conus

            conus[place_code] = is_conus(place_data.lat, place_data.lon)
        else:
            conus[place_code] = place_data.country == "US"

    geod = pyproj.Geod(ellps="WGS84")

    leg_lines = {}
    international_lines = {}
    market_count = {}
    market_seats = {}
    n_legs = 0
    n_legs_by_place = {}
    for leg in legs:
        if carrier is not None and leg.carrier != carrier:
            continue
        market = f"{leg.orig}-{leg.dest}"  # (leg.orig, leg.dest)
        market_count[market] = market_count.get(market, 0) + 1
        market_seats[market] = market_seats.get(market, 0) + leg.capacity
        n_legs += 1
        n_legs_by_place[leg.orig] = n_legs_by_place.get(leg.orig, 0) + 1
        n_legs_by_place[leg.dest] = n_legs_by_place.get(leg.dest, 0) + 1
        orig = points[leg.orig]
        dest = points[leg.dest]
        dist = leg.distance or 1000
        npts = max(30, int(dist / 100))  # at least 10 points, more for longer distances
        leg_coords = geod.npts(orig.x, orig.y, dest.x, dest.y, npts=npts)
        leg_coords.insert(0, (orig.x, orig.y))
        leg_coords.append((dest.x, dest.y))
        if conus[leg.orig] and conus[leg.dest]:
            leg_lines[market] = LineString(leg_coords)
        else:
            international_lines[market] = LineString(leg_coords)

    gdf_conus = gpd.GeoDataFrame(
        list(leg_lines.keys()),
        geometry=list(leg_lines.values()),
        crs="EPSG:4326",
        columns=["market"],
    ).set_index("market")

    gdf_intl = gpd.GeoDataFrame(
        list(international_lines.keys()),
        geometry=list(international_lines.values()),
        crs="EPSG:4326",
        columns=["market"],
    ).set_index("market")

    point_data = []
    for pointcode, pointvalue in points.items():
        if pointcode not in n_legs_by_place:
            continue
        shape = "circle"
        if n_legs_by_place[pointcode] > n_legs * 0.2:
            shape = "star"
        point_data.append(
            {
                "place_code": pointcode,
                "latitude": pointvalue.y,
                "longitude": pointvalue.x,
                "place_shape": shape,
            }
        )

    if color_on == "n_legs":
        color_tag = "n_legs"
        color_label = "Number of Legs"
        df = pd.DataFrame(
            list(market_count.items()),
            columns=["market", "n_legs"],
        ).set_index("market")
    elif color_on == "n_seats":
        color_tag = "n_seats"
        color_label = "Number of Seats"
        df = pd.DataFrame(list(market_seats.items()), columns=["market", "n_seats"]).set_index("market")
    elif color_on is None:
        color_tag = "constant"
        color_label = ""
        df = pd.DataFrame(
            {"market": list(leg_lines.keys()), "constant": [1] * len(leg_lines)},
            columns=["market", "constant"],
        ).set_index("market")
    else:
        raise ValueError(f"Unknown color_on value: {color_on}")

    # print("gdf_conus")
    # print(gdf_conus)
    # print("gdf_intl")
    # print(gdf_intl)

    flights_geojson = json.loads(gdf_conus.to_json())
    intl_geojson = json.loads(gdf_intl.to_json())
    if raw_geojson:
        return flights_geojson, intl_geojson

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
        if color_on is None:
            return (
                alt.Chart(alt.Data(values=geojson["features"]))
                .mark_geoshape(
                    stroke=line_color,
                    strokeWidth=1,
                    fill="none",
                )
                .project(**proj_args)
            )
        else:
            return (
                alt.Chart(alt.Data(values=geojson["features"]))
                .mark_geoshape(
                    strokeWidth=1,
                    fill="none",
                )
                .transform_lookup(lookup="market", from_=alt.LookupData(df, "market", [color_tag]))
                .encode(
                    stroke=alt.Stroke(
                        f"{color_tag}:Q",
                        scale=alt.Scale(scheme="reds"),
                        legend=alt.Legend(title=color_label),
                    ),
                )
                .project(**proj_args)
            )

    place_points_conus = _make_place_points(point_data, projection_args)
    place_points_intl = _make_place_points(point_data, projection2_args)
    route_lines = _make_route_lines(flights_geojson, projection_args)
    route_lines_intl = _make_route_lines(intl_geojson, projection2_args)

    if title:
        if carrier is not None:
            title = f"{carrier} {title}"
        route_lines = route_lines.properties(title=title)
    return (route_lines + basemap + place_points_conus) | (route_lines_intl + basemap2 + place_points_intl)


PROJECTION1_CONUS = dict(
    type="albers",
    center=[0, 38],
    scale=550,
)
PROJECTION2_CONUS = dict(
    type="albers",
    center=[-27, 35],
    scale=550,
)
PROJECTION2_WORLD = dict(
    type="equalEarth",
    center=[-160, 0],
    scale=90,
)
PROJECTION1_WORLD = dict(
    type="equalEarth",
    center=[0, 0],
    scale=100,
)


def conus_only() -> alt.Chart:
    source_conus = alt.topo_feature(vega_datasets.data.us_10m.url, "states")
    base_usa = (
        alt.Chart(source_conus, width=430, height=275)
        .mark_geoshape(fill="none", stroke="#cccccc")
        .project(**PROJECTION1_CONUS)
    )
    return base_usa


def world_only() -> alt.Chart:
    source_world = alt.topo_feature(vega_datasets.data.world_110m.url, "countries")
    base_world = (
        alt.Chart(source_world, width=450, height=275)
        .mark_geoshape(fill="none", stroke="#cccccc")
        .project(**PROJECTION1_WORLD)
    )
    return base_world


def conus_and_world() -> tuple[alt.Chart, alt.Chart]:
    # due to some weirdness with hconcat maps in altair, the center points
    # of concatenated maps need to be adjusted
    source_conus = alt.topo_feature(vega_datasets.data.us_10m.url, "states")
    source_world = alt.topo_feature(vega_datasets.data.world_110m.url, "countries")
    base_usa = (
        alt.Chart(source_conus, width=430, height=275)
        .mark_geoshape(fill="none", stroke="#cccccc")
        .project(**PROJECTION2_CONUS)
    )
    base_world = (
        alt.Chart(source_world, width=450, height=275)
        .mark_geoshape(fill="none", stroke="#cccccc")
        .project(**PROJECTION2_WORLD)
    )
    return base_usa, base_world


def legs_conus_and_world(
    legs: list[Leg],
    places: dict[str, Place],
    *,
    carrier: str | None = None,
    line_color: str = "red",
    title: str = "Flight Routes",
    raw_geojson: bool = False,
):
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
            from .geocoding import is_conus

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
    intl_geojson = json.loads(gdf_intl.to_json())
    if raw_geojson:
        return conus_geojson, intl_geojson

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

    if not intl_geojson["features"]:
        # there are no international legs, so just return the CONUS map
        place_points_conus = _make_place_points(point_data_conus, PROJECTION1_CONUS)
        route_lines = _make_route_lines(conus_geojson, PROJECTION1_CONUS)
        base_conus = conus_only()
        if title:
            if carrier is not None:
                title = f"{carrier} {title}"
            route_lines = route_lines.properties(title=title)
        return route_lines + base_conus + place_points_conus

    place_points_conus = _make_place_points(point_data_conus, PROJECTION2_CONUS)
    place_points_intl = _make_place_points(point_data_world, PROJECTION2_WORLD)
    route_lines = _make_route_lines(conus_geojson, PROJECTION2_CONUS)
    route_lines_intl = _make_route_lines(intl_geojson, PROJECTION2_WORLD)

    base_conus, base_world = conus_and_world()
    if title:
        if carrier is not None:
            title = f"{carrier} {title}"
        route_lines = route_lines.properties(title=title)
    return (route_lines + base_conus + place_points_conus) | (route_lines_intl + base_world + place_points_intl)
