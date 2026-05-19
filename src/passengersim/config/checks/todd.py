from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
from passengersim_core import Demand as CoreDemand
from passengersim_core import Offer

from passengersim import Simulation
from passengersim.config import Config
from passengersim.config.demands import Demand
from passengersim.config.fares import Fare
from passengersim.config.legs import Leg
from passengersim.config.places import get_mileage
from passengersim.utils.airport_lookup import lookup_airport

from .choice_models import _create_sim_for_only_one_demand

try:
    from KDEpy import FFTKDE
except ImportError as e:
    raise ValueError("KDEpy must be installed to use these checks") from e


def check_todd_curves(
    cfg: Config,
    orig: str = "ISP",
    dest: str = "LSE",
    carrier: str | None = None,
    segment: str = "business",
    *,
    raw_df: bool = False,
    also_none: bool = False,
    n_draws: int = 100_000,
):

    cfg = cfg.model_copy(deep=True)

    # check that there are todd curves defined
    if len(cfg.todd_curves) == 0:
        raise ValueError("No todd curves defined in config")

    # check that a choice model exists for this segment
    if segment not in cfg.choice_models:
        raise ValueError(f"Invalid choice model: {segment}")

    # ensure the two places are included
    place_o = cfg.places[orig] = lookup_airport(orig)
    place_d = cfg.places[dest] = lookup_airport(dest)

    distance = get_mileage(cfg.places, orig, dest)
    if distance < cfg.simulation_controls.speed_limits.SHORT_HAUL_MAX_DISTANCE:
        mph = cfg.simulation_controls.speed_limits.SHORT_HAUL_MAXIMUM_SPEED - 25
    elif distance > cfg.simulation_controls.speed_limits.LONG_HAUL_MIN_DISTANCE:
        mph = cfg.simulation_controls.speed_limits.LONG_HAUL_MINIMUM_SPEED + 25
    else:
        mph = (
            cfg.simulation_controls.speed_limits.SHORT_HAUL_MAXIMUM_SPEED
            + cfg.simulation_controls.speed_limits.LONG_HAUL_MINIMUM_SPEED
        ) / 2
    duration = distance / mph * 3600

    # pick a carrier if none was set
    if carrier is None:
        carrier = next(iter(cfg.carriers))

    cfg.carriers[carrier].classes = ["Y99"]

    # set legs departing every hour
    new_legs = []
    for h in range(1, 24):
        dep_timestamp = datetime.fromisoformat(f"2023-01-07T{h:02d}:06:00").replace(tzinfo=ZoneInfo(place_o.time_zone))
        dep_unixtime = dep_timestamp.timestamp()
        arr_unixtime = dep_unixtime + duration
        arr_timestamp = datetime.fromtimestamp(arr_unixtime, tz=ZoneInfo(place_d.time_zone))
        arr_day = arr_timestamp.day - 7

        leg = Leg(
            orig=orig,
            dest=dest,
            dep_time=f"{h:02d}:06",
            arr_time=f"{arr_timestamp.hour:02d}:{arr_timestamp.minute:02d}",
            arr_day=arr_day,
            date="2023-01-07",
            capacity=70,
            leg_id=9000 + h,
            fltno=9000 + h,
            carrier=carrier,
        )
        new_legs.append(leg)

    cfg.legs = new_legs
    cfg.paths = []
    cfg.simulation_controls.connection_builder.existing_paths = "keep"

    # create a demand
    any_booking_curve = next(iter(cfg.booking_curves))
    any_todd_curve = next(iter(cfg.todd_curves))
    cfg.demands = [
        Demand(
            orig=orig,
            dest=dest,
            base_demand=100,
            segment=segment,
            reference_price=100,
            curve=any_booking_curve,
            todd_curve=any_todd_curve,
        ),
    ]

    f = Fare(
        carrier=carrier,
        orig=orig,
        dest=dest,
        booking_class="Y99",
        price=100.0,
        advance_purchase=0,
        restrictions=[],
        cabin="Y",
    )
    cfg.fares.append(f)

    sim = Simulation(cfg)
    sim.setup_scenario()
    pths = [p for p in sim.paths.set_filters(**{"orig": orig, "dest": dest})]
    dmd = sim.demands.select(orig=orig, dest=dest, segment=segment)
    f = sim.fares.select(orig=orig, dest=dest, booking_class="Y99")

    offers = [Offer(p, f, 100.0) for p in pths]

    # t = time.time()

    choo = []
    for tc in sim.todd_curves.values():
        dmd.dwm = tc
        # print(f"Simulating choices for {tc.name}...", end="")
        choo.append(
            pd.Series(dmd.simulate_choices(offers, n_draws=n_draws), index=[p.path_id for p in pths], name=dmd.dwm.name)
        )
        # print(f"Done in {time.time() - t:.2f} seconds.")
        # t = time.time()

    if also_none:
        dmd.dwm = None
        # print("Simulating choices for No TODD Curve...", end="")
        choo.append(
            pd.Series(dmd.simulate_choices(offers, n_draws=n_draws), index=[p.path_id for p in pths], name="None")
        )
        # print(f"Done in {time.time() - t:.2f} seconds.")

    df = (
        pd.concat(choo, axis=1)
        .rename_axis(index="path_id")
        .reset_index()
        .melt(id_vars="path_id", var_name="todd_curve", value_name="choices")
    )
    df["dep_hour"] = df["path_id"] % 9000

    if raw_df:
        return df

    step = 20
    overlap = 0

    return (
        alt.Chart(df, height=step)
        .mark_area(interpolate="monotone", fillOpacity=0.8, stroke="lightgray", strokeWidth=0.5)
        .encode(
            alt.X("dep_hour:Q").title("Departure Hour"),
            alt.Y("choices:Q", stack="center", axis=None, scale=alt.Scale(range=[step, -step * overlap])),
            # alt.Fill('mean_temp:Q')
            #     .legend(None)
            #     .scale(domain=[30, 5], scheme='redyellowblue')
        )
        .facet(
            row=alt.Row("todd_curve:N", sort=sorted(sim.todd_curves.keys()))
            .title(None)
            .header(labelAngle=0, labelAlign="left")
        )
        .properties(
            title="Choice Distributions Across TODD Curves",
            # bounds='flush'
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )


def generate_time_windows(dmd: CoreDemand, n: int = 60) -> pd.DataFrame:

    if not isinstance(dmd, CoreDemand):
        raise TypeError("dmd must be a core.Demand object")

    prng = np.random.default_rng(seed=0)
    dw = dmd.dwm
    if dw is None:
        raise ValueError("demand has no decision window model")

    # draw random windows
    windows = np.empty([n, 3])
    for i in range(n):
        midpoint = windows[i, 0] = dw.get_midpoint(prng.uniform())
        windows[i, 1:] = dw.get_window(dmd, int(midpoint), 0)

    df = (
        pd.DataFrame(windows, columns=["midpoint", "start", "end"])
        .apply(lambda x: pd.to_datetime(x, unit="s"))
        .rename_axis(index="draw")
        .reset_index()
    )
    df["total_seconds"] = (df["end"] - df["start"]).dt.total_seconds()
    df["wiggle_seconds"] = df["total_seconds"] - dmd.get_min_delta_t()
    return df


def _pack_windows_min_groups(windows, *, min_space: float | timedelta = 1.0):
    """
    Pack windows into the minimum number of non-overlapping groups.

    Uses the classic interval-partitioning greedy algorithm with a min-heap of
    group end times, which runs in O(n log k) time (n windows, k groups).

    Assumption:
    - Each Window is interpreted as an interval from min(start, end) to max(start, end).
    - If allow_touching=True, [a, b] and [b, c] are considered non-overlapping.
    """
    # Normalize windows so intervals are always (lo <= hi), preserving original objects.
    normalized = [(min(w.start, w.end), max(w.start, w.end), w) for w in windows]
    normalized.sort(key=lambda x: (x[0], x[1]))  # sort by interval start, then end

    groups = []  # groups[group_id] -> list[Window]
    active_groups = []  # min-heap of (current_group_end, group_id)

    for lo, hi, w in normalized:
        if active_groups:
            earliest_end, gid = active_groups[0]
            can_reuse = lo > earliest_end + min_space
            if can_reuse:
                # Reuse the group that frees up first.
                heapq.heapreplace(active_groups, (hi, gid))
                groups[gid].append(w)
                continue

        # Otherwise create a new group.
        gid = len(groups)
        groups.append([w])
        heapq.heappush(active_groups, (hi, gid))

    return groups


def _pack_window_rows(df: pd.DataFrame, *, min_space: float | timedelta = 1.0):
    recombine = {}
    if isinstance(min_space, float):
        min_space = timedelta(seconds=min_space)
    for rownum, grp in enumerate(_pack_windows_min_groups(df.itertuples(index=False), min_space=min_space), start=1):
        recombine[rownum] = pd.DataFrame(grp)
    return pd.concat(recombine).reset_index(level=0).rename(columns={"level_0": "rownum"})


def _check_time_windows(
    df: pd.DataFrame,
    dmd: CoreDemand,
    color="#AA0000",
    n: int = 60,
    width: int | None = None,
    height: int | None = None,
    pack: bool = False,
) -> alt.LayerChart:
    if len(df) > n:
        df = df.sample(n)

    df = df.drop(columns=["draw"])
    df = df.reset_index(drop=True)
    df = df.rename_axis(index="draw")
    df = df.reset_index()
    if pack:
        df = _pack_window_rows(df, min_space=timedelta(seconds=900))
    else:
        df["rownum"] = df.index

    props = {}
    if width is not None:
        props["width"] = width
    if height is not None:
        props["height"] = height

    chart = alt.Chart(df, title=f"Decision Windows for {dmd.identifier} Demand", **props)
    caps = chart.mark_point(shape="stroke", angle=90, color=color, strokeWidth=1).encode(y=alt.Y("rownum", axis=None))
    figure = (
        chart.mark_rule(color=color).encode(
            x=alt.X("start:T", axis=alt.Axis(title="Time of Day", format="%H:%M")),
            x2=alt.X2("end:T"),
            y=alt.Y("rownum", axis=None),
        )
        + caps.encode(x=alt.X("start:T"))
        + caps.encode(x=alt.X("end:T"))
    )
    return figure


def _check_wiggle_room(df: pd.DataFrame):
    # Compute kernel density estimate on schedule wiggle room
    x, y1 = FFTKDE(bw="silverman").fit(df["wiggle_seconds"].to_numpy().reshape(-1, 1)).evaluate(2**10)
    return (
        alt.Chart(pd.DataFrame({"wiggle_hours": x / 3600, "density": y1}))
        .mark_area()
        .encode(
            x=alt.X("wiggle_hours:Q", title="Schedule Tolerance (hours)", scale=alt.Scale(domain=(0, x.max() / 3600))),
            y=alt.Y("density:Q", axis=None),
        )
    )


def _check_edge_times(df: pd.DataFrame, side: Literal["start", "end"] = "start"):
    starts = (df[side].astype("int64") // 10**9).to_numpy().reshape(-1, 1)
    # module for overnight wraparound
    starts = starts % (24 * 60 * 60)
    # mirror the data in a second block to account for wraparound
    data_extended = np.concatenate([starts, starts + 86400])
    data_extended = np.where(data_extended < 129600, data_extended, data_extended - 172800)
    data_extended = data_extended.astype(np.float64)
    data_extended /= 3600
    data_extended = np.clip(data_extended, -11.99, 35.99)
    # Compute kernel density estimate on start times
    x = np.linspace(-12, 36, 2048)
    kde = FFTKDE(bw="silverman").fit(data_extended)
    try:
        y1 = kde.evaluate(x)
    except ValueError:
        print(f"{x.min()=}, {x.max()=}")
        print(f"{data_extended.min()=}, {data_extended.max()=}")
        raise
    mask = (x >= 0) & (x < 24)
    # convert to datetime formatting
    df2 = pd.DataFrame({side: x[mask] * 3600, "density": y1[mask]})
    df2[side] = pd.to_datetime(df2[side], unit="s")
    return (
        alt.Chart(df2)
        .mark_area()
        .encode(
            x=alt.X(f"{side}:T", axis=alt.Axis(title=f"Window {side.title()} Times", format="%H:%M")),
            y=alt.Y("density:Q", axis=None),
        )
    )


def check_time_windows(
    obj: CoreDemand | Config,
    *,
    orig: str | None = None,
    dest: str | None = None,
    carrier: str | None = None,
    segment: str | None = None,
    n_draws: int = 100_000,
) -> alt.ConcatChart:
    """Generate a dashboard of checks for the decision windows of a demand.

    This dashboard includes a plot of a sampling of the windows themselves, as well as
    checks on the distribution of start and end times and the amount of schedule
    tolerance (i.e. wiggle room within each window above the minimum travel time).

    Parameters
    ----------
    dmd : passengersim.core.Demand
        The core demand object to check. Must have a decision window model defined.
    n_draws : int, optional
        The number of random windows to draw for the checks, by default 100_000.

    Returns
    -------
    alt.Chart
    """
    if isinstance(obj, Config):
        if orig is None:
            orig = "ISP"
        if dest is None:
            dest = "LSE"
        if segment is None:
            segment = set(obj.choice_models.keys()).pop()
        sim = _create_sim_for_only_one_demand(obj, orig=orig, dest=dest, carrier=carrier, segment=segment)
        dmd = sim.demands.select(orig=orig, dest=dest, segment=segment)
    else:
        dmd = obj
    df = generate_time_windows(dmd, n=n_draws)

    start_end_times = _check_edge_times(df, side="start").properties(height=125, width=200) | _check_edge_times(
        df, side="end"
    ).properties(height=125, width=200)
    wiggle_room = _check_wiggle_room(df).properties(height=120, width=420)

    dashboard_right = start_end_times & wiggle_room

    return _check_time_windows(df, dmd) | dashboard_right
