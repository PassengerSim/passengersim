from __future__ import annotations

import warnings
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
from passengersim_core import Offer
from passengersim_core._Zoo import _random_max_wtp

from passengersim import Simulation
from passengersim.config import Config
from passengersim.config.choice_model import LogitChoiceModel, PodsChoiceModel
from passengersim.config.legs import Leg
from passengersim.config.places import get_mileage
from passengersim.utils.airport_lookup import lookup_airport


def _interpolate_series_with_new_index(s: pd.Series, new_index: np.ndarray) -> pd.Series:

    if isinstance(new_index, (int, float, np.integer, np.floating)):
        new_index = np.asarray([new_index])
    # concat + sort creates NaN at new index locations
    # interpolate(method='index') uses index values for linear calculation
    interpolated_series = (
        pd.concat([s, pd.Series(index=new_index, dtype=float)])
        .sort_index()
        .interpolate(method="index")  # 'index' uses numerical distance
    )
    # Remove duplicate entries if old and new indices overlapped
    interpolated_series = interpolated_series[~interpolated_series.index.duplicated(keep="first")]
    return interpolated_series.reindex(new_index)  # only keep new points


def _get_wtp_from_quantile(quantile: float, reference_price: float, emult: float):
    assert 0 <= quantile <= 1
    return reference_price - ((np.log(quantile) * reference_price * (emult - 1.0)) / 0.6931471806)  # log(2)


def _get_quantile_from_wtp(wtp_price: float, reference_price: float, emult: float):
    return min(1.0, np.exp((-np.log(2) * (wtp_price - reference_price)) / (reference_price * (emult - 1.0))))


def _get_frat5(wtp_price: float, reference_price: float, emult: float):
    quantile = _get_quantile_from_wtp(wtp_price, reference_price, emult)
    wtp2 = _get_wtp_from_quantile(quantile / 2, reference_price, emult)
    return wtp2 / wtp_price


def _create_sim_for_only_one_demand(
    cfg: Config,
    orig: str = "ISP",
    dest: str = "LSE",
    via: str | None = None,
    carrier: str | None = None,
    segment: str = "business",
):
    # don't manipulate the config you were given, make a copy
    cfg = cfg.model_copy(deep=True)
    cfg.simulation_controls.allow_unused_restrictions = True

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

    # check if there are any paths on this carrier
    use_paths = cfg.paths.set_filters(orig=orig, dest=dest)
    if use_paths:
        use_leg_ids = set()
        for path in use_paths:
            use_leg_ids.update(path.legs)
        new_legs = [leg for leg in cfg.legs if leg.leg_id in use_leg_ids]
    else:
        # no paths serve this market, so make one up
        # set a legs departing at 9 am
        new_legs = []
        h = 9
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

    # check that a demand exist for this market and segment
    use_demands = [d for d in cfg.demands if d.orig == orig and d.dest == dest and d.segment == segment]
    cfg.demands = use_demands

    # check that fares exist for this market and carrier
    use_fares = [f for f in cfg.fares if f.carrier == carrier and f.orig == orig and f.dest == dest]
    cfg.fares = use_fares

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore")
        sim = Simulation(cfg)
    sim.setup_scenario()

    return sim


def check_choice_models(
    cfg: Config,
    orig: str = "ISP",
    dest: str = "LSE",
    via: str | None = None,
    carrier: str | None = None,
    segment: str = "business",
    *,
    raw_data: bool = False,
    n_draws: int = 100_000,
) -> alt.HConcatChart | tuple:
    """Analyze the choice model for a single demand.

    This tool will compare the distribution of choices across fares and the
    distribution of maximum willingness to pay used by the demand's choice
    model.  This is intended as a diagnostic tool to help understand how the
    choice model is working for a given demand, and how it relates to the
    fares available in the market.

    In the resulting dashboard, the first table shows the relative proportion
    of customers choosing each booking class, conditional on various possibilities
    for what is the lowest available booking class. This distribution will
    show both overall purchase rates and buyup probabilities driven by fare
    restrictions. The second table shows the customer's distribution of
    maximum willingness to pay, and the prices of the various fares contrasted
    against this distribution.

    Parameters
    ----------
    cfg : Config
    orig, dest, segment : str
        The origin, destination, and segment of the demand to be analyzed.
    via : str
        Not implemented.
    carrier : str
        Name of the carrier to analyze.  Choices will be generated based on
        this carrier's fares in the relevant market. If None, the first carrier
        in the config will be used.
    raw_data : bool, default False
        If True, return the raw data used to create the figures instead of the
        figures themselves.
    n_draws : int, optional
        The number of random draws to use for choice distributions.  The choice
        model distribution table is created by simulating this number of customers
        facing each pool of offers.

    Returns
    -------
    alt.Chart | tuple
        If `raw_data` is False, an Altair chart object containing the resulting
        dashboard.  If `raw_data` is True, a tuple containing the raw data used
        to create the figures: a DataFrame of choice model results, a Series of
        willingness to pay distribution points, and a DataFrame of fare prices
        and related information.
    """
    sim = _create_sim_for_only_one_demand(cfg, orig, dest, via, carrier, segment)

    pth = sim.paths.select(**{"orig": orig, "dest": dest})
    dmd = sim.demands.select(orig=orig, dest=dest, segment=segment)
    far = sim.fares(orig=orig, dest=dest, carrier=carrier)

    # ensure fares are sorted in decreasing order of price (i.e. increasing order of attractiveness)
    far = sorted(far, key=lambda f: f.price, reverse=True)
    far_classes = [f.booking_class for f in far]

    offers = [Offer(pth, f, f.price) for f in far]

    choices = {}
    for n in range(1, len(far) + 1):
        choices[far_classes[n - 1]] = pd.Series(
            dmd.simulate_choices(offers[:n], n_draws=n_draws), index=far_classes[:n]
        )

    df = pd.concat(choices, axis=1).rename_axis(columns="lowest_avail_class", index="chosen_class")

    cm = dmd.choice_model
    wtp = _random_max_wtp(dmd.reference_price, n_draws=n_draws, raw=True, emult=dmd.emult or cm.emult)

    lower_bound = 0
    upper_bound = far[0].price * 1.2

    wtp_raw_sorted = np.sort(wtp["raw"])
    x_values = np.linspace(lower_bound, upper_bound, 200)
    exact_prices = np.asarray([f.price for f in far])
    x_values = np.unique(np.concatenate([x_values, exact_prices]))
    wtp_pct_greater = pd.Series(
        100.0 * (1 - np.searchsorted(wtp_raw_sorted, x_values, side="left") / wtp_raw_sorted.size),
        index=x_values,
        name="pct_wtp",
    ).rename_axis(index="price_point")

    fare_prices = (
        pd.Series({f.booking_class: f.price for f in far}, name="price")
        .rename_axis(index="booking_class")
        .reset_index()
    )

    fare_prices["wtp_share"] = _interpolate_series_with_new_index(wtp_pct_greater, fare_prices["price"]).to_numpy()
    fare_prices["zero"] = 0
    fare_prices["frat5"] = fare_prices["price"].apply(
        lambda x: _get_frat5(x, dmd.reference_price, dmd.emult or cm.emult)
    )

    if raw_data:
        return df, wtp_pct_greater, fare_prices

    fare_rules = (
        alt.Chart(fare_prices)
        .transform_calculate(wtp_percent="(datum.wtp_share) / 100")
        .mark_rule(strokeWidth=3)
        .encode(
            x=alt.X(
                "price:Q",
                title="Price Point",
                axis=alt.Axis(format="$.0f"),
                scale=alt.Scale(domain=[lower_bound, upper_bound]),
            ),
            y=alt.Y("wtp_share:Q"),
            y2=alt.Y2("zero:Q"),
            color=alt.Color("booking_class:N", title="Booking Class"),
            tooltip=[
                alt.Tooltip("booking_class", title="Booking Class"),
                alt.Tooltip("price", format="$.0f"),
                alt.Tooltip("wtp_percent:Q", title="WTP Share", format=".2%"),
                alt.Tooltip("frat5:Q", title="Frat5 Value", format=".2f"),
            ],
        )
    )

    wtp_figure = fare_rules + (
        alt.Chart(
            wtp_pct_greater.reset_index(),
            width=300,
            title=alt.TitleParams("Willingness to Pay Distribution", anchor="middle", fontSize=14),
        )
        .mark_line(
            color="black",
        )
        .encode(
            x=alt.X("price_point:Q", title="Price Point", axis=alt.Axis(format="$.0f")),
            y=alt.Y("pct_wtp:Q", title="Percentage Willing to Pay"),
        )
    )

    choice_figure = (
        alt.Chart(
            (df / n_draws).stack().to_frame("count").reset_index(),
            width=300,
            title=alt.TitleParams("Choice Model Distribution", anchor="middle", fontSize=14),
        )
        .mark_bar()
        .encode(
            x=alt.X("lowest_avail_class:N", title="Lowest Available Class", sort="descending"),
            y=alt.Y("count:Q", title="Percentage Choosing", axis=alt.Axis(format="0.0%")),
            color=alt.Color("chosen_class:N", title="Booking Class"),
            tooltip=[
                alt.Tooltip("lowest_avail_class", title="Lowest Available Class"),
                alt.Tooltip("chosen_class", title="Chosen Class"),
                alt.Tooltip("count", title="Percentage Choosing", format=".2%"),
            ],
        )
    )

    return (
        (choice_figure | wtp_figure)
        .configure_axis(grid=False)
        .properties(title=f"Choice Analysis for {orig}~{dest}@{segment}")
    )


def check_choice_model_restrictions(cfg: Config) -> dict[str, dict[str, int]]:
    """Check the restrictions found over all the choice models.

    Parameters
    ----------
    cfg : Config
        The config to check.

    Returns
    -------
    dict[str, dict[str, int]]
        A dictionary mapping restriction names to dictionaries, where the inner dictionaries
        map choice model names to restriction values.
    """
    restrictions = defaultdict(dict)
    for name, cm in cfg.choice_models.items():
        assert isinstance(cm, PodsChoiceModel | LogitChoiceModel)
        if cm.restrictions:
            for r_name, r_value in cm.restrictions.items():
                restrictions[r_name][name] = r_value
    return dict(restrictions)
