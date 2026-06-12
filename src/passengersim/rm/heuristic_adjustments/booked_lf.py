"""
Simulate an analyst manipulating forecasts on atypical demand days.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from passengersim.config.named import DictOfNamed, Named
from passengersim.rm._common import RmActionCfg

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation
    from passengersim.summaries import SimulationTables


class BookedLoadFactorCurves(Named, extra="forbid"):
    """Define a Booked Load Factor Curve."""

    lower_bound: dict[int, float] = {}
    """For each number of days prior, a lower bound on booked load factor.

    At each day across the booking process, if the actual booked load factor is
    below this lower bound, and if the time since the last adjustment (if any) is
    sufficient, then the forecasted demand for this leg is reduced for the
    remainder of the booking curve for this departure.
    """

    upper_bound: dict[int, float] = {}
    """For each number of days prior, an upper bound on booked load factor.

    At each day across the booking process, if the actual booked load factor is
    above this upper bound, and if the time since the last adjustment (if any) is
    sufficient, then the forecasted demand for this leg is increased for the
    remainder of the booking curve for this departure.
    """


class BookedLoadFactorAdjustment(RmActionCfg):
    frequency: Literal["dcp", "daily", "daily_pre_dep", "non_dcp", "begin_sample", "end_sample", "weekly"] = "daily"

    cfg_name: str = "booked_load_factor_curves"
    cfg_type: type = DictOfNamed[BookedLoadFactorCurves]

    def __init__(
        self,
        *,
        carrier: str,
        minimum_sample: int = 10,
        cfg: Config | None = None,
        min_days_between_actions: int = 7,
        adjustment_multiplier: float = 1.1,
        key_tag: str = "BLF_Curve",
    ):
        """
        Apply an adjustment to leg forecasted demand when the bookings are atypical.

        Parameters
        ----------
        carrier : str
            Apply the adjustment only to legs for this carrier.
        minimum_sample : int
            Do not apply any adjustment until at least this many samples have
            been simulated.
        cfg : Config
            The configuration object for the simulation.  The booked load factor
            curves will be stored in the `other_controls` section of this config.
        min_days_between_actions : int, default 7
            When a demand adjustment is applied (in either direction), no other
            demand adjustments will be applied by this action until this number
            of days has elapsed. This prevents the heuristic from swinging the
            forecast too hard, as instances of the booked load factor exceeding
            thresholds will typically be strongly serially correlated.
        adjustment_multiplier : float, default 1.1
            The multiplier to apply to demand when bookings exceed the upper bound
            curve of the booked load factor range. The inverse of this multiplier
            will be applied when bookings fail to meet the lower bound curve.
        """

        super().__init__(carrier=carrier, minimum_sample=minimum_sample, cfg=cfg)
        self.min_days_between_actions = min_days_between_actions
        """Minimum wait period before applying another demand adjustment.

        When a demand adjustment is applied (in either direction), no other demand
        adjustments will be applied by this action until this number of days has
        elapsed. This prevents the heuristic from swinging the forecast too hard,
        as instances of the booked load factor exceeding thresholds will typically
        be strongly serially correlated.
        """

        if adjustment_multiplier <= 1.0:
            raise ValueError("adjustment_multiplier should be greater than 1.0")
        self.hot_multiplier = adjustment_multiplier
        self.cold_multiplier = 1.0 / adjustment_multiplier

        self.key_tag = key_tag

        # check that curves exist for all legs with a BLF curve tag
        missing_curves = set()
        for leg in cfg.legs:
            if leg.carrier != carrier:
                # legs from other carriers are not checked
                continue
            if curve := leg.tags.get(self.key_tag):
                if curve not in self.action_cfg:
                    missing_curves.add(curve)
        if missing_curves:
            raise ValueError(f"The following BLF curves are missing: {missing_curves}")

    def clear_multipliers(self, sim: Simulation):
        """Clear all forecast multipliers for this carrier.

        This is typically done at the end of a simulation sample (i.e. at
        departure).
        """
        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            leg.forecast.clear_forecast_means_adjustment()

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        # At the end of each sample, set all multipliers back to zero.
        if days_prior == 0:
            self.clear_multipliers(sim)
            return

        current_time = sim.eng.last_event_time

        tot_legs, num_hot, num_cold = 0, 0, 0
        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            # for each leg, find the name of the appropriate BLF curve
            leg_blf_curve = leg.tags.get(self.key_tag)
            # if there is no BLF curve tag, apply no adjustment
            if leg_blf_curve is None:
                continue
            # find the named curve and apply the adjust based on current BLF
            if curve := self.action_cfg.get(leg_blf_curve):
                leg_booked = leg.sold / leg.capacity
                if leg_booked > curve.upper_bound[days_prior]:
                    leg.forecast.adjust_forecast_means(self.hot_multiplier, current_time, self.min_days_between_actions)
                    num_hot += 1
                if leg_booked < curve.lower_bound[days_prior]:
                    leg.forecast.adjust_forecast_means(
                        self.cold_multiplier, current_time, self.min_days_between_actions
                    )
                    num_cold += 1
            tot_legs += 1


# The booked load factor action uses an exogenously provided curve (actually,
# a pair of curves) to see if demand is running unusually hot or cold for a
# given leg.  The idea here is to approximate a RM analyst making a manual
# intervention.  In practice, the actual RM analysts do some science, or maybe
# some educated guessing, to know when things are too hot or too cold.  For the
# simulation, we can either get these curves from actual carriers who can try
# to replicate the RM analysts, or we can use the helper functions below to get
# reasonable curves by running the simulation.


def collect_blf_detail(sim: Simulation, days_prior: int) -> dict | None:
    """A daily callback function to collect booked load factor for all legs."""
    if sim.eng.sample < sim.eng.burn_samples:
        return None
    out = {}
    for leg in sim.eng.legs:
        key = leg.leg_id
        out[f"leg-{key}"] = leg.sold / leg.capacity
    return out


def map_leg_blf_groups(cfg: Config, tag_key: str = "BLF_Curve") -> dict[int, str]:
    """Get the mapping of leg ids to BLF groups, based on the leg tags."""
    leg_blf = {}
    for leg in cfg.legs:
        if tag := leg.tags.get(tag_key):
            leg_blf[leg.leg_id] = tag
    return leg_blf


def set_minimum_bandwidth(bounds: pd.DataFrame, bandwidth: float = 0.1) -> pd.DataFrame:
    """Set a minimum bandwidth between lower and upper bounds to the specified value."""

    if bandwidth <= 0 or bandwidth >= 1:
        raise ValueError("Bandwidth must be between 0 and 1")

    calc_bandwidth = bounds["upper_bound"] - bounds["lower_bound"]
    augmentation = np.maximum(0, bandwidth - calc_bandwidth)
    lb = bounds["lower_bound"] - augmentation / 2
    ub = bounds["upper_bound"] + augmentation / 2

    # shift both lower and upper bounds up if lower bound is negative after augmentation
    shift_up = np.maximum(0, -lb)
    lb += shift_up
    ub += shift_up

    # shift both lower and upper bounds down if upper bound is above 1 after augmentation
    shift_down = np.maximum(0, ub - 1)
    lb -= shift_down
    ub -= shift_down

    bounds["lower_bound"] = lb
    bounds["upper_bound"] = ub
    return bounds


def process_blf_detail(
    summary: SimulationTables,
    leg_blf_groups: dict[int, str],
    lower_bound: float = 0.1,
    upper_bound: float = 0.9,
    key_tag: str = "BLF_Curve",
    minimum_bandwidth: float | None = None,
) -> pd.DataFrame:
    """Process data from the `collect_blf_detail` callback to compute BLF curves.

    This will analyze the results from the simulation and suggest a pair of curves
    to use for the booked load factor adjustments.

    Parameters
    ----------
    summary : SimulationTables
        The summary output from a simulation run that used the `collect_blf_detail`
        daily callback.
    leg_blf_groups : dict[int, str]
        The booked load factor group for each leg to be analyzed. This could be
        a unique group for each leg, or several legs can be aggregated in a common
        group.
    lower_bound : float, default 0.1
        The quantile from the observed data to use as the lower bound.
    upper_bound : float, default 0.9
        The quantile from the observed data to use as the upper bound.
    key_tag : str, default "BLF_Curve"
        The name of the column to use for the BLF group in the output dataframe.
        This should match the tag key used in the leg tags to identify BLF groups.
    minimum_bandwidth : float, optional
        If provided, ensure that the distance between the lower and upper bound is
        at least this value. This can be helpful to prevent the heuristic from
        overreacting to small deviations in booked load factor when the bounds are
        very tight (as is typical very early in the booking curve).  For example,
        if this is set at 0.05, then you won't ever decide that demand is running
        hot until you have sold at least 5% of capacity. You also won't ever decide
        that demand is running cold until you reach the time when you can have sold
        5% of capacity and still not be considered as running hot.

    Returns
    -------
    pd.Dataframe
        A dataframe that shows the computed lower and upper bounds for each BLF
        group, by days prior to departure.
    """
    df = pd.DataFrame(summary.callback_data.daily)
    df = df.melt(id_vars=["trial", "sample", "days_prior"], var_name="leg", value_name="blf")
    df[key_tag] = df["leg"].str.strip("leg-").astype(int).apply(leg_blf_groups.get)
    part = df.drop(columns=["sample", "trial", "leg"]).groupby([key_tag, "days_prior"])["blf"]
    lb = part.quantile(lower_bound).rename("lower_bound")
    ub = part.quantile(upper_bound).rename("upper_bound")
    out = pd.concat({"lower_bound": lb, "upper_bound": ub}, axis=1).reset_index()
    if minimum_bandwidth is not None:
        out = set_minimum_bandwidth(out, minimum_bandwidth)
    return out


def attach_blf_bounds(cfg: Config, bounds: pd.DataFrame) -> Config:
    """
    Attach bounds from `process_blf_detail` for use by booked load factor adjustment.
    """
    if len(bounds.columns) != 4:
        raise ValueError("bounds must have exactly 4 columns: key, days_prior, lower_bound, upper_bound")
    key_col, days_col, lower_bound_col, upper_bound_col = bounds.columns
    bound_cols = [lower_bound_col, upper_bound_col]
    bounds_nested = {}
    for curve, grp in bounds.groupby(key_col, sort=False):
        by_day = grp.set_index(days_col)[bound_cols].to_dict()
        bounds_nested[curve] = {}
        for col_name, day_map in by_day.items():
            bounds_nested[curve][col_name] = day_map
    cfg.other_controls["booked_load_factor_curves"] = bounds_nested
    return cfg


def fig_blf_detail(df: pd.DataFrame | Config):
    """Visualize the outputs from `process_blf_detail`."""
    import altair as alt

    from passengersim.config import Config

    if isinstance(df, Config):
        # convert from config nested dict format into dataframe
        data = df.other_controls["booked_load_factor_curves"]
        df = pd.concat({k: pd.DataFrame(v) for k, v in data.items()}, names=["haul", "days_prior"]).reset_index()

    key_tag = df.columns[0]

    chart = alt.Chart(df)

    tooltips = [
        f"{key_tag}:N",
        alt.Tooltip("days_prior", title="Days Prior"),
        alt.Tooltip("lower_bound", format=".2%"),
        alt.Tooltip("upper_bound", format=".2%"),
    ]

    return (
        chart.mark_area(opacity=0.5, line={"color": "black"})
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y=alt.Y("lower_bound", title="Booked Load Factor", axis=alt.Axis(format=".0%")),
            y2="upper_bound",
            tooltip=tooltips,
        )
        .properties(title="Booked Load Factor Curves by Haul Type")
        + chart.mark_line(color="black").encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y=alt.Y("upper_bound", title="Booked Load Factor"),
            tooltip=tooltips,
        )
    ).facet(facet=alt.Facet(f"{key_tag}:N", title="Bound Range"), columns=2)
