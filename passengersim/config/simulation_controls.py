# TITLE: Simulation Controls
# DOC-NAME: 01-simulation-controls
from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import ValidationInfo, confloat, conint, field_validator

from passengersim.utils import iso_to_unix

from .pretty import PrettyModel


class SimulationSettings(PrettyModel, extra="allow", validate_assignment=True):
    num_trials: conint(ge=1, le=1000) = 1
    """The overall number of trials to run.

    Each trial is a complete simulation, including burn-in training time as well
    as study time.  It will have a number of sequentially developed samples, each of
    which represents one "typical" day of travel.

    See [Counting Simulations][counting-simulations] for more details.
    """

    num_samples: conint(ge=1, le=10000) = 600
    """The number of samples to run within each trial.

    Each sample represents one "typical" day of travel.
    See [Counting Simulations][counting-simulations] for more details.
    """

    burn_samples: conint(ge=0, le=10000) = 100
    """The number of samples to burn when starting each trial.

    Burned samples are used to populate a stable history of data to support
    forecasting and optimization algorithms, but are not used to evaluate
    performance results.

    See [Counting Simulations][counting-simulations] for more details.
    """

    double_capacity_until: int | None = None
    """
    Double the capacity on all legs until this sample.

    The extra capacity may reduce the statistical noise of untruncation
    within the burn period and allow the simulation to achieve a stable
    steady state faster.  If used, this should be set to a value at least
    26 below the `burn_samples` value to avoid polluting the results.
    """

    @field_validator("double_capacity_until")
    @classmethod
    def _avoid_capacity_pollution(cls, v: int | None, info: ValidationInfo):
        if v and v >= info.data["burn_samples"] - 25:
            raise ValueError("doubled capacity will pollute results")
        return v

    sys_k_factor: confloat(ge=0, le=5.0) = 0.10
    """
    System-level randomness factor.

    This factor controls the level of correlation in demand levels across the
    entire system.

    See [k-factors][demand-generation-k-factors]
    for more details.
    """

    mkt_k_factor: confloat(ge=0, le=5.0) = 0.20
    """
    Market-level randomness factor.

    This factor controls the level of correlation in demand levels across origin-
    destination markets.

    See [k-factors][demand-generation-k-factors]
    for more details.
    """

    pax_type_k_factor: confloat(ge=0, le=5.0) = 0.0
    """
    Passenger-type randomness factor.

    Deprecated: use `simple_k_factor` instead.

    This factor add uncorrelated variance to every demand, unless there are
    multiple demands in the same market and with the same passenger segment.

    See [k-factors][demand-generation-k-factors]
    for more details.
    """

    segment_k_factor: confloat(ge=0, le=5.0) = 0.0
    """
    Passenger segment randomness factor.

    This factor controls the level of correlation in demand levels across
    passenger segments.
    """

    simple_k_factor: confloat(ge=0, le=5.0) = 0.40
    """
    Passenger-type randomness factor.

    This factor add uncorrelated variance to every demand.

    See [k-factors][demand-generation-k-factors]
    for more details.
    """

    simple_cv100: confloat(ge=0, le=1.0) = 0.0
    """THIS IS A TEST"""

    tf_k_factor: confloat(ge=0) = 0.1
    """
    Time frame randomness factor.

    This factor controls the dispersion of bookings over time, given a previously
    identified level of total demand. See [k-factors]() for more details.
    """

    tot_z_factor: confloat(ge=0, le=100.0) = 2.0
    """
    Base level demand variance control.

    This factor scales the variance in the amount of total demand for any given
    market segment.

    See [k-factors][demand-generation-k-factors] for more details.
    """

    tf_z_factor: confloat(ge=0, le=100.0) = 2.0
    """
    Timeframe demand variance control.

    This factor scales the variance in the allocation of total demand to the
    various arrival timeframes.

    See [k-factors][demand-generation-k-factors] for more details.
    """

    prorate_revenue: bool = True

    save_orders: bool = False

    save_all_offers: bool = False
    """
    This will save all Offers, including those that would fail fare rules or availability.
    The output choice set data will have all of these, so you can find first choice demand,
    recapture, etc.
    False by default
    """

    dwm_lite: bool = True
    """
    Use the "lite" decision window model.

    The structure of this model is the same as that use by Boeing.
    """

    max_connect_time: conint(ge=0) = 240
    """
    Maximum connection time for automatically generated paths.

    Any generated path that has a connection time greater than this value (expressed
    in minutes) is invalidated.
    """

    disable_ap: bool = False
    """
    Remove all advance purchase settings used in the simulation.

    This applies to all carriers and all fare products.  If active, this filter
    is applied to all Fare definitions at the time the Config is loaded into to a
    Simulation object.
    """

    demand_multiplier: confloat(gt=0) = 1.0
    """
    Scale all demand by this value.

    Setting to a value other than 1.0 will increase or decrease
    all demand inputs uniformly by the same multiplicative amount.
    This is helpful when exploring how simulation results vary
    when you have "low demand" scenarios
    (e.g, demand_multiplier = 0.8), or "high demand" scenarios
    (e.g., demand multiplier = 1.1).
    """

    capacity_multiplier: confloat(gt=0) = 1.0
    """
    Scale all capacities by this value.

    Setting to a value other than 1.0 will increase or decrease all capacity inputs
    uniformly by the same multiplicative amount.
    Business class and/or first class can be quickly simulated with this option
    """

    manual_paths: bool = False
    """
    The user has provided explicit paths and connections.

    If set to False, the automatic path generation algorithm is applied.
    """

    generate_3seg: bool | None = False
    """
    Use the new A* search to build connections, it can create 3seg connects
    """

    @property
    def use_3seg(self) -> bool:
        return self.generate_3seg

    @use_3seg.setter
    def use_3seg(self, value: bool):
        # deprecated
        if value:
            warnings.warn(
                "`use_3seg` is deprecated, use `generate_3seg` instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self.generate_3seg = bool(value)

    write_raw_files: bool = False

    random_seed: int | None = None
    """
    Integer used to control the reproducibility of simulation results.

    A seed is base value used by a pseudo-random generator to generate random
    numbers. A fixed random seed is used to ensure the same randomness pattern
    is reproducible and does not change between simulation runs, i.e. allows
    subsequent runs to be conducted with the same randomness pattern as a
    previous one. Any value set here will allow results to be repeated.

    The random number generator is re-seeded at the beginning of every sample
    in every trial with a fixed tuple of three values: this "global" random seed,
    plus the sample number and trial number.  This ensures that partial results
    are also reproducible: the simulation of sample 234 in trial 2 will be the
    same regardless of how many samples are in trial 1.
    """

    update_frequency: int | None = None

    controller_time_zone: int | float = -21600
    """
    The reference time zone for the controller (seconds relative to UTC).

    Data collection points will be trigger at approximately midnight in this time zone.

    This value can be input in hours instead of seconds, any absolute value less
    than or equal to 12 will be assumed to be hours and scaled to seconds.

    The default value is -6 hours, or US Central Standard Time.
    """

    base_date: str = "2020-03-01"
    """
    The default date used to compute relative times for travel.

    Future enhancements may include multi-day modeling.
    """

    dcp_hour: float = 0.0
    """
    The hour of the day that the RM recalculation events are triggered.

    If set to zero, the events happen at midnight.  Other values can
    delay the recalculation into later in the night (or the next day).
    """

    capture_competitor_data: bool = False
    """
    Turns on the capturing of competitor data.

    This feature captures lowest available fare data captured by market, for potential
    use in competitive analysis RM strategies.
    """

    capture_choice_set_file: str = ""
    """
    Turns on the capturing of the choice set and writes the data to the specified file
    """

    capture_choice_set_obs: int | None = None
    """
    If this is set, PassengerSim will randomly sample the ChoiceSet data and output
    APPROXIMATELY this many choice sets (each will have multiple items and all items
    for the choice set will be saved and output)
    """

    capture_choice_set_mkts: list[tuple] | None = []
    """Capture only these markets (O&D pairs)"""

    show_progress_bar: bool = True
    """
    Show a progress bar while running.

    The progress display requires `rich` is installed.
    """

    # A bunch of debug flags, these are only used for development !!!
    debug_availability: bool | None = False
    debug_choice: bool | None = False
    debug_connections: bool | None = False
    debug_events: bool | None = False
    debug_fares: bool | None = False
    debug_offers: bool | None = False
    debug_orders: bool | None = False

    additional_settings: dict[str, Any] = {}
    """
    Additional settings to pass to the simulation.

    These settings are passed directly to the simulation object and can be used to
    set various parameters that are not directly exposed in the configuration.
    """

    @field_validator("controller_time_zone", mode="before")
    def _time_zone_convert_hours_to_seconds(cls, v):
        if -12 <= v <= 12:
            v *= 3600
        return v

    def reference_epoch(self) -> int:
        """Get the reference travel datetime in unix time."""
        return iso_to_unix(self.base_date) - self.controller_time_zone

    timeframe_demand_allocation: Literal["v2", "pods"] = "v2"
    """
    Which algorithm to use for time frame demand allocation.
    """

    allow_unused_restrictions: bool = False
    """
    Allow restrictions to be defined but not used.

    If set to False, any restriction that is defined as a parameter of a choice
    model but not present on any fare, or vice versa, will raise a ValueError.
    Users may override this behavior by setting this parameter to True, which
    will emit a warning instead of an error.
    """
