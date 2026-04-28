# TITLE: Simulation Controls
# DOC-NAME: 01-simulation-controls
from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, confloat, conint, field_validator, model_validator

from passengersim.utils import iso_to_unix

from .pretty import PrettyModel
from .speed_limits import SpeedLimits


class ConnectionBuilderSettings(PrettyModel, extra="forbid", validate_assignment=True):
    """
    Settings for the automatic connection builder.

    This tool generates connections in the simulation, converting legs into paths.
    """

    max_legs: Annotated[int, Field(ge=1, le=6)] = 2
    """The maximum number of legs to include in any generated path."""

    max_legs_if_nonstop_exists: Annotated[int, Field(ge=1, le=6)] = 2
    """The maximum number of legs to include in any generated path if a nonstop path exists in that market.

    The nonstop path can be on any carrier."""

    existing_paths: Literal["keep", "replace", "required", "none"] = "keep"
    """What to do with existing paths when generating new ones.

    The default value is "keep", which means that for any market where paths
    already exist they will be used, and no new paths will be generated. For
    markets where no paths exist, new paths will be generated as normal.
    Alternatively, set this to "none", which has the same behavior as "keep"
    but will raise an error if the configuration includes any defined paths.

    Other options are "replace", which will remove all existing paths and then
    generating new ones for all markets, and "required", which will prevent
    the generation of any new paths.  If set to "required", the connection
    builder will only serve as a check that all markets have paths, and will
    raise an error if any market is missing paths.
    """

    circuity_function: str = "default_circuity_function"
    """The function to use when deciding if a path is allowable due to circuity.

    Circuity is the ratio of the total distance of the path to the direct distance
    between the origin and destination. The default function disallows paths that are
    excessively circuitous, with thresholds that vary based on the direct distance.
    Users can provide their own function with the same signature to implement custom
    circuity rules.

    The circuity function is specified by name here and should be a registered
    circuity function.  See `passengersim.connection_builder.circuity` for more
    details on circuity functions and how to register custom ones.
    """

    nonstop_leg_path_id_alignment: bool = True
    """Whether to align path IDs with leg IDs for nonstop paths.

    By default, this is set to True, which means that any nonstop path (corresponds
    to a single leg) will be assigned the same ID as that leg by the path building
    algorithm. This can make it easier to identify and analyze nonstop paths in the
    simulation results. If set to False, nonstop paths will be assigned unique IDs
    that do not necessarily align with leg IDs. This generally corresponds to the
    behavior of the previous path building algorithm, and may be desirable in cases
    where there are existing results to compare against.
    """

    verbosity: int = 0
    """The level of detail to include in connection builder logging."""

    min_paths_per_market: int = 1
    """The minimum number of paths to generate for each market.

    This is not a hard minimum, but the connection builder will make an effort to
    generate at least this many paths for each market, if possible given the other
    settings.  This could be by progressively relaxing circuity rules, maximum
    connection times, or other tweaks. If the connection builder is unable to generate
    at least this many paths for a market, it will log a warning.
    """

    extra_max_connect_time_per_iteration: int = 0
    """Extra time added to all maximum connection times at each iteration.

    The connection builder iterates when the `min_paths_per_market` value is not
    met, potentially relaxing circuity rules at each iteration.  This setting also
    allows for the relaxation of maximum connect times, by adding this many minutes
    to all maximum connection times at each iteration.
    """


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

    @model_validator(mode="after")
    def _migrate_pax_type_k_factor(self):
        """Migrate deprecated pax_type_k_factor to simple_k_factor.

        If a non-zero value is provided for pax_type_k_factor, emit a
        DeprecationWarning and move the value to simple_k_factor.  If the
        user also supplies a non-zero simple_k_factor, that is an error
        because we cannot determine which value should take precedence.
        """
        if self.pax_type_k_factor != 0.0:
            if self.simple_k_factor != 0.0:
                raise ValueError(
                    "Cannot set both `pax_type_k_factor` (deprecated) and "
                    "`simple_k_factor` to non-zero values. "
                    "Use `simple_k_factor` only."
                )
            warnings.warn(
                "`pax_type_k_factor` is deprecated and will be removed in a "
                "future version. Use `simple_k_factor` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Use object.__setattr__ to bypass Pydantic's validate_assignment,
            # which would otherwise re-trigger this model validator and cause
            # a spurious "both non-zero" conflict error mid-migration.
            object.__setattr__(self, "simple_k_factor", self.pax_type_k_factor)
            object.__setattr__(self, "pax_type_k_factor", 0.0)
        return self

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

    revenue_alpha: float = 0.0
    """Used to exponentially smooth revenue per PathClass, to get optimizationFare"""

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

    connection_builder: ConnectionBuilderSettings = ConnectionBuilderSettings()
    """Settings related to the automatic generation of paths and connections in the simulation."""

    manual_paths: Annotated[
        bool | None,
        Field(
            deprecated=(
                "`manual_paths` is deprecated and will be removed in a future version. "
                "Use `connection_builder.existing_paths` instead"
            )
        ),
    ] = None
    """
    Deprecated. See `connection_builder.existing_paths` instead.
    """

    @model_validator(mode="after")
    def _migrate_manual_paths(self):
        """Migrate manual_paths to connection_builder.existing_paths."""
        # Access via __dict__ to avoid triggering the deprecation warning
        # when the value is just None (i.e. the user didn't set it).
        manual_paths_value = self.__dict__.get("manual_paths")
        if manual_paths_value is not None:
            warnings.warn(
                "`manual_paths` is deprecated and will be removed in a future "
                "version. Use `connection_builder.existing_paths` instead",
                DeprecationWarning,
                stacklevel=2,
            )
            # Migrate the value to connection_builder
            if manual_paths_value:
                self.connection_builder.existing_paths = "required"
            else:
                self.connection_builder.existing_paths = "none"
            # Clear the deprecated field
            self.__dict__["manual_paths"] = None
        return self

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

    random_seed: int | None = 42
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

    speed_limits: SpeedLimits = SpeedLimits()
    """
    Speed limits for short, medium, and long travel legs.

    These are only used for data quality checks at Config load time.
    """
