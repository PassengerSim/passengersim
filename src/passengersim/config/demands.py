from __future__ import annotations

import ast
import warnings
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, field_serializer, field_validator, model_validator

if TYPE_CHECKING:
    from .base import Config

_REFERENCE_FARE_DEPRECATION_MSG = (
    "The 'reference_fare' field on Demand is deprecated and has been renamed "
    "to 'reference_price'. Please update your code/configuration to use "
    "'reference_price' instead."
)


class DemandOverride(BaseModel, extra="forbid"):
    carrier: str
    """Carrier code for the override."""

    discount_pct: float = 0.0
    """Discount percentage to apply for this override."""

    pref_adj: float = 0.0
    """Preference adjustment to apply for this override."""


class Demand(BaseModel, extra="forbid"):
    orig: str
    """Origin location for this demand.

    This is commonly a three letter airport code, but it need not be limited
    to airports.  It can be any location that is relevant to the simulation.

    If using 'places' for locations, this should match the 'name' field of
    a Place object."""

    dest: str
    """Destination location for this demand.

    This is commonly a three letter airport code, but it need not be limited
    to airports.  It can be any location that is relevant to the simulation.

    If using 'places' for locations, this should match the 'name' field of
    a Place object."""

    segment: str
    """Customer segment that this demand belongs to.

    For many applications, segments include 'business' and 'leisure', but
    they are not limited to these two categories."""

    @property
    def identifier(self):
        """Unique identifier for this demand."""
        return f"{self.orig}~{self.dest}@{self.segment}"

    @property
    def market_identifier(self):
        """Unique identifier for the market of this demand."""
        return f"{self.orig}~{self.dest}"

    base_demand: float

    reference_price: float
    """Reference price used for willingness-to-pay and choice model scaling.

    This field was previously named ``reference_fare``; that name is still
    accepted as input for backward compatibility but is deprecated and will
    emit a :class:`DeprecationWarning`."""

    emult: float | None = None
    """An 'emult' value for this demand.

    This value scales the decay rate of maximum willingness to pay above the
    reference price.  If not provided, the `emult` attached to the choice model
    will be used."""

    distance: float | None = 0.0
    """O-D distance."""

    choice_model: str | None = None
    """The name of the choice model that is applied for this demand."""

    dwm_tolerance: float | None = 0.0
    """The Decision Window is the shortest elapsed time, plus the tolerance (random draw)."""

    todd_curve: str | None = None
    """Time Of Day curve to be used in the choice model.  These
       can vary by length of haul, day of week, E-W directionality, etc.
       If specified here, it will override the curve in the ChoiceModel"""

    curve: str | None = None
    """The name of the booking curve that is applied for this demand.

    Each demand is attached to a booking curve that describes the temporal
    distribution of customer arrivals."""

    group_sizes: list[float] | None = None
    """Probability of each group size.
    i.e. [0.5, 0.3, 0.2] will give 50% one pax, 30% 2 pax, etc"""

    prob_saturday_night: float | None = None
    """Probability that the customer has a R/T itinerary with a Saturday night stay.
       Using this for choice modeling and CP experiments"""

    prob_num_days: list[float] = []
    """Probability of durations.
       [0.1, 0.3, 0.4, 0.2] will have durations of 1, 2, 3, 4 days
       and probability of each is specified explicitly
       Using this for choice modeling and CP experiments"""

    deterministic: bool = False
    """Whether the total amount of demand generated in each sample should be constant.

    If this is true, there will be no variance in the total demand generated,
    Which will be equal to the base demand modified by any market multipliers.
    There still can be some randomness in the timeframe arrival distribution of
    this demand, but the total demand generated in each sample will be the same.
    """

    overrides: list[DemandOverride] = []
    """Used for some specialized tests.
       Each dictionary should have 'carrier', 'discount_pct' and 'pref_adj'"""

    @field_validator("overrides", "prob_num_days", mode="before")
    def _accept_strings(cls, v):
        if isinstance(v, str):
            v = ast.literal_eval(v)
        return v

    @field_serializer("overrides", "prob_num_days")
    def _serialize_overrides(self, v):
        return [str(o.model_dump() if isinstance(o, BaseModel) else str(o)) for o in v]

    @property
    def choice_model_(self):
        """Choice model, falling back to segment name if not set explicitly."""
        return self.choice_model or self.segment

    @field_validator("curve", mode="before")
    def curve_integer_name(cls, v):
        """Booking curves can have integer names, treat as string."""
        if isinstance(v, int):
            v = str(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def _migrate_reference_fare(cls, data):
        """Accept the legacy ``reference_fare`` key as an alias for
        ``reference_price`` for backward compatibility.

        If a caller supplies ``reference_fare`` (and not ``reference_price``),
        the value is moved to ``reference_price`` and a
        :class:`DeprecationWarning` is emitted. If both are supplied,
        ``reference_price`` wins and the legacy key is discarded with a
        warning."""
        if isinstance(data, dict) and "reference_fare" in data:
            warnings.warn(
                _REFERENCE_FARE_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            legacy = data.pop("reference_fare")
            if "reference_price" not in data:
                data["reference_price"] = legacy
        return data

    def __getattr__(self, item):
        """Route access to the deprecated ``reference_fare`` attribute to
        ``reference_price``, emitting a :class:`DeprecationWarning`.

        Note: ``__getattr__`` is only invoked when the attribute is not found
        through the normal mechanism, so this does not shadow real fields."""
        if item == "reference_fare":
            warnings.warn(
                _REFERENCE_FARE_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            return self.reference_price
        # Delegate to Pydantic's default __getattr__ behavior for anything else.
        return super().__getattr__(item)

    def __setattr__(self, key, value):
        """Route assignment to the deprecated ``reference_fare`` attribute to
        ``reference_price``, emitting a :class:`DeprecationWarning`."""
        if key == "reference_fare":
            warnings.warn(
                _REFERENCE_FARE_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            key = "reference_price"
        super().__setattr__(key, value)


def assign_standard_todd_curves(cfg: Config) -> Config:
    """For all demands with no TODD curve, assign the appropriate standard TODD curve."""

    if not cfg.simulation_controls.use_standard_todd_curves:
        # if disabled, do nothing
        return cfg

    todd_curve_queue = set()

    # for each demand, check if it has a defined TODD curve.
    # if not, assign the Standard_TODD_Curve based on the mkt.delta_t
    for dmd in cfg.demands:
        if dmd.todd_curve is None:
            mkt = cfg.markets_dict[f"{dmd.orig}~{dmd.dest}"]
            todd_curve = f"Standard_TODD_Curve_{mkt.delta_t:02d}"
            dmd.todd_curve = todd_curve
            if todd_curve not in cfg.todd_curves:
                todd_curve_queue.add(todd_curve)

    # load all required standard configs
    if todd_curve_queue:
        from passengersim import Config, demo_network

        std_cfg = Config.from_yaml(demo_network("standard-todd.yaml"))
        for q in todd_curve_queue:
            cfg.todd_curves[q] = std_cfg.todd_curves[q].model_copy(deep=True)

    return cfg


def assign_standard_dwm_tolerances(cfg: Config, segment_mapping: dict[str, str] | None = None) -> Config:
    """For all demands with no DWM tolerance, assign the appropriate standard DWM tolerance."""

    if not cfg.simulation_controls.use_standard_todd_curves:
        # if disabled, do nothing
        return cfg

    if segment_mapping is None:
        segment_mapping = {}

    std_tols = {
        "business": [1.240, 3.318, 3.544, 3.765, 3.971, 4.159, 5.356, 10.5, 10.78, 10.96, 24],
        "leisure": [2.034, 4.617, 4.936, 5.245, 5.536, 5.801, 6.379, 12.7, 13.04, 13.327, 24],
    }
    std_miles = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, np.inf]

    for dmd in cfg.demands:
        if not dmd.dwm_tolerance:
            segment = segment_mapping.get(dmd.segment, dmd.segment)
            if segment not in std_tols:
                segment = "leisure"
            mile_category = np.searchsorted(std_miles, dmd.distance or 0)
            dmd.dwm_tolerance = std_tols[segment][mile_category]

    return cfg
