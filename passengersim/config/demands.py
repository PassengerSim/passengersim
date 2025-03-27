from __future__ import annotations

from pydantic import BaseModel, field_validator


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

    base_demand: float
    reference_fare: float
    distance: float | None = 0.0
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

    prob_saturday_night: bool = False
    """Probability that the customer has a R/T itinerary with a Saturday night stay.
       Using this for choice modeling and CP experiments"""

    prob_num_days: list[float] = []
    """Probability of durations.
       [0.1, 0.3, 0.4, 0.2] will have durations of 1, 2, 3, 4 days
       and probability of each is specified explicitly
       Using this for choice modeling and CP experiments"""

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
