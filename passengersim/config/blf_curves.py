#
# TITLE: Booked Load Factor Curves
#
# PassengerSim LLC
#

from __future__ import annotations

from .named import Named


class BlfCurve(Named, extra="forbid"):
    """Define a Booked Load Factor Curve. Used for UserAction."""

    #    name: str
    min_distance: int = 0
    max_distance: int = 25000
    min_duration: float = 0.0
    max_duration: float = 25.0
    type: str | None = None
    k_factor: float = 0.3
    curve: dict[int, float] | list[float] = None
