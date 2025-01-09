#
# TITLE: Booked Load Factor Curves
#
# PassengerSim LLC
#

from __future__ import annotations

from .pretty import PrettyModel


class BlfCurve(PrettyModel, extra="forbid"):
    """Define a Booked Load Factor Curve. Used for UserAction."""

    name: str = "???"
    type: str
    min_distance: int = 0
    max_distance: int = 25000
    min_duration: float = 0.0
    max_duration: float = 25.0
    k_factor: float = 3.0
    curve: dict[int, float] | list[float] = None
