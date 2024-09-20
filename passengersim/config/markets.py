from __future__ import annotations

from pydantic import BaseModel


class Market(BaseModel, extra="forbid"):
    orig: str
    """Origin location for this market.

    This is commonly a three letter airport code, but it need not be limited
    to airports.  It can be any location that is relevant to the simulation.

    If using 'places' for locations, this should match the 'name' field of
    a Place object."""

    dest: str
    """Destination location for this market.

    This is commonly a three letter airport code, but it need not be limited
    to airports.  It can be any location that is relevant to the simulation.

    If using 'places' for locations, this should match the 'name' field of
    a Place object."""

    demand_multiplier: float = 1.0
    """Multiplier on base demand for all demand segments in this market."""

    @property
    def identifier(self) -> str:
        return f"{self.orig}~{self.dest}"
