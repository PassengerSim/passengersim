from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class Path(BaseModel, extra="forbid"):
    """A travel path from an origin to a destination, comprised of one or more legs."""

    path_id: int | None = None
    """Optional numeric identifier for the path."""

    orig: str
    """Identifier code for the path origin.

    This is typically the IATA airport code for the origin airport, but it may be a
    different code if the path is being used in a non-air-travel context, or if the
    airport is a hypothetical place that does not have an assigned IATA code.
    """

    dest: str
    """Identifier code for the path destination.

    This is typically the IATA airport code for the destination airport, but it may
    be a different code if the path is being used in a non-air-travel context, or if
    the airport is a hypothetical place that does not have an assigned IATA code.
    """

    path_quality_index: float
    """A numeric index representing the quality of the path.

    Lower values typically indicate a more desirable itinerary. This is typically
    the number of connections on the path but could also be some other quality
    measure.
    """

    legs: list[int]
    """Leg IDs of the legs comprising the path, in order from origin to destination.

    At least one leg is required.
    """

    @field_validator("legs", mode="before")
    @classmethod
    def _allow_single_leg(cls, v: Any) -> list[int]:
        """Coerce a bare integer into a one-element list.

        Pydantic calls this validator before type coercion, so callers may
        supply a single flight number instead of a one-element list.

        Parameters
        ----------
        v : int or list of int or Any
            The raw (pre-coercion) value supplied for ``legs``.

        Returns
        -------
        list of int
            ``v`` wrapped in a list when it is a plain integer, otherwise
            ``v`` unchanged.
        """
        if isinstance(v, int):
            return [v]
        return list(v)

    @field_validator("legs")
    @classmethod
    def _at_least_one_leg(cls, v: Any) -> list[int]:
        """Ensure the path contains at least one leg.

        Parameters
        ----------
        v : list of int
            The post-coercion list of flight numbers.

        Returns
        -------
        list of int
            ``v`` unchanged if it contains at least one element.

        Raises
        ------
        ValueError
            If ``v`` is an empty list.
        """
        legs: list[int] = list(v)
        if len(legs) < 1:
            raise ValueError("path must have at least one leg")
        return legs

    @property
    def market_identifier(self) -> str:
        """A tilde-separated string identifying the origin–destination market.

        Returns
        -------
        str
            A string of the form ``"<orig>~<dest>"``, e.g. ``"JFK~LAX"``.
        """
        return f"{self.orig}~{self.dest}"
