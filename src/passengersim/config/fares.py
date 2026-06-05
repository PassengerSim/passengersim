from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, field_serializer, field_validator


class Fare(BaseModel, extra="forbid"):
    """A single fare rule connecting a carrier, market, and booking class.

    A :class:`Fare` record defines the price and conditions under which a
    passenger may purchase a seat in a specific booking class on a specific
    carrier between an origin and destination.
    """

    carrier: str
    """IATA carrier code for the airline offering this fare."""

    orig: str
    """Origin airport or location code for this fare."""

    dest: str
    """Destination airport or location code for this fare."""

    booking_class: str
    """Booking class (fare class) identifier, e.g. ``"Y"``, ``"B"``, ``"Q"``."""

    brand: str | None = ""
    """Optional brand name associated with this fare (e.g. a cabin product name).

    Defaults to an empty string, indicating no specific brand."""

    price: float
    """Base price of this fare in the simulation's currency units."""

    advance_purchase: int
    """Minimum number of days before departure that this fare must be purchased."""

    restrictions: list[str] = []
    """Named restrictions that apply to this fare.

    These are typically related to refundability, changeability, and other conditions
    that may apply to the fare and which may be subject to random preference variation
    across customers.

    The names may not contain pipe or slash characters, as these are used as separators
    when parsing from strings. The names should match those used in choice models.
    """

    category: str | None = None
    """Optional category label for grouping fares (e.g. by product tier)."""

    cabin: str | None = "Y"
    """Cabin code for this fare, e.g. ``"Y"`` for economy.  Defaults to ``"Y"``."""

    min_stay: int = 0
    """Minimum number of nights the passenger must stay at the destination.

    A value of ``0`` indicates no minimum stay requirement."""

    saturday_night_required: bool | None = False
    """Whether a Saturday night stay is required to purchase this fare."""

    @field_validator("restrictions", mode="before")
    @classmethod
    def _allow_unrestricted(cls, v: list[str] | None) -> Any:
        """Coerce ``None`` or missing restriction values to an empty list.

        Parameters
        ----------
        v : list of str or None
            The raw value supplied for the ``restrictions`` field.

        Returns
        -------
        list of str
            An empty list when *v* is ``None``; otherwise *v* unchanged.
        """
        if v is None:
            v = []
        return v

    @field_validator("restrictions", mode="before")
    @classmethod
    def _allow_pipe_sep(cls, v: list[str] | str) -> Any:
        """Parse pipe- or slash-separated restriction strings into a list.

        Configuration files sometimes express restrictions as a single
        delimited string (e.g. ``"NON_REF|NON_CHG"``).  This validator
        splits such strings on ``|`` or ``/`` and removes empty tokens,
        so callers may use either the list or string form.

        Parameters
        ----------
        v : list of str or str
            The raw value supplied for the ``restrictions`` field.  When it
            is a :class:`str`, it is split on ``|`` or ``/``; otherwise it
            is returned unchanged.

        Returns
        -------
        list of str
            The parsed list of restriction name strings.
        """
        if isinstance(v, str):
            v = list(filter(None, re.split(r"[|/]", v)))
        return v

    @field_serializer("restrictions", when_used="always")
    def serialize_restrictions(self, value: list[str]) -> str:
        """Serialize the restrictions list to a pipe-separated string.

        Pydantic calls this serializer whenever the model is converted to a
        dictionary or JSON.  The resulting string is compatible with the
        :meth:`allow_pipe_sep` validator for round-trip fidelity.

        Parameters
        ----------
        value : list of str
            The list of restriction name strings to serialize.

        Returns
        -------
        str
            A single string with restriction names joined by ``"|"``, or an
            empty string if *value* is empty.
        """
        return "|".join(value)

    @property
    def market_identifier(self) -> str:
        """Unique identifier for the origin–destination market of this fare.

        Returns
        -------
        str
            Identifier string in the format ``"<orig>~<dest>"``.
        """
        return f"{self.orig}~{self.dest}"
