from __future__ import annotations

import re

from pydantic import BaseModel, field_serializer, field_validator


class Fare(BaseModel, extra="forbid"):
    carrier: str
    orig: str
    dest: str
    booking_class: str
    brand: str | None = ""
    price: float
    advance_purchase: int
    restrictions: list[str] = []
    """Named restrictions that apply to this fare.

    These are typically related to refundability, changeability, and other conditions
    that may apply to the fare and which may be subject to random preference variation
    across customers.

    The names may not contain pipe or slash characters, as these are used as separators
    when parsing from strings. The names should match those used in choice models.
    """

    category: str | None = None
    cabin: str | None = "Y"
    min_stay: int = 0
    saturday_night_required: bool | None = False

    @field_validator("restrictions", mode="before")
    @classmethod
    def allow_unrestricted(cls, v):
        """Allow restrictions to be None or missing."""
        if v is None:
            v = []
        return v

    @field_validator("restrictions", mode="before")
    @classmethod
    def allow_pipe_sep(cls, v):
        """Allow restrictions to be a string of pipe or slash separated values."""
        if isinstance(v, str):
            v = list(filter(None, re.split(r"[|/]", v)))
        return v

    @field_serializer("restrictions", when_used="always")
    def serialize_restrictions(self, value: list[str]) -> str:
        """Serialize restrictions as a string."""
        return "|".join(value)

    @property
    def market_identifier(self):
        return f"{self.orig}~{self.dest}"
