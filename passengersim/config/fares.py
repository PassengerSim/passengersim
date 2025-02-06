from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


class Fare(BaseModel, extra="forbid"):
    carrier: str
    orig: str
    dest: str
    booking_class: str
    price: float
    advance_purchase: int
    restrictions: list[str] = []
    category: str | None = None
    cabin: str | None = "Y"

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
