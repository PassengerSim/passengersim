from __future__ import annotations

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
    def allow_unrestricted(cls, v):
        """Allow restrictions to be None or missing."""
        if v is None:
            v = []
        return v
