from __future__ import annotations

from pydantic import BaseModel, field_validator


class Path(BaseModel, extra="forbid"):
    path_id: int | None = None
    orig: str
    dest: str
    path_quality_index: float

    legs: list[int]
    """Flight numbers of legs comprising the path."""

    @field_validator("legs", mode="before")
    def allow_single_leg(cls, v):
        """Allow a single leg path to be just an int not a list of one int."""
        if isinstance(v, int):
            v = [v]
        return v

    @field_validator("legs")
    def at_least_one_leg(cls, v):
        """There must be at least one leg."""
        if len(v) < 1:
            raise ValueError("path must have at least one leg")
        return v
