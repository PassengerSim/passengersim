import math
from zoneinfo import ZoneInfo
from typing import List
from pydantic import BaseModel, field_validator


class Place(BaseModel, extra="forbid", validate_assignment=True):
    name: str
    """Identifying code for this place.

    For airports, typically the three letter code."""

    label: str
    """A descriptive label for this place."""

    lat: float | None = None
    """Latitude in degrees."""

    lon: float | None = None
    """Longitude in degrees."""

    time_zone: str | None = None
    """
    The time zone for this location.
    """

    mct: List[int] | None = None
    """
    Default Minimum Connect Time (MCT) for this location (Airport)
    Specified as a list of 4 integers, [DD, DI, ID, II], with time in minutes
    Domestic->Domestic, Domestic->International, etc.
    Future version of PassengerSim will also allow specific exceptions by airline / route / etc.
    """

    @field_validator("time_zone")
    def _valid_time_zone(cls, v: str):
        """Check for valid time zones."""
        if isinstance(v, str):
            ZoneInfo(v)
        return v

    @field_validator("mct")
    def _valid_mct(cls, m: str):
        if m is None:
            return True
        elif not isinstance(m, list):
            return False
        elif len(m) != 4:
            raise ValueError("When specified, MCT should be a list of 4 integer values, [DD, DI, ID, II], in minutes")

    @property
    def time_zone_info(self):
        return ZoneInfo(self.time_zone)


def great_circle(place1: Place, place2: Place):
    """Using Haversine formula, to get distance between points in miles."""
    lon1 = math.radians(place1.lon)
    lat1 = math.radians(place1.lat)
    lon2 = math.radians(place2.lon)
    lat2 = math.radians(place2.lat)
    lon_diff = lon2 - lon1
    lat_diff = lat2 - lat1
    a = math.sin((lat_diff) / 2.0) ** 2.0 + (
        math.cos(lat1) * math.cos(lat2) * (math.sin((lon_diff) / 2.0) ** 2.0)
    )
    angle2 = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    # Convert back to degrees.
    angle2 = math.degrees(angle2)
    # Each degree on a great circle of Earth is 69.0468 miles. ( 60 nautical miles )
    distance2 = 69.0468 * angle2
    return distance2
