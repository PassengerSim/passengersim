import math
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator
from pydantic.functional_validators import AfterValidator, BeforeValidator


class MinConnectTime(BaseModel, extra="forbid", validate_assignment=True):
    domestic_domestic: int
    """Minimum connect time for domestic to domestic connections in minutes."""

    domestic_international: int
    """Minimum connect time for domestic to international connections in minutes."""

    international_domestic: int
    """Minimum connect time for international to domestic connections in minutes."""

    international_international: int
    """Minimum connect time for international to international connects in minutes."""


def _inflate_simple_mct(mct: int):
    if isinstance(mct, int):
        return MinConnectTime(
            domestic_domestic=mct,
            domestic_international=mct,
            international_domestic=mct,
            international_international=mct,
        )
    return mct


def _reformat_mct(mct: Any) -> Any:
    if isinstance(mct, dict):
        # reformat shorthand keys
        if "DD" in mct:
            mct["domestic_domestic"] = mct.pop("DD")
        if "DI" in mct:
            mct["domestic_international"] = mct.pop("DI")
        if "ID" in mct:
            mct["international_domestic"] = mct.pop("ID")
        if "II" in mct:
            mct["international_international"] = mct.pop("II")
        if "dd" in mct:
            mct["domestic_domestic"] = mct.pop("dd")
        if "di" in mct:
            mct["domestic_international"] = mct.pop("di")
        if "id" in mct:
            mct["international_domestic"] = mct.pop("id")
        if "ii" in mct:
            mct["international_international"] = mct.pop("ii")
    elif isinstance(mct, list):
        # handle list of 4 integers
        mct = dict(
            zip(
                [
                    "domestic_domestic",
                    "domestic_international",
                    "international_domestic",
                    "international_international",
                ],
                mct,
            )
        )
    return mct


class Place(BaseModel, extra="forbid", validate_assignment=True):
    name: str
    """Identifying code for this place.

    For airports, typically the three letter code."""

    label: str
    """A descriptive label for this place."""

    country: str | None = None
    """Country code.

    Recommended to use ISO 3166-1 alpha-2 codes, ie. US / GB / AU / MX / etc."""

    state: str | None = None
    """State code"""

    lat: float | None = None
    """Latitude in degrees."""

    lon: float | None = None
    """Longitude in degrees."""

    time_zone: str | None = None
    """
    The time zone for this location.
    """

    tz_offset: int | None = None
    """Hours offset from GMT"""

    mct: Annotated[
        MinConnectTime | int | None,
        AfterValidator(_inflate_simple_mct),
        BeforeValidator(_reformat_mct),
    ] = None
    """
    Default Minimum Connect Time (MCT) in minutes for this location (Airport).

    This can be given as a single integer, which will be applied to all
    connections, or differentiated by connection type (domestic-domestic,
    domestic-international, etc.).  Connection types can be given using their
    full name (with underscore) or using shorthand codes (DD, DI, ID, II), or
    as a list of 4 integers in the order DD, DI, ID, II.

    Future version of PassengerSim will also allow specific exceptions by
    airline / route / etc.
    """

    @field_validator("time_zone")
    def _valid_time_zone(cls, v: str):
        """Check for valid time zones."""
        if isinstance(v, str):
            ZoneInfo(v)
        return v

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
