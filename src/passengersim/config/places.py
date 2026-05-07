from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator
from pydantic.functional_validators import AfterValidator, BeforeValidator

if TYPE_CHECKING:
    from passengersim.core import Airport


class LimitConnectTime(BaseModel, extra="forbid", validate_assignment=True):
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
        return LimitConnectTime(
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
        LimitConnectTime | int | None,
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

    max_connect_time: Annotated[
        LimitConnectTime | int | None,
        AfterValidator(_inflate_simple_mct),
        BeforeValidator(_reformat_mct),
    ] = None
    """
    Maximum connection time (MCT) in minutes for this place.

    This can be given as a single integer, which will be applied to all
    connections, or differentiated by connection type (domestic-domestic,
    domestic-international, etc.).  Connection types can be given using their
    full name (with underscore) or using shorthand codes (DD, DI, ID, II), or
    as a list of 4 integers in the order DD, DI, ID, II.

    If the maximum connection time for any category is missing or set to -1,
    then the simulation default maximum connection time will be used for that
    category.

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
    def time_zone_info(self) -> ZoneInfo | None:
        if self.time_zone is None:
            return None  # No time zone set
        return ZoneInfo(self.time_zone)

    @property
    def latitude(self) -> float:
        """Alias for `lat`."""
        return self.lat

    @property
    def longitude(self) -> float:
        """Alias for `lon`."""
        return self.lon


def great_circle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Using Haversine formula, to get distance between points in miles."""
    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    lon2 = math.radians(lon2)
    lat2 = math.radians(lat2)
    lon_diff = lon2 - lon1
    lat_diff = lat2 - lat1
    a = math.sin((lat_diff) / 2.0) ** 2.0 + (math.cos(lat1) * math.cos(lat2) * (math.sin((lon_diff) / 2.0) ** 2.0))
    angle2 = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    # Convert back to degrees.
    angle2 = math.degrees(angle2)
    # Each degree on a great circle of Earth is 69.0468 miles. ( 60 nautical miles )
    distance2 = 69.0468 * angle2
    return distance2


def get_mileage(airports: dict[str, Place | Airport], orig: str, dest: str) -> float:
    """Get the distance between two places in statue miles.

    Parameters
    ----------
    airports : dict[str, Place or Airport]
        A dictionary mapping airport codes to Place objects containing their location information.
    orig : str
        The code of the origin airport.
    dest : str
        The code of the destination airport.

    Returns
    -------
    float
        The great circle distance between the origin and destination airports in miles.
        Returns 0 if either airport is not found in the provided dictionary.
    """
    if orig not in airports or dest not in airports:
        return 0
    a1 = airports[orig]
    a2 = airports[dest]
    dist = great_circle(a1.latitude, a1.longitude, a2.latitude, a2.longitude)
    return dist


def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    initial_bearing = math.atan2(y, x)
    # Convert radians to degrees and normalize to 0-360°
    initial_bearing = math.degrees(initial_bearing)
    return (initial_bearing + 360) % 360


def calculate_final_bearing(lat1, lon1, lat2, lon2):
    initial_bearing = calculate_initial_bearing(lat2, lon2, lat1, lon1)
    # reverse the bearing to get the final bearing
    final_bearing = (initial_bearing + 180) % 360
    return final_bearing


def calculate_mean_bearing(lat1, lon1, lat2, lon2):
    initial_bearing = calculate_initial_bearing(lat1, lon1, lat2, lon2)
    final_bearing = calculate_final_bearing(lat1, lon1, lat2, lon2)
    mean_bearing = (final_bearing + initial_bearing) / 2
    return mean_bearing


def calculate_predominant_bearing(lat1, lon1, lat2, lon2):
    mean_bearing = calculate_mean_bearing(lat1, lon1, lat2, lon2)
    if mean_bearing < 45:
        return "N"
    elif mean_bearing < 135:
        return "E"
    elif mean_bearing < 225:
        return "S"
    elif mean_bearing < 315:
        return "W"
    else:
        return "N"
