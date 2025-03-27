from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationInfo, field_serializer, field_validator


def create_timestamp(base_date, offset, hh, mm) -> int:
    """Create Unix time from base date, offset (days) and time"""
    td = timedelta(days=offset, hours=hh, minutes=mm)
    tmp = base_date + td
    result = int(tmp.timestamp())
    return result


class Leg(BaseModel, extra="forbid"):
    leg_id: int | None = None
    """A unique identifier for this leg.

    Each leg in a network should have a globally unique identifier (i.e. even
    if the carrier is different, `leg_id` values should be unique.  Note this
    is not the same as the `fltno`, which is a label analogous to flight numbers
    in practice, which don't need to be unique.  If the leg_id is not provided,
    the `fltno` is used, unless the simulation already has a leg with that
    `leg_id`, in which case a new unique leg_id will be generated.
    """

    carrier: str
    """The carrier name for this leg."""

    fltno: int
    """A flight number identifier for this leg.

    Flight numbers do not need to be unique.
    """

    orig: str
    """Origination location for this leg."""

    orig_timezone: str | None = None
    """Timezone name for the origination location for this leg."""

    dest: str
    """Destination location for this leg."""

    dest_timezone: str | None = None
    """Timezone name for the destination location for this leg."""

    date: datetime = datetime.fromisoformat("2020-03-01")
    """Departure date for this leg."""

    arr_day: int = 0
    """If the arrival time is on a different day, this is offset in days.

    This will usually be zero (arrival date is same as departure date) but will
    sometimes be 1 (arrives next day) or in a few pathological cases -1 or +2
    (for travel across the international dateline).
    """

    dep_time: int
    """Departure time for this leg in Unix time.

    In input files, this can be specified as a string in the format "HH:MM",
    with the hour in 24-hour format.

    Unix time is the number of seconds since 00:00:00 UTC on 1 Jan 1970."""

    dep_time_offset: int = 0

    @property
    def dep_localtime(self) -> datetime:
        """Departure time for this leg in local time at the origin."""
        t = datetime.fromtimestamp(self.dep_time, tz=timezone.utc)
        if self.orig_timezone is not None:
            z = ZoneInfo(self.orig_timezone)
            t = t.astimezone(z)
        return t

    arr_time: int
    """Arrival time for this leg in Unix time.

    In input files, this can be specified as a string in the format "HH:MM",
    with the hour in 24-hour format.

    Unix time is the number of seconds since 00:00:00 UTC on 1 Jan 1970."""

    arr_time_offset: int = 0

    @property
    def arr_localtime(self) -> datetime:
        """Arrival time for this leg in local time at the destination."""
        t = datetime.fromtimestamp(self.arr_time, tz=timezone.utc)
        if self.dest_timezone is not None:
            z = ZoneInfo(self.dest_timezone)
            t = t.astimezone(z)
        return t

    time_adjusted: bool = False
    capacity: int | dict[str, int]
    distance: float | None = None

    @field_validator("date", mode="before")
    def _date_from_string(cls, v):
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            # when no timezone is specified, assume UTC (not naive)
            v = v.replace(tzinfo=timezone.utc)
        return v

    @field_serializer("date", when_used="always")
    def serialize_date(self, date: datetime) -> str:
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return date.isoformat()

    @field_validator("dep_time", "arr_time", mode="before")
    def _timestring_to_int(cls, v, info: ValidationInfo):
        if isinstance(v, str) and ":" in v:
            dep_time_str = v.split(":")
            hh, mm = int(dep_time_str[0]), int(dep_time_str[1])
            v = create_timestamp(info.data["date"], 0, hh, mm)
        if info.field_name == "arr_time":
            if v < info.data["dep_time"] and info.data["arr_day"] == 0:
                v += 86400  # add a day (in seconds) as arr time is next day
            elif info.data["arr_day"] != 0:
                v += 86400 * info.data["arr_day"]  # adjust day[s] (in seconds)
        return v
