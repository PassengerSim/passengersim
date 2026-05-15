from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
)


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

    #    date: datetime = datetime.fromisoformat("2020-03-01")
    date: datetime = datetime(2020, 3, 1, tzinfo=UTC)
    """Departure date for this leg.

    This should be the date of the departure, in local time at the origin
    (not the UTC date). When the `dep_time` is specified as a string in the
    format "HH:MM", the date is used in conjunction with this time to create
    the preferred `dep_time` in Unix time.
    """

    arr_day: int = 0
    """If the arrival time is on a different day, this is offset in days.

    This is based on the local time at the destination. For example, if a
    flight departs on March 1st in the late evening from New York and arrives
    in London in the early morning of March 2nd, the `arr_day` would be +1,
    even though if you look only at UTC, the departure and arrival might
    be on the same day. This should correspond to the number of days (plus
    or minus) that would typically be shown the a customer booking on this leg.

    This will usually be zero (arrival date is same as departure date) but will
    sometimes be 1 (arrives next day) or in a few pathological cases -1 or +2
    (for travel across the international dateline).
    """

    dep_time: int
    """Departure time for this leg in Unix time.

    In input files, this can be specified as a string in the format "HH:MM",
    with the hour in 24-hour format, and given in local time at the origin.
    The date field is used in conjunction with this time to create the
    preferred `dep_time` in Unix time.

    Unix time is the number of seconds since 00:00:00 UTC on 1 Jan 1970."""

    dep_time_offset: int = 0

    @property
    def dep_localtime(self) -> datetime:
        """Departure time for this leg in local time at the origin."""
        t = datetime.fromtimestamp(self.dep_time, tz=UTC)
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
        t = datetime.fromtimestamp(self.arr_time, tz=UTC)
        if self.dest_timezone is not None:
            z = ZoneInfo(self.dest_timezone)
            t = t.astimezone(z)
        return t

    time_adjusted: bool = False

    @property
    def duration_minutes(self) -> int | None:
        """Duration of this leg in minutes, or None if dep_time or arr_time is not set."""
        if self.dep_time is not None and self.arr_time is not None:
            result = (self.arr_time - self.dep_time) // 60
            if result < 0:
                # this can happen if arr_time is on the next day but arr_day is not set correctly
                result += 24 * 60
            return result
        else:
            return None

    capacity: int | dict[str, int]
    """The capacity of this leg.

    If provided as an int, the leg is assumed to have a single cabin with that many seats.
    If provided as a dict, the keys are cabin names and the values are the number of seats
    in each cabin. For example, `{"Y": 150, "J": 8}` would indicate 150 Y-cabin seats and
    20 J-cabin seats.
    """

    @property
    def total_capacity(self) -> int:
        """The total capacity of this leg across all cabins."""
        if isinstance(self.capacity, dict):
            return sum(self.capacity.values())
        else:
            return self.capacity

    distance: float | None = None

    @field_validator("date", mode="before")
    def _date_from_string(cls, v):
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            # when no timezone is specified, assume UTC (not naive)
            v = v.replace(tzinfo=UTC)
        return v

    @field_serializer("date", when_used="always")
    def serialize_date(self, date: datetime) -> str:
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        return date.isoformat()

    @field_validator("dep_time", "arr_time", mode="before")
    def _timestring_to_int(cls, v, info: ValidationInfo):
        if isinstance(v, str) and ":" in v:
            dep_time_str = v.split(":")
            hh, mm = int(dep_time_str[0]), int(dep_time_str[1])
            v = create_timestamp(info.data.get("date", datetime(1492, 1, 1)), 0, hh, mm)
            if info.field_name == "arr_time":
                # if v < info.data["dep_time"] and info.data["arr_day"] == 0:
                #     v += 86400  # add a day (in seconds) as arr time is next day
                if info.data["arr_day"] != 0:
                    v += 86400 * info.data["arr_day"]  # adjust day[s] (in seconds)
        return v

    tags: dict[str, str] = {}
    """Optional dictionary of tags associated with this leg.

    Tags can be used to store arbitrary key-value pairs of information related to
    this leg. This may be useful for categorization, filtering, or adding metadata.
    Tags can also be used in RM actions to apply different strategies based on
    leg characteristics. For example, a tag could indicate whether a leg is
    a "short-haul" or "long-haul" flight, allowing RM actions to adjust their
    behavior accordingly.
    """

    @field_validator("tags", mode="before")
    def validate_tags(cls, v):
        if isinstance(v, str):
            # attempt to parse a string representation of a dict, e.g. "{'key1': 'value1', 'key2': 'value2'}"
            try:
                v = ast.literal_eval(v)
            except Exception as e:
                raise ValueError(f"Invalid tags string: {v}") from e
        if not isinstance(v, dict):
            raise ValueError(f"Tags must be a dict or a string representation of a dict, got {type(v)}")
        # ensure all keys and values are strings
        return {str(k): str(v) for k, v in v.items()}

    def __str__(self) -> str:
        ident = f"{self.carrier}{self.fltno}"
        if self.leg_id and self.leg_id != self.fltno:
            ident += f" (leg_id {self.leg_id})"
        s = (
            f"{ident:{20 if len(ident) > 6 else 6}} "
            f"depart {self.orig} at {self.dep_localtime.time()}, "
            f"arrive {self.dest} at {self.arr_localtime.time()}"
        )
        net_days = self.arr_localtime.toordinal() - self.dep_localtime.toordinal()
        if net_days:
            if abs(net_days) == 1:
                s += f" ({net_days:+d} day)"
            else:
                s += f" ({net_days:+d} days)"
        minutes = (self.arr_time - self.dep_time) // 60
        hours = minutes // 60
        minutes = minutes % 60
        s += f" [{self.distance:5.0f} miles, {hours}h {minutes:02d}m, {self.capacity:3d} seats]"
        return s

    def __repr__(self) -> str:
        arr_day = f", arr_day={self.arr_day:+d}" if self.arr_day else ""
        distance = f", distance={round(self.distance, 3):g}" if self.distance else ""
        tags = f", tags={self.tags}" if self.tags else ""
        return (
            f"Leg("
            f"leg_id={self.leg_id}, "
            f"carrier='{self.carrier}', "
            f"fltno={self.fltno}, "
            f"orig={self.orig!r}, "
            f"dest={self.dest!r}, "
            f"date={self.date.strftime('%Y-%m-%d')!r}, "
            f"dep_time={self.dep_localtime.strftime('%H:%M')!r}, "
            f"arr_time={self.arr_localtime.strftime('%H:%M')!r}{arr_day}, "
            f"capacity={self.capacity}"
            f"{distance}{tags}"
            f")"
        )

    @model_serializer(mode="wrap")
    def serialize_leg(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict:
        # check if the context is asking for human readable serialized outputs.
        if isinstance(info.context, dict) and info.context.get("human_readable", False):
            out = {}

            # basic data remains as-is
            out["leg_id"] = self.leg_id
            out["carrier"] = self.carrier
            out["fltno"] = self.fltno
            out["orig"] = self.orig
            out["dest"] = self.dest
            out["capacity"] = self.capacity
            if self.distance:
                out["distance"] = round(self.distance, 3)

            # convert the departure date in local time to YYYY-MM-DD
            dep_date = self.dep_localtime.strftime("%Y-%m-%d")
            arr_date = self.arr_localtime.strftime("%Y-%m-%d")
            out["date"] = dep_date
            # convert dep_time to a string "HH:MM" in 24h local time at the origin
            out["dep_time"] = self.dep_localtime.strftime("%H:%M")
            if self.orig_timezone:
                out["orig_timezone"] = self.orig_timezone
            # convert arr_time to a string "HH:MM" in 24h local time at the destination
            out["arr_time"] = self.arr_localtime.strftime("%H:%M")
            if self.dest_timezone:
                out["dest_timezone"] = self.dest_timezone

            # include add_day only if nonzero
            days = (datetime.fromisoformat(arr_date) - datetime.fromisoformat(dep_date)).days
            if days:
                out["arr_day"] = f"{days:+d}"

            if self.tags:
                out["tags"] = str(self.tags)

            # do not serialize time_adjusted, dep_time_offset, or arr_time_offset, these are
            # created automatically during validation.
            return out

        # otherwise, normal serialization
        serialized = nxt(self)
        return serialized
