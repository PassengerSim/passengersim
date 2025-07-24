import calendar
from datetime import UTC, datetime


def iso_to_unix(datestring: str, assume_utc: bool = True) -> int:
    """Convert ISO datetime directly to Unix timestamp"""
    tmp = datetime.fromisoformat(datestring)
    if assume_utc and tmp.tzinfo is None:
        tmp = tmp.replace(tzinfo=UTC)
    return int(calendar.timegm(tmp.utctimetuple()))
