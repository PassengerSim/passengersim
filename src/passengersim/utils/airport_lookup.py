from typing import Any

from passengersim.config.places import Place

_AIRPORTS: dict[str, Any] = None


def lookup_airport(iata_code: str):
    global _AIRPORTS

    try:
        import airportsdata
    except ImportError:
        raise ImportError("airports module not available; try `pip install airportsdata`") from None

    if _AIRPORTS is None:
        _AIRPORTS = airportsdata.load("IATA")

    if iata_code not in _AIRPORTS:
        raise KeyError(f"Invalid iata code: {iata_code}")

    airport = _AIRPORTS[iata_code]

    return Place(
        name=iata_code,
        label=airport["name"],
        country=airport["country"],
        state=airport["subd"],
        lat=airport["lat"],
        lon=airport["lon"],
        time_zone=airport["tz"],
    )
