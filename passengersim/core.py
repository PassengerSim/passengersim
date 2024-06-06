import warnings

try:
    import passengersim_core  # noqa: F401
except ImportError:
    warnings.warn("passengersim.core is not available", stacklevel=2)
    Airline = None
    BookingCurve = None
    Bucket = None
    Cabin = None
    ChoiceModel = None
    DecisionWindow = None
    Demand = None
    DynamicProgram = None
    Event = None
    Fare = None
    Forecast = None
    Frat5 = None
    Generator = None
    Leg = None
    LicenseError = None
    Market = None
    Path = None
    PathClass = None
    ProBP = None
    SimulationEngine = None
else:
    from passengersim_core import (
        Airline,
        BookingCurve,
        Bucket,
        Cabin,
        ChoiceModel,
        DecisionWindow,
        Demand,
        DynamicProgram,
        Event,
        Fare,
        Forecast,
        Frat5,
        Generator,
        Leg,
        LicenseError,
        Market,
        Path,
        PathClass,
        ProBP,
        SimulationEngine,
    )

__all__ = [
    "Airline",
    "BookingCurve",
    "Bucket",
    "Cabin",
    "ChoiceModel",
    "DecisionWindow",
    "Demand",
    "DynamicProgram",
    "Event",
    "Fare",
    "Forecast",
    "Frat5",
    "Generator",
    "Leg",
    "LicenseError",
    "Market",
    "Path",
    "PathClass",
    "ProBP",
    "SimulationEngine",
]
