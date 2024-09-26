import warnings

try:
    import passengersim_core  # noqa: F401
except ImportError:
    warnings.warn("passengersim.core is not available", stacklevel=2)
    Airport = None
    BookingCurve = None
    Bucket = None
    Cabin = None
    Carrier = None
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
    UserAdjustment = None
else:
    from passengersim_core import (
        Airport,
        BookingCurve,
        Bucket,
        Cabin,
        Carrier,
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
        UserAdjustment,
    )

__all__ = [
    "Airport",
    "BookingCurve",
    "Bucket",
    "Cabin",
    "Carrier",
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
    "UserAdjustment",
]
