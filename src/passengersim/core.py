"""Low-level interface to the PassengerSim C++ simulation engine.

Re-exports the key classes and objects from the compiled
``passengersim_core`` extension module so they are accessible under
the ``passengersim.core`` namespace.
"""

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
    ForecastData = None
    ForecastGroup = None
    Frat5 = None
    Generator = None
    Leg = None
    LicenseError = None
    Market = None
    Path = None
    PathClass = None
    ProBP = None
    Profiler = None
    SimulationEngine = None
    UserAction = None
    __version__ = None
    build_connections = None
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
        ForecastData,
        ForecastGroup,
        Frat5,
        Generator,
        History,
        Leg,
        LicenseError,
        Market,
        Path,
        PathClass,
        ProBP,
        Profiler,
        SimulationEngine,
        UserAction,
        __version__,
        build_connections,
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
    "ForecastData",
    "ForecastGroup",
    "Frat5",
    "Generator",
    "History",
    "Leg",
    "LicenseError",
    "Market",
    "Path",
    "PathClass",
    "ProBP",
    "Profiler",
    "SimulationEngine",
    "UserAction",
    "__version__",
    "build_connections",
]
