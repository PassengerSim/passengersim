from collections.abc import Callable

from passengersim.config.places import get_mileage
from passengersim.core import Airport, Leg

## circuity function registration ##

CircuityFunc = Callable[[dict[str, Airport], tuple[Leg, ...], str | None, int], bool]
"""Type alias for circuity functions.

A circuity function accepts a dictionary of airport places, a tuple of legs
representing the proposed path, an optional destination airport code, and an
iteration counter.  It returns ``True`` if the path is allowable and ``False``
if the path should be rejected.  It may raise ``StopIteration`` to signal that
no further iterations will yield new acceptable paths.
"""
_REGISTERED_CIRCUITY_FUNCS: dict[str, CircuityFunc] = {}
"""Registry mapping circuity function names to their implementations."""


def register_circuity_function(func: CircuityFunc, name: str | None = None) -> CircuityFunc:
    """Register a circuity function.

    This can be used to register a user-defined circuity function that can then
    be used in the connection builder. The name used for registration is the
    provided `name` parameter, or the function's `__name__` if `name` is not
    provided. The registered circuity function can then be retrieved by name
    using `get_registered_circuity_function(name)`.

    Parameters
    ----------
    func : CircuityFunc
        The circuity function to register.
    name : str | None, optional
        The name to register the circuity function under. If None, the function's
        `__name__` attribute will be used. Defaults to None.

    Returns
    -------
    CircuityFunc
        The same circuity function that was registered. This allows this function
        to be used as a decorator to register a circuity function but otherwise
        act transparently.

    Raises
    ------
    ValueError
        If a circuity function is already registered under *name*.
    """
    global _REGISTERED_CIRCUITY_FUNCS
    if name is None:
        name = func.__name__
    if name in _REGISTERED_CIRCUITY_FUNCS:
        raise ValueError(f"Circuity function {name!r} is already registered.")
    _REGISTERED_CIRCUITY_FUNCS[name] = func
    return func


def get_registered_circuity_function(name: str) -> CircuityFunc:
    """Retrieve a registered circuity function by name.

    Parameters
    ----------
    name : str
        The name under which the circuity function was registered.

    Returns
    -------
    CircuityFunc
        The circuity function registered under *name*.

    Raises
    ------
    KeyError
        If no circuity function has been registered under *name*.
    """
    global _REGISTERED_CIRCUITY_FUNCS
    if name not in _REGISTERED_CIRCUITY_FUNCS:
        raise KeyError(f"Circuity function {name!r} is not registered.")
    return _REGISTERED_CIRCUITY_FUNCS[name]


## circuity function implementations ##


@register_circuity_function
def default_circuity_function(
    places: dict[str, Airport], legs: tuple[Leg, ...], dest: str | None = None, iteration: int = 0
) -> bool:
    """Decide if the path is allowable.

    Parameters
    ----------
    places : dict[str, Airport]
        A dictionary mapping airport codes to Airport objects containing
        their location information.
    legs : tuple[Leg,...]
        A tuple of Leg objects representing the sequence of legs in the path.
    dest : str | None, optional
        The code of the destination airport, if available. Defaults to the
        `dest` attribute of the last leg in `legs` if not provided. When
        building 3+ leg paths, the destination may be different from the
        final leg's destination, in which case this function is evaluating
        not whether the proposed path is acceptable (it isn't) but whether
        the proposed path could be the beginning of an acceptable path.
    iteration : int, optional
        The current iteration of the connection builder. This can be used to
        allow for different circuity rules in different iterations, for example
        to allow more circuitous paths in later iterations when not enough
        valid paths were created in earlier iterations. Defaults to 0.

    Returns
    -------
    bool
         True if the path is allowable, False if it should be disallowed.

    Raises
    ------
    ValueError
        If *legs* is empty, or if *orig* or *dest* is not found in *places*.
    StopIteration
        If the iteration number is large enough that further iterations are
        no longer going to produce newly acceptable paths.

    Notes
    -----
    This function should evaluate the directness of the path given by the
    sequence of legs, and False if the path should be disallowed (e.g. if
    it is too circuitous).  The determination of "too circuitous" is up to
    the user, and can be based on the total distance traveled, the ratio of
    total distance to great circle distance, or any other metric derived from
    the legs and the places. This means you can allow special rules for certain
    airports, airports in certain states or countries, individual carriers,
    or any other available static criteria.

    This default function can be replaced in the connection builder by any
    user-defined function with the same signature.
    """
    if len(legs) < 1:
        raise ValueError("proposed path has no legs")
    orig = legs[0].orig
    if dest is None:
        dest = legs[-1].dest

    # check that the origin and destination are known places.
    if orig not in places:
        raise ValueError("orig not found in places")
    if dest not in places:
        raise ValueError("dest not found in places")

    # compute the distance from the last leg's destination to the final destination.
    # This allows for evaluating the proposed path as a potential beginning of an
    # acceptable path, even if the final destination is not yet reached.
    remaining_distance = 0
    if dest != legs[-1].dest:
        remaining_distance = get_mileage(places, legs[-1].dest, dest)

    # compute the great circle distance from origin to destination, and the total
    # distance traveled along the legs plus remaining distance.
    market_distance = get_mileage(places, orig, dest)
    total_distance = sum(leg.distance for leg in legs) + remaining_distance

    # check for excessive circuity. The thresholds here are somewhat arbitrary,
    # and can be adjusted as needed.

    if iteration < 5:
        nudge = 0.25 * iteration
        if market_distance < 500 and total_distance > (1.8 + nudge) * market_distance:
            # Short legs allow for more circuitous paths as hub connection are unlikely to be aligned.
            return False
        if market_distance < 2000 and total_distance > (1.5 + nudge) * market_distance:
            return False
        if market_distance >= 2000 and total_distance > (1.3 + nudge) * market_distance:
            return False
    else:
        raise StopIteration

    return True


@register_circuity_function
def unlimited_circuity(
    places: dict[str, Airport], legs: tuple[Leg, ...], dest: str | None = None, iteration: int = 0
) -> bool:
    """Allow unlimited circuity in the network.

    This function can be used as a simple way to disable circuity checks,
    for example in testing or in cases where the user does not have strong
    preferences about circuity and wants to allow all paths. Note that this
    will likely lead to a very large number of possible paths, which may
    increase computation time and memory usage.

    Parameters
    ----------
    places : dict[str, Airport]
        A dictionary mapping airport codes to Airport objects containing
        their location information.  Not used by this implementation, but
        required by the :data:`CircuityFunc` interface.
    legs : tuple[Leg, ...]
        A tuple of Leg objects representing the sequence of legs in the
        proposed path.  Must contain at least one leg.
    dest : str | None, optional
        The code of the destination airport.  Not used by this
        implementation, but required by the :data:`CircuityFunc` interface.
        Defaults to None.
    iteration : int, optional
        The current iteration of the connection builder.  When greater than
        zero, ``StopIteration`` is raised to signal that no new paths will
        be discovered in subsequent iterations.  Defaults to 0.

    Returns
    -------
    bool
        Always returns ``True`` (all paths are accepted) on the first
        iteration.

    Raises
    ------
    ValueError
        If *legs* is empty.
    StopIteration
        If *iteration* is greater than zero, signaling that further
        iterations will not produce new paths.
    """
    if len(legs) < 1:
        raise ValueError("proposed path has no legs")
    if iteration > 0:
        raise StopIteration
    return True
