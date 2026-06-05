# TITLE: Booking Curves
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import Field, field_validator, model_validator

from .pretty import PrettyModel

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from .base import Config
    from .legs import Leg
    from .paths import Path

from collections import namedtuple

_TimeInfo = namedtuple("_TimeInfo", ["dep_unix", "arr_unix", "dep_local_hour", "arr_local_hour"])

_LegTimeType = dict[int, _TimeInfo]
"""Mapping of Leg ID to departure and arrival time data for TODD analysis."""


class ToddCurve(PrettyModel, extra="forbid"):
    """
    Customer preference data for Time Of Day for departure (TODD).

    These curves are used to model the probability of a customer choosing various
    departure times based on the distance of the trip.
    """

    name: str = "???"
    """The name of this TODD curve."""

    min_distance: Annotated[int, Field(deprecated="no effect")] = 0
    """Deprecated, no effect."""

    max_distance: int = 25000
    """Deprecated, no effect."""

    k_factor: float = 0.3
    """The k-factor controlling the spread of the TODD preference distribution.

    Larger values of ``k_factor`` produce a wider spread of departure-time
    preferences, while smaller values concentrate preferences more tightly
    around the peak hours.
    """

    probabilities: dict[int, float] | None = None
    """Define a TODD curve.

    This must be a dictionary with integer keys in the range (0, 23) inclusive,
    and float values in the range [0, 1]. The keys represent hours of the day,
    and the values represent the probability of a customer choosing to depart
    at that hour for trips within the specified distance range.
    """

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_distance_fields(cls, values: dict) -> dict:
        """Emit a deprecation warning when ``min_distance`` or ``max_distance`` are set.

        Both fields are retained for backward compatibility with existing
        configuration files but have no effect on simulation behavior.

        Parameters
        ----------
        values : dict
            The raw input dictionary supplied to the model constructor.

        Returns
        -------
        dict
            ``values`` unchanged.
        """
        for field in ("min_distance", "max_distance"):
            if field in values:
                warnings.warn(
                    f"ToddCurve field '{field}' is deprecated and has no effect; "
                    "it will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return values

    @field_validator("probabilities", mode="before")
    @classmethod
    def _validate_probabilities(cls, v: dict | None) -> dict[int, float] | None:
        """Validate and normalize the probabilities dictionary.

        - Any missing keys in 0-23 are filled in with 0.0.
        - All values must be non-negative floats.
        - The values are rescaled so they sum to 1.0.
        - A warning is issued if rescaling changes any individual value by more
          than a small tolerance (1e-6), indicating the original data was not
          already normalized.

        Parameters
        ----------
        v : dict or None
            The raw probabilities value to validate. May be ``None`` (no curve
            defined) or a mapping of hour keys to probability values. Keys may
            be integers or any type coercible to ``int``; values may be any type
            coercible to ``float``.

        Returns
        -------
        dict[int, float] or None
            ``None`` if ``v`` is ``None``; otherwise a fully populated dict
            with integer keys 0–23 and non-negative float values that sum to
            exactly 1.0.

        Raises
        ------
        ValueError
            If any key is outside the range 0–23 after coercion to ``int``.
        ValueError
            If any value is negative after coercion to ``float``.
        ValueError
            If all provided values are zero (the distribution cannot be
            normalized).
        """
        if v is None:
            return v

        # Ensure all keys are integers in the valid range 0-23
        normalized: dict[int, float] = {}
        for key, val in v.items():
            int_key = int(key)
            if int_key < 0 or int_key > 23:
                raise ValueError(
                    f"TODD curve probability key {key!r} is out of range; "
                    "keys must be integers in the range 0-23 inclusive."
                )
            float_val = float(val)
            if float_val < 0:
                raise ValueError(
                    f"TODD curve probability value for hour {int_key} must be non-negative, got {float_val}."
                )
            normalized[int_key] = float_val

        # Fill in any missing hours with 0.0
        for hour in range(24):
            normalized.setdefault(hour, 0.0)

        # Rescale so that values sum to 1.0
        total = sum(normalized.values())
        if total == 0:
            raise ValueError(
                "TODD curve probabilities must not all be zero; the values cannot be rescaled to sum to 1.0."
            )

        # Warn if the rescaling is notable (i.e. the data was not already normalized)
        if abs(total - 1.0) > 1e-6:
            warnings.warn(
                f"TODD curve '{normalized}' probabilities sum to {total:.6g}, not 1.0; "
                "the values will be rescaled to sum to 1.0.",
                UserWarning,
                stacklevel=2,
            )

        rescaled = {hour: normalized[hour] / total for hour in range(24)}
        return rescaled


def _get_leg_time_info(legs: Collection[Leg]) -> _LegTimeType:
    """Build a mapping from leg ID to departure and arrival time data.

    Parameters
    ----------
    legs : Collection[Leg]
        The collection of legs from which time information is extracted.
        Every leg must have a non-``None`` ``leg_id``.

    Returns
    -------
    _LegTimeType
        A dict mapping each leg's ``leg_id`` to a ``_TimeInfo`` named tuple
        containing the departure Unix timestamp, arrival Unix timestamp,
        local departure hour, and local arrival hour.

    Raises
    ------
    ValueError
        If any leg has ``leg_id`` set to ``None``.
    """
    info = _LegTimeType()
    for leg in legs:
        if leg.leg_id is None:
            raise ValueError("leg_ids must be set")
        info[leg.leg_id] = _TimeInfo(leg.dep_time, leg.arr_time, leg.dep_localtime.hour, leg.arr_localtime.hour)
    return info


def _path_duration(path: Path, legtimes: _LegTimeType) -> int:
    """Get the duration of a path, in seconds.

    The duration is measured from the Unix departure time of the first leg to
    the Unix arrival time of the last leg.

    Parameters
    ----------
    path : Path
        The path whose total travel duration is to be computed.
    legtimes : _LegTimeType
        Mapping of leg IDs to their time information, as produced by
        :func:`_get_leg_time_info`.

    Returns
    -------
    int
        Duration of the path in seconds.
    """
    path_depart = legtimes[path.legs[0]].dep_unix
    path_arrive = legtimes[path.legs[-1]].arr_unix
    return path_arrive - path_depart


def _min_duration_path(paths: Sequence[Path], legtimes: _LegTimeType) -> Path:
    """Get the minimum duration path from a list of paths.

    Parameters
    ----------
    paths : Sequence[Path]
        The candidate paths to compare. Must contain at least one element.
    legtimes : _LegTimeType
        Mapping of leg IDs to their time information, as produced by
        :func:`_get_leg_time_info`.

    Returns
    -------
    Path
        The path with the shortest total travel duration. If two or more paths
        share the same minimum duration, the first one (lowest index) is
        returned.
    """
    durations = [_path_duration(path, legtimes) for path in paths]
    n = np.argmin(durations)
    return paths[n]


def _path_delta_local_time(path: Path, legtimes: _LegTimeType) -> int:
    """Get the delta local time of a path, in hours.

    The delta is computed as ``(local arrival hour) - (local departure hour)``,
    taken modulo 24 so the result is always in the range [0, 23].

    Parameters
    ----------
    path : Path
        The path for which the local-time delta is computed.
    legtimes : _LegTimeType
        Mapping of leg IDs to their time information, as produced by
        :func:`_get_leg_time_info`.

    Returns
    -------
    int
        The difference between the local arrival hour of the last leg and the
        local departure hour of the first leg, expressed as a value in the
        range [0, 23].
    """
    path_depart = legtimes[path.legs[0]].dep_local_hour
    path_arrive = legtimes[path.legs[-1]].arr_local_hour
    delta_t = path_arrive - path_depart
    return delta_t % 24


def clean_todd_curves(cfg: Config) -> Config:
    """Remove all TODD curves that are not used by any demand.

    Iterates over all TODD curves defined in the configuration and removes any
    that are not referenced by at least one demand entry. This keeps the
    configuration lean and avoids processing unused curves during simulation.

    Parameters
    ----------
    cfg : Config
        The simulation configuration to prune.

    Returns
    -------
    Config
        The same ``cfg`` object with unused TODD curves removed in-place.
    """
    logger = logging.getLogger("passengersim.config.preprocess")
    used_todd_curves = set([d.todd_curve for d in cfg.demands])
    for k in list(cfg.todd_curves):
        if k not in used_todd_curves:
            logger.info("Removing unused TODD curve '%s' from config.", k)
            cfg.todd_curves.pop(k)
    return cfg
