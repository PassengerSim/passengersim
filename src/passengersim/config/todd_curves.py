# TITLE: Booking Curves
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from pydantic import field_validator

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
    min_distance: int = 0
    max_distance: int = 25000
    k_factor: float = 0.3

    probabilities: dict[int, float] | None = None
    """Define a TODD curve.

    This must be a dictionary with integer keys in the range (0,23) inclusive,
    and float values in the range [0,1]. The keys represent hours of the day,
    and the values represent the probability of a customer choosing to depart
    at that hour for trips within the specified distance range.
    """

    @field_validator("probabilities", mode="before")
    @classmethod
    def _validate_probabilities(cls, v):
        """Validate and normalize the probabilities dictionary.

        - Any missing keys in 0-23 are filled in with 0.0.
        - All values must be non-negative floats.
        - The values are rescaled so they sum to 1.0.
        - A warning is issued if rescaling changes any individual value by more
          than a small tolerance (1e-6), indicating the original data was not
          already normalized.
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
    info = _LegTimeType()
    for leg in legs:
        if leg.leg_id is None:
            raise ValueError("leg_ids must be set")
        info[leg.leg_id] = _TimeInfo(leg.dep_time, leg.arr_time, leg.dep_localtime.hour, leg.arr_localtime.hour)
    return info


def _path_duration(path: Path, legtimes: _LegTimeType) -> int:
    """Get the duration of a path, in seconds."""
    path_depart = legtimes[path.legs[0]].dep_unix
    path_arrive = legtimes[path.legs[-1]].arr_unix
    return path_arrive - path_depart


def _min_duration_path(paths: Sequence[Path], legtimes: _LegTimeType) -> Path:
    """Get the minimum duration path from a list of paths."""
    durations = [_path_duration(path, legtimes) for path in paths]
    n = np.argmin(durations)
    return paths[n]


def _path_delta_local_time(path: Path, legtimes: _LegTimeType) -> int:
    """Get the delta local time of a path, in hours."""
    path_depart = legtimes[path.legs[0]].dep_local_hour
    path_arrive = legtimes[path.legs[-1]].arr_local_hour
    delta_t = path_arrive - path_depart
    return delta_t % 24


def clean_todd_curves(cfg: Config) -> Config:
    """Remove all TODD curves that are not used by any demand."""
    logger = logging.getLogger("passengersim.config.preprocess")
    used_todd_curves = set([d.todd_curve for d in cfg.demands])
    for k in list(cfg.todd_curves):
        if k not in used_todd_curves:
            logger.info("Removing unused TODD curve '%s' from config.", k)
            cfg.todd_curves.pop(k)
    return cfg
