# TITLE: Booking Curves
from __future__ import annotations

from pydantic import ValidationInfo, field_validator

from .named import Named


class ToddCurve(Named, extra="forbid"):
    """
    Customer preference data for Time Of Day
    """

    min_distance: int = 0
    max_distance: int = 25000
    k_factor: float = 3.0
    earlyDepMultiplier: float = 0.0
    lateDepMultiplier: float = 0.0
    earlyArrMultiplier: float = 0.0
    lateArrMultiplier: float = 0.0

    curve: dict[int, float]
    """Define a TODD curve.


    Example
    -------
    ```{yaml}
    - name: business
      curve:
        63: 0.01
        56: 0.02
        49: 0.05
        42: 0.13
        35: 0.19
        31: 0.23
        28: 0.29
        24: 0.35
        21: 0.45
        17: 0.54
        14: 0.67
        10: 0.79
        7: 0.86
        5: 0.91
        3: 0.96
        1: 1.0
    ```
    """
