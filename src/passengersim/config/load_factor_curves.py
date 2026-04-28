# TITLE: Load Factor Curves
from __future__ import annotations

from .named import Named


class LoadFactorCurve(Named, extra="forbid"):
    """
    LF Curve specifies the load factor at which to close a booking class.
    This is designed to simulate a naive LCC that really doesn't have automated RM
    """

    algorithm: str
    min_accordion: float
    max_accordion: float
    target_load_factor: float
    convergence_constant: float
    curve: dict[str, float]
    """Define a Load Factor curve.

    Example
    -------
    ```{yaml}
    - name: lf_curve_1
      curve:
        Y0: 1.0
        Y1: 0.85
        Y2: 0.75
    ```
    """
