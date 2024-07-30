# TITLE: Booking Curves
from __future__ import annotations

from pydantic import ValidationInfo, field_validator

from .named import Named


class BookingCurve(Named, extra="forbid"):
    """
    A mathematical description of the relative arrival rate of customers over time.
    """

    curve: dict[int, float]
    """Define a booking curve.

    For a given customer type, the booking curve gives the cumulative fraction
    of those customers who are expected to have already "arrived" at any given
    data collection point (DCP / days_prior to departure).
    An "arriving" customer is one who is interested
    in booking, but may or may not actually purchase a travel product from one
    of the carriers, depending on the availability of products at the time of their
    arrival.

    The values (cumulative fraction of customers arriving) should increase
    monotonically as the keys (days_prior, e.g. days to departure) decrease.

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

    @field_validator("curve")
    def _booking_curves_accumulate(cls, v: dict[int, float], info: ValidationInfo):
        """Check that all curve values do not decrease as DCP keys decrease."""
        sorted_days_prior = reversed(sorted(v.keys()))
        i = 0
        for days_prior in sorted_days_prior:
            assert (
                v[days_prior] >= i
            ), f"booking curve {info.data['name']} moves backwards at dcp {days_prior}"
            i = v[days_prior]
        return v
