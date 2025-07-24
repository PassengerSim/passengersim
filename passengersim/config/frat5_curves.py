# TITLE: Frat5 Curves
from __future__ import annotations

from pydantic import ValidationInfo, field_validator

from .named import Named


class Frat5Curve(Named, extra="forbid"):
    """
    FRAT5 = Fare Ratio at which 50% of customers will buy up to the higher fare.
    """

    enforce_monotonic: bool = True
    """Enforce monotonicity of Frat 5 curves.

    Typically it is expected that the Frat5 curve is monotonic, i.e. that the
    average willingness to pay only increases as the departure date approaches.
    It is easy to accidentally define the Frat5 curve "backwards", and thus
    PassengerSim will check that the Frat5 curve is monotonically increasing
    by default. To violate this assumption, set `enforce_monotonic` to False,
    which will disable the check that the Frat5 curve is monotonic.
    """

    curve: dict[int, float]
    """Define a Frat5 curve.

    To be consistent with the econometric interpretation of the Frat5 curve,
    the values should increase as the keys (DCPs, e.g. days to departure) decrease.
    This implies that average willingness to pay increases as the departure date
    approaches.

    Example
    -------
    ```{yaml}
    - name: curve_C
      curve:
        63: 1.4
        56: 1.4
        49: 1.5
        42: 1.5
        35: 1.6
        31: 1.7
        28: 1.8
        24: 1.9
        21: 2.3
        17: 2.7
        14: 3.2
        10: 3.3
        7: 3.4
        5: 3.4
        3: 3.5
        1: 3.5
    ```
    """

    max_cap: float = 10.0
    """
    Maximum Q-equivalent demand implied by any unit of demand in any fare class.

    This cap is applied only on the recording of Q-equivalent demand that occurs
    within the simulation engine itself, and not as part of any RM step.
    Simulation-recorded Q-equivalent demand can be used by RM steps, such as
    within PODS-like hybrid forecasting models, but the max-cap filter transform
    is implicitly already baked in to the Q-equivalent demand before the RM step
    can use it.

    This can be contrasted against a `max_cap` parameter used in the RM step,
    which can applied against observed demand within the RM step, but the RM step
    receives the "raw" sales data, without adulteration by the simulation engine.
    """

    @field_validator("curve")
    def _frat5_curves_accumulate(cls, v: dict[int, float], info: ValidationInfo):
        """Check that all curve values do not decrease as DCP keys decrease."""
        if "enforce_monotonic" in info.data and not info.data["enforce_monotonic"]:
            # if the user has explicitly set enforce_monotonic to False, then
            # we do not check that the Frat5 curve is monotonic
            return v
        sorted_dcps = reversed(sorted(v.keys()))
        i = 0
        for dcp in sorted_dcps:
            assert v[dcp] >= i, f"frat5 curve {info.data['name']} moves backwards at dcp {dcp}"
            i = v[dcp]
        return v

    @field_validator("curve")
    def _frat5_curves_gt_1(cls, v: dict[int, float], info: ValidationInfo):
        """Check that all curve values are greater than 1.0.

        Values that are less than 1.0 imply that lowering the fare will
        reduce demand, which is not consistent with the econometric interpretation
        of the Frat5 curve.  Similarly, values that are exactly 1.0 imply that
        any fare increase no matter how small will instantly reduce demand to zero,
        which is theoretically plausible as a corner case but in practice is not
        realistic, and will cause numerical issues in simulation.
        """
        for dcp, val in v.items():
            assert val > 1.0, f"frat5 curve {info.data['name']} is not greater than 1 at {dcp}"
        return v
