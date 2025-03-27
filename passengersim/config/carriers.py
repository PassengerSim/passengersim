from __future__ import annotations

from typing import Any, Literal

from .named import Named


class Carrier(Named, extra="forbid"):
    """Configuration for passengersim.Carrier object."""

    rm_system: str
    """Name of the revenue management system used by this carrier."""

    control: str = ""
    """Deprecated.  No effect"""

    cp_algorithm: str | None = "None"
    """Used to select continuous pricing"""

    cp_quantize: int | None = 0
    """Controls quantization (rounding) for Continuous Pricing
       Example: If you set it to 5, the price will be rounded to the nearest $5"""

    cp_bounds: float = 0.5
    """Controls upper and lower bounds for continuous pricing.
       Example:  Y1 fare = $400, Y2 fare = $300
                 The difference is $100, and a 0.25 multiplier will set the lower bound
                 for Y1 as $375 and the upper bound for Y2 as $325"""

    cp_scale: float = 1.0
    """Scales the fare modifier, which was computed using WTP"""

    cp_record: str = "highest_closed"
    """Do we record the sale in the highest_closed class, lowest_open or nearest?"""

    cp_elasticity: dict | None = None
    """Parameters to esimate customer price elasticity for CP
         - Defaults to being off
         - {'accuracy': 0.8, 'multiplier': 0.5} will guess 80% accurate and multiply
             the Frat5 value for *leisure* by 0.5
         - Other algorithms to come in the future :-) """

    frat5: str | None = ""
    """Name of the FRAT5 curve to use.

    This is the default that will be applied if not found at a more detailed level.
    If not specified, the default frat5 from the carrier's RM system is used.
    """

    frat5_map: dict | None = {}
    """Experimenting with different Frat5 curves by market"""

    fare_adjustment_scale: float | None = 1.0

    load_factor_curve: Any | None = None
    """Named Load Factor curve.
    This is the default that will be applied if not found at a more detailed level
    """

    brand_preference: float | None = 1.0
    """Used for airline preference to give premium airlines a bump"""

    ancillaries: dict[str, float] | None = {}
    """Specifies ancillaries offered by the carrier, codes are ANC1 .. ANC4"""

    classes: list[str] | list[tuple[str, str]] = []
    """A list of fare classes.

    This list can be a simple list of fare classes, or a list of 2-tuples where
    the first element is the fare class and the second element is the cabin.

    One convention is to use Y0, Y1, ... to label fare classes from the highest
    fare (Y0) to the lowest fare (Yn).  You can also use Y, B, M, H,... etc.
    An example of classes is below.

    Example
    -------
    ```{yaml}
    classes:
      - Y0
      - Y1
      - Y2
      - Y3
      - Y4
      - Y5
    ```

    If using cabins, it is reasonable to name the classes in consistent manner,
    but this is optional, and arbitrary class names are still allowed. All class
    names should still be unique, and cabin identifiers should be replicated
    identically for classes that share a cabin.  Thus the list might look like this:

    ```{yaml}
    classes:
      - (F0, F)
      - (F1, F)
      - (Y0, Y)
      - (Y1, Y)
      - (Y2, Y)
      - (Y3, Y)
    ```
    """

    truncation_rule: Literal[1, 2, 3] = 3
    """How to handle marking truncation of demand in timeframes.

    If 1, then the demand is marked as truncated if the bucket or pathclass is closed at
    the DCP that is the beginning of the timeframe.

    If 2, then the demand is marked as truncated if the bucket or pathclass is closed at
    the DCP that is the end of the timeframe.

    If 3, then the demand is marked as truncated if the bucket or pathclass is closed at
    either of the DCPs that are at the beginning or the end of the timeframe.
    """

    proration_rule: Literal["distance", "sqrt_distance"] = "distance"
    """How to prorate revenue to legs and buckets for connecting paths.

    If "distance", then the revenue is prorated based on the relatives distance
    of the legs.  So if the first leg is 100 miles and the second leg is 400 miles,
    then the first leg gets 20% of the revenue and the second leg gets 80%.

    If "sqrt_distance", then the revenue is prorated based on the relative square
    root of distance of the legs.  So if the first leg is 100 miles and the
    second leg is 400 miles, then the first leg gets 1/3 of the revenue and the
    second leg gets 2/3.
    """

    history_length: int = 26
    """The number of samples to keep in the carrier's history buffers."""
