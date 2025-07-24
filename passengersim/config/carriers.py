from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import (
    field_validator,
)

from .named import Named
from .optional_literal import Optional


class Carrier(Named, extra="forbid"):
    """Configuration for passengersim.Carrier object."""

    rm_system: str
    """Name of the revenue management system used by this carrier."""

    control: str = ""
    """Deprecated.  No effect"""

    cp_algorithm: Optional[Literal["BP", "CBC", "OPT"]] = None
    """Used to select continuous pricing.

    The default is None, which means that continuous pricing is not used.
    If set to "BP", then the continuous pricing is based on the bid price,
    i.e. the fare of the continuous priced product offered to the customer
    is equal to the bid price of the product, modified potentially by the
    `cp_quantize` and/or `cp_bounds` settings.  If set to "CBC", then the
    continuous price is set via a class-based continuous pricing algorithm,
    which adjusts the price of the continuous priced product based on the
    expected willingness to pay of the customer, as defined by the `frat5`
    curve. In constrast to the "BP" algorithm, the "CBC" algorithm sets
    the price of the continuous priced product to be equal to the bid price
    plus the expected marginal revenue of the product,
    
    OPT is an experimental algorithm that uses web-shopping (similar to Infare) and
    a choice model to try and improve the expected contribution of an airline's offer(s)
    """

    cp_record: Literal["highest_closed", "lowest_open"] = "highest_closed"
    """Where to record sales of continuous-prices products.

    When a sale is made of a product that is offered at some modified price,
    (i.e., a continuous price), it can be recorded as a sale in the highest
    closed class, or in the lowest open class.  This recording is relevant
    when forecasting demand in the various fare classes.  If recording in the
    lowest open class, no other adjustments are made to the recording, as we
    are selling a fare class that is open.  If recording in the highest closed,
    users should also review the `cp_record_highest_closed_as_open` setting,
    which controls whether the highest closed fare class is recorded as "open"
    in the history data, even though it otherwise would appear to be closed.
    """

    cp_record_highest_closed_as_open: bool = False
    """Record the highest closed fare class as open.

    When recording history data, by default the continuous pricing algorithm
    is ignored when getting closure status of each fare at the end of each DCP.
    If `cp_algorithm` is set to "highest_closed", then the highest closed fare
    is actually being offered (with a modified price) to the customer, and the
    carrier may want to record this fare as open in the history data.

    This setting has no effect on the actual continuous pricing algorithm at the
    time of making offers.  It only affects the history data that is recorded.
    This setting is only used when `cp_record` is set to "highest_closed",
    otherwise it has no effect.
    """

    cp_quantize: int | None = 0
    """Controls quantization (rounding) for Continuous Pricing
       Example: If you set it to 5, the price will be rounded to the nearest $5"""

    cp_bounds: float = 1.0
    """DEPRECATED - Controls upper and lower bounds for continuous pricing.
       Example:  Y1 fare = $400, Y2 fare = $300
                 The difference is $100, and a 0.25 multiplier will set the lower bound
                 for Y1 as $375 and the upper bound for Y2 as $325"""

    cp_upper_bound: float = 1.0
    """Controls upper bound for continuous pricing.
       Example:  If the highest fare, Y0 = $400,
                 then a 1.1 multiplier will allow CP to go up to $440"""

    cp_scale: float = 1.0
    """Continuous pricing modifier scale factor.

    This is used to scale the fare modifier when using CBC.

    Scales the fare modifier, which was computed using WTP"""

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

    @field_validator("cp_upper_bound")
    def _check_cp_upper_bound(cls, v: str):
        x = float(v)
        if x < 0.0 or x > 2.0:
            warnings.warn("cp_upper_bound should be in the range (0, 2)", stacklevel=2)
        return v
