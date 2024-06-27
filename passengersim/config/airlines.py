from __future__ import annotations

from typing import Literal, Optional, Any

from .named import Named


class Airline(Named, extra="forbid"):
    """Configuration for passengersim.Airline object."""

    rm_system: str
    """Name of the revenue management system used by this airline."""

    control: str = ""
    """Deprecated.  No effect"""

    continuous_pricing: bool | None = False
    """Used to select continuous pricing"""

    frat5: str | None = ""
    """Named FRAT5 curve.
    This is the default that will be applied if not found at a more detailed level
    """

    fare_adjustment_scale: Optional[float] = 1.0

    load_factor_curve: Any | None = None
    """Named Load Factor curve.
    This is the default that will be applied if not found at a more detailed level
    """

    ancillaries: dict[str, float] | None = {}
    """Specifies ancillaries offered by the airline, codes are ANC1 .. ANC4"""

    classes: list[str] | list[tuple[str, str]] = []
    """A list of fare classes.

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