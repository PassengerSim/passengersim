from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, Self

from pydantic import (
    StringConstraints,
    field_validator,
    model_validator,
)

from passengersim.rm.systems import RmSys

from .named import Named
from .optional_literal import Optional
from .pretty import PrettyModel

CabinCode = Annotated[str, StringConstraints(min_length=1, max_length=1)]


class CustomerModel(Named, extra="forbid"):
    """MNL models used in CP"""

    price: float = 0.0
    nonstop: float = 0.0


class ContextualOptimizer(PrettyModel, extra="forbid"):
    pct_up: float | None = 0.0
    pct_down: float | None = 0.0
    only_locals: bool | None = False


class Carrier(Named, extra="forbid"):
    """Configuration for passengersim.Carrier object."""

    rm_system: str
    """Name of the revenue management system used by this carrier.

    If using a callback-style RM system, this can be given as a dict
    instead, in which case the `name` key is extracted and the rest of
    the dict is stored in `rm_system_options`. If a `name` key is not found,
    a validation error is raised.
    """

    rm_system_options: dict[str, Any] | None = None
    """Definition of the revenue management system used by this carrier.

    This can be used to declare parameters for this carrier's RM system.
    """

    @field_validator("rm_system_options")
    def _rm_system_options_not_false(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is False:
            raise ValueError("rm_system_options cannot be False; use None or an empty dict instead")
        return v

    control: str = ""
    """Deprecated.  No effect.

    The control method for availability management is defined in the RM system,
    not in the carrier.  This ensures that the correct control method is always
    used for each RM system.
    """

    cp_algorithm: Optional[Literal["BP", "CBC", "OPT", "CLASSLESS"]] = None
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

    cp_record: Literal["highest_closed", "lowest_open", "nearest"] = "highest_closed"
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
    """Parameters to estimate customer price elasticity for CP
         - Defaults to being off
         - {'accuracy': 0.8, 'multiplier': 0.5} will guess 80% accurate and multiply
             the Frat5 value for *leisure* by 0.5
         - Other algorithms to come in the future :-) """

    cp_markets: Optional[Literal["all", "local", "connect"]] = "all"
    """Limit the Continuous Pricing to local markets, or conecting markets
       Not currently implemented for every CP algorithm"""

    customer_models: list[CustomerModel] | None = []
    """Customer behavior models for Offer generation and optimization"""

    contextual_optimizer: ContextualOptimizer | None = None
    """Parameters for the contextual optimizer"""

    frat5: str | None = ""
    """Name of the FRAT5 curve to use.

    This is the default that will be applied if not found at a more detailed level.
    If not specified, the default frat5 from the carrier's RM system is used.
    """

    frat5_map: dict | None = {}
    """Experimenting with different Frat5 curves by market"""

    fare_adjustment_scale: float | None = 1.0

    store_q_history: bool = False
    """Store Q history for this carrier.

    This needs to be turned on Q forecasting RM systems.  For other RM systems,
    the storage of this data is not needed, so it can be left off to save memory
    and processing time.
    """

    load_factor_curve: Any | None = None
    """Named Load Factor curve.
    This is the default that will be applied if not found at a more detailed level
    """

    brand_preference: float | None = 1.0
    """Used for airline preference to give premium airlines a bump"""

    ancillaries: dict[str, float] | None = {}
    """Specifies ancillaries offered by the carrier, codes are ANC1 .. ANC4"""

    cabin_ordering: list[CabinCode] = ["Y"]
    """The ordering of cabins by quality, from best to worst.

    The cabin code for each cabin must be a string of length 1.

    For example, this could be ["F", "J", "W", "Y"] for first, business, premium economy,
    and economy, in that order.  We assume that any customer who books a fare
    for a given cabin will be satisfied by a seat in that specific cabin, or any
    better cabin.  The default value of ["Y"] signals that there is only one cabin
    type in this carrier's fleet.

    This ordering must be comprehensive for all cabins on all this carrier's legs. There
    cannot be a cabin on any leg operated by this carrier which does not appear on this
    list. The converse need not hold; it is acceptable for some legs to have fewer than
    the complete set of cabins.
    """

    classes: list[str] | list[tuple[str, str]] = []
    """A list of fare classes.

    This list can be a simple list of fare classes, or a list of 2-tuples where
    the first element is the fare class and the second element is the cabin.

    If using the simple list format, the first character of each class must be
    one of the cabin codes defined in `cabin_ordering`.  For example, if two cabins
    of types "F" and "Y" are possible, the fare classes could be:

    Example
    -------
    ```{yaml}
    classes:
      - F0
      - F1
      - Y2
      - Y3
      - Y4
      - Y5
    ```

    If using the 2-tuple format, the second item in each tuple must match one of the
    cabin codes, but the class names can be anything (including replicating the cabin names).
    For example:

    ```{yaml}
    classes:
      - (F, F)
      - (A, F)
      - (Y, Y)
      - (H, Y)
      - (L, Y)
      - (Q, Y)
    ```

    Either way, all class names must be unique across the complete set of classes; you
    cannot have a class "X" associated with two different cabins.
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

    proration_rule: Literal["distance", "sqrt_distance", "off"] = "distance"
    """How to prorate revenue to legs and buckets for connecting paths.

    If "distance", then the revenue is prorated based on the relatives distance
    of the legs.  So if the first leg is 100 miles and the second leg is 400 miles,
    then the first leg gets 20% of the revenue and the second leg gets 80%.

    If "sqrt_distance", then the revenue is prorated based on the relative square
    root of distance of the legs.  So if the first leg is 100 miles and the
    second leg is 400 miles, then the first leg gets 1/3 of the revenue and the
    second leg gets 2/3.

    If "off", then no proration is done, and each leg and bucket gets the full
    revenue of the path. This will lead to double counting of revenue in legs,
    but is useful for some analyses.
    """

    history_length: int = 26
    """The number of samples to keep in the carrier's history buffers."""

    @model_validator(mode="before")
    @classmethod
    def _populate_rm_system_from_def(cls, values: Any):
        """Pre-process input mapping for rm_systems.

        If `rm_system` is an actual RmSys class instead of a string, convert
        to use just the registered name here.

        If `rm_system` is given as a dict instead of a string, extract the name
        and put the rest of the dict into `rm_system_options`.

        If `rm_system` is missing or empty, but `rm_system_options` is provided and
        has a `name` key, copy that into `rm_system`.

        This is a 'before' validator so it operates on the raw input data.
        """
        try:
            if isinstance(values, dict):
                rm_system = values.get("rm_system")
                if isinstance(rm_system, type) and issubclass(rm_system, RmSys):
                    # convert to registered name
                    rm_name = rm_system.get_name()
                    values["rm_system"] = rm_name
                    rm_system = rm_name
                if isinstance(rm_system, dict):
                    # if there is an existing rm_system_options, we need to merge it
                    existing_def = values.get("rm_system_options", {})
                    # merge existing_def into rm_system, checking for conflicts
                    for k, v in existing_def.items():
                        if k in rm_system and rm_system[k] != v:
                            raise ValueError(f"Conflict merging rm_system and rm_system_options for key '{k}'")
                        if k not in rm_system:
                            rm_system[k] = v
                    # move the dict into rm_system_options
                    values["rm_system_options"] = rm_system
                    # extract the name into rm_system
                    if "name" in rm_system:
                        values["rm_system"] = rm_system["name"]
                    else:
                        raise ValueError("`rm_system` dict must have a 'name' key")
                if not rm_system:
                    rm_def = values.get("rm_system_options")
                    if isinstance(rm_def, dict) and "name" in rm_def:
                        # copy the name into rm_system so the rest of the model
                        # validation has a populated value
                        values["rm_system"] = rm_def["name"]
        except Exception:
            # Keep behavior tolerant: if anything unexpected happens, don't
            # raise here — let later validators raise clearer errors.
            pass
        return values

    @model_validator(mode="after")
    def _check_rm_system_from_def(self) -> Self:
        """Check that if `rm_system_options` is provided with a name, it matches."""
        if isinstance(self.rm_system_options, dict) and "name" in self.rm_system_options:
            if self.rm_system != self.rm_system_options["name"]:
                raise ValueError(
                    "`rm_system` must match `rm_system_options['name']` if `rm_system_options` is provided"
                )
            # now remove the name from rm_system_options to avoid duplication
            del self.rm_system_options["name"]
        return self

    @field_validator("cp_upper_bound")
    def _check_cp_upper_bound(cls, v: str):
        x = float(v)
        if x < 0.0 or x > 2.0:
            warnings.warn("cp_upper_bound should be in the range (0, 2)", stacklevel=2)
        return v

    @model_validator(mode="after")
    def _check_cabin_code_alignment(self) -> Self:
        # check that there is at least one cabin
        if not self.cabin_ordering:
            raise ValueError("cabin_ordering must have at least one cabin code")
        # check all cabin codes are unique
        if len(set(self.cabin_ordering)) != len(self.cabin_ordering):
            raise ValueError("cabin codes must be unique")
        # check all classes
        class_codes = set()
        for c in self.classes:
            if isinstance(c, str):
                if len(c) < 1:
                    raise ValueError("class code must have at least one character")
                if c[0] not in self.cabin_ordering:
                    raise ValueError("class codes must begin with a cabin code character")
                class_codes.add(c)
            elif isinstance(c, (list, tuple)):
                if len(c[0]) < 1:
                    raise ValueError("class code must have at least one character")
                if c[1] not in self.cabin_ordering:
                    raise ValueError(f"cabin code {c[1]} not found in cabin ordering {list(self.cabin_ordering)}")
                class_codes.add(c[0])
        if len(class_codes) != len(self.classes):
            raise ValueError("class codes must be unique")
        return self
