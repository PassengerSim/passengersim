from __future__ import annotations

import ast
import re
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, field_serializer, field_validator, model_validator

if TYPE_CHECKING:
    from .base import Config

_REFERENCE_FARE_DEPRECATION_MSG = (
    "The 'reference_fare' field on Demand is deprecated and has been renamed "
    "to 'reference_price'. Please update your code/configuration to use "
    "'reference_price' instead."
)


class DemandOverride(BaseModel, extra="forbid"):
    """Per-carrier override applied on top of a base :class:`Demand`.

    Used for special situations that require a particular carrier to offer
    a discounted price or an adjusted preference weight relative to the
    base demand specification.
    """

    carrier: str
    """Carrier code for the override."""

    discount_pct: float = 0.0
    """Discount percentage to apply for this override."""

    pref_adj: float = 0.0
    """Preference adjustment to apply for this override."""


class Demand(BaseModel, extra="forbid"):
    """Specification of passenger demand between an origin–destination pair.

    A :class:`Demand` record captures all parameters that govern how many
    customers arrive in a market, how their willingness-to-pay is modeled,
    and which simulation primitives (booking curves, choice models, TODD
    curves, etc.) are attached to those customers.
    """

    orig: str
    """Origin location for this demand.

    This is commonly a three letter airport code, but it need not be limited
    to airports.  It can be any location that is relevant to the simulation.

    If using 'places' for locations, this should match the 'name' field of
    a Place object."""

    dest: str
    """Destination location for this demand.

    This is commonly a three letter airport code, but it need not be limited
    to airports.  It can be any location that is relevant to the simulation.

    If using 'places' for locations, this should match the 'name' field of
    a Place object."""

    segment: str
    """Customer segment that this demand belongs to.

    For many applications, segments include 'business' and 'leisure', but
    they are not limited to these two categories."""

    @property
    def identifier(self) -> str:
        """Unique identifier for this demand.

        The identifier encodes the origin, destination, and segment in a
        single string of the form ``"ORIG~DEST@SEGMENT"``.

        Returns
        -------
        str
            Identifier string in the format ``"<orig>~<dest>@<segment>"``.
        """
        return f"{self.orig}~{self.dest}@{self.segment}"

    @property
    def market_identifier(self) -> str:
        """Unique identifier for the market of this demand.

        The market identifier encodes only the origin and destination,
        omitting the segment, in a string of the form ``"ORIG~DEST"``.

        Returns
        -------
        str
            Identifier string in the format ``"<orig>~<dest>"``.
        """
        return f"{self.orig}~{self.dest}"

    base_demand: float
    """Mean number of customers arriving in this market per sample."""

    reference_price: float
    """Reference price used for willingness-to-pay and choice model scaling.

    This field was previously named ``reference_fare``; that name is still
    accepted as input for backward compatibility but is deprecated and will
    emit a :class:`DeprecationWarning`."""

    emult: float | None = None
    """An 'emult' value for this demand.

    This value scales the decay rate of maximum willingness to pay above the
    reference price.  If not provided, the `emult` attached to the choice model
    will be used."""

    distance: float | None = 0.0
    """O-D distance."""

    choice_model: str | None = None
    """The name of the choice model that is applied for this demand."""

    dwm_tolerance: float | None = 0.0
    """The Decision Window is the shortest elapsed time, plus the tolerance (random draw)."""

    todd_curve: str | None = None
    """Time Of Day curve to be used in the choice model.  These
       can vary by length of haul, day of week, E-W directionality, etc.
       If specified here, it will override the curve in the ChoiceModel"""

    curve: str | None = None
    """The name of the booking curve that is applied for this demand.

    Each demand is attached to a booking curve that describes the temporal
    distribution of customer arrivals."""

    group_sizes: list[float] | None = None
    """Probability of each group size.
    i.e. [0.5, 0.3, 0.2] will give 50% one pax, 30% 2 pax, etc"""

    prob_saturday_night: float | None = None
    """Probability that the customer has a R/T itinerary with a Saturday night stay.
       Using this for choice modeling and CP experiments"""

    prob_num_days: list[float] = []
    """Probability of durations.
       [0.1, 0.3, 0.4, 0.2] will have durations of 1, 2, 3, 4 days
       and probability of each is specified explicitly
       Using this for choice modeling and CP experiments"""

    deterministic: bool = False
    """Whether the total amount of demand generated in each sample should be constant.

    If this is true, there will be no variance in the total demand generated,
    Which will be equal to the base demand modified by any market multipliers.
    There still can be some randomness in the timeframe arrival distribution of
    this demand, but the total demand generated in each sample will be the same.
    """

    overrides: list[DemandOverride] = []
    """Used for some specialized tests.
       Each dictionary should have 'carrier', 'discount_pct' and 'pref_adj'"""

    @field_validator("overrides", "prob_num_days", mode="before")
    @classmethod
    def _accept_strings(cls, v: list | str) -> Any:
        """Parse list-valued fields that arrive as a serialized string.

        Pydantic validators run *before* type coercion when ``mode="before"``
        is set.  This validator allows the ``overrides`` and ``prob_num_days``
        fields to be specified as a Python literal string (e.g. from a CSV or
        database column) in addition to the normal list form.

        Parameters
        ----------
        v : list or str
            The raw field value supplied by the caller.  If it is a
            :class:`str`, it is parsed with :func:`ast.literal_eval`;
            otherwise it is returned unchanged.

        Returns
        -------
        list or any
            The parsed value.  In normal usage this will be a list, but
            :func:`ast.literal_eval` may return any Python literal type if
            the input string represents a non-list literal.

        Raises
        ------
        ValueError
            If ``v`` is a string that cannot be parsed as a Python literal.
        """
        if isinstance(v, str):
            v = ast.literal_eval(v)
        return v

    @field_serializer("overrides", "prob_num_days")
    def _serialize_overrides(self, v: list) -> list[str]:
        """Serialize list-valued fields to a list of strings.

        Pydantic calls this serializer when converting the model to a
        dictionary or JSON.  Each element is converted to its string
        representation so that the round-trip through
        :meth:`_accept_strings` is lossless.

        Parameters
        ----------
        v : list
            The list of values to serialize.  Elements may be
            :class:`~pydantic.BaseModel` instances (serialized via
            :meth:`~pydantic.BaseModel.model_dump`) or plain scalars
            (converted with :class:`str`).

        Returns
        -------
        list of str
            A list where every element has been converted to a string.
        """
        return [str(o.model_dump() if isinstance(o, BaseModel) else str(o)) for o in v]

    @property
    def choice_model_(self) -> str:
        """Effective choice model name, falling back to segment name if not set.

        Returns the explicitly configured :attr:`choice_model` when present,
        otherwise falls back to :attr:`segment` so that callers always receive
        a non-``None`` model name.

        Returns
        -------
        str
            The name of the choice model to use for this demand.
        """
        return self.choice_model or self.segment

    @field_validator("curve", mode="before")
    @classmethod
    def _curve_integer_name(cls, v: int | str | None) -> str | None:
        """Coerce integer booking-curve names to strings.

        Booking curves are identified by string keys, but configuration files
        sometimes provide integer values (e.g. ``curve: 1``).  This validator
        converts any integer value to its string equivalent so downstream code
        can rely on a uniform type.

        Parameters
        ----------
        v : int, str, or None
            The raw value supplied for the ``curve`` field.

        Returns
        -------
        str or None
            The curve name as a string, or ``None`` if no curve was specified.
        """
        if isinstance(v, int):
            v = str(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def _migrate_reference_fare(cls, data: Any) -> Any:
        """Accept the legacy ``reference_fare`` key as an alias for ``reference_price``.

        Provides backward compatibility for configuration files and code that
        still use the old ``reference_fare`` field name.

        If a caller supplies ``reference_fare`` (and not ``reference_price``),
        the value is moved to ``reference_price`` and a
        :class:`DeprecationWarning` is emitted.  If both are supplied,
        ``reference_price`` wins and the legacy key is discarded with a
        warning.

        Parameters
        ----------
        data : dict or any
            The raw input data passed to the model constructor.  When it is a
            :class:`dict`, the validator inspects and potentially mutates it;
            non-dict values are returned unchanged.

        Returns
        -------
        dict or any
            The (possibly modified) input data with ``reference_fare`` removed
            and its value promoted to ``reference_price`` when applicable.
        """
        if isinstance(data, dict) and "reference_fare" in data:
            warnings.warn(
                _REFERENCE_FARE_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            legacy = data.pop("reference_fare")
            if "reference_price" not in data:
                data["reference_price"] = legacy
        return data

    def __getattr__(self, item: str) -> Any:
        """Route access to the deprecated ``reference_fare`` attribute to ``reference_price``.

        :meth:`__getattr__` is only invoked when the attribute is not found
        through the normal lookup mechanism, so this does not shadow real
        Pydantic fields.

        Parameters
        ----------
        item : str
            Name of the attribute being accessed.

        Returns
        -------
        any
            The value of ``reference_price`` when ``item`` is
            ``"reference_fare"``; otherwise the result of the parent class's
            ``__getattr__`` implementation.

        Raises
        ------
        AttributeError
            If ``item`` is not ``"reference_fare"`` and is not found by the
            parent :class:`~pydantic.BaseModel` implementation.
        """
        if item == "reference_fare":
            warnings.warn(
                _REFERENCE_FARE_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            return self.reference_price
        # Delegate to Pydantic's default __getattr__ behavior for anything else.
        # BaseModel.__getattr__ exists at runtime but is absent from type stubs.
        return super().__getattr__(item)  # type: ignore[attr-defined]

    def __setattr__(self, key: str, value: Any) -> None:
        """Route assignment to the deprecated ``reference_fare`` attribute to ``reference_price``.

        Intercepts attribute assignment so that code using the old
        ``reference_fare`` name continues to work while emitting a
        :class:`DeprecationWarning`.

        Parameters
        ----------
        key : str
            Name of the attribute being set.  When equal to
            ``"reference_fare"``, it is silently redirected to
            ``"reference_price"`` after emitting a warning.
        value : any
            The value to assign.
        """
        if key == "reference_fare":
            warnings.warn(
                _REFERENCE_FARE_DEPRECATION_MSG,
                DeprecationWarning,
                stacklevel=2,
            )
            key = "reference_price"
        super().__setattr__(key, value)


def assign_standard_todd_curves(cfg: Config) -> Config:
    """Assign standard TODD curves to all demands that do not have one set.

    For each :class:`Demand` in *cfg* whose :attr:`~Demand.todd_curve` is
    ``None``, this function looks up the associated market's ``delta_t`` and
    assigns the matching ``Standard_TODD_Curve_<delta_t>`` curve.  Any
    standard curves that are not already present in *cfg* are loaded from the
    bundled ``standard-todd.yaml`` demo network and added to
    :attr:`~Config.todd_curves`.

    If :attr:`~SimulationControls.use_standard_todd_curves` is ``False`` on
    *cfg*, the function returns *cfg* unchanged.

    Parameters
    ----------
    cfg : Config
        The simulation configuration to update in-place.

    Returns
    -------
    Config
        The same *cfg* object, with TODD curves assigned where they were
        missing.
    """

    if not cfg.simulation_controls.use_standard_todd_curves:
        # if disabled, do nothing
        return cfg

    todd_curve_queue = set()

    # for each demand, check if it has a defined TODD curve.
    # if not, assign the Standard_TODD_Curve based on the mkt.delta_t
    for dmd in cfg.demands:
        if dmd.todd_curve is None:
            mkt = cfg.markets_dict[f"{dmd.orig}~{dmd.dest}"]
            todd_curve = f"Standard_TODD_Curve_{mkt.delta_t:02d}"
            dmd.todd_curve = todd_curve
            if todd_curve not in cfg.todd_curves:
                todd_curve_queue.add(todd_curve)
        elif dmd.todd_curve not in cfg.todd_curves and re.match(r"Standard_TODD_Curve_[0-9][0-9]", dmd.todd_curve):
            todd_curve_queue.add(dmd.todd_curve)

    # load all required standard configs
    if todd_curve_queue:
        from passengersim import Config, demo_network

        std_cfg = Config.from_yaml(demo_network("standard-todd.yaml"))
        for q in todd_curve_queue:
            cfg.todd_curves[q] = std_cfg.todd_curves[q].model_copy(deep=True)

    return cfg


def _assign_standard_dwm_tolerances(
    cfg: Config,
    segment_mapping: dict[str, str] | None = None,
) -> Config:
    """Assign standard Decision Window Model tolerances to demands that lack one.

    For each :class:`Demand` in *cfg* whose :attr:`~Demand.dwm_tolerance` is
    falsy (zero or ``None``), a tolerance value is selected from a lookup
    table indexed by segment type and O-D distance.  Segment names are first
    translated through *segment_mapping*; any name not found in the table is
    treated as ``"leisure"``.

    If :attr:`~SimulationControls.use_standard_todd_curves` is ``False`` on
    *cfg*, the function returns *cfg* unchanged (the same flag controls both
    TODD curves and DWM tolerances).

    Parameters
    ----------
    cfg : Config
        The simulation configuration to update in-place.
    segment_mapping : dict of {str: str}, optional
        A mapping from segment names used in *cfg* to the canonical segment
        names recognized by the lookup table (``"business"`` or
        ``"leisure"``).  Segments absent from the mapping are used as-is,
        and any segment not found in the table falls back to ``"leisure"``.
        Defaults to an empty mapping (i.e. no translation).

    Returns
    -------
    Config
        The same *cfg* object, with DWM tolerances filled in where they were
        missing.
    """

    if not cfg.simulation_controls.use_standard_todd_curves:
        # if disabled, do nothing
        return cfg

    if segment_mapping is None:
        segment_mapping = {}

    std_tols = {
        "business": [1.240, 3.318, 3.544, 3.765, 3.971, 4.159, 5.356, 10.5, 10.78, 10.96, 24],
        "leisure": [2.034, 4.617, 4.936, 5.245, 5.536, 5.801, 6.379, 12.7, 13.04, 13.327, 24],
    }
    std_miles = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, np.inf]

    for dmd in cfg.demands:
        if not dmd.dwm_tolerance:
            segment = segment_mapping.get(dmd.segment, dmd.segment)
            if segment not in std_tols:
                segment = "leisure"
            mile_category = np.searchsorted(std_miles, dmd.distance or 0)
            dmd.dwm_tolerance = std_tols[segment][mile_category]

    return cfg
