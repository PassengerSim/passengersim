# TITLE: Choice Models
from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal

from pydantic import Field, SerializeAsAny, field_validator, model_validator

from .named import Named

TwoFloats = SerializeAsAny[tuple[float, float] | None]


class CommonChoiceModel(Named, extra="forbid"):
    """A common base class for choice models.

    Defines restrictions and other parameters that are common to all
    choice models."""

    restrictions: dict[str, float] | None = None

    @property
    def r1(self):
        """Restriction 1.
        This is deprecated in favor of the `restrictions` dictionary."""
        return self.restrictions.get("r1", None)

    @r1.setter
    def r1(self, value: float):
        self.restrictions["r1"] = value

    @property
    def r2(self):
        """Restriction 2.
        This is deprecated in favor of the `restrictions` dictionary."""
        return self.restrictions.get("r2", None)

    @r2.setter
    def r2(self, value: float):
        self.restrictions["r2"] = value

    @property
    def r3(self):
        """Restriction 3.
        This is deprecated in favor of the `restrictions` dictionary."""
        return self.restrictions.get("r3", None)

    @r3.setter
    def r3(self, value: float):
        self.restrictions["r3"] = value

    @property
    def r4(self):
        """Restriction 4.
        This is deprecated in favor of the `restrictions` dictionary."""
        return self.restrictions.get("r4", None)

    @r4.setter
    def r4(self, value: float):
        self.restrictions["r4"] = value

    @property
    def r5(self):
        """Restriction 5.
        This is deprecated in favor of the `restrictions` dictionary."""
        return self.restrictions.get("r5", None)

    @r5.setter
    def r5(self, value: float):
        self.restrictions["r5"] = value

    @model_validator(mode="before")
    @classmethod
    def _named_restrictions(cls, data: Any) -> Any:
        restricts = data.get("restrictions", {})
        if "r1" in data:
            restricts["r1"] = data.pop("r1")
        if "r2" in data:
            restricts["r2"] = data.pop("r2")
        if "r3" in data:
            restricts["r3"] = data.pop("r3")
        if "r4" in data:
            restricts["r4"] = data.pop("r4")
        data["restrictions"] = restricts
        return data


class PodsChoiceModel(CommonChoiceModel, extra="forbid"):
    kind: Literal["pods"]

    emult: float | None = None

    basefare_mult: float | None = Field(
        default=None,
        deprecated="basefare_mult is deprecated; use reference_price_multiplier instead.",
    )

    @field_validator("basefare_mult", mode="before")
    @classmethod
    def _warn_basefare_mult_is_bad(cls, v: Any) -> Any:
        """Raise a Warning whenever a value is provided for this field."""
        if v is not None:
            warnings.warn(
                "PodsChoiceModel.basefare_mult does not work correctly; use reference_price_multiplier instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        return v

    reference_price_multiplier: float | None = 1.0
    """The multiplier against the reference fare, which sets the pricing anchor point for this choice model.

    All customers using this choice model will have a maximum willingness to pay (WTP) of at least
    this multiple of the reference price.

    .. note::
        This field was previously named ``basefare_mult2``.  Configs that still use the old name
        will have their value silently migrated; specifying **both** names at the same time is an
        error.
    """

    @model_validator(mode="before")
    @classmethod
    def _migrate_basefare_mult2(cls, data: Any) -> Any:
        """Silently migrate the old ``basefare_mult2`` field name to ``reference_price_multiplier``.

        A ``None`` value for the old name is treated as "not specified" and is ignored.
        Raises ``ValueError`` if both the old and new names are supplied with non-None values
        at the same time.
        """
        if "basefare_mult2" in data:
            old_value = data.pop("basefare_mult2")
            if old_value is not None:
                if "reference_price_multiplier" in data and data["reference_price_multiplier"] is not None:
                    raise ValueError(
                        "Cannot specify both 'basefare_mult2' (deprecated) and "
                        "'reference_price_multiplier' at the same time; use 'reference_price_multiplier' only."
                    )
                # Only migrate when the new key is absent or explicitly None
                if "reference_price_multiplier" not in data or data["reference_price_multiplier"] is None:
                    data["reference_price_multiplier"] = old_value
        return data

    connect_disutility: float | None = None
    path_quality: TwoFloats = None
    airline_pref_pods: TwoFloats = None
    airline_pref_hhi: TwoFloats = None
    airline_pref_seat_share: TwoFloats = None
    elapsed_time: TwoFloats = None
    buffer_threshold: int | None = None
    buffer_time: TwoFloats = None
    tolerance: float | None = None
    non_stop_multiplier: float | None = None
    connection_multiplier: float | None = None

    # DWM info
    todd_curve: str | None = None
    early_dep: dict | None = None
    late_arr: dict | None = None
    replanning: TwoFloats = None

    # Ancillaries
    anc1_relevance: float | None = None
    anc2_relevance: float | None = None
    anc3_relevance: float | None = None
    anc4_relevance: float | None = None


class LogitChoiceModel(CommonChoiceModel, extra="forbid"):
    kind: Literal["logit"]

    emult: float | None = None
    """Using for WTP, a bit of a quick and dirty until we have a better approach"""

    anc1_relevance: float | None = None
    anc2_relevance: float | None = None
    anc3_relevance: float | None = None
    anc4_relevance: float | None = None

    intercept: float = 0
    """This is the alternative specific constant for the no-purchase alternative."""

    nonstop: float = 0
    duration: float = 0
    price: float = 0
    """This is the parameter for the price of each alternative."""

    tod_sin2p: float = 0
    r"""Schedule parameter.

    If $t$ is departure time (in minutes after midnight local time) divided
    by 1440, this parameter is multiplied by $sin( 2 \pi t)$ and the result is
    added to the utility of the particular alternative."""

    tod_sin4p: float = 0
    r"""Schedule parameter.

    If $t$ is departure time (in minutes after midnight local time) divided
    by 1440, this parameter is multiplied by $sin( 4 \pi t)$ and the result is
    added to the utility of the particular alternative."""

    tod_sin6p: float = 0
    tod_cos2p: float = 0
    tod_cos4p: float = 0
    tod_cos6p: float = 0
    free_bag: float = 0
    early_boarding: float = 0
    same_day_change: float = 0


ChoiceModel = Annotated[PodsChoiceModel | LogitChoiceModel, Field(discriminator="kind")]
"""
Two types of choice models are available in PassengerSim.

Use the `kind` key to select which kind of choice model you wish to parameterize.
"""
