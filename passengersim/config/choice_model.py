# TITLE: Choice Models
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, SerializeAsAny, model_validator

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

    basefare_mult: float | None = None
    basefare_mult2: float | None = 1.0
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
